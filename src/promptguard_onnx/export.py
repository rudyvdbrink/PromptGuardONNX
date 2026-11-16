"""Export Meta's Llama Prompt Guard 2 classifiers to ONNX.

The pipeline reproduces the gravitee-io export (Hugging Face revision da68d0f):

1. Download the model and tokenizer and re-save them with ``save_pretrained``.
2. Export ``model.onnx`` with the TorchScript exporter at opset 17.
3. Run the onnxoptimizer fuse and elimination passes on a temporary copy.
4. Dynamically quantize the optimized graph to ``model.quant.onnx``
   (per-tensor, unsigned 8-bit weights, for portability across CPUs).
5. Write the BENIGN / MALICIOUS labels into ``config.json``.
6. Add the Llama 4 license, the required NOTICE and a model card.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import onnx
import onnxoptimizer
import torch
from huggingface_hub import auth_check, hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from promptguard_onnx import DEFAULT_OUTPUT_DIR, MODELS, SourceModel
from promptguard_onnx.model_card import render_model_card

OPSET_VERSION = 17
LABELS = ("BENIGN", "MALICIOUS")
NOTICE_TEXT = (
    "Llama 4 is licensed under the Llama 4 Community License, "
    "Copyright © Meta Platforms, Inc. All Rights Reserved."
)


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding the environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def has_access(source: SourceModel, token: str | None) -> bool:
    """Return True if the token can read the (gated) source repository."""
    try:
        auth_check(source.repo_id, token=token)
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        print(f"[!] No access to {source.repo_id}, skipping ({type(exc).__name__}).")
        return False
    return True


def save_hf_model(source: SourceModel, model_dir: Path, token: str | None) -> None:
    print(f"[*] Downloading {source.repo_id}@{source.revision[:7]}...")
    tokenizer = AutoTokenizer.from_pretrained(source.repo_id, revision=source.revision, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        source.repo_id, revision=source.revision, token=token
    )
    print(f"[*] Saving model and tokenizer to {model_dir}...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    # transformers writes JSON in text mode, which gives CRLF line endings on Windows.
    for path in model_dir.glob("*.json"):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def export_onnx(model_dir: Path) -> Path:
    print("[*] Exporting model.onnx...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer("Hello, World!", return_tensors="pt")
    onnx_path = model_dir / "model.onnx"
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=OPSET_VERSION,
        dynamo=False,
    )
    return onnx_path


def optimize_onnx(onnx_path: Path, optimized_path: Path) -> None:
    print("[*] Optimizing graph (fuse and elimination passes)...")
    model = onnx.load(str(onnx_path))
    optimized = onnxoptimizer.optimize(model, onnxoptimizer.get_fuse_and_elimination_passes())
    onnx.save(optimized, str(optimized_path))


def quantize_onnx(onnx_path: Path, quant_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        optimized_path = Path(tmp) / "model.optim.onnx"
        optimize_onnx(onnx_path, optimized_path)
        print("[*] Quantizing model.quant.onnx (dynamic, per-tensor, QUInt8)...")
        quantize_dynamic(
            optimized_path,
            quant_path,
            per_channel=False,
            weight_type=QuantType.QUInt8,
        )


def update_config_labels(model_dir: Path) -> None:
    print("[*] Writing labels to config.json...")
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["id2label"] = {str(i): label for i, label in enumerate(LABELS)}
    config["label2id"] = {label: i for i, label in enumerate(LABELS)}
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8", newline="\n")


def write_license_files(source: SourceModel, model_dir: Path, token: str | None) -> None:
    print("[*] Writing LICENSE, NOTICE and README.md...")
    license_path = hf_hub_download(source.repo_id, "LICENSE", revision=source.revision, token=token)
    (model_dir / "LICENSE").write_bytes(Path(license_path).read_bytes())
    (model_dir / "NOTICE").write_text(NOTICE_TEXT, encoding="utf-8", newline="\n")
    readme_path = model_dir / "README.md"
    if readme_path.exists():
        print("[*] Keeping the existing README.md.")
    else:
        readme_path.write_text(render_model_card(source), encoding="utf-8", newline="\n")


def export_model(source: SourceModel, output_dir: Path, token: str | None) -> Path:
    model_dir = output_dir / source.export_name
    model_dir.mkdir(parents=True, exist_ok=True)

    save_hf_model(source, model_dir, token)
    onnx_path = export_onnx(model_dir)
    quantize_onnx(onnx_path, model_dir / "model.quant.onnx")
    update_config_labels(model_dir)
    write_license_files(source, model_dir, token)

    print(f"[+] Done: {model_dir}")
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS),
        default=sorted(MODELS),
        help="Model sizes to export. Models without access are skipped.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--env-file", type=Path, default=Path(".env"), help="File that holds HF_TOKEN."
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[!] HF_TOKEN is not set. The Meta repositories are gated.", file=sys.stderr)

    exported = [
        export_model(MODELS[size], args.output_dir, token)
        for size in args.models
        if has_access(MODELS[size], token)
    ]
    if not exported:
        sys.exit("[!] Nothing was exported.")


if __name__ == "__main__":
    main()
