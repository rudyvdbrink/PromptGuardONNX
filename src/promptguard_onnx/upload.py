"""Upload exported model directories to the Hugging Face Hub."""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from promptguard_onnx import DEFAULT_OUTPUT_DIR
from promptguard_onnx.export import load_dotenv


def upload(api: HfApi, model_dir: Path, namespace: str, private: bool) -> str:
    repo_id = f"{namespace}/{model_dir.name}"
    print(f"[*] Uploading {model_dir} to {repo_id}...")
    url = api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=model_dir,
        commit_message=f"Upload {model_dir.name}",
    )
    print(f"[+] {url}")
    return repo_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dirs",
        nargs="*",
        type=Path,
        help=f"Exported model directories (default: every export under {DEFAULT_OUTPUT_DIR}/).",
    )
    parser.add_argument(
        "--namespace", help="User or organization to upload to (default: the token owner)."
    )
    parser.add_argument(
        "--private", action="store_true", help="Create new repositories as private."
    )
    parser.add_argument(
        "--env-file", type=Path, default=Path(".env"), help="File that holds HF_TOKEN."
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    namespace = args.namespace or api.whoami()["name"]

    model_dirs = args.model_dirs or sorted(
        path.parent for path in Path(DEFAULT_OUTPUT_DIR).glob("*/model.onnx")
    )
    if not model_dirs:
        sys.exit("[!] No exported models found.")

    for model_dir in model_dirs:
        upload(api, model_dir, namespace, args.private)


if __name__ == "__main__":
    main()
