# PromptGuardONNX

Exports Meta's [Llama Prompt Guard 2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M)
classifiers (22M and 86M) to ONNX. The output reproduces
[gravitee-io/Llama-Prompt-Guard-2-22M-onnx](https://huggingface.co/gravitee-io/Llama-Prompt-Guard-2-22M-onnx/tree/da68d0f6023c7aeaf6b256eec549de295d5e8740)
(revision `da68d0f`) and
[gravitee-io/Llama-Prompt-Guard-2-86M-onnx](https://huggingface.co/gravitee-io/Llama-Prompt-Guard-2-86M-onnx/tree/45a05fbd5337a864edc608f994911f009c37ca57)
(revision `45a05fb`) byte for byte. This applies to `model.onnx`, `model.quant.onnx`,
`model.safetensors`, the config and the tokenizer files. Only `README.md` is different.
`LICENSE` is Meta's file verbatim.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The Meta repositories are gated, so put a
Hugging Face token with access in `.env`:

```
HF_TOKEN=hf_...
```

```sh
uv sync
```

## Export

```sh
uv run promptguard-export                 # both sizes; sizes without access are skipped
uv run promptguard-export --models 22M    # one size
```

Each model is written to `models/Llama-Prompt-Guard-2-<size>-onnx/`:

| File | Description |
|------|-------------|
| `model.onnx` | fp32 graph, opset 17, dynamic `batch_size` and `sequence_length`. |
| `model.quant.onnx` | Dynamic quantization, per-tensor, `QUInt8` weights. Lossy, see below. |
| `model.safetensors`, `config.json` | PyTorch weights and config (labels `BENIGN`, `MALICIOUS`). |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | Tokenizer. |
| `README.md`, `LICENSE`, `NOTICE` | Model card and the Llama 4 license files. |

The source commits of the Meta models are pinned in `src/promptguard_onnx/__init__.py`.

## Verify

```sh
uv run promptguard-verify
uv run promptguard-verify models/Llama-Prompt-Guard-2-22M-onnx --reference path/to/other-export
```

This compares the logits of both ONNX files with the PyTorch model on a set of benign and
malicious prompts. It fails if the fp32 graph deviates by more than 1e-4 in probability.

## Upload

```sh
uv run promptguard-upload                          # every export, public, to the token owner
uv run promptguard-upload --namespace my-org --private models/Llama-Prompt-Guard-2-22M-onnx
```

The token needs write access. Each directory goes to `<namespace>/<directory name>`.
`promptguard-export` creates `README.md` only when it does not exist, so edits to a model
card survive a new export.

## Development

```sh
uv run ruff check
uv run ruff format
```

## Export pipeline

1. Download the model and tokenizer at a pinned commit, then re-save them with
   `save_pretrained`. JSON files are written with LF line endings on every platform.
2. `torch.onnx.export` with the TorchScript exporter at opset 17.
3. Apply the `onnxoptimizer` fuse and elimination passes to a temporary copy.
4. `quantize_dynamic(per_channel=False, weight_type=QuantType.QUInt8)` on the optimized copy.
5. Write the label names into `config.json` and add the license files and model card.

### Why these versions

The pins in `pyproject.toml` are the toolchain that reproduces the reference files. They are
not the versions in the reference GitHub `requirements.txt`, because those do not reproduce
the published files:

- `transformers==4.40.2`. Newer versions trace a different DeBERTa graph (4599 nodes instead
  of 4384). The output is the same, but the file is not.
- `onnxoptimizer==0.4.2`. Version 0.3.13 gives one tensor in `model.quant.onnx` a different
  name.
- The reference GitHub script quantizes per channel to `QInt8`. The published files use
  per-tensor `QUInt8`, which the reference changed for portability across CPUs.

### Quantization quality

`model.onnx` matches PyTorch to about 1e-7 in probability. `model.quant.onnx` is lossy. For
22M the labels agree on the test prompts. For 86M, 2 of the 11 test prompts get a different
label. The reference reports the same loss (86M accuracy 0.98 for fp32 and 0.90 for the
quantized model). Use `model.onnx` for 86M unless you have validated the quantized model.

## License

The code in this repository exports Llama Materials. The exported models are subject to the
Llama 4 Community License. Built with Llama.
