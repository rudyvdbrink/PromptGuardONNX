# PromptGuardONNX

Exports Meta's [Llama Prompt Guard 2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M)
classifiers (22M and 86M) to ONNX. 

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
```

This compares the logits of both ONNX files with the PyTorch model on a set of benign and
malicious prompts. It fails if the fp32 graph deviates by more than 1e-4 in probability.

## Upload

```sh
uv run promptguard-upload                          
```
If you want to push your model to huggingface hub, your huggingface token needs write access. 



