"""ONNX export tooling for Meta's Llama Prompt Guard 2 classifiers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceModel:
    """A Meta Prompt Guard 2 checkpoint on the Hugging Face Hub, pinned to a commit."""

    repo_id: str
    revision: str

    @property
    def name(self) -> str:
        return self.repo_id.split("/")[1]

    @property
    def export_name(self) -> str:
        return f"{self.name}-onnx"


MODELS: dict[str, SourceModel] = {
    "22M": SourceModel(
        "meta-llama/Llama-Prompt-Guard-2-22M", "11614a155199674a0a95e6602d6ab0417b790ed0"
    ),
    "86M": SourceModel(
        "meta-llama/Llama-Prompt-Guard-2-86M", "a8ded8e697ce7c355e395a0df51f94adb4a2fd27"
    ),
}

DEFAULT_OUTPUT_DIR = "models"
