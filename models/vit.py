"""Canonical ViT-B/16 checkpoint loaders for the supported pretraining setups."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import timm
import torch

MODEL_NAME = "vit_base_patch16_224.augreg_in21k"
IMAGENET_1K_MODEL_NAME = "vit_base_patch16_224"


@dataclass(frozen=True)
class CheckpointProvenance:
    model_name: str
    path: str
    sha256: str | None
    source_url: str | None
    timm_version: str

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_inputs(model: torch.nn.Module) -> None:
    """Normalise [0, 1] images with the checkpoint's own mean/std inside the
    model, so the data pipeline stays identical across backbones."""
    mean = torch.tensor(model.pretrained_cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(model.pretrained_cfg["std"]).view(1, 3, 1, 1)
    model.register_forward_pre_hook(
        lambda _m, inputs: ((inputs[0] - mean.to(inputs[0])) / std.to(inputs[0]),)
    )


def load_frozen_vit(
    checkpoint: str | Path | None = None,
    *,
    model_name: str = MODEL_NAME,
    expected_sha256: str | None = None,
    source_url: str | None = None,
    input_norm: bool = False,
) -> tuple[torch.nn.Module, CheckpointProvenance]:
    """Load the canonical backbone from timm, or an explicitly supplied file.

    ``input_norm`` applies the checkpoint's mean/std (off for the canonical
    supervised backbones, which the protocol feeds unnormalised [0, 1] images).
    """
    model_name = model_name or MODEL_NAME
    if checkpoint is None:
        if expected_sha256:
            raise ValueError(
                "A local checkpoint path is required to verify the expected SHA-256"
            )
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        if input_norm:
            _normalize_inputs(model)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, CheckpointProvenance(
            model_name=model_name,
            path=f"timm://{model_name}",
            sha256=None,
            source_url=source_url,
            timm_version=timm.__version__,
        )

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical checkpoint does not exist: {path}")
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch for {path}")

    state = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = (
        state.get("state_dict", state.get("model", state))
        if isinstance(state, dict)
        else state
    )
    if not isinstance(state_dict, dict):
        raise ValueError(
            "Checkpoint must be a tensor state dictionary or contain model/state_dict"
        )
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }

    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    incompatible = model.load_state_dict(state_dict, strict=False)
    disallowed = [
        key
        for key in (*incompatible.missing_keys, *incompatible.unexpected_keys)
        if not key.startswith(("head.", "fc."))
    ]
    if disallowed:
        raise ValueError(
            f"Checkpoint is not compatible with {model_name}: {disallowed[:5]}"
        )
    if input_norm:
        _normalize_inputs(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, CheckpointProvenance(
        model_name=model_name,
        path=str(path.resolve()),
        sha256=actual_sha256,
        source_url=source_url,
        timm_version=timm.__version__,
    )
