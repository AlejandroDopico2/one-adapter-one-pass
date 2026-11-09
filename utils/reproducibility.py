"""Deterministic setup for protocol-validation runs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parse_device(value: str | int) -> torch.device:
    text = str(value).strip().lower()
    if text == "cpu":
        return torch.device("cpu")
    if text.isdigit():
        text = f"cuda:{text}"
    elif text == "cuda":
        text = "cuda:0"
    device = torch.device(text)
    if device.type != "cuda":
        raise ValueError(f"Unknown device: {value}")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    index = 0 if device.index is None else device.index
    visible = torch.cuda.device_count()
    if index >= visible:
        raise ValueError(f"GPU {index} requested but only {visible} visible")
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    return device
