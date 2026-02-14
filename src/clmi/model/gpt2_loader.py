"""GPT-2 loading helpers."""

from __future__ import annotations

import os
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Avoid background online safetensors-conversion checks in offline/repro settings.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


DeviceOption = Literal["auto", "cpu", "cuda"]


def resolve_device(device: DeviceOption | str = "auto") -> str:
    """Resolve concrete device string from user choice."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return str(device)


def load_model_tokenizer(
    model_name: str = "gpt2",
    device: DeviceOption | str = "auto",
    torch_dtype: torch.dtype | None = None,
    local_files_only: bool = False,
):
    """Load causal LM model + tokenizer and place model on target device."""
    resolved_device = resolve_device(device)
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "local_files_only": local_files_only,
    }
    try:
        # Prefer legacy .bin weights first (works for tiny/local checkpoints).
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=False,
            **load_kwargs,
        )
    except OSError as exc:
        # Fallback for models distributed only as safetensors (e.g. current gpt2 hub files).
        msg = str(exc)
        if "pytorch_model.bin" in msg or "model.safetensors" in msg:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_safetensors=True,
                **load_kwargs,
            )
        else:
            raise
    model.to(resolved_device)
    return model, tokenizer, resolved_device
