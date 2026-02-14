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
    """Load causal LM model + tokenizer and place model on target device.

    When *local_files_only* is ``False`` (default), the function first tries to
    load from the local HuggingFace cache.  Only if that fails it falls back to
    downloading the model — so repeated runs never hit the network.
    """
    resolved_device = resolve_device(device)

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        effective_local = True
    else:
        # Try local cache first; download only on miss.
        effective_local = True  # optimistic

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=effective_local,
        )
    except (OSError, ValueError):
        if local_files_only:
            raise  # user explicitly asked for offline — propagate error
        print(f"[gpt2_loader] Model '{model_name}' not in local cache — downloading…")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch_dtype,
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=effective_local,
            **load_kwargs,
        )
    except (OSError, ValueError):
        if local_files_only:
            raise
        print(f"[gpt2_loader] Model weights '{model_name}' not in local cache — downloading…")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=False,
            **load_kwargs,
        )

    model.to(resolved_device)
    return model, tokenizer, resolved_device
