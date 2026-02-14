#!/usr/bin/env python3
"""Ensure a HuggingFace model is downloaded into the local cache.

Run this once (with network access) before launching offline experiments.
Subsequent calls are a no-op if the model is already cached.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download a HF model into cache.")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name.")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = args.model
    print(f"[ensure_model_cached] Checking '{name}'…")

    # Try local first
    try:
        AutoTokenizer.from_pretrained(name, local_files_only=True)
        AutoModelForCausalLM.from_pretrained(name, local_files_only=True)
        print(f"[ensure_model_cached] '{name}' already in cache. Nothing to download.")
        return
    except (OSError, ValueError):
        pass

    # Download
    print(f"[ensure_model_cached] Downloading '{name}'…")
    AutoTokenizer.from_pretrained(name)
    AutoModelForCausalLM.from_pretrained(name)
    print(f"[ensure_model_cached] '{name}' cached successfully.")


if __name__ == "__main__":
    main()
