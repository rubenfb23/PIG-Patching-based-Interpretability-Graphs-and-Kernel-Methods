#!/usr/bin/env python3
"""Compare model memory footprints for PIG models."""

from __future__ import annotations

import argparse

from pig.model import create_model


def model_memory_stats(model_name: str, device: str) -> dict[str, float]:
    model = create_model(model_name=model_name, device=device)
    wrapped = getattr(model, "model", model)

    param_count = sum(param.numel() for param in wrapped.parameters())
    param_bytes = sum(param.numel() * param.element_size() for param in wrapped.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in wrapped.buffers())

    return {
        "parameters": float(param_count),
        "parameter_bytes": float(param_bytes),
        "buffer_bytes": float(buffer_bytes),
        "total_mb": (param_bytes + buffer_bytes) / (1024**2),
    }


def _format_count(value: float) -> str:
    return f"{int(value):,}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare memory usage for two PIG models",
    )
    parser.add_argument("--model-a", default="gpt2")
    parser.add_argument("--model-b", default="toy_transformer")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    first = model_memory_stats(args.model_a, args.device)
    second = model_memory_stats(args.model_b, args.device)

    print(f"Model A: {args.model_a}")
    print(f"  parameters: {_format_count(first['parameters'])}")
    print(f"  parameter_bytes: {_format_count(first['parameter_bytes'])}")
    print(f"  buffer_bytes: {_format_count(first['buffer_bytes'])}")
    print(f"  total_model_memory_mb(fp32): {first['total_mb']:.2f}")

    print(f"Model B: {args.model_b}")
    print(f"  parameters: {_format_count(second['parameters'])}")
    print(f"  parameter_bytes: {_format_count(second['parameter_bytes'])}")
    print(f"  buffer_bytes: {_format_count(second['buffer_bytes'])}")
    print(f"  total_model_memory_mb(fp32): {second['total_mb']:.2f}")

    ratio = first["parameter_bytes"] / second["parameter_bytes"]
    print(f"ratio_param_memory_{args.model_a}_vs_{args.model_b}: {ratio:.1f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
