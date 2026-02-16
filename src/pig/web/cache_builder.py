"""Cache builder for the local interactive graph viewer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pig.model import ALLOWED_NODE_TYPES, create_model
from pig.patching import compute_patch_effects
from pig.prompts import PromptPair, create_ioi_dataset


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_node_types(node_types: Sequence[str]) -> None:
    invalid = set(node_types) - ALLOWED_NODE_TYPES
    if invalid:
        raise ValueError(
            "Unsupported node types: "
            f"{sorted(invalid)}. Allowed: {sorted(ALLOWED_NODE_TYPES)}"
        )


def _build_pairs(
    corruptions: Sequence[str], n_examples: int, seed: int
) -> list[PromptPair]:
    pairs: list[PromptPair] = []
    for index, corruption in enumerate(corruptions):
        corruption_seed = seed + index
        pairs.extend(
            create_ioi_dataset(
                n_examples=n_examples,
                corruption=corruption,
                seed=corruption_seed,
            )
        )
    return pairs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pig-viewer-cache",
        description="Generate patch-effect cache files for local web viewer",
    )
    parser.add_argument(
        "--model-name",
        default="toy_transformer",
        help="Model identifier (e.g. toy_transformer, gpt2, local model path)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/patch_effects",
        help="Output directory for patch-effect cache JSON files",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=16,
        help="Number of examples per corruption slice",
    )
    parser.add_argument(
        "--corruptions",
        default="name_swap,abba",
        help="Comma-separated IOI corruptions to include",
    )
    parser.add_argument(
        "--node-types",
        default="res",
        help="Comma-separated node types (res,mlp,att)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    corruptions = _parse_csv(args.corruptions)
    if not corruptions:
        raise SystemExit("At least one corruption must be provided")

    node_types = _parse_csv(args.node_types)
    if not node_types:
        raise SystemExit("At least one node type must be provided")

    _validate_node_types(node_types)

    if args.num_examples <= 0:
        raise SystemExit("--num-examples must be greater than zero")

    print(f"Loading model: {args.model_name}")
    model = create_model(model_name=args.model_name)

    pairs = _build_pairs(corruptions, args.num_examples, args.seed)
    print(
        "Building cache for "
        f"{len(pairs)} prompt pairs across {len(corruptions)} slices"
    )

    dataset = compute_patch_effects(
        model=model,
        prompt_pairs=pairs,
        cache_dir=args.cache_dir,
        show_progress=True,
        node_types=node_types,
    )
    print(
        "Cache generation completed: "
        f"{len(dataset)} tensors written to {args.cache_dir}"
    )


if __name__ == "__main__":
    main()
