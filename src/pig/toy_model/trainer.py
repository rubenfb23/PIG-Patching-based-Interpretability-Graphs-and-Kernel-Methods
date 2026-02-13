"""Training utilities for `ToyHookedModel`."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as F

from pig.toy_model.config import TinyTrainingConfig, TinyTransformerConfig
from pig.toy_model.model import ToyHookedModel


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated metrics for one training epoch."""

    epoch: int
    mean_loss: float
    num_sequences: int


def _build_next_token_sequences(
    model: ToyHookedModel,
    texts: Iterable[str],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    sequences: list[tuple[torch.Tensor, torch.Tensor]] = []
    for text in texts:
        token_ids = model.tokenize(text)[0]
        if token_ids.numel() < 2:
            continue
        input_ids = token_ids[:-1].unsqueeze(0)
        labels = token_ids[1:].unsqueeze(0)
        sequences.append((input_ids, labels))
    return sequences


def train_toy_model(
    model: ToyHookedModel,
    texts: Iterable[str],
    config: TinyTrainingConfig | None = None,
) -> list[EpochMetrics]:
    """Train the toy model with a next-token objective over raw text strings."""
    cfg = config or TinyTrainingConfig()

    sequences = _build_next_token_sequences(model, texts)
    if not sequences:
        raise ValueError("No valid training sequences (need at least 2 tokens per text)")

    model.set_trainable(True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history: list[EpochMetrics] = []
    base_indices = list(range(len(sequences)))

    for epoch in range(cfg.epochs):
        indices = base_indices.copy()
        if cfg.shuffle:
            random.Random(cfg.seed + epoch).shuffle(indices)

        epoch_loss = 0.0
        for idx in indices:
            input_ids, labels = sequences[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model.logits(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )
            loss.backward()

            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            optimizer.step()
            epoch_loss += float(loss.item())

        history.append(
            EpochMetrics(
                epoch=epoch + 1,
                mean_loss=epoch_loss / float(len(indices)),
                num_sequences=len(indices),
            )
        )

    model.eval()
    return history


def load_text_corpus(data_file: str | Path) -> list[str]:
    """Load a plain-text corpus (one training sample per non-empty line)."""
    path = Path(data_file)
    if not path.is_file():
        raise FileNotFoundError(f"Training corpus not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    corpus = [line.strip() for line in lines if line.strip()]
    if not corpus:
        raise ValueError(f"No non-empty lines found in training corpus: {path}")
    return corpus


def _build_parser() -> argparse.ArgumentParser:
    default_model_cfg = TinyTransformerConfig()
    default_train_cfg = TinyTrainingConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Train ToyHookedModel from a plain-text corpus (one sample per line)."
        )
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to training text file (one sample per line).",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional path to save checkpoint (.pt).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example 'cpu' or 'cuda'.",
    )

    parser.add_argument("--vocab-size", type=int, default=default_model_cfg.vocab_size)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=default_model_cfg.max_seq_len,
    )
    parser.add_argument("--d-model", type=int, default=default_model_cfg.d_model)
    parser.add_argument("--n-layers", type=int, default=default_model_cfg.n_layers)
    parser.add_argument("--n-heads", type=int, default=default_model_cfg.n_heads)
    parser.add_argument("--mlp-dim", type=int, default=default_model_cfg.mlp_dim)
    parser.add_argument("--seed", type=int, default=default_model_cfg.seed)

    parser.add_argument("--epochs", type=int, default=default_train_cfg.epochs)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=default_train_cfg.learning_rate,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=default_train_cfg.weight_decay,
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=default_train_cfg.grad_clip_norm,
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable per-epoch shuffling.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for training the toy model from text."""
    args = _build_parser().parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no CUDA device is available.")

    corpus = load_text_corpus(args.data_file)

    model_cfg = TinyTransformerConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_dim=args.mlp_dim,
        seed=args.seed,
    )
    train_cfg = TinyTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )

    model = ToyHookedModel(
        config=model_cfg,
        device=args.device,
        freeze_weights=False,
    )
    history = train_toy_model(model=model, texts=corpus, config=train_cfg)

    print(
        f"Training complete: epochs={len(history)}, "
        f"samples={len(corpus)}, final_loss={history[-1].mean_loss:.4f}"
    )
    for metric in history:
        print(
            f"  epoch={metric.epoch} "
            f"mean_loss={metric.mean_loss:.4f} "
            f"sequences={metric.num_sequences}"
        )

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_cfg),
                "training_config": asdict(train_cfg),
                "history": [asdict(item) for item in history],
            },
            save_path,
        )
        print(f"Saved checkpoint: {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
