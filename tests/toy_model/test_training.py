"""Unit tests for toy model training utilities."""

import pytest
import torch

from pig.toy_model import TinyTrainingConfig, TinyTransformerConfig, ToyHookedModel, train_toy_model
from pig.toy_model.trainer import main as train_main


def _make_model() -> ToyHookedModel:
    return ToyHookedModel(
        config=TinyTransformerConfig(
            vocab_size=257,
            max_seq_len=96,
            d_model=32,
            n_layers=2,
            n_heads=4,
            mlp_dim=64,
            seed=3,
        ),
        device="cpu",
        freeze_weights=True,
    )


def test_train_toy_model_updates_parameters():
    model = _make_model()
    before = model.lm_head.weight.detach().clone()

    history = train_toy_model(
        model,
        texts=[
            "Alice gave the book to Bob .",
            "Bob gave the book to Alice .",
            "Carol sent a letter to Dan .",
            "Dan replied to Carol .",
        ],
        config=TinyTrainingConfig(epochs=2, learning_rate=1e-2, seed=11),
    )

    after = model.lm_head.weight.detach()

    assert len(history) == 2
    assert all(item.mean_loss > 0.0 for item in history)
    assert not torch.equal(before, after)


def test_train_toy_model_requires_valid_sequences():
    model = _make_model()

    with pytest.raises(ValueError, match="No valid training sequences"):
        train_toy_model(
            model,
            texts=["hello"],
            config=TinyTrainingConfig(epochs=1),
        )


def test_training_cli_main_saves_checkpoint(tmp_path):
    data_file = tmp_path / "toy_corpus.txt"
    data_file.write_text(
        "Alice gave the book to Bob .\n"
        "Bob gave the book to Alice .\n",
        encoding="utf-8",
    )
    save_path = tmp_path / "toy_model.pt"

    exit_code = train_main(
        [
            "--data-file",
            str(data_file),
            "--save-path",
            str(save_path),
            "--epochs",
            "1",
            "--d-model",
            "32",
            "--n-layers",
            "2",
            "--n-heads",
            "4",
            "--mlp-dim",
            "64",
        ]
    )

    assert exit_code == 0
    assert save_path.is_file()
