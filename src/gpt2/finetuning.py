"""Optimized GPT-2 fine-tuning with torchrun/DDP.

Example (this machine has 4x RTX 3090):

    torchrun --standalone --nproc_per_node=4 \
      -m gpt2.finetuning \
      --data-path data/corpus.txt \
      --output-dir outputs/gpt2_finetune \
      --model-name gpt2 \
      --seq-len 1024 \
      --epochs 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sysconfig
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
PACKING_VERSION = 3


@dataclass(frozen=True)
class DistContext:
    """Distributed execution context."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device


@dataclass(frozen=True)
class StructuredEvalExample:
    """One structured example for epoch-wise generation comparison."""

    dataset_index: int
    prompt: str
    gold_final: str


@dataclass(frozen=True)
class GenerationEvalResult:
    """Aggregated generation metrics on a fixed evaluation subset."""

    parsed: int
    correct: int
    accuracy: float
    parsed_rate: float
    predictions: list[str]
    correct_flags: list[bool]


@dataclass(frozen=True)
class PromptCompletionSample:
    """A structured supervised sample with prompt and completion fields."""

    prompt: str
    completion: str
    fallback_text: str


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 with DDP/torchrun and hardware-aware defaults."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to dataset (.txt/.jsonl/.json).",
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="Field name containing text when using .jsonl/.json input.",
    )
    parser.add_argument(
        "--response-only-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When prompt/completion fields are available, mask prompt tokens and "
            "optimize loss only on completion tokens."
        ),
    )
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--output-dir", default="outputs/gpt2_finetune")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--test-ratio", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="If > 0, stop after this many optimizer steps.",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=-1,
        help="Per-GPU micro batch. -1 means auto-tune from VRAM and seq-len.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=-1,
        help="Gradient accumulation steps. -1 means auto from target tokens.",
    )
    parser.add_argument(
        "--target-global-batch-tokens",
        type=int,
        default=-1,
        help=(
            "Target tokens per optimizer step across all GPUs. "
            "-1 means hardware-aware default."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers per process. -1 auto-tunes by CPU/world size.",
    )

    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fused AdamW on CUDA when available.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.compile when possible.",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
        help="torch.compile mode.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="If > 0, cap validation batches for faster eval.",
    )
    parser.add_argument(
        "--compare-base-on-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare test metrics against an unfine-tuned base model.",
    )
    parser.add_argument(
        "--base-model-name",
        default="",
        help="Base model used for test comparison. Defaults to --model-name.",
    )
    parser.add_argument(
        "--test-eval-every-epochs",
        type=int,
        default=1,
        help="Run test comparison every N epochs (>=1).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop if validation does not improve for this many epochs (<=0 disables).",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation loss improvement to reset patience.",
    )
    parser.add_argument(
        "--early-stopping-warmup-epochs",
        type=int,
        default=1,
        help="Do not apply early stopping before this epoch count.",
    )
    parser.add_argument(
        "--wandb-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases logging on rank 0.",
    )
    parser.add_argument("--wandb-project", default="pig-finetune")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument(
        "--wandb-tags",
        default="finetune,gpt2,gsm8k",
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default="online",
        help="W&B mode.",
    )
    parser.add_argument(
        "--epoch-compare-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run generation comparison (original vs finetuned) every epoch.",
    )
    parser.add_argument(
        "--epoch-compare-num-examples",
        type=int,
        default=64,
        help="Number of held-out structured examples for epoch comparison.",
    )
    parser.add_argument(
        "--epoch-compare-batch-size",
        type=int,
        default=8,
        help="Batch size used during epoch generation comparison.",
    )
    parser.add_argument(
        "--epoch-compare-max-input-length",
        type=int,
        default=768,
        help="Max prompt tokens for epoch generation comparison.",
    )
    parser.add_argument(
        "--epoch-compare-max-new-tokens",
        type=int,
        default=192,
        help="Max generation tokens for epoch generation comparison.",
    )
    parser.add_argument(
        "--epoch-compare-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for epoch generation comparison.",
    )
    parser.add_argument(
        "--epoch-compare-top-p",
        type=float,
        default=0.95,
        help="Top-p for epoch generation comparison when sampling is enabled.",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bar on rank 0 during training.",
    )
    parser.add_argument("--save-every-steps", type=int, default=200)
    return parser.parse_args()


def setup_dist() -> DistContext:
    """Initialize distributed process group when launched with torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return DistContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_dist() -> None:
    """Destroy process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(ctx: DistContext) -> bool:
    """Return True on rank 0."""
    return ctx.rank == 0


def rank0_print(ctx: DistContext, msg: str) -> None:
    """Print only from rank 0."""
    if is_rank0(ctx):
        print(msg, flush=True)


def maybe_barrier(ctx: DistContext) -> None:
    """Synchronize all workers when in distributed mode."""
    if ctx.world_size > 1 and dist.is_initialized():
        dist.barrier()


def distributed_mean(value: float, device: torch.device) -> float:
    """Average a scalar across distributed workers."""
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def init_wandb_run(
    args: argparse.Namespace,
    ctx: DistContext,
    precision: str,
    micro_batch_size: int,
    grad_accum_steps: int,
    target_tokens: int,
) -> Any | None:
    """Initialize optional W&B run on rank 0."""
    if not args.wandb_enabled or not is_rank0(ctx):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `wandb`. Install with: uv pip install wandb"
        ) from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        group=args.wandb_group or None,
        tags=tags,
        mode=args.wandb_mode,
        config={
            "data_path": args.data_path,
            "text_key": args.text_key,
            "response_only_loss": args.response_only_loss,
            "model_name": args.model_name,
            "seq_len": args.seq_len,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "precision": precision,
            "micro_batch_size": micro_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "target_global_batch_tokens": target_tokens,
            "world_size": ctx.world_size,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "early_stopping_warmup_epochs": args.early_stopping_warmup_epochs,
            "compare_base_on_test": args.compare_base_on_test,
            "base_model_name": args.base_model_name or args.model_name,
            "test_eval_every_epochs": args.test_eval_every_epochs,
            "epoch_compare_enabled": args.epoch_compare_enabled,
            "epoch_compare_num_examples": args.epoch_compare_num_examples,
            "epoch_compare_batch_size": args.epoch_compare_batch_size,
            "epoch_compare_max_input_length": args.epoch_compare_max_input_length,
            "epoch_compare_max_new_tokens": args.epoch_compare_max_new_tokens,
            "epoch_compare_temperature": args.epoch_compare_temperature,
            "epoch_compare_top_p": args.epoch_compare_top_p,
        },
    )
    run.define_metric("train/global_step")
    run.define_metric("train/*", step_metric="train/global_step")
    run.define_metric("epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("test/*", step_metric="epoch")
    run.define_metric("compare/*", step_metric="epoch")
    return run


def set_global_seed(seed: int, rank: int) -> None:
    """Set deterministic seeds per process."""
    final_seed = seed + rank
    random.seed(final_seed)
    torch.manual_seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)


def choose_precision(arg_precision: str, device: torch.device) -> str:
    """Choose final precision mode."""
    if arg_precision != "auto":
        return arg_precision
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    return "fp32"


def python_headers_available() -> tuple[bool, str]:
    """Return whether Python development headers are available."""
    include_dir = sysconfig.get_path("include") or ""
    header_path = Path(include_dir) / "Python.h" if include_dir else Path()
    return header_path.is_file(), str(header_path)


def pick_micro_batch_size(
    seq_len: int,
    vram_gb: float,
    user_value: int,
) -> int:
    """Auto-select per-GPU micro batch from VRAM and seq length."""
    if user_value > 0:
        return user_value
    if seq_len <= 1024:
        if vram_gb >= 23:
            return 8
        if vram_gb >= 16:
            return 4
        return 2
    if seq_len <= 1536:
        if vram_gb >= 23:
            return 4
        if vram_gb >= 16:
            return 2
        return 1
    if vram_gb >= 23:
        return 2
    return 1


def pick_target_tokens(
    user_value: int,
    world_size: int,
    vram_gb: float,
) -> int:
    """Auto-select target global tokens per optimizer step."""
    if user_value > 0:
        return user_value
    if world_size >= 4 and vram_gb >= 23:
        return 262_144
    if world_size >= 2:
        return 131_072
    return 65_536


def pick_grad_accum_steps(
    seq_len: int,
    world_size: int,
    micro_batch_size: int,
    target_tokens: int,
    user_value: int,
) -> int:
    """Auto-select grad accumulation to hit target global tokens/update."""
    if user_value > 0:
        return user_value
    tokens_per_micro_step = seq_len * world_size * micro_batch_size
    if tokens_per_micro_step <= 0:
        return 1
    return max(1, math.ceil(target_tokens / tokens_per_micro_step))


def pick_num_workers(user_value: int, world_size: int) -> int:
    """Auto-select DataLoader workers per process."""
    if user_value >= 0:
        return user_value
    cpu_count = os.cpu_count() or 4
    workers = max(2, cpu_count // max(1, world_size * 2))
    return min(8, workers)


def read_text_samples(
    data_path: Path,
    text_key: str,
) -> list[str]:
    """Read text samples from txt/jsonl/json."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    suffix = data_path.suffix.lower()
    samples: list[str] = []

    if suffix in {".txt", ".text", ".md"}:
        lines = data_path.read_text(encoding="utf-8").splitlines()
        samples = [line.strip() for line in lines if line.strip()]
    elif suffix == ".jsonl":
        with data_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if text_key not in record:
                    raise KeyError(
                        f"Missing key '{text_key}' in jsonl line {idx}: {data_path}"
                    )
                value = str(record[text_key]).strip()
                if value:
                    samples.append(value)
    elif suffix == ".json":
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    text = item.strip()
                elif isinstance(item, dict):
                    if text_key not in item:
                        raise KeyError(f"Missing key '{text_key}' in json item.")
                    text = str(item[text_key]).strip()
                else:
                    continue
                if text:
                    samples.append(text)
        elif isinstance(payload, dict) and text_key in payload:
            value = payload[text_key]
            if isinstance(value, list):
                samples = [str(item).strip() for item in value if str(item).strip()]
            elif isinstance(value, str) and value.strip():
                samples = [value.strip()]
        else:
            raise ValueError(
                "Unsupported JSON format. Use list[str], list[dict], or dict[text_key]."
            )
    else:
        raise ValueError(
            f"Unsupported extension: {suffix}. Use .txt, .jsonl, or .json."
        )

    if not samples:
        raise ValueError(f"No non-empty text samples found in: {data_path}")
    return samples


def read_prompt_completion_samples(
    data_path: Path,
    text_key: str,
) -> list[PromptCompletionSample]:
    """Read structured prompt/completion pairs from jsonl/json."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    suffix = data_path.suffix.lower()
    parsed_rows: list[PromptCompletionSample] = []

    def maybe_append(record: dict[str, Any]) -> None:
        prompt = str(record.get("prompt", "")).strip()
        completion = str(record.get("completion", "")).strip()
        if not prompt or not completion:
            return
        fallback_text = str(record.get(text_key, f"{prompt}{completion}")).strip()
        if not fallback_text:
            fallback_text = f"{prompt}{completion}"
        parsed_rows.append(
            PromptCompletionSample(
                prompt=prompt,
                completion=completion,
                fallback_text=fallback_text,
            )
        )

    if suffix == ".jsonl":
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if isinstance(record, dict):
                    maybe_append(record)
    elif suffix == ".json":
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    maybe_append(item)
        elif isinstance(payload, dict):
            maybe_append(payload)
    return parsed_rows


def split_samples(
    samples: list[Any],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Create deterministic train/validation/test split."""
    if len(samples) < 3:
        return samples, [], []

    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)

    val_ratio = max(0.0, min(0.49, val_ratio))
    test_ratio = max(0.0, min(0.49, test_ratio))

    n_total = len(shuffled)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)

    if val_ratio > 0 and n_val == 0:
        n_val = 1
    if test_ratio > 0 and n_test == 0:
        n_test = 1

    max_held_out = max(0, n_total - 1)
    if n_val + n_test > max_held_out:
        overflow = n_val + n_test - max_held_out
        reduce_test = min(overflow, n_test)
        n_test -= reduce_test
        overflow -= reduce_test
        if overflow > 0:
            n_val = max(0, n_val - overflow)

    n_train = n_total - n_val - n_test
    if n_train <= 0:
        return shuffled, [], []

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def build_distill_prompt(question: str) -> str:
    """Build the strict prompt format used in distillation."""
    return (
        "Solve this math problem.\n"
        "ALWAYS return exactly this format:\n"
        "<reasoning>\n"
        "- short, concrete steps (maximum 5 lines)\n"
        "</reasoning>\n"
        "<final>\n"
        "- only the final number\n"
        "</final>\n\n"
        f"Problem: {question}\n"
    )


def extract_tag(text: str, tag: str) -> str | None:
    """Extract a tag body from XML-like text."""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def extract_last_number(text: str) -> str | None:
    """Extract final number-like token from text."""
    cleaned = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    return matches[-1]


def canonicalize_number(text: str) -> str | None:
    """Normalize numeric strings to stable canonical form."""
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    raw = match.group(0)
    if "." not in raw:
        return str(int(raw))
    value = float(raw)
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12):
        return str(int(round(value)))
    return str(value).rstrip("0").rstrip(".")


def extract_pred_final(text: str) -> str | None:
    """Extract predicted final answer from model generation."""
    pred_final_raw = extract_tag(text, "final")
    if pred_final_raw is None:
        pred_final_raw = extract_last_number(text)
    return canonicalize_number(pred_final_raw or "")


def extract_gold_final(record: dict[str, Any]) -> str | None:
    """Extract gold numeric final answer from a structured record."""
    if "gold_final" in record and str(record["gold_final"]).strip():
        return canonicalize_number(str(record["gold_final"]))
    if "answer" in record:
        answer = str(record["answer"])
        if "####" in answer:
            answer = answer.split("####")[-1]
        return canonicalize_number(answer)
    if "completion" in record and str(record["completion"]).strip():
        completion = str(record["completion"])
        completion_final = extract_tag(completion, "final") or extract_last_number(
            completion
        )
        return canonicalize_number(completion_final or "")
    return None


def is_correct(pred_final: str | None, gold_final: str | None) -> bool:
    """Check exact numeric correctness with float tolerance."""
    if pred_final is None or gold_final is None:
        return False
    if pred_final == gold_final:
        return True
    try:
        return math.isclose(
            float(pred_final),
            float(gold_final),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    except ValueError:
        return False


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when wrapped in DDP."""
    return model.module if isinstance(model, DDP) else model


def load_structured_eval_examples(
    *,
    data_path: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_examples: int,
) -> list[StructuredEvalExample]:
    """Load a held-out structured subset for epoch-wise comparison."""
    if data_path.suffix.lower() != ".jsonl":
        return []

    examples: list[StructuredEvalExample] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue

            gold_final = extract_gold_final(record)
            if gold_final is None:
                continue

            prompt = str(record.get("prompt", "")).strip()
            if not prompt:
                question = str(record.get("question", "")).strip()
                if question:
                    prompt = build_distill_prompt(question)
            if not prompt:
                continue

            examples.append(
                StructuredEvalExample(
                    dataset_index=int(record.get("dataset_index", line_idx - 1)),
                    prompt=prompt,
                    gold_final=gold_final,
                )
            )

    if not examples:
        return []

    _, _, test_examples = split_samples(
        samples=examples,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    pool = test_examples if test_examples else examples

    if max_examples > 0 and len(pool) > max_examples:
        shuffled = pool.copy()
        random.Random(seed + 17).shuffle(shuffled)
        pool = shuffled[:max_examples]

    return pool


@torch.no_grad()
def evaluate_generation_subset(
    *,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    examples: list[StructuredEvalExample],
    device: torch.device,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> GenerationEvalResult:
    """Evaluate exact-answer metrics on a fixed prompt subset."""
    if not examples:
        return GenerationEvalResult(
            parsed=0,
            correct=0,
            accuracy=0.0,
            parsed_rate=0.0,
            predictions=[],
            correct_flags=[],
        )

    eval_model = unwrap_model(model)
    was_training = eval_model.training
    eval_model.eval()

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    prompts = [example.prompt for example in examples]
    outputs: list[str] = []
    try:
        for start in range(0, len(prompts), max(1, batch_size)):
            batch_prompts = prompts[start : start + max(1, batch_size)]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            if temperature > 0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p
            else:
                generate_kwargs["do_sample"] = False

            generated = eval_model.generate(**encoded, **generate_kwargs)
            prompt_tokens = encoded["input_ids"].shape[1]
            generated_only = generated[:, prompt_tokens:]
            outputs.extend(
                tokenizer.batch_decode(generated_only, skip_special_tokens=True)
            )

        parsed = 0
        correct = 0
        correct_flags: list[bool] = []
        predictions: list[str] = []

        for output_text, example in zip(outputs, examples):
            pred = extract_pred_final(output_text)
            predictions.append(pred or "")
            if pred is not None:
                parsed += 1
            ok = is_correct(pred, example.gold_final)
            correct_flags.append(ok)
            if ok:
                correct += 1

        evaluated = max(1, len(examples))
        return GenerationEvalResult(
            parsed=parsed,
            correct=correct,
            accuracy=correct / evaluated,
            parsed_rate=parsed / evaluated,
            predictions=predictions,
            correct_flags=correct_flags,
        )
    finally:
        tokenizer.padding_side = original_padding_side
        if was_training:
            eval_model.train()


def write_epoch_compare_history(
    path: Path,
    history: list[dict[str, float | int]],
) -> None:
    """Persist epoch-wise original-vs-finetuned comparison metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def log_wandb_epoch_compare_charts(
    run: Any,
    history: list[dict[str, float | int]],
    epoch: int,
    global_step: int,
) -> None:
    """Log explicit line charts to W&B for original vs finetuned trends."""
    if not history:
        return
    try:
        import wandb
    except ImportError:
        return

    xs = [int(item["epoch"]) for item in history]
    original_acc = [float(item["original_accuracy"]) for item in history]
    finetuned_acc = [float(item["finetuned_accuracy"]) for item in history]
    original_parsed = [float(item["original_parsed_rate"]) for item in history]
    finetuned_parsed = [float(item["finetuned_parsed_rate"]) for item in history]

    run.log(
        {
            "epoch": epoch,
            "train/global_step": global_step,
            "compare/charts/accuracy": wandb.plot.line_series(
                xs=xs,
                ys=[original_acc, finetuned_acc],
                keys=["original", "finetuned"],
                title="Accuracy by Epoch (Original vs Finetuned)",
                xname="epoch",
            ),
            "compare/charts/parsed_rate": wandb.plot.line_series(
                xs=xs,
                ys=[original_parsed, finetuned_parsed],
                keys=["original", "finetuned"],
                title="Parsed Rate by Epoch (Original vs Finetuned)",
                xname="epoch",
            ),
        }
    )


def pack_causal_lm_blocks(
    samples: Iterable[str],
    tokenizer: AutoTokenizer,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack token stream into fixed-size causal LM blocks."""
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")

    token_stream: list[int] = []
    for text in samples:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        token_stream.extend(token_ids)
        token_stream.append(eos_token_id)

    block_len = seq_len
    usable_tokens = (len(token_stream) // block_len) * block_len
    if usable_tokens < block_len:
        raise ValueError(
            f"Not enough tokens to build one block of size {block_len}. "
            f"Current token count: {len(token_stream)}."
        )

    all_tokens = torch.tensor(token_stream[:usable_tokens], dtype=torch.long)
    blocks = all_tokens.view(-1, block_len)
    input_ids = blocks.contiguous()
    # HF CausalLM applies internal shift for next-token prediction.
    labels = blocks.clone().contiguous()
    return input_ids, labels


def pack_prompt_completion_blocks(
    samples: Iterable[PromptCompletionSample],
    tokenizer: AutoTokenizer,
    seq_len: int,
    response_only_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack prompt/completion pairs into blocks, with optional prompt masking."""
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")

    token_stream: list[int] = []
    label_stream: list[int] = []

    for sample in samples:
        if response_only_loss:
            prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
            completion_ids = tokenizer.encode(
                sample.completion,
                add_special_tokens=False,
            )
            if not prompt_ids and not completion_ids:
                continue
            token_stream.extend(prompt_ids)
            label_stream.extend([-100] * len(prompt_ids))
            token_stream.extend(completion_ids)
            label_stream.extend(completion_ids)
        else:
            text = sample.fallback_text or f"{sample.prompt}{sample.completion}"
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            token_stream.extend(token_ids)
            label_stream.extend(token_ids)

        token_stream.append(eos_token_id)
        label_stream.append(eos_token_id)

    block_len = seq_len
    usable_tokens = (len(token_stream) // block_len) * block_len
    if usable_tokens < block_len:
        raise ValueError(
            f"Not enough tokens to build one block of size {block_len}. "
            f"Current token count: {len(token_stream)}."
        )

    all_tokens = torch.tensor(token_stream[:usable_tokens], dtype=torch.long)
    all_labels = torch.tensor(label_stream[:usable_tokens], dtype=torch.long)
    input_ids = all_tokens.view(-1, block_len).contiguous()
    labels = all_labels.view(-1, block_len).contiguous()
    return input_ids, labels


def cache_file_for_dataset(
    data_path: Path,
    model_name: str,
    seq_len: int,
    val_ratio: float,
    test_ratio: float,
    text_key: str,
    response_only_loss: bool,
    output_dir: Path,
) -> Path:
    """Build deterministic cache path for tokenized dataset."""
    stat = data_path.stat()
    cache_payload = {
        "packing_version": PACKING_VERSION,
        "path": str(data_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "model_name": model_name,
        "seq_len": seq_len,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "text_key": text_key,
        "response_only_loss": response_only_loss,
    }
    cache_key = hashlib.sha1(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"packed_tokens_{cache_key}.pt"


def build_or_load_tokenized_data(
    args: argparse.Namespace,
    ctx: DistContext,
    tokenizer: AutoTokenizer,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Prepare and cache tokenized train/val/test tensors."""
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    cache_path = cache_file_for_dataset(
        data_path=data_path,
        model_name=args.model_name,
        seq_len=args.seq_len,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        text_key=args.text_key,
        response_only_loss=args.response_only_loss,
        output_dir=output_dir,
    )

    if not cache_path.exists() and is_rank0(ctx):
        use_structured = False
        structured_samples: list[PromptCompletionSample] = []
        if args.response_only_loss:
            structured_samples = read_prompt_completion_samples(
                data_path=data_path,
                text_key=args.text_key,
            )
            use_structured = len(structured_samples) > 0

        if use_structured:
            train_samples, val_samples, test_samples = split_samples(
                samples=structured_samples,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
            rank0_print(
                ctx,
                (
                    "[data] using structured prompt/completion tokenization "
                    f"(response_only_loss={args.response_only_loss})"
                ),
            )
            train_inputs, train_labels = pack_prompt_completion_blocks(
                samples=train_samples,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                response_only_loss=args.response_only_loss,
            )

            def build_eval_split(
                split_name: str,
                split_samples: list[PromptCompletionSample],
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if not split_samples:
                    return (
                        torch.empty((0, args.seq_len), dtype=torch.long),
                        torch.empty((0, args.seq_len), dtype=torch.long),
                    )
                try:
                    return pack_prompt_completion_blocks(
                        samples=split_samples,
                        tokenizer=tokenizer,
                        seq_len=args.seq_len,
                        response_only_loss=args.response_only_loss,
                    )
                except ValueError:
                    rank0_print(
                        ctx,
                        f"[data] {split_name} split too small after tokenization; skipping.",
                    )
                    return (
                        torch.empty((0, args.seq_len), dtype=torch.long),
                        torch.empty((0, args.seq_len), dtype=torch.long),
                    )

            val_inputs, val_labels = build_eval_split("validation", val_samples)
            test_inputs, test_labels = build_eval_split("test", test_samples)
        else:
            if args.response_only_loss:
                rank0_print(
                    ctx,
                    (
                        "[data] prompt/completion fields not found; falling back to text "
                        "tokenization without prompt masking."
                    ),
                )
            samples = read_text_samples(data_path=data_path, text_key=args.text_key)
            train_samples, val_samples, test_samples = split_samples(
                samples=samples,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )

            train_inputs, train_labels = pack_causal_lm_blocks(
                samples=train_samples,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
            )

            def build_eval_split(
                split_name: str,
                split_samples: list[str],
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if not split_samples:
                    return (
                        torch.empty((0, args.seq_len), dtype=torch.long),
                        torch.empty((0, args.seq_len), dtype=torch.long),
                    )
                try:
                    return pack_causal_lm_blocks(
                        samples=split_samples,
                        tokenizer=tokenizer,
                        seq_len=args.seq_len,
                    )
                except ValueError:
                    rank0_print(
                        ctx,
                        f"[data] {split_name} split too small after tokenization; skipping.",
                    )
                    return (
                        torch.empty((0, args.seq_len), dtype=torch.long),
                        torch.empty((0, args.seq_len), dtype=torch.long),
                    )

            val_inputs, val_labels = build_eval_split("validation", val_samples)
            test_inputs, test_labels = build_eval_split("test", test_samples)

        payload = {
            "train_inputs": train_inputs,
            "train_labels": train_labels,
            "val_inputs": val_inputs,
            "val_labels": val_labels,
            "test_inputs": test_inputs,
            "test_labels": test_labels,
        }
        torch.save(payload, cache_path)
        rank0_print(
            ctx,
            (
                f"[data] token cache built: {cache_path} | "
                f"train_blocks={train_inputs.size(0)} "
                f"val_blocks={val_inputs.size(0)} test_blocks={test_inputs.size(0)}"
            ),
        )

    maybe_barrier(ctx)
    payload = torch.load(cache_path, map_location="cpu")
    train_dataset = TensorDataset(payload["train_inputs"], payload["train_labels"])
    val_dataset = TensorDataset(payload["val_inputs"], payload["val_labels"])
    test_dataset = TensorDataset(payload["test_inputs"], payload["test_labels"])
    return train_dataset, val_dataset, test_dataset


def build_dataloader(
    dataset: TensorDataset,
    batch_size: int,
    ctx: DistContext,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build DataLoader and DistributedSampler."""
    sampler: DistributedSampler | None = None
    if ctx.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": ctx.device.type == "cuda",
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4

    dataloader = DataLoader(**kwargs)
    return dataloader, sampler


def split_weight_decay_params(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Create AdamW parameter groups with/without weight decay."""
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    no_decay_keywords = ("bias", "ln_", "layer_norm", "LayerNorm", "norm")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    ctx: DistContext,
    global_step: int,
    epoch: int,
    val_loss: float | None = None,
) -> None:
    """Save training checkpoint on rank 0."""
    if not is_rank0(ctx):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped_model = model.module if isinstance(model, DDP) else model
    payload = {
        "model_state_dict": wrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "args": vars(args),
        "val_loss": val_loss,
    }
    torch.save(payload, path)
    print(f"[checkpoint] saved {path}", flush=True)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    ctx: DistContext,
    precision: str,
    max_batches: int,
) -> float:
    """Evaluate average validation loss."""
    if len(val_loader.dataset) == 0:
        return float("nan")

    was_training = model.training
    model.eval()
    dtype = None
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16

    total_loss = torch.tensor(0.0, device=ctx.device)
    total_batches = torch.tensor(0.0, device=ctx.device)

    for idx, (input_ids, labels) in enumerate(val_loader):
        if max_batches > 0 and idx >= max_batches:
            break
        input_ids = input_ids.to(ctx.device, non_blocking=True)
        labels = labels.to(ctx.device, non_blocking=True)
        attention_mask = torch.ones_like(input_ids)

        with (
            torch.autocast(device_type="cuda", dtype=dtype)
            if (ctx.device.type == "cuda" and dtype is not None)
            else nullcontext()
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        total_loss += loss.detach()
        total_batches += 1.0

    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

    if was_training:
        model.train()
    if total_batches.item() <= 0:
        return float("nan")
    return float((total_loss / total_batches).item())


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError("--val-ratio and --test-ratio must be >= 0")
    if args.val_ratio + args.test_ratio >= 0.95:
        raise ValueError("--val-ratio + --test-ratio must be < 0.95")
    if args.test_eval_every_epochs < 1:
        raise ValueError("--test-eval-every-epochs must be >= 1")
    if args.epoch_compare_enabled and args.epoch_compare_num_examples < 1:
        raise ValueError("--epoch-compare-num-examples must be >= 1")
    if args.epoch_compare_enabled and args.epoch_compare_batch_size < 1:
        raise ValueError("--epoch-compare-batch-size must be >= 1")
    ctx = setup_dist()
    wandb_run: Any | None = None

    try:
        set_global_seed(args.seed, ctx.rank)
        if args.progress_bar and tqdm is None and is_rank0(ctx):
            rank0_print(
                ctx,
                "[train] tqdm is not installed; disabling progress bar.",
            )

        if ctx.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        vram_gb = 0.0
        if ctx.device.type == "cuda":
            props = torch.cuda.get_device_properties(ctx.device)
            vram_gb = props.total_memory / 1024**3

        precision = choose_precision(args.precision, ctx.device)
        micro_batch_size = pick_micro_batch_size(
            seq_len=args.seq_len,
            vram_gb=vram_gb,
            user_value=args.micro_batch_size,
        )
        target_tokens = pick_target_tokens(
            user_value=args.target_global_batch_tokens,
            world_size=ctx.world_size,
            vram_gb=vram_gb,
        )
        grad_accum_steps = pick_grad_accum_steps(
            seq_len=args.seq_len,
            world_size=ctx.world_size,
            micro_batch_size=micro_batch_size,
            target_tokens=target_tokens,
            user_value=args.grad_accum_steps,
        )
        num_workers = pick_num_workers(args.num_workers, ctx.world_size)

        rank0_print(
            ctx,
            (
                f"[hardware] world_size={ctx.world_size} device={ctx.device} "
                f"vram_gb={vram_gb:.2f}"
            ),
        )
        rank0_print(
            ctx,
            (
                f"[train] precision={precision} micro_batch={micro_batch_size} "
                f"grad_accum={grad_accum_steps} target_tokens={target_tokens} "
                f"num_workers={num_workers}"
            ),
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_dataset, val_dataset, test_dataset = build_or_load_tokenized_data(
            args=args,
            ctx=ctx,
            tokenizer=tokenizer,
        )

        train_loader, train_sampler = build_dataloader(
            dataset=train_dataset,
            batch_size=micro_batch_size,
            ctx=ctx,
            num_workers=num_workers,
            shuffle=True,
            drop_last=False,
        )
        val_loader, _ = build_dataloader(
            dataset=val_dataset,
            batch_size=micro_batch_size,
            ctx=ctx,
            num_workers=max(1, num_workers // 2),
            shuffle=False,
            drop_last=False,
        )
        test_loader, _ = build_dataloader(
            dataset=test_dataset,
            batch_size=micro_batch_size,
            ctx=ctx,
            num_workers=max(1, num_workers // 2),
            shuffle=False,
            drop_last=False,
        )

        if len(train_loader) == 0:
            raise RuntimeError(
                "Training loader has 0 batches. "
                "Lower --micro-batch-size or --seq-len, "
                "or increase training data."
            )

        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model.to(ctx.device)

        compile_enabled = args.compile
        if compile_enabled and ctx.device.type == "cuda":
            headers_ok, header_path = python_headers_available()
            if not headers_ok:
                compile_enabled = False
                rank0_print(
                    ctx,
                    (
                        "[train] torch.compile disabled: missing Python.h "
                        f"({header_path}). Install python3-dev to enable compile."
                    ),
                )

        if compile_enabled:
            try:
                model = torch.compile(model, mode=args.compile_mode)
                rank0_print(ctx, f"[train] torch.compile enabled ({args.compile_mode})")
            except Exception as exc:  # pragma: no cover - runtime dependent
                rank0_print(ctx, f"[train] torch.compile disabled due to: {exc}")

        if ctx.world_size > 1:
            model = DDP(
                model,
                device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
                output_device=ctx.local_rank if ctx.device.type == "cuda" else None,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        param_groups = split_weight_decay_params(
            model=model,
            weight_decay=args.weight_decay,
        )

        use_fused = args.fused_adamw and ctx.device.type == "cuda"
        try:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=args.lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=use_fused,
            )
        except TypeError:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=args.lr,
                betas=(0.9, 0.95),
                eps=1e-8,
            )

        updates_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        total_updates = updates_per_epoch * max(1, args.epochs)
        if args.max_steps > 0:
            total_updates = min(total_updates, args.max_steps)
        warmup_steps = max(1, int(total_updates * args.warmup_ratio))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_updates),
        )

        scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=(ctx.device.type == "cuda" and precision == "fp16"),
        )
        amp_dtype = None
        if precision == "bf16":
            amp_dtype = torch.bfloat16
        elif precision == "fp16":
            amp_dtype = torch.float16

        tokenizer.save_pretrained(output_dir / "tokenizer")
        rank0_print(
            ctx,
            (
                f"[train] train_blocks={len(train_dataset)} val_blocks={len(val_dataset)} "
                f"test_blocks={len(test_dataset)} updates_per_epoch={updates_per_epoch} "
                f"total_updates={total_updates}"
            ),
        )

        wandb_run = init_wandb_run(
            args=args,
            ctx=ctx,
            precision=precision,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            target_tokens=target_tokens,
        )

        epoch_compare_examples: list[StructuredEvalExample] = []
        epoch_compare_history: list[dict[str, float | int]] = []
        original_compare_result: GenerationEvalResult | None = None
        original_compare_flags: list[bool] = []
        if args.epoch_compare_enabled and is_rank0(ctx):
            epoch_compare_examples = load_structured_eval_examples(
                data_path=Path(args.data_path),
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                max_examples=args.epoch_compare_num_examples,
            )
            if epoch_compare_examples:
                rank0_print(
                    ctx,
                    (
                        "[epoch-compare] loaded structured eval subset "
                        f"(examples={len(epoch_compare_examples)})"
                    ),
                )
            else:
                rank0_print(
                    ctx,
                    "[epoch-compare] no structured examples found; comparison disabled.",
                )

        base_test_loss = float("nan")
        base_test_ppl = float("nan")
        if args.compare_base_on_test and len(test_dataset) > 0:
            base_model_name = args.base_model_name or args.model_name
            rank0_print(ctx, f"[test-compare] loading base model: {base_model_name}")
            base_eval_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            base_eval_model.config.pad_token_id = tokenizer.pad_token_id
            base_eval_model.to(ctx.device)
            base_test_loss = evaluate(
                model=base_eval_model,
                val_loader=test_loader,
                ctx=ctx,
                precision=precision,
                max_batches=args.max_eval_batches,
            )
            if not math.isnan(base_test_loss):
                base_test_ppl = math.exp(min(base_test_loss, 20))
                rank0_print(
                    ctx,
                    f"[test-compare] base_test_loss={base_test_loss:.4f} base_test_ppl={base_test_ppl:.2f}",
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "epoch": 0,
                            "test/base_loss": base_test_loss,
                            "test/base_ppl": base_test_ppl,
                            "train/global_step": 0,
                        }
                    )
            del base_eval_model
            if ctx.device.type == "cuda":
                torch.cuda.empty_cache()

        if args.epoch_compare_enabled and epoch_compare_examples:
            if is_rank0(ctx):
                try:
                    original_compare_result = evaluate_generation_subset(
                        model=model,
                        tokenizer=tokenizer,
                        examples=epoch_compare_examples,
                        device=ctx.device,
                        batch_size=args.epoch_compare_batch_size,
                        max_input_length=args.epoch_compare_max_input_length,
                        max_new_tokens=args.epoch_compare_max_new_tokens,
                        temperature=args.epoch_compare_temperature,
                        top_p=args.epoch_compare_top_p,
                    )
                    original_compare_flags = original_compare_result.correct_flags
                    rank0_print(
                        ctx,
                        (
                            "[epoch-compare] original "
                            f"acc={original_compare_result.accuracy:.4f} "
                            f"parsed={original_compare_result.parsed_rate:.4f}"
                        ),
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "epoch": 0,
                                "train/global_step": 0,
                                "compare/accuracy_original": original_compare_result.accuracy,
                                "compare/parsed_rate_original": (
                                    original_compare_result.parsed_rate
                                ),
                                "compare/accuracy_finetuned": original_compare_result.accuracy,
                                "compare/parsed_rate_finetuned": (
                                    original_compare_result.parsed_rate
                                ),
                                "compare/delta_accuracy_vs_original": 0.0,
                                "compare/improved_count_vs_original": 0,
                                "compare/regressed_count_vs_original": 0,
                                "compare/evaluated_examples": len(epoch_compare_examples),
                            }
                        )
                except Exception as exc:
                    rank0_print(
                        ctx,
                        f"[epoch-compare] disabled (baseline generation failed): {exc}",
                    )
                    epoch_compare_examples = []
                    original_compare_result = None
                    original_compare_flags = []

        global_step = 0
        best_val_loss = float("inf")
        best_epoch = 0
        best_step = 0
        epochs_without_improve = 0
        early_stopped = False
        stop_reason = ""
        last_epoch = 0
        train_start = time.perf_counter()

        for epoch in range(args.epochs):
            last_epoch = epoch + 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            progress_bar: Any | None = None
            epoch_updates_target = updates_per_epoch
            if args.max_steps > 0:
                epoch_updates_target = min(
                    updates_per_epoch,
                    max(0, args.max_steps - global_step),
                )
            if (
                args.progress_bar
                and tqdm is not None
                and is_rank0(ctx)
                and epoch_updates_target > 0
            ):
                progress_bar = tqdm(
                    total=epoch_updates_target,
                    desc=f"finetune e{epoch + 1}/{args.epochs}",
                    unit="step",
                    dynamic_ncols=True,
                    leave=False,
                )

            accum_counter = 0
            interval_loss_sum = 0.0
            interval_loss_count = 0
            interval_tokens = 0
            interval_start = time.perf_counter()

            try:
                for batch_idx, (input_ids, labels) in enumerate(train_loader):
                    input_ids = input_ids.to(ctx.device, non_blocking=True)
                    labels = labels.to(ctx.device, non_blocking=True)
                    attention_mask = torch.ones_like(input_ids)

                    with (
                        torch.autocast(device_type="cuda", dtype=amp_dtype)
                        if (ctx.device.type == "cuda" and amp_dtype is not None)
                        else nullcontext()
                    ):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss

                    loss_scaled = loss / grad_accum_steps
                    if scaler.is_enabled():
                        scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()

                    accum_counter += 1
                    interval_loss_sum += float(loss.item())
                    interval_loss_count += 1
                    interval_tokens += int(input_ids.numel()) * ctx.world_size

                    should_step = (
                        accum_counter >= grad_accum_steps
                        or batch_idx + 1 == len(train_loader)
                    )
                    if not should_step:
                        continue

                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)

                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    accum_counter = 0
                    if progress_bar is not None and progress_bar.n < progress_bar.total:
                        progress_bar.update(1)

                    if global_step % args.log_interval == 0:
                        local_avg_loss = interval_loss_sum / max(1, interval_loss_count)
                        mean_loss = distributed_mean(local_avg_loss, ctx.device)
                        elapsed = max(1e-6, time.perf_counter() - interval_start)
                        tok_per_s = interval_tokens / elapsed
                        lr = scheduler.get_last_lr()[0]
                        rank0_print(
                            ctx,
                            (
                                f"[step {global_step}/{total_updates}] "
                                f"loss={mean_loss:.4f} lr={lr:.3e} "
                                f"tok/s={tok_per_s:,.0f}"
                            ),
                        )
                        if progress_bar is not None:
                            progress_bar.set_postfix(
                                {
                                    "loss": f"{mean_loss:.4f}",
                                    "lr": f"{lr:.2e}",
                                    "tok/s": f"{tok_per_s:,.0f}",
                                }
                            )
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "train/global_step": global_step,
                                    "train/loss": mean_loss,
                                    "train/lr": lr,
                                    "train/tokens_per_sec": tok_per_s,
                                    "train/epoch": epoch + 1,
                                }
                            )
                        interval_loss_sum = 0.0
                        interval_loss_count = 0
                        interval_tokens = 0
                        interval_start = time.perf_counter()

                    if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                        save_checkpoint(
                            path=output_dir / "checkpoints" / "last.pt",
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            args=args,
                            ctx=ctx,
                            global_step=global_step,
                            epoch=epoch + 1,
                        )

                    if args.max_steps > 0 and global_step >= args.max_steps:
                        break
            finally:
                if progress_bar is not None:
                    progress_bar.close()

            val_loss = evaluate(
                model=model,
                val_loader=val_loader,
                ctx=ctx,
                precision=precision,
                max_batches=args.max_eval_batches,
            )
            if not math.isnan(val_loss):
                val_ppl = math.exp(min(val_loss, 20))
                rank0_print(
                    ctx,
                    (
                        f"[epoch {epoch + 1}/{args.epochs}] "
                        f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}"
                    ),
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "epoch": epoch + 1,
                            "val/loss": val_loss,
                            "val/ppl": val_ppl,
                            "val/best_loss_so_far": min(best_val_loss, val_loss),
                            "train/global_step": global_step,
                        }
                    )

                improved = val_loss < (best_val_loss - args.early_stopping_min_delta)
                if improved:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    best_step = global_step
                    epochs_without_improve = 0
                    save_checkpoint(
                        path=output_dir / "checkpoints" / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        ctx=ctx,
                        global_step=global_step,
                        epoch=epoch + 1,
                        val_loss=val_loss,
                    )
                else:
                    epochs_without_improve += 1

            should_eval_test_now = (
                len(test_dataset) > 0
                and (
                    (epoch + 1) % args.test_eval_every_epochs == 0
                    or (epoch + 1) == args.epochs
                    or (args.max_steps > 0 and global_step >= args.max_steps)
                )
            )
            if should_eval_test_now:
                distilled_test_loss = evaluate(
                    model=model,
                    val_loader=test_loader,
                    ctx=ctx,
                    precision=precision,
                    max_batches=args.max_eval_batches,
                )
                if not math.isnan(distilled_test_loss):
                    distilled_test_ppl = math.exp(min(distilled_test_loss, 20))
                    log_payload: dict[str, float | int] = {
                        "epoch": epoch + 1,
                        "train/global_step": global_step,
                        "test/distilled_loss": distilled_test_loss,
                        "test/distilled_ppl": distilled_test_ppl,
                    }
                    status = "n/a"
                    if not math.isnan(base_test_loss):
                        delta_loss = distilled_test_loss - base_test_loss
                        delta_ppl = distilled_test_ppl - base_test_ppl
                        improvement_pct = (
                            ((base_test_loss - distilled_test_loss) / base_test_loss) * 100.0
                            if base_test_loss > 0
                            else 0.0
                        )
                        if abs(delta_loss) <= 1e-9:
                            status = "no_change"
                        elif delta_loss < 0:
                            status = "improved"
                        else:
                            status = "deteriorated"
                        log_payload.update(
                            {
                                "test/base_loss": base_test_loss,
                                "test/base_ppl": base_test_ppl,
                                "test/delta_loss_vs_base": delta_loss,
                                "test/delta_ppl_vs_base": delta_ppl,
                                "test/improvement_pct_vs_base": improvement_pct,
                                "compare/loss_original": base_test_loss,
                                "compare/loss_finetuned": distilled_test_loss,
                                "compare/loss_delta_vs_original": delta_loss,
                                "compare/loss_improvement_pct_vs_original": (
                                    improvement_pct
                                ),
                            }
                        )
                        rank0_print(
                            ctx,
                            (
                                f"[test epoch {epoch + 1}] base_loss={base_test_loss:.4f} "
                                f"distilled_loss={distilled_test_loss:.4f} "
                                f"delta_loss={delta_loss:+.4f} status={status}"
                            ),
                        )
                    else:
                        rank0_print(
                            ctx,
                            (
                                f"[test epoch {epoch + 1}] "
                                f"distilled_loss={distilled_test_loss:.4f} "
                                f"distilled_ppl={distilled_test_ppl:.2f}"
                            ),
                        )
                    if wandb_run is not None:
                        wandb_run.log(log_payload)

            should_eval_compare_now = (
                args.epoch_compare_enabled
                and len(epoch_compare_examples) > 0
                and original_compare_result is not None
                and (
                    (epoch + 1) % args.test_eval_every_epochs == 0
                    or (epoch + 1) == args.epochs
                    or (args.max_steps > 0 and global_step >= args.max_steps)
                )
            )
            if should_eval_compare_now:
                if is_rank0(ctx):
                    try:
                        finetuned_compare_result = evaluate_generation_subset(
                            model=model,
                            tokenizer=tokenizer,
                            examples=epoch_compare_examples,
                            device=ctx.device,
                            batch_size=args.epoch_compare_batch_size,
                            max_input_length=args.epoch_compare_max_input_length,
                            max_new_tokens=args.epoch_compare_max_new_tokens,
                            temperature=args.epoch_compare_temperature,
                            top_p=args.epoch_compare_top_p,
                        )
                    except Exception as exc:
                        rank0_print(
                            ctx,
                            f"[epoch-compare] disabled (epoch eval failed): {exc}",
                        )
                        epoch_compare_examples = []
                        original_compare_result = None
                        original_compare_flags = []
                    else:
                        improved = 0
                        regressed = 0
                        for base_ok, finetuned_ok in zip(
                            original_compare_flags,
                            finetuned_compare_result.correct_flags,
                        ):
                            if (not base_ok) and finetuned_ok:
                                improved += 1
                            elif base_ok and (not finetuned_ok):
                                regressed += 1

                        delta_accuracy = (
                            finetuned_compare_result.accuracy
                            - original_compare_result.accuracy
                        )
                        compare_payload: dict[str, float | int] = {
                            "epoch": epoch + 1,
                            "train/global_step": global_step,
                            "compare/evaluated_examples": len(epoch_compare_examples),
                            "compare/accuracy_original": original_compare_result.accuracy,
                            "compare/parsed_rate_original": (
                                original_compare_result.parsed_rate
                            ),
                            "compare/accuracy_finetuned": (
                                finetuned_compare_result.accuracy
                            ),
                            "compare/parsed_rate_finetuned": (
                                finetuned_compare_result.parsed_rate
                            ),
                            "compare/delta_accuracy_vs_original": delta_accuracy,
                            "compare/improved_count_vs_original": improved,
                            "compare/regressed_count_vs_original": regressed,
                        }
                        epoch_compare_history.append(
                            {
                                "epoch": epoch + 1,
                                "evaluated_examples": len(epoch_compare_examples),
                                "original_accuracy": original_compare_result.accuracy,
                                "finetuned_accuracy": (
                                    finetuned_compare_result.accuracy
                                ),
                                "original_parsed_rate": (
                                    original_compare_result.parsed_rate
                                ),
                                "finetuned_parsed_rate": (
                                    finetuned_compare_result.parsed_rate
                                ),
                                "delta_accuracy_vs_original": delta_accuracy,
                                "improved_count_vs_original": improved,
                                "regressed_count_vs_original": regressed,
                            }
                        )
                        write_epoch_compare_history(
                            output_dir / "compare_by_epoch.json",
                            epoch_compare_history,
                        )
                        rank0_print(
                            ctx,
                            (
                                f"[epoch-compare {epoch + 1}] "
                                f"orig_acc={original_compare_result.accuracy:.4f} "
                                f"ft_acc={finetuned_compare_result.accuracy:.4f} "
                                f"delta={delta_accuracy:+.4f} "
                                f"improved={improved} regressed={regressed}"
                            ),
                        )
                        if wandb_run is not None:
                            wandb_run.log(compare_payload)
                            log_wandb_epoch_compare_charts(
                                run=wandb_run,
                                history=epoch_compare_history,
                                epoch=epoch + 1,
                                global_step=global_step,
                            )

            save_checkpoint(
                path=output_dir / "checkpoints" / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                ctx=ctx,
                global_step=global_step,
                epoch=epoch + 1,
                val_loss=val_loss if not math.isnan(val_loss) else None,
            )

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_reason = f"max_steps_reached:{args.max_steps}"
                break

            can_early_stop = (
                args.early_stopping_patience > 0
                and not math.isnan(val_loss)
                and (epoch + 1) >= max(1, args.early_stopping_warmup_epochs)
            )
            if can_early_stop and epochs_without_improve >= args.early_stopping_patience:
                early_stopped = True
                stop_reason = (
                    "early_stopping:"
                    f"patience={args.early_stopping_patience},"
                    f"min_delta={args.early_stopping_min_delta}"
                )
                rank0_print(ctx, f"[early-stop] {stop_reason}")
                break

        train_minutes = (time.perf_counter() - train_start) / 60.0
        wrapped_model = model.module if isinstance(model, DDP) else model

        best_ckpt_path = output_dir / "checkpoints" / "best.pt"
        if best_ckpt_path.exists():
            best_payload = torch.load(best_ckpt_path, map_location="cpu")
            wrapped_model.load_state_dict(best_payload["model_state_dict"], strict=True)

        test_loss = evaluate(
            model=model,
            val_loader=test_loader,
            ctx=ctx,
            precision=precision,
            max_batches=args.max_eval_batches,
        )
        test_ppl = math.exp(min(test_loss, 20)) if not math.isnan(test_loss) else float("nan")
        final_delta_loss_vs_base = (
            (test_loss - base_test_loss)
            if (not math.isnan(test_loss) and not math.isnan(base_test_loss))
            else float("nan")
        )
        final_delta_ppl_vs_base = (
            (test_ppl - base_test_ppl)
            if (not math.isnan(test_ppl) and not math.isnan(base_test_ppl))
            else float("nan")
        )
        final_improvement_pct_vs_base = (
            ((base_test_loss - test_loss) / base_test_loss) * 100.0
            if (
                not math.isnan(test_loss)
                and not math.isnan(base_test_loss)
                and base_test_loss > 0
            )
            else float("nan")
        )
        final_status_vs_base = "unknown"
        if not math.isnan(final_delta_loss_vs_base):
            if abs(final_delta_loss_vs_base) <= 1e-9:
                final_status_vs_base = "no_change"
            elif final_delta_loss_vs_base < 0:
                final_status_vs_base = "improved"
            else:
                final_status_vs_base = "deteriorated"

        final_metrics = {
            "global_step": global_step,
            "train_minutes": train_minutes,
            "train_blocks": len(train_dataset),
            "val_blocks": len(val_dataset),
            "test_blocks": len(test_dataset),
            "best_val_loss": None if math.isinf(best_val_loss) else best_val_loss,
            "best_val_ppl": (
                None if math.isinf(best_val_loss) else math.exp(min(best_val_loss, 20))
            ),
            "best_epoch": best_epoch,
            "best_step": best_step,
            "base_test_loss": None if math.isnan(base_test_loss) else base_test_loss,
            "base_test_ppl": None if math.isnan(base_test_ppl) else base_test_ppl,
            "test_loss": None if math.isnan(test_loss) else test_loss,
            "test_ppl": None if math.isnan(test_loss) else test_ppl,
            "test_delta_loss_vs_base": (
                None if math.isnan(final_delta_loss_vs_base) else final_delta_loss_vs_base
            ),
            "test_delta_ppl_vs_base": (
                None if math.isnan(final_delta_ppl_vs_base) else final_delta_ppl_vs_base
            ),
            "test_improvement_pct_vs_base": (
                None
                if math.isnan(final_improvement_pct_vs_base)
                else final_improvement_pct_vs_base
            ),
            "test_status_vs_base": final_status_vs_base,
            "early_stopped": early_stopped,
            "stop_reason": stop_reason,
        }
        if epoch_compare_history:
            latest_compare = epoch_compare_history[-1]
            final_metrics.update(
                {
                    "compare_evaluated_examples": int(
                        latest_compare.get(
                            "evaluated_examples", len(epoch_compare_examples)
                        )
                    ),
                    "compare_original_accuracy": float(
                        latest_compare["original_accuracy"]
                    ),
                    "compare_finetuned_accuracy": float(
                        latest_compare["finetuned_accuracy"]
                    ),
                    "compare_delta_accuracy_vs_original": float(
                        latest_compare["delta_accuracy_vs_original"]
                    ),
                    "compare_improved_count_vs_original": int(
                        latest_compare["improved_count_vs_original"]
                    ),
                    "compare_regressed_count_vs_original": int(
                        latest_compare["regressed_count_vs_original"]
                    ),
                }
            )

        if is_rank0(ctx):
            wrapped_model.save_pretrained(output_dir / "model_final")
            metrics_path = output_dir / "metrics_final.json"
            metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
            if wandb_run is not None:
                wandb_run.summary.update(final_metrics)
                if not math.isnan(test_loss):
                    wandb_run.log(
                        {
                            "epoch": last_epoch,
                            "test/distilled_loss": test_loss,
                            "test/distilled_ppl": test_ppl,
                            "test/base_loss": base_test_loss,
                            "test/base_ppl": base_test_ppl,
                            "test/delta_loss_vs_base": final_delta_loss_vs_base,
                            "test/delta_ppl_vs_base": final_delta_ppl_vs_base,
                            "test/improvement_pct_vs_base": final_improvement_pct_vs_base,
                            "compare/loss_original": base_test_loss,
                            "compare/loss_finetuned": test_loss,
                            "compare/loss_delta_vs_original": final_delta_loss_vs_base,
                            "compare/loss_improvement_pct_vs_original": (
                                final_improvement_pct_vs_base
                            ),
                            "train/global_step": global_step,
                        }
                    )
                if epoch_compare_history:
                    log_wandb_epoch_compare_charts(
                        run=wandb_run,
                        history=epoch_compare_history,
                        epoch=last_epoch,
                        global_step=global_step,
                    )
                wandb_run.finish()

            if not math.isnan(test_loss):
                rank0_print(
                    ctx,
                    (
                        f"[test final] base_loss={base_test_loss:.4f} "
                        f"distilled_loss={test_loss:.4f} "
                        f"delta_loss={final_delta_loss_vs_base:+.4f} "
                        f"status={final_status_vs_base}"
                    ),
                )
            print(
                (
                    "[done] training completed | "
                    f"steps={global_step} time_min={train_minutes:.2f} "
                    f"output={output_dir}"
                ),
                flush=True,
            )

        return 0
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
        cleanup_dist()


if __name__ == "__main__":
    raise SystemExit(main())
