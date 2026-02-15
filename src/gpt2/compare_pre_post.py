"""Compare GPT-2 base vs finetuned model outputs on GSM8K.

Example:
    PYTHONPATH=src uv run python -m gpt2.compare_pre_post \
      --base-model gpt2 \
      --finetuned-model-path outputs/gpt2_gsm8k_distilled/model_final \
      --finetuned-tokenizer-path outputs/gpt2_gsm8k_distilled/tokenizer \
      --num-examples 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve each problem correctly and be concise."
)


@dataclass(frozen=True)
class ComparisonRow:
    """Single example comparison record."""

    dataset_index: int
    question: str
    gold_final: str
    base_pred_final: str
    finetuned_pred_final: str
    base_correct: bool
    finetuned_correct: bool
    improved: bool
    regressed: bool
    base_output: str
    finetuned_output: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare outputs from base GPT-2 and finetuned GPT-2 on GSM8K."
    )
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument(
        "--finetuned-model-path",
        default="outputs/gpt2_gsm8k_distilled/model_final",
    )
    parser.add_argument(
        "--finetuned-tokenizer-path",
        default="outputs/gpt2_gsm8k_distilled/tokenizer",
    )
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply each tokenizer chat template when available.",
    )
    parser.add_argument(
        "--prepend-system-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prepend SYSTEM_PROMPT when chat template is disabled.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Number of improved/regressed examples to print.",
    )
    parser.add_argument(
        "--report-path",
        default="outputs/gpt2_pre_vs_post_compare.json",
        help="JSON report path with metrics and sampled rows.",
    )
    parser.add_argument(
        "--csv-path",
        default="outputs/gpt2_pre_vs_post_compare_rows.csv",
        help="CSV path with per-example comparison rows.",
    )
    parser.add_argument(
        "--plots-dir",
        default="outputs/figures",
        help="Directory for generated comparison plots.",
    )
    parser.add_argument(
        "--plot-prefix",
        default="gpt2_pre_vs_post",
        help="Filename prefix for generated plot PNG files.",
    )
    parser.add_argument(
        "--write-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write PNG comparison plots.",
    )
    return parser.parse_args()


def choose_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    """Resolve dtype from CLI and hardware."""
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "fp32":
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def choose_device(device_arg: str) -> torch.device:
    """Resolve torch device."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_prompt(question: str) -> str:
    """Build strict prompt used in distillation/finetuning."""
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


def maybe_apply_chat_template(
    tokenizer: AutoTokenizer,
    prompt: str,
    use_chat_template: bool,
    prepend_system_prompt: bool,
) -> str:
    """Format prompt with chat template when available."""
    if use_chat_template and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if prepend_system_prompt:
        return f"{SYSTEM_PROMPT}\n\n{prompt}"
    return prompt


def extract_tag(text: str, tag: str) -> str | None:
    """Extract content from XML-like tags."""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def extract_last_number(text: str) -> str | None:
    """Extract final numeric token from free-form text."""
    cleaned = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    return matches[-1]


def canonicalize_number(text: str) -> str | None:
    """Convert numeric string to canonical format."""
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


def extract_gold_final(answer: str) -> str | None:
    """Extract GSM8K final answer canonical number."""
    if "####" in answer:
        answer = answer.split("####")[-1]
    return canonicalize_number(answer)


def is_correct(pred_final: str | None, gold_final: str | None) -> bool:
    """Check exact/floating numeric match."""
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


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load model/tokenizer pair."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[str]:
    """Generate outputs for a batch of prompts."""
    encoded = tokenizer(
        prompts,
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

    with torch.inference_mode():
        generated = model.generate(**encoded, **generate_kwargs)

    prompt_len = encoded["input_ids"].shape[1]
    generated_only = generated[:, prompt_len:]
    return tokenizer.batch_decode(generated_only, skip_special_tokens=True)


def predict_all(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[str]:
    """Run batched generation for all prompts."""
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        outputs.extend(
            generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )
        )
    return outputs


def extract_pred_final(text: str) -> str | None:
    """Parse predicted final number from model output."""
    pred_final_raw = extract_tag(text, "final")
    if pred_final_raw is None:
        pred_final_raw = extract_last_number(text)
    return canonicalize_number(pred_final_raw or "")


def to_status_label(row: ComparisonRow) -> str:
    """Map row to transition class."""
    if row.base_correct and row.finetuned_correct:
        return "both_correct"
    if row.base_correct and (not row.finetuned_correct):
        return "regressed"
    if (not row.base_correct) and row.finetuned_correct:
        return "improved"
    return "both_wrong"


def write_rows_csv(rows: list[ComparisonRow], csv_path: Path) -> None:
    """Write detailed per-example rows as CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_index",
        "question",
        "gold_final",
        "base_pred_final",
        "finetuned_pred_final",
        "base_correct",
        "finetuned_correct",
        "improved",
        "regressed",
        "status",
        "base_output",
        "finetuned_output",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_dict = asdict(row)
            row_dict["status"] = to_status_label(row)
            writer.writerow(row_dict)


def write_plots(
    *,
    plots_dir: Path,
    plot_prefix: str,
    evaluated: int,
    base_correct: int,
    finetuned_correct: int,
    base_parsed: int,
    finetuned_parsed: int,
    improved: int,
    regressed: int,
) -> list[str]:
    """Write comparison PNG plots and return created file paths."""
    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Missing dependency `matplotlib`. Install with: uv pip install matplotlib"
        ) from exc

    plots_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[str] = []

    # Plot 1: accuracy and parse rates
    rate_labels = ["accuracy", "parsed_rate"]
    base_rates = [
        (base_correct / evaluated) if evaluated else 0.0,
        (base_parsed / evaluated) if evaluated else 0.0,
    ]
    finetuned_rates = [
        (finetuned_correct / evaluated) if evaluated else 0.0,
        (finetuned_parsed / evaluated) if evaluated else 0.0,
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = list(range(len(rate_labels)))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x],
        base_rates,
        width=width,
        label="gpt2_base",
        color="#4c78a8",
        edgecolor="none",
    )
    ax.bar(
        [i + width / 2 for i in x],
        finetuned_rates,
        width=width,
        label="gpt2_finetuned",
        color="#f58518",
        edgecolor="none",
    )
    ax.set_xticks(x, rate_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Base vs Finetuned: Accuracy and Parse Rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    metrics_path = plots_dir / f"{plot_prefix}__metrics.png"
    fig.savefig(metrics_path, dpi=160)
    plt.close(fig)
    created_paths.append(str(metrics_path))

    # Plot 2: transition counts
    transition_labels = ["both_wrong", "improved", "regressed", "both_correct"]
    both_correct = max(0, base_correct - regressed)
    transition_counts = [
        max(0, evaluated - improved - regressed - both_correct),
        improved,
        regressed,
        both_correct,
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(
        transition_labels,
        transition_counts,
        color=["#9d9d9d", "#54a24b", "#e45756", "#4c78a8"],
        edgecolor="none",
    )
    ax.set_ylabel("count")
    ax.set_title("Prediction Transitions (Base -> Finetuned)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    transitions_path = plots_dir / f"{plot_prefix}__transitions.png"
    fig.savefig(transitions_path, dpi=160)
    plt.close(fig)
    created_paths.append(str(transitions_path))

    return created_paths


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "Missing dependency `datasets`. Install with: uv pip install datasets"
        ) from exc

    finetuned_model_path = Path(args.finetuned_model_path)
    if not finetuned_model_path.exists():
        raise FileNotFoundError(
            f"Finetuned model path not found: {finetuned_model_path}"
        )

    finetuned_tokenizer_path = Path(args.finetuned_tokenizer_path)
    if finetuned_tokenizer_path.exists():
        finetuned_tokenizer_id = str(finetuned_tokenizer_path)
    else:
        finetuned_tokenizer_id = args.base_model

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    print(
        f"[setup] device={device} dtype={dtype} batch_size={args.batch_size} "
        f"split={args.split}",
        flush=True,
    )

    base_tokenizer, base_model = load_model_and_tokenizer(
        model_name_or_path=args.base_model,
        tokenizer_name_or_path=args.base_model,
        device=device,
        dtype=dtype,
    )
    finetuned_tokenizer, finetuned_model = load_model_and_tokenizer(
        model_name_or_path=str(finetuned_model_path),
        tokenizer_name_or_path=finetuned_tokenizer_id,
        device=device,
        dtype=dtype,
    )

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    total = len(dataset)
    start = max(0, args.start_index)
    end = min(total, start + max(1, args.num_examples))
    if start >= end:
        raise ValueError(f"Invalid range start={start} end={end} total={total}")

    questions = dataset[start:end]["question"]
    answers = dataset[start:end]["answer"]
    prompts = [build_prompt(question) for question in questions]
    base_inputs = [
        maybe_apply_chat_template(
            base_tokenizer,
            prompt,
            args.use_chat_template,
            args.prepend_system_prompt,
        )
        for prompt in prompts
    ]
    finetuned_inputs = [
        maybe_apply_chat_template(
            finetuned_tokenizer,
            prompt,
            args.use_chat_template,
            args.prepend_system_prompt,
        )
        for prompt in prompts
    ]

    print(f"[run] generating base model outputs for {len(prompts)} examples...", flush=True)
    base_outputs = predict_all(
        model=base_model,
        tokenizer=base_tokenizer,
        prompts=base_inputs,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    print("[run] generating finetuned model outputs...", flush=True)
    finetuned_outputs = predict_all(
        model=finetuned_model,
        tokenizer=finetuned_tokenizer,
        prompts=finetuned_inputs,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    rows: list[ComparisonRow] = []
    base_parsed = 0
    finetuned_parsed = 0
    base_correct = 0
    finetuned_correct = 0
    improved = 0
    regressed = 0

    for i, (question, answer, base_text, finetuned_text) in enumerate(
        zip(questions, answers, base_outputs, finetuned_outputs)
    ):
        gold = extract_gold_final(answer) or ""
        base_pred = extract_pred_final(base_text)
        finetuned_pred = extract_pred_final(finetuned_text)
        if base_pred is not None:
            base_parsed += 1
        if finetuned_pred is not None:
            finetuned_parsed += 1

        base_ok = is_correct(base_pred, gold)
        finetuned_ok = is_correct(finetuned_pred, gold)
        if base_ok:
            base_correct += 1
        if finetuned_ok:
            finetuned_correct += 1

        is_improved = (not base_ok) and finetuned_ok
        is_regressed = base_ok and (not finetuned_ok)
        if is_improved:
            improved += 1
        if is_regressed:
            regressed += 1

        rows.append(
            ComparisonRow(
                dataset_index=start + i,
                question=question,
                gold_final=gold,
                base_pred_final=base_pred or "",
                finetuned_pred_final=finetuned_pred or "",
                base_correct=base_ok,
                finetuned_correct=finetuned_ok,
                improved=is_improved,
                regressed=is_regressed,
                base_output=base_text.strip(),
                finetuned_output=finetuned_text.strip(),
            )
        )

    evaluated = len(rows)
    base_acc = base_correct / evaluated if evaluated else 0.0
    finetuned_acc = finetuned_correct / evaluated if evaluated else 0.0

    print("[summary]", flush=True)
    print(
        f"  evaluated={evaluated} range=[{start}, {end}) split={args.split}",
        flush=True,
    )
    print(
        f"  base: parsed={base_parsed}/{evaluated} correct={base_correct}/{evaluated} "
        f"acc={base_acc:.4f}",
        flush=True,
    )
    print(
        f"  finetuned: parsed={finetuned_parsed}/{evaluated} "
        f"correct={finetuned_correct}/{evaluated} acc={finetuned_acc:.4f}",
        flush=True,
    )
    print(
        f"  delta_acc={finetuned_acc - base_acc:+.4f} improved={improved} regressed={regressed}",
        flush=True,
    )

    if args.show_examples > 0:
        print("[examples] improved", flush=True)
        for row in [r for r in rows if r.improved][: args.show_examples]:
            print(
                (
                    f"  idx={row.dataset_index} gold={row.gold_final} "
                    f"base={row.base_pred_final or 'N/A'} -> "
                    f"finetuned={row.finetuned_pred_final or 'N/A'}"
                ),
                flush=True,
            )

        print("[examples] regressed", flush=True)
        for row in [r for r in rows if r.regressed][: args.show_examples]:
            print(
                (
                    f"  idx={row.dataset_index} gold={row.gold_final} "
                    f"base={row.base_pred_final or 'N/A'} -> "
                    f"finetuned={row.finetuned_pred_final or 'N/A'}"
                ),
                flush=True,
            )

    report_path = Path(args.report_path)
    csv_path = Path(args.csv_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    write_rows_csv(rows, csv_path)
    print(f"[done] wrote rows CSV to {csv_path}", flush=True)

    plot_paths: list[str] = []
    if args.write_plots:
        plot_paths = write_plots(
            plots_dir=Path(args.plots_dir),
            plot_prefix=args.plot_prefix,
            evaluated=evaluated,
            base_correct=base_correct,
            finetuned_correct=finetuned_correct,
            base_parsed=base_parsed,
            finetuned_parsed=finetuned_parsed,
            improved=improved,
            regressed=regressed,
        )
        for path in plot_paths:
            print(f"[done] wrote plot to {path}", flush=True)

    report = {
        "base_model": args.base_model,
        "finetuned_model_path": str(finetuned_model_path),
        "finetuned_tokenizer_path": finetuned_tokenizer_id,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "start_index": start,
        "end_index": end,
        "evaluated": evaluated,
        "base_parsed": base_parsed,
        "finetuned_parsed": finetuned_parsed,
        "base_correct": base_correct,
        "finetuned_correct": finetuned_correct,
        "base_accuracy": base_acc,
        "finetuned_accuracy": finetuned_acc,
        "delta_accuracy": finetuned_acc - base_acc,
        "improved": improved,
        "regressed": regressed,
        "csv_path": str(csv_path),
        "plot_paths": plot_paths,
        "rows": [asdict(row) for row in rows[: max(20, args.show_examples * 2)]],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote report to {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
