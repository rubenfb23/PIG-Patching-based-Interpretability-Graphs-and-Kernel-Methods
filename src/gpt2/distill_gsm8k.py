"""Teacher distillation pipeline for GSM8K using openai/gpt-oss-20b.

This script generates teacher rationales and filtered final answers, then writes
JSONL ready for `gpt2.finetuning` (`text` field contains prompt+completion).

Example:
    python -m gpt2.distill_gsm8k \
      --teacher-model openai/gpt-oss-20b \
      --output-path outputs/gsm8k_distilled_gptoss20b.jsonl \
      --max-examples 2000 \
      --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency at runtime
    tqdm = None


SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve each problem correctly and be concise."
)


@dataclass(frozen=True)
class DistillationExample:
    """Record stored in the distilled JSONL file."""

    dataset_index: int
    question: str
    prompt: str
    completion: str
    text: str
    gold_final: str
    pred_final: str
    correct: bool


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a filtered teacher-distilled GSM8K dataset for GPT-2 SFT."
        )
    )
    parser.add_argument("--teacher-model", default="openai/gpt-oss-20b")
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-path", default="outputs/gsm8k_distilled_gptoss20b.jsonl")
    parser.add_argument("--meta-path", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
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
        "--attn-implementation",
        choices=("auto", "eager", "sdpa"),
        default="eager",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Device map for model loading (e.g., "auto", "balanced", "balanced_low_0").',
    )
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=float,
        default=0.0,
        help="Per-GPU max memory cap in GiB. 0 means auto from current free memory.",
    )
    parser.add_argument(
        "--gpu-memory-reserve-gib",
        type=float,
        default=2.0,
        help="Reserved free memory per GPU when auto-calculating max memory.",
    )
    parser.add_argument(
        "--cpu-offload-gib",
        type=float,
        default=96.0,
        help="CPU RAM budget for offload in GiB.",
    )
    parser.add_argument(
        "--offload-folder",
        default="outputs/offload/gpt_oss_teacher",
        help="Folder used by accelerate when dispatching/offloading modules.",
    )
    parser.add_argument(
        "--mxfp4-dequantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For GPT-OSS checkpoints, force MXFP4 dequantize=True (stable, no kernel "
            "compilation needed). Disable only if Triton kernels toolchain is fully set."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use tokenizer chat template when available.",
    )
    parser.add_argument(
        "--include-incorrect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write incorrect teacher outputs (default keeps only correct).",
    )
    parser.add_argument(
        "--normalize-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewrite completion to strict <reasoning>/<final> tags when parsable.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bar during distillation.",
    )
    parser.add_argument(
        "--wandb-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases metric logging.",
    )
    parser.add_argument(
        "--wandb-project",
        default="pig-distill",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default="",
        help="W&B entity/team (optional).",
    )
    parser.add_argument(
        "--wandb-run-name",
        default="",
        help="W&B run name (optional).",
    )
    parser.add_argument(
        "--wandb-group",
        default="",
        help="W&B group name (optional).",
    )
    parser.add_argument(
        "--wandb-tags",
        default="distill,gsm8k",
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default="online",
        help="W&B mode. Use offline to avoid immediate upload.",
    )
    return parser.parse_args()


def choose_dtype(dtype_arg: str) -> torch.dtype:
    """Resolve model dtype from CLI and available hardware."""
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "fp32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def build_prompt(question: str) -> str:
    """Build strict teacher prompt with parseable tags."""
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
    return f"{SYSTEM_PROMPT}\n\n{prompt}"


def extract_tag(text: str, tag: str) -> str | None:
    """Extract tagged content from teacher output."""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def canonicalize_number(text: str) -> str | None:
    """Convert a numeric string into a canonical comparable format."""
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


def extract_last_number(text: str) -> str | None:
    """Extract the last numeric token from free-form model output."""
    cleaned = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    return matches[-1]


def extract_gold_final(answer: str) -> str | None:
    """Extract canonical GSM8K gold final number."""
    if "####" in answer:
        answer = answer.split("####")[-1]
    return canonicalize_number(answer)


def is_correct(pred_final: str | None, gold_final: str | None) -> bool:
    """Check exact numeric match with fallback float tolerance."""
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


def render_completion(reasoning: str | None, pred_final_raw: str | None, raw_text: str) -> str:
    """Render normalized completion when both tags are available."""
    if reasoning is None or pred_final_raw is None:
        return raw_text.strip()
    return (
        "<reasoning>\n"
        f"{reasoning.strip()}\n"
        "</reasoning>\n\n"
        "<final>\n"
        f"{pred_final_raw.strip()}\n"
        "</final>"
    )


def load_teacher(
    teacher_model: str,
    dtype: torch.dtype,
    attn_implementation: str,
    device_map: str,
    max_gpu_memory_gib: float,
    gpu_memory_reserve_gib: float,
    cpu_offload_gib: float,
    offload_folder: Path,
    mxfp4_dequantize: bool,
    trust_remote_code: bool,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer/model with multi-GPU aware defaults."""
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_model,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation
    if "gpt-oss" in teacher_model.lower():
        model_kwargs["quantization_config"] = Mxfp4Config(dequantize=mxfp4_dequantize)
        print(
            f"[setup] mxfp4_dequantize={mxfp4_dequantize}",
            flush=True,
        )

    if torch.cuda.is_available():
        model_kwargs["dtype"] = dtype
        model_kwargs["device_map"] = device_map

        max_memory: dict[int | str, str] = {}
        for gpu_idx in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_idx)
            free_gib = free_bytes / 1024**3
            total_gib = total_bytes / 1024**3

            if max_gpu_memory_gib > 0:
                limit_gib = min(max_gpu_memory_gib, total_gib - 0.5)
            else:
                limit_gib = min(total_gib - 0.5, free_gib - gpu_memory_reserve_gib)
                limit_gib = max(4.0, limit_gib)

            max_memory[gpu_idx] = f"{int(limit_gib)}GiB"

        max_memory["cpu"] = f"{int(max(8.0, cpu_offload_gib))}GiB"
        model_kwargs["max_memory"] = max_memory
        offload_folder.mkdir(parents=True, exist_ok=True)
        model_kwargs["offload_folder"] = str(offload_folder)
        model_kwargs["offload_state_dict"] = True
        print(f"[setup] max_memory map: {max_memory}", flush=True)
    else:
        model_kwargs["dtype"] = torch.float32
        model_kwargs["device_map"] = {"": "cpu"}

    try:
        model = AutoModelForCausalLM.from_pretrained(teacher_model, **model_kwargs)
    except ValueError as exc:
        requested_attn = str(model_kwargs.get("attn_implementation", "auto"))
        can_fallback = (
            requested_attn not in ("auto", "eager")
            and "attn_implementation" in model_kwargs
            and (
                "scaled_dot_product_attention" in str(exc)
                or "attn_implementation=\"eager\"" in str(exc)
                or "does not support an attention implementation" in str(exc)
            )
        )
        if not can_fallback:
            raise
        model_kwargs["attn_implementation"] = "eager"
        print(
            (
                f"[warn] teacher model does not support attn_implementation="
                f"{requested_attn}; retrying with eager"
            ),
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(teacher_model, **model_kwargs)
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
) -> list[str]:
    """Generate teacher outputs for one prompt batch."""
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}

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


def init_wandb_run(
    args: argparse.Namespace,
    dtype: torch.dtype,
    total_examples: int,
) -> Any | None:
    """Initialize optional Weights & Biases run."""
    if not args.wandb_enabled:
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
            "teacher_model": args.teacher_model,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "start_index": args.start_index,
            "max_examples": args.max_examples,
            "batch_size": args.batch_size,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "dtype": str(dtype),
            "attn_implementation": args.attn_implementation,
            "use_chat_template": args.use_chat_template,
            "include_incorrect": args.include_incorrect,
            "normalize_tags": args.normalize_tags,
            "total_examples": total_examples,
            "output_path": args.output_path,
        },
    )
    run.define_metric("attempted")
    run.define_metric("distill/*", step_metric="attempted")
    if getattr(run, "url", None):
        print(f"[wandb] run_url={run.url}", flush=True)
    return run


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "Missing dependency `datasets`. Install with: uv pip install datasets accelerate"
        ) from exc

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = (
        Path(args.meta_path)
        if args.meta_path
        else output_path.with_suffix(output_path.suffix + ".meta.json")
    )

    dtype = choose_dtype(args.dtype)
    print(
        f"[setup] teacher={args.teacher_model} dtype={dtype} batch_size={args.batch_size}",
        flush=True,
    )
    tokenizer, teacher = load_teacher(
        teacher_model=args.teacher_model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
        gpu_memory_reserve_gib=args.gpu_memory_reserve_gib,
        cpu_offload_gib=args.cpu_offload_gib,
        offload_folder=Path(args.offload_folder),
        mxfp4_dequantize=args.mxfp4_dequantize,
        trust_remote_code=args.trust_remote_code,
    )

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
    )
    total_available = len(dataset)
    start_index = max(0, args.start_index)
    end_index = total_available
    if args.max_examples > 0:
        end_index = min(total_available, start_index + args.max_examples)
    if start_index >= end_index:
        raise ValueError(
            f"Invalid range: start_index={start_index} end_index={end_index}"
        )

    attempted = 0
    kept = 0
    correct = 0
    parsed_final = 0
    progress = None
    total_examples = end_index - start_index
    wandb_run = init_wandb_run(args=args, dtype=dtype, total_examples=total_examples)
    started_at = time.perf_counter()
    last_logged_attempted = 0

    def log_metrics(force: bool = False) -> None:
        nonlocal last_logged_attempted
        if wandb_run is None:
            return
        if attempted == 0 and not force:
            return
        if not force and (attempted - last_logged_attempted) < max(1, args.log_interval):
            return
        elapsed_s = max(1e-9, time.perf_counter() - started_at)
        keep_rate = (kept / attempted) if attempted else 0.0
        acc_estimate = (correct / attempted) if attempted else 0.0
        parse_rate = (parsed_final / attempted) if attempted else 0.0
        ex_per_sec = (attempted / elapsed_s) if attempted else 0.0
        wandb_run.log(
            {
                "attempted": attempted,
                "distill/kept": kept,
                "distill/correct": correct,
                "distill/parsed_final": parsed_final,
                "distill/keep_rate": keep_rate,
                "distill/accuracy_estimate": acc_estimate,
                "distill/parse_rate": parse_rate,
                "distill/examples_per_sec": ex_per_sec,
            }
        )
        last_logged_attempted = attempted

    if args.progress_bar and tqdm is not None:
        progress = tqdm(
            total=total_examples,
            desc="distill",
            unit="ex",
            dynamic_ncols=True,
        )
    log_metrics(force=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for batch_start in range(start_index, end_index, args.batch_size):
            batch_end = min(end_index, batch_start + args.batch_size)
            batch = dataset[batch_start:batch_end]
            questions: list[str] = batch["question"]
            answers: list[str] = batch["answer"]

            prompts = [build_prompt(question) for question in questions]
            formatted_prompts = [
                maybe_apply_chat_template(tokenizer, prompt, args.use_chat_template)
                for prompt in prompts
            ]
            outputs = generate_batch(
                model=teacher,
                tokenizer=tokenizer,
                prompts=formatted_prompts,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            for local_i, output_text in enumerate(outputs):
                dataset_index = batch_start + local_i
                attempted += 1

                question = questions[local_i]
                answer = answers[local_i]
                prompt = prompts[local_i]

                reasoning = extract_tag(output_text, "reasoning")
                pred_final_raw = extract_tag(output_text, "final")
                if pred_final_raw is None:
                    pred_final_raw = extract_last_number(output_text)
                pred_final = canonicalize_number(pred_final_raw or "")
                gold_final = extract_gold_final(answer)

                if pred_final is not None:
                    parsed_final += 1

                ok = is_correct(pred_final, gold_final)
                if ok:
                    correct += 1
                if not ok and not args.include_incorrect:
                    continue

                if args.normalize_tags:
                    completion = render_completion(reasoning, pred_final_raw, output_text)
                else:
                    completion = output_text.strip()
                if not completion:
                    continue

                record = DistillationExample(
                    dataset_index=dataset_index,
                    question=question,
                    prompt=prompt,
                    completion=completion,
                    text=f"{prompt}{completion}",
                    gold_final=gold_final or "",
                    pred_final=pred_final or "",
                    correct=ok,
                )
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                kept += 1

            processed_this_batch = len(outputs)
            if progress is not None:
                progress.update(processed_this_batch)
                if (attempted - last_logged_attempted) >= max(1, args.log_interval):
                    progress.set_postfix(
                        attempted=attempted,
                        kept=kept,
                        correct=correct,
                        parsed=parsed_final,
                    )
                    log_metrics()
            elif (attempted - last_logged_attempted) >= max(1, args.log_interval):
                print(
                    (
                        f"[progress] attempted={attempted} kept={kept} "
                        f"correct={correct} parsed_final={parsed_final}"
                    ),
                    flush=True,
                )
                log_metrics()

    if progress is not None:
        progress.set_postfix(
            attempted=attempted,
            kept=kept,
            correct=correct,
            parsed=parsed_final,
        )
        progress.close()

    metadata = {
        "teacher_model": args.teacher_model,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "start_index": start_index,
        "end_index": end_index,
        "attempted": attempted,
        "kept": kept,
        "correct": correct,
        "parsed_final": parsed_final,
        "keep_rate": (kept / attempted) if attempted else 0.0,
        "teacher_accuracy_estimate": (correct / attempted) if attempted else 0.0,
        "include_incorrect": args.include_incorrect,
        "output_path": str(output_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if wandb_run is not None:
        log_metrics(force=True)
        wandb_run.summary.update(metadata)
        wandb_run.finish()
    print(f"[done] wrote dataset to {output_path}", flush=True)
    print(f"[done] wrote metadata to {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
