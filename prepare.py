"""
Fixed benchmark harness for ornith-mlx-autoresearch.
DO NOT MODIFY — agents edit optimize.py only.

Usage:
    uv run prepare.py          # verify model + print baseline metrics
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

# ---------------------------------------------------------------------------
# Fixed constants (do not modify)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ornith-9b-4bit"
WARMUP_MAX_TOKENS = 32
EVAL_REPEATS = 3
TIMEOUT_SECONDS = 600

BENCHMARK_CASES: list[dict[str, Any]] = [
    {
        "name": "code_prime",
        "user": "Write a Python function is_prime(n). Return only code, no explanation.",
        "max_tokens": 128,
    },
    {
        "name": "short_qa",
        "user": "What is the capital of France? One word only.",
        "max_tokens": 8,
    },
    {
        "name": "json_extract",
        "user": 'Extract {"name": "Ada", "role": "engineer"} as compact JSON only.',
        "max_tokens": 32,
    },
]


@dataclass
class CaseResult:
    name: str
    decode_tok_s: float
    prompt_tok_s: float
    generation_tokens: int
    prompt_tokens: int
    peak_memory_gb: float
    wall_s: float
    output_chars: int


@dataclass
class BenchmarkResult:
    decode_tok_s: float
    prompt_tok_s: float
    peak_memory_gb: float
    total_seconds: float
    case_results: list[CaseResult]
    status: str = "ok"


def _count_reasoning_chars(text: str) -> int:
    start = text.find("<think>")
    if start == -1:
        return 0
    end = text.find("</think>", start)
    if end == -1:
        return len(text) - start
    return end - start + len("</think>")


def run_case(
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    *,
    messages: list[dict[str, str]],
    sampler: Any,
    generation_kwargs: dict[str, Any],
    template_kwargs: dict[str, Any],
) -> CaseResult:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    max_tokens = case["max_tokens"]
    last: Any = None
    output_text = ""
    tic = time.perf_counter()
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        **generation_kwargs,
    ):
        last = response
        output_text += response.text
    wall_s = time.perf_counter() - tic
    if last is None:
        raise RuntimeError(f"no tokens generated for case {case['name']}")

    return CaseResult(
        name=case["name"],
        decode_tok_s=last.generation_tps,
        prompt_tok_s=last.prompt_tps,
        generation_tokens=last.generation_tokens,
        prompt_tokens=last.prompt_tokens,
        peak_memory_gb=last.peak_memory,
        wall_s=wall_s,
        output_chars=len(output_text),
    )


def evaluate_decode_tps(
    model: Any,
    tokenizer: Any,
    optimize_module: Any,
) -> BenchmarkResult:
    """
    Fixed evaluation metric: median decode tokens/sec across benchmark cases.
    Higher is better.
    """
    mx.reset_peak_memory()
    generation_kwargs = optimize_module.build_generation_kwargs()
    sampler = optimize_module.build_sampler()
    template_kwargs = optimize_module.chat_template_kwargs()

    # Warmup (not scored)
    warmup_messages = optimize_module.build_messages(
        user_text="Say hi.",
        case_name="warmup",
    )
    warmup_prompt = tokenizer.apply_chat_template(
        warmup_messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    for _ in stream_generate(
        model,
        tokenizer,
        warmup_prompt,
        max_tokens=WARMUP_MAX_TOKENS,
        sampler=sampler,
        **generation_kwargs,
    ):
        pass
    mx.clear_cache()

    pass_medians: list[float] = []
    pass_prompt_medians: list[float] = []
    peak_mem = 0.0
    last_case_results: list[CaseResult] = []
    t0 = time.perf_counter()

    for _ in range(EVAL_REPEATS):
        case_results: list[CaseResult] = []
        for case in BENCHMARK_CASES:
            messages = optimize_module.build_messages(
                user_text=case["user"],
                case_name=case["name"],
            )
            case_results.append(
                run_case(
                    model,
                    tokenizer,
                    case,
                    messages=messages,
                    sampler=sampler,
                    generation_kwargs=generation_kwargs,
                    template_kwargs=template_kwargs,
                )
            )
            mx.clear_cache()

        decode_rates = [c.decode_tok_s for c in case_results]
        prompt_rates = [c.prompt_tok_s for c in case_results]
        pass_medians.append(statistics.median(decode_rates))
        pass_prompt_medians.append(statistics.median(prompt_rates))
        peak_mem = max(peak_mem, max(c.peak_memory_gb for c in case_results))
        last_case_results = case_results

    return BenchmarkResult(
        decode_tok_s=statistics.median(pass_medians),
        prompt_tok_s=statistics.median(pass_prompt_medians),
        peak_memory_gb=peak_mem,
        total_seconds=time.perf_counter() - t0,
        case_results=last_case_results,
    )


def verify_model_exists() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Convert Ornith-9B first (see convert-35b.sh / mlx_lm.convert)."
        )


def main() -> None:
    import optimize

    from mlx_lm import load

    verify_model_exists()
    print(f"Model: {MODEL_PATH}")
    model, tokenizer = load(str(MODEL_PATH))
    optimize.post_load(model, tokenizer)
    result = evaluate_decode_tps(model, tokenizer, optimize)
    print("---")
    print(f"decode_tok_s:     {result.decode_tok_s:.3f}")
    print(f"prompt_tok_s:     {result.prompt_tok_s:.3f}")
    print(f"peak_memory_gb:   {result.peak_memory_gb:.3f}")
    print(f"total_seconds:    {result.total_seconds:.1f}")
    for case in result.case_results:
        print(
            f"  {case.name}: decode={case.decode_tok_s:.2f} tok/s, "
            f"gen={case.generation_tokens} tok, wall={case.wall_s:.1f}s"
        )


if __name__ == "__main__":
    main()