"""
Ornith MLX inference optimization script. Agent-editable.
Usage: uv run optimize.py
"""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

from prepare import TIMEOUT_SECONDS, evaluate_decode_tps, verify_model_exists

# ---------------------------------------------------------------------------
# Hyperparameters — edit these (and helper functions below)
# ---------------------------------------------------------------------------

MODEL_PATH = None  # None => use prepare.MODEL_PATH

# Chat / reasoning control
SYSTEM_PROMPT = "You are a concise assistant. Answer directly."
ENABLE_THINKING = False
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}

# Sampling
TEMPERATURE = 0.1
TOP_P = 1.0
TOP_K = 0
MIN_P = 0.0
SEED = 42

# Generation engine knobs (mlx_lm.generate_step / stream_generate)
PREFILL_STEP_SIZE = 4096
KV_BITS: int | None = None
KV_GROUP_SIZE = 64
QUANTIZED_KV_START = 5000
EXTRA_EOS_TOKENS: list[str] = []

# Metal memory wiring (Apple Silicon). None = mlx_lm default.
WIRED_LIMIT: bool | None = True

# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def resolve_model_path() -> str:
    if MODEL_PATH is not None:
        return str(MODEL_PATH)
    from prepare import MODEL_PATH as default_path

    return str(default_path)


def build_messages(user_text: str, case_name: str) -> list[dict[str, str]]:
    """Format chat messages sent to the tokenizer."""
    messages: list[dict[str, str]] = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_text})
    return messages


def build_sampler():
    if SEED is not None:
        mx.random.seed(SEED)
    return make_sampler(
        temp=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        min_p=MIN_P,
    )


def chat_template_kwargs() -> dict[str, Any]:
    """Kwargs forwarded to tokenizer.apply_chat_template."""
    kwargs = dict(CHAT_TEMPLATE_KWARGS)
    if not ENABLE_THINKING:
        kwargs["enable_thinking"] = False
    return kwargs


def build_generation_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prefill_step_size": PREFILL_STEP_SIZE,
        "kv_group_size": KV_GROUP_SIZE,
        "quantized_kv_start": QUANTIZED_KV_START,
    }
    if KV_BITS is not None:
        kwargs["kv_bits"] = KV_BITS
    if EXTRA_EOS_TOKENS:
        kwargs["extra_eos_tokens"] = EXTRA_EOS_TOKENS
    return kwargs


def post_load(model: Any, tokenizer: Any) -> None:
    """Optional post-load tweaks (wired memory, model patches, etc.)."""
    if WIRED_LIMIT and mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    verify_model_exists()
    model_path = resolve_model_path()

    t_start = time.perf_counter()
    model, tokenizer = load(model_path)
    load_seconds = time.perf_counter() - t_start

    post_load(model, tokenizer)

    t_eval = time.perf_counter()
    result = evaluate_decode_tps(model, tokenizer, __import__(__name__))
    eval_seconds = time.perf_counter() - t_eval
    total_seconds = time.perf_counter() - t_start

    print("---")
    print(f"decode_tok_s:     {result.decode_tok_s:.6f}")
    print(f"prompt_tok_s:     {result.prompt_tok_s:.6f}")
    print(f"peak_memory_gb:   {result.peak_memory_gb:.3f}")
    print(f"load_seconds:     {load_seconds:.1f}")
    print(f"eval_seconds:     {eval_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    for case in result.case_results:
        print(
            f"case_{case.name}_decode_tok_s: {case.decode_tok_s:.3f} "
            f"(gen={case.generation_tokens}, wall={case.wall_s:.1f}s)"
        )


if __name__ == "__main__":
    main()