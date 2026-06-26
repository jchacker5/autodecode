#!/usr/bin/env python3
"""Benchmark MLX vs Ollama for Ornith research."""

import json
import subprocess
import time
import urllib.request

PROMPT = "Write a Python function is_prime(n). Return only code."
MAX_TOKENS = 128


def bench_ollama(model: str) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "prompt": PROMPT,
            "stream": False,
            "options": {
                "num_predict": MAX_TOKENS,
                "num_ctx": 4096,
                "temperature": 0.1,
            },
        }
    ).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 1) / 1e9
    return {
        "backend": "ollama",
        "model": model,
        "tokens": eval_count,
        "tok_per_s": eval_count / eval_duration if eval_duration else 0,
        "wall_s": elapsed,
        "load_s": data.get("load_duration", 0) / 1e9,
    }


def bench_mlx(model_path: str) -> dict:
    cmd = [
        "/Users/jchacker5/Documents/ornith-mlx-research/.venv/bin/mlx_lm.generate",
        "--model",
        model_path,
        "--prompt",
        PROMPT,
        "--max-tokens",
        str(MAX_TOKENS),
        "--temp",
        "0.1",
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    out = proc.stdout + proc.stderr
    tok_s = None
    for line in out.splitlines():
        if "tokens/s" in line.lower() or "tok/s" in line.lower():
            for token in line.replace(",", "").split():
                try:
                    tok_s = float(token)
                    break
                except ValueError:
                    continue
    return {
        "backend": "mlx",
        "model": model_path,
        "tok_per_s": tok_s,
        "wall_s": elapsed,
        "exit_code": proc.returncode,
        "tail": "\n".join(out.splitlines()[-8:]),
    }


def main() -> None:
    results = []
    results.append(
        bench_ollama("hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:Q8_0")
    )
    mlx_path = "/Users/jchacker5/Documents/ornith-mlx-research/models/ornith-9b-4bit"
    try:
        results.append(bench_mlx(mlx_path))
    except Exception as exc:  # noqa: BLE001
        results.append({"backend": "mlx", "error": str(exc)})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()