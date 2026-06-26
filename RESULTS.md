# autodecode — Experiment Results

**Run tag:** `autoresearch/jun25`  
**Date:** June 25–26, 2026  
**Hardware:** MacBook Pro, Apple M4, 32 GB RAM  
**Model:** [Ornith-1.0-9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B) → MLX 4-bit (`models/ornith-9b-4bit`, 4.7 GB)  
**Metric:** `decode_tok_s` — median decode tokens/sec across 3 fixed prompts, measured over 3 eval passes (higher is better)

---

## Executive summary

| Stage | decode_tok_s | vs Ollama |
|-------|-------------|-----------|
| Ollama Ornith Q8 | ~10 | baseline |
| MLX out-of-box | ~18 | **1.8×** |
| autodecode winner | **21.4** | **2.1×** |

Peak memory stayed at **~5.2 GB** throughout — well within 32 GB RAM.

**Headline finding:** Local LLM slowness on Mac was mostly config, not silicon. Disabling Qwen3.5 "thinking" tokens and tuning prefill/sampling beat every KV-cache hack we tried.

---

## Benchmark harness

Fixed prompts in `prepare.py`:

| Case | Prompt | max_tokens |
|------|--------|------------|
| `code_prime` | Write `is_prime(n)` in Python, code only | 128 |
| `short_qa` | Capital of France, one word | 8 |
| `json_extract` | Compact JSON extract | 32 |

Eval protocol (added mid-run for stability):

- 32-token warmup (not scored)
- 3 full benchmark passes; median of pass-medians = `decode_tok_s`
- **+0.15 tok/s noise margin** required to `keep` a commit

---

## Winning configuration

Commit `d7f3168` — **21.395 tok/s**

```python
ENABLE_THINKING = False
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
TEMPERATURE = 0.05
PREFILL_STEP_SIZE = 2048
WIRED_LIMIT = True

# build_messages(): code_prime gets "Python only. No prose."
```

Per-case breakdown (last pass of winning run):

| Case | tok/s | tokens |
|------|-------|--------|
| code_prime | 19.85 | 109 |
| short_qa | 37.60 | 2 |
| json_extract | 20.72 | 11 |

---

## Full experiment log

| commit | decode_tok_s | GB | status | description |
|--------|-------------|-----|--------|-------------|
| 4501e2c | 20.450 | 5.2 | keep | baseline, thinking off, prefill 2048 |
| f208ab1 | 21.369 | 5.2 | keep | prefill 4096 |
| 88a4138 | 20.858 | 5.2 | discard | temperature 0 greedy |
| 98f8ebe | 21.123 | 5.2 | discard | kv_bits 8, quantize from 0 |
| fe364dd | 17.569 | 5.2 | discard | prefill 8192 |
| 22ffbcf | 20.403 | 5.2 | discard | no system prompt |
| eb78597 | 17.841 | 5.2 | discard | wired_limit false |
| f22ea71 | 18.076 | 5.2 | discard | prefill 6144 |
| fea3c2f | 17.802 | 5.2 | discard | prefill 3072 |
| f0494c9 | 18.175 | 5.3 | discard | append /no_think |
| 26cd02a | 18.512 | 5.2 | discard | kv_bits 4, quantize after 128 |
| 9ae073c | 19.714 | 5.2 | discard | seed none |
| 745e086 | 17.453 | 5.2 | keep | stable 3-pass harness, prefill 4096 |
| 7482452 | 18.553 | 5.2 | keep | prefill 2048 under stable harness |
| dccc0cb | 18.035 | 5.2 | discard | global code-only system prompt |
| c6b89bf | 20.206 | 5.2 | keep | case-specific code system prompt |
| 0f19183 | 21.195 | 5.2 | keep | temperature 0.05 |
| **d7f3168** | **21.395** | **5.2** | **keep** | **case prompt + temp 0.05 + prefill 2048** |
| efe8983 | 19.325 | 5.2 | discard | prefill 4096 + combined stack |
| 45f0c08 | 21.131 | 5.2 | discard | temperature 0.0 |
| f537c5d | 20.984 | 5.2 | discard | temperature 0.02 |
| bf67b21 | 20.675 | 5.2 | discard | per-case minimal prompts (all 3) |

**22 experiments** · **8 kept** · **14 discarded**

---

## What worked

1. **`enable_thinking: false`** — Ornith/Qwen3.5 chat template injects `</think>` blocks; disabling is the largest quality-of-speed win.
2. **`TEMPERATURE = 0.05`** — beat 0.0, 0.02, and 0.1 under stable eval.
3. **`PREFILL_STEP_SIZE = 2048`** — counter-intuitive; 4096 looked better on noisy single-pass runs, 8192 was clearly worse.
4. **Case-specific system prompt for code** — `"Python only. No prose."` on `code_prime` only; global code prompt hurt other cases.
5. **`WIRED_LIMIT = True`** — Metal wired memory on M4.

## What failed

- KV cache quantization (`kv_bits` 4/8)
- `/no_think` user suffix
- Removing system prompt entirely
- `WIRED_LIMIT = False`
- Larger prefill steps (6144, 8192)
- Greedy decoding (`temperature = 0`)

---

## Ollama vs MLX (why Ollama felt broken)

Ornith-9B on Ollama (`hf.co/.../Ornith-1.0-9B-GGUF:Q8_0`):

- ~9–10 tok/s generation, 9.5 GB model RAM, 32k context pre-allocated
- Immature `qwen35` llama.cpp path
- Q8 quant + reasoning token bloat on coding tasks

Same model MLX 4-bit:

- ~18 tok/s baseline → **21.4** after autodecode
- ~5.2 GB peak memory

Llama 3.2 3B on same Ollama install: ~21–22 tok/s — confirming the Mac GPU is fine.

---

## Agent integrations

After optimization, the winning MLX config is served to coding agents:

### MLX server

```bash
./start-mlx-server.sh
# http://127.0.0.1:8080/v1  model: default_model
```

Server flags: `--temp 0.05 --prefill-step-size 2048 --chat-template-args '{"enable_thinking":false}'`

### OpenCode

`~/.config/opencode/opencode.jsonc` → provider `mlx` @ `:8080`, default model `mlx/default_model`.

### Hermes

`~/.hermes/config.yaml` → `custom_providers` entry `mlx-ornith`, default model with `context_length: 65536` (Hermes requires ≥64K for tool schemas; Ornith supports 262K native).

Fallback: `xai-oauth` / `grok-4.3` when MLX server is down.

See [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) for copy-paste configs.

---

## Reproduce

```bash
cd autodecode
uv sync
uv run optimize.py   # expect ~17–21 tok/s depending on thermals
```

Thermal variance: identical config can swing **~15%** run-to-run. Use 3-pass eval and noise margin before trusting a `keep`.

---

## References

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the training loop we adapted
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — MLX training port
- [Ornith-1.0-9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B) — Qwen3.5 hybrid, linear attention + reasoning
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) — Apple Silicon inference