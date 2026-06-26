# autodecode

**Karpathy-style autonomous research for Apple Silicon inference.**

> *autoresearch trains models overnight. autodecode makes them faster overnight.*

Give an AI agent one editable file (`optimize.py`), a fixed benchmark harness (`prepare.py`), and a single metric — **median decode tokens/sec**. It commits experiments, keeps winners, reverts losers, and loops until you stop it.

No fine-tuning. No new weights. Just config archaeology on MLX.

Built and validated on **Ornith-9B** (Qwen3.5 hybrid) on an **M4 MacBook Pro · 32 GB RAM**.

---

## Results (Jun 2026)

| Stack | Decode tok/s | Notes |
|-------|-------------|-------|
| Ollama Q8 GGUF | ~10 | Default local path most people hit |
| MLX 4-bit baseline | ~18 | `enable_thinking: false` |
| **autodecode winner** | **21.4** | +23% vs stable MLX baseline |

**22 agent experiments** in one session. Best config fits in ~15 lines of `optimize.py` changes.

See [RESULTS.md](RESULTS.md) for the full experiment log, lessons, and agent-integration setup.

---

## How it works

Deliberately tiny — same spirit as [karpathy/autoresearch](https://github.com/karpathy/autoresearch):

```
prepare.py    → fixed benchmark harness + metric (do not edit)
optimize.py   → inference knobs the agent edits
program.md    → autonomous loop instructions (the "skill")
results.tsv   → experiment log (gitignored during runs)
```

Each experiment:

```bash
uv run optimize.py > run.log 2>&1
grep "^decode_tok_s:" run.log
```

**Goal:** maximize `decode_tok_s` (higher is better).  
**Loop:** edit → commit → benchmark → keep or `git reset --hard`.

---

## Quick start

**Requirements:** Apple Silicon Mac, Python 3.11+, [uv](https://docs.astral.sh/uv/), ~6 GB disk for Ornith-9B MLX weights.

```bash
git clone https://github.com/jchacker5/autodecode.git
cd autodecode
uv sync

# Convert Ornith-9B once (~5 min, ~4.7 GB)
uv run mlx_lm.convert \
  --hf-path deepreinforce-ai/Ornith-1.0-9B \
  --mlx-path models/ornith-9b-4bit \
  -q --trust-remote-code

# Baseline benchmark
uv run optimize.py
```

Point your agent at `program.md`:

```
Read program.md. Create branch autoresearch/<tag>, run baseline, then loop forever.
Only edit optimize.py.
```

Or use the helper:

```bash
./run_loop.sh opencode
```

---

## Winning config (snapshot)

The current best on M4 32GB — all in `optimize.py`:

- `ENABLE_THINKING = False` / `enable_thinking: false`
- `TEMPERATURE = 0.05`
- `PREFILL_STEP_SIZE = 2048` (4096 and 8192 were *slower*)
- Case-specific system prompt for code: `"Python only. No prose."`
- `WIRED_LIMIT = True`

---

## Serve to coding agents

Start the OpenAI-compatible MLX server (uses autodecode settings):

```bash
./start-mlx-server.sh
# → http://127.0.0.1:8080/v1
```

Works with **OpenCode**, **Hermes**, and any OpenAI-compatible client.  
See [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md).

---

## What we learned

1. **Your Mac isn't the bottleneck** — Ollama's immature `qwen35` path + Q8 + 32k ctx + reasoning bloat dominates.
2. **MLX is ~2× faster** than Ollama for the same model before autodecode even runs.
3. **Bigger prefill ≠ faster** — `prefill_step_size=8192` lost to `2048` under stable eval.
4. **Disable thinking** — Qwen3.5's `</think>` blocks are the biggest single win.
5. **Benchmark variance is real** — 3-pass eval + 0.15 tok/s noise margin before `keep`.

---

## Project structure

```
prepare.py           # Fixed harness (3 prompts, 3 eval passes, median metric)
optimize.py          # Agent-editable inference config
program.md           # Autonomous research instructions
start-mlx-server.sh  # MLX server for OpenCode / Hermes
run_loop.sh          # Kick off agent loop
RESULTS.md           # Full experiment log + hardware notes
docs/INTEGRATIONS.md # OpenCode + Hermes wiring
results/             # Committed experiment TSV snapshot
```

---

## Fork it

Swap `MODEL_PATH` in `prepare.py` and convert any MLX model. The loop doesn't care if you're optimizing Ornith, Qwen, or Llama — only `decode_tok_s`.

Inspired by [@karpathy/autoresearch](https://github.com/karpathy/autoresearch) · MLX by Apple · Ornith by [DeepReinforce](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B)

## License

MIT