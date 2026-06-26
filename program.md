# ornith-mlx-autoresearch

Autonomous agent loop to optimize **Ornith-9B MLX inference speed** on Apple Silicon.

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — instead of training GPT for 5 minutes, each experiment benchmarks fixed prompts and optimizes **median decode tokens/sec**.

## Setup

Work with the user to:

1. **Agree on a run tag** — e.g. `jun25`. Branch `autoresearch/<tag>` must not exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `main`.
3. **Read in-scope files**:
   - `prepare.py` — fixed benchmark harness, prompts, metric. **Do not modify.**
   - `optimize.py` — inference knobs you edit (sampling, chat template, KV cache, prefill, memory).
4. **Verify model exists**: `models/ornith-9b-4bit/` must be present. If missing, tell the human to convert with `mlx_lm.convert`.
5. **Initialize `results.tsv`**: header only; baseline recorded after first run.
6. **Confirm and go.**

## Experimentation

Run each experiment:

```bash
uv run optimize.py > run.log 2>&1
```

**What you CAN do:**
- Modify **`optimize.py` only** — system prompt, `enable_thinking`, temperature, `prefill_step_size`, KV quant (`kv_bits`), `EXTRA_EOS_TOKENS`, `WIRED_LIMIT`, `post_load()` hooks, etc.

**What you CANNOT do:**
- Modify `prepare.py` (fixed metric + prompts).
- Install new packages beyond `pyproject.toml`.
- Change model weights or re-quantize (out of scope for this loop).

**Goal: maximize `decode_tok_s`** (median decode tokens/sec across 3 fixed prompts). **Higher is better.**

**VRAM / RAM** is a soft constraint on M4 32GB — peak_memory_gb should stay under ~28 GB.

**Simplicity criterion**: prefer changes that improve speed without ugly complexity. Deleting reasoning bloat beats exotic hacks.

**First run**: establish baseline with unmodified `optimize.py`.

## Output format

```
---
decode_tok_s:     18.432100
prompt_tok_s:     412.500000
peak_memory_gb:   5.234
load_seconds:     2.1
eval_seconds:     45.3
total_seconds:    47.4
```

Extract metrics:

```bash
grep "^decode_tok_s:\|^peak_memory_gb:" run.log
```

## Logging results

Append to `results.tsv` (tab-separated). **Do not commit `results.tsv`.**

```
commit	decode_tok_s	memory_gb	status	description
```

- `commit` — short git hash (7 chars)
- `decode_tok_s` — median decode tok/s; `0.000000` on crash
- `memory_gb` — peak memory; `0.0` on crash
- `status` — `keep`, `discard`, or `crash`
- `description` — what this experiment tried

Example:

```
commit	decode_tok_s	memory_gb	status	description
a1b2c3d	18.432100	5.2	keep	baseline
b2c3d4e	19.801200	5.3	keep	disable thinking + prefill 4096
c3d4e5f	17.100000	5.2	discard	kv_bits=4 (slower)
```

## The experiment loop

Branch: `autoresearch/<tag>` (e.g. `autoresearch/jun25`).

LOOP FOREVER:

1. Inspect git state (branch, last `keep` commit).
2. Edit `optimize.py` with one experimental idea.
3. `git commit -am "exp: <description>"`
4. `uv run optimize.py > run.log 2>&1`
5. `grep "^decode_tok_s:\|^peak_memory_gb:" run.log`
6. If empty → crash. `tail -n 50 run.log`, fix or discard.
7. Append row to `results.tsv`.
8. If `decode_tok_s` **improved** → keep commit (advance branch).
9. If equal or worse → `git reset --hard HEAD~1`.

**Timeout**: If a run exceeds 10 minutes, kill and treat as `crash`.

**NEVER STOP**: Once the loop starts, do not ask the human to continue. Run autonomously until interrupted.

## Ideas to try

- `enable_thinking: false` / system prompts that forbid `</think>` blocks
- `EXTRA_EOS_TOKENS` to stop after `</think>` or code fence
- `prefill_step_size` sweep: 512, 1024, 2048, 4096
- `kv_bits=4` / `kv_bits=8` with low `quantized_kv_start`
- Lower `max_tokens` via shorter system prompt (less generation overhead in metric)
- `TEMPERATURE=0` greedy decoding
- Batch-like reuse: warm model once (already done); avoid `mx.clear_cache()` in `post_load` experiments
- Compare `WIRED_LIMIT=True` vs `False`

## Kickoff prompt (for OpenCode / Codex / Claude)

```
Read program.md in /Users/jchacker5/Documents/ornith-mlx-research.
Do setup (branch autoresearch/jun25), run baseline, then loop experiments
optimizing decode_tok_s. Only edit optimize.py.
```