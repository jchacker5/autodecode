# autodecode v0.1.0

**Ornith-9B on Apple Silicon — first public release**

## Highlights

- Karpathy-style autonomous inference loop (`prepare.py` + `optimize.py` + `program.md`)
- **21.4 decode tok/s** on M4 32GB (vs ~10 Ollama, ~17.5 MLX baseline)
- 22 documented experiments in `results/experiments.tsv`
- 3-pass stable benchmark harness with noise margin
- MLX server script for OpenCode / Hermes
- Full results write-up in `RESULTS.md`

## Hardware validated

- MacBook Pro, Apple M4, 32 GB RAM
- Ornith-1.0-9B MLX 4-bit (~4.7 GB)

## Breaking / known

- Model weights not included — run `mlx_lm.convert` per README
- Hermes requires `context_length: 65536` in config (tool schema overhead)
- Thermal variance ±15% — use 3-pass eval before trusting keeps

## Install

```bash
git clone https://github.com/jchacker5/autodecode.git
cd autodecode && uv sync
```

## Links

- [RESULTS.md](../RESULTS.md)
- [docs/INTEGRATIONS.md](INTEGRATIONS.md)
- [program.md](../program.md)