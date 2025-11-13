LLM Cultural Probing (India)
============================

Evaluate OpenRouter and Gemini models on values/norms, regional variations, and kinship/pronouns using Jupyter notebooks. Includes discriminative (MCQ), generative, robustness (prompt sensitivity), and multi‑turn probing.

Quickstart
----------
1) Install uv (package manager): see `https://docs.astral.sh/uv/`.
2) Create and sync environment:
   - `uv venv`
   - `uv sync`
3) Configure environment:
   - Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` and `GOOGLE_API_KEY`.
4) Launch notebooks:
   - `uv run jupyter lab` (or `uv run jupyter notebook`)
   - Start with `notebooks/00_setup.ipynb`.

Structure
---------
- `config/` model and evaluation configs
- `data/` raw and processed datasets
- `notebooks/` end‑to‑end build, probing, and analysis
- `results/` cached runs and figures
- `src/` provider clients, evaluation helpers, utilities
- `tests/` minimal tests for metrics/cache

Providers
---------
Unified `ModelClient` interface with implementations:
- OpenRouter (`src/providers/openrouter_client.py`)
- Gemini (`src/providers/gemini_client.py`)

Caching and Retries
-------------------
All calls are cached by content hash in `.cache/`. Transient errors are retried with exponential backoff.

Config
------
- `config/models.yaml` controls model ids and default params.
- `config/eval.yaml` sets tasks, seeds, and dataset paths.

Ethics
------
This project studies cultural behavior. Avoid harmful stereotypes, document limitations, and use neutral, verifiable items where possible.

Notes
-----
- Package management is via uv and `pyproject.toml` (no `requirements.txt` needed).

