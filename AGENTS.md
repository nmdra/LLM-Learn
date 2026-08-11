# AGENTS.md

This repo is a collection of small, self-contained projects for learning how
LLMs work (e.g. `01-tiny-embeddings/`). Each project must work on its own.

## Tooling

- Every project is a uv-managed Python project with its own `pyproject.toml`.
- Run code from the project directory: `uv run <script>` — never pip or manual
  venvs.
- Add a dependency with `uv add <package>` (updates `pyproject.toml` and
  `uv.lock`).
- Commit `uv.lock`; never commit `.venv/` or `__pycache__/`.

## Layout

- New projects go in numbered folders: `NN-topic/` (e.g. `02-tokenizer/`).
- No imports or shared code between folders — every example is self-contained.

## Code style

- Write for a learner: readable over clever, explanatory variable names.
- Comments explain the *idea*, not the syntax — detailed but not noisy.
- Ruff formatting is fine.

## Docs

- English only, plain English.

## Commits

- Conventional Commits, e.g. `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`.
