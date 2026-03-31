# AGENTS Guide for `DNN`

## Current project state (read first)
- This repository is an early scaffold: there is no training/inference pipeline yet.
- `README.md` and `requirements.txt` are currently empty, so agent-generated changes should establish missing project context.
- `sample.ipynb` is the only executable artifact and currently contains a minimal `print("Hello World!")` notebook cell.
- `data/` and `models/` exist but are empty; use them as stable locations instead of introducing new top-level storage folders.

## Architecture expectations for new work
- Treat notebooks as the current entry point (`sample.ipynb`), but move reusable logic into `.py` modules once logic exceeds a few cells.
- Keep a clear boundary between:
  - `data/` for datasets, intermediate artifacts, and metadata
  - `models/` for saved weights/checkpoints/exported model files
  - notebook/script code for orchestration only
- Prefer deterministic paths rooted at repository top-level (e.g., `data/...`, `models/...`) to avoid environment-specific absolute paths.

## Developer workflows (what is actually available now)
- There is no discovered test suite, linter config, or CI workflow in this repo yet.
- There is no discovered package manager lockfile or pinned dependency set yet.
- If you introduce runnable Python code, also introduce the minimal command(s) needed to execute it and document them in `README.md`.
- If you add dependencies, update `requirements.txt` in the same change.

## Project-specific conventions to follow
- Keep notebook outputs non-essential: code and markdown should be sufficient to rerun results.
- When adding substantial code, include a tiny runnable entry point (script or notebook cell block) so the repository is not left as incomplete snippets.
- Avoid creating parallel folder structures (for example both `model/` and `models/`); extend existing top-level directories first.
- Any structural decision (new directories, training pipeline layout, experiment tracking files) should be reflected in `README.md` immediately.

## Integration points to preserve
- Jupyter/PyCharm notebook workflow is currently the only explicit workflow (`sample.ipynb` includes JetBrains notebook usage guidance).
- Assume local filesystem-based artifacts for now; no external service integrations are discoverable in the current codebase.

## When generating future changes
- Reference concrete files in commit-sized changes (for example: update `requirements.txt`, add `src/train.py`, document run command in `README.md`).
- Prefer small, end-to-end increments that run locally over large partial scaffolding.

