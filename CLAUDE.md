# Project: 3D Multimodal Medical AI

3D self-supervised CT encoder (MAE) → low-data segmentation benchmark → 3D vision-language report generation.
Plan: `medical_ai_portfolio_project_plan.md`. Decisions: `DECISIONS.md`.

## Development workflow (follow strictly)

The user wants to understand and track all code. Work in small, reviewed steps:

1. **Propose** the next step first: what, why, inputs/outputs, decisions needed, and who writes it (user / Claude / together). No code before approval.
2. **Keep steps small**: one function/class/script, roughly under 150 lines changed.
3. **Explain** non-obvious parts after writing: tensor shapes, ordering, design choices.
4. **Verify visibly**: a test, printed shapes, a plotted slice, or an overfit-one-batch run. Not done until verified.
5. **Commit** one small conventional commit per step (`feat:`, `fix:`, `chore:`, `docs:`, ...), only after the user approves.
6. **Stop** after each step. Never chain steps without the user's go-ahead.

When a choice is made (dataset, architecture, hyperparameters, tooling), add an entry to `DECISIONS.md`.

## Environment

- Python via `uv` (`pyproject.toml`); no `requirements.txt`.
- Local GPU: GTX 1070 (8 GB, Pascal sm_61). fp16/bf16 give no speedup; use fp32 locally. Use PyTorch cu126 wheels (newer CUDA builds may drop Pascal).
- Heavy preprocessing runs locally (12 cores); cloud (Kaggle / rented GPU) only receives preprocessed data.
- Never commit datasets, checkpoints, or credentials.

## Git

- `main` is the stable branch; one branch per milestone (e.g. `m1-data-pipeline`), merged after reviewing the full diff.
