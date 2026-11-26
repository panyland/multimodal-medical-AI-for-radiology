# Decisions

Short log of project choices. Newest at the bottom. Status: **decided** or **tentative**.

## 2026-11-26: Development workflow
**Status:** decided
Small reviewed steps (propose → approve → write → explain → verify → commit), documented in `CLAUDE.md`.

## 2026-11-26: Compute split
**Status:** decided
3D vision work (MAE pretraining, segmentation) runs locally on the GTX 1070 (no session limits, 12 CPU cores). VLM training runs on a rented GPU (RunPod/Vast.ai, budget ~$20–50) or Kaggle.

## 2026-11-26: Datasets
**Status:** tentative
- Segmentation benchmark: an MSD task (e.g. Task09 Spleen).
- Report generation: CT-RATE subset (3D chest CT + real reports), replacing MIMIC-CXR (2D) / M3D-Cap.
- MAE pretraining: mixed unlabeled CT pool (CT-RATE + MSD) so one encoder serves both branches.

## 2026-11-26: Architecture
**Status:** tentative
- MAE-pretrained 3D ViT encoder, used inside UNETR for segmentation; scratch UNETR as control.
- Small instruct LLM (1.5–3B, Qwen class) + MLP projector, LLaVA-style two-stage training.
