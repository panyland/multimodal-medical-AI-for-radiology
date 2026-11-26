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

## 2026-11-26: Chest-only domain
**Status:** decided
Supersedes the segmentation/pretraining parts of "Datasets" above.
- Segmentation benchmark: MSD Task06 Lung (lung tumours), replacing Task09 Spleen.
- MAE pretraining pool: chest CT only (CT-RATE subset + MSD Lung images).
- Report generation: CT-RATE (access granted), one reconstruction per scan, subset downloaded locally; only preprocessed data goes to cloud.
**Why:** one anatomical domain across all three phases, so the encoder that serves segmentation also serves report generation. Trade-off: lung tumours are small and hard, so low-data Dice will be lower and noisier than spleen would have been.

## 2026-11-26: Preprocessing and code layout
**Status:** decided
- Orientation RAS; resample to 1.5 mm isotropic (image bilinear, label nearest). A 96³ crop covers 14.4 cm; smallest MSD Lung tumour stays ~7 voxels wide.
- HU window [-1000, 1000] clipped and scaled to [0, 1]: keeps lung, soft tissue/tumour and bone, clips metal outliers (max seen: 20349).
- Crop to body bounding box (> -500 HU) to drop surrounding air.
- Training crops: 96³ via `RandCropByPosNegLabeld`, pos:neg = 1:1, because tumours are a median 0.011% of the volume.
- Importable code lives in an installable package `src/ctvlm/` (hatchling build); `scripts/` holds entry points.
