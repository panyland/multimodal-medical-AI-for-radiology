# 3D Multimodal Medical AI: Self-Supervised Foundation Model & Radiology Report Generator

## Executive Summary
This project demonstrates an end-to-end medical AI pipeline designed to overcome two major real-world bottlenecks in healthcare machine learning: **extreme data scarcity (annotation cost)** and **clinical report generation**. 

Rather than training a standard single-label classifier on pre-annotated 2D images, this project implements:
1. A **3D Self-Supervised Foundation Model** that learns spatial anatomical structures from unlabeled CT/MRI scans using a Masked Autoencoder (MAE) approach.
2. A **Data-Efficient Fine-Tuning Benchmark** proving high-accuracy organ/tumor segmentation using only 5–10 annotated scans.
3. A **3D Vision-Language Model (VLM)** bridging the 3D visual encoder to a lightweight Large Language Model (LLM) via a trained Projection Layer to generate structured, clinical radiology reports.
4. **Clinical Information Extraction Evaluation** using medical entity-aware metrics (**RadGraph F1**) rather than standard word-overlap NLP metrics.

---

## Technical Architecture Overview

```
                      ┌──────────────────────────────────────────┐
                      │          UNLABELED 3D CT/MRI             │
                      └────────────────────┬─────────────────────┘
                                           │
                                           ▼
                      ┌──────────────────────────────────────────┐
                      │    PHASE 1: 3D SSL FOUNDATION MODEL      │
                      │   - 3D Masked Autoencoder (ViT/MONAI)    │
                      │   - Reconstructs masked 3D patches       │
                      └────────────────────┬─────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
┌──────────────────────────────────────────┐   ┌──────────────────────────────────────────┐
│     BRANCH A: SEGMENTATION PROOF         │   │   PHASE 2: MULTIMODAL REPORT GENERATION  │
│  - Downstream fine-tuning (5-10 scans)   │   │  - Extracted 3D Embeddings (Encoder)     │
│  - Benchmarked vs. scratch baseline      │   │  - MLP Projection Layer (Visual-Text)    │
│  - Evaluated via Dice Similarity Coeff   │   │  - Fine-Tuned LLM via LoRA (Report Gen)  │
└──────────────────────────────────────────┘   └────────────────────┬─────────────────────┘
                                                                    │
                                                                    ▼
                                               ┌──────────────────────────────────────────┐
                                               │     EVALUATION: CLINICAL ACCURACY        │
                                               │  - RadGraph F1 / RadCliQ Evaluation      │
                                               └──────────────────────────────────────────┘
```

---

## Hardware & Compute Setup (Hybrid Workflow)

To maximize training efficiency without needing expensive cloud infrastructure:

* **Local Machine (GTX 1070 - 8GB VRAM):** Used for repository structure setup, pipeline prototyping, dataset preprocessing, MONAI transform validation, and single-epoch debugging.
* **Google Colab (Free T4 GPU - 16GB VRAM):** Used for lightweight 3D patch SSL pre-training and projection layer alignment.
* **Kaggle (Free Dual T4 GPUs - 32GB VRAM / 30 hrs/wk):** Used for multi-GPU fine-tuning of the 3D visual encoder and LoRA fine-tuning of the language model.

---

## Detailed Step-by-Step Execution Plan

### Phase 0: Data Engineering & Pipeline Setup
* **Datasets:**
  * **Phase 1 (SSL & Segmentation):** Medical Segmentation Decathlon (MSD) (e.g., Task03 Liver or Task09 Spleen) or OrganMNIST3D (for ultra-lightweight iterations).
  * **Phase 2 (Report Generation):** M3D-Cap or a curated subset of MIMIC-CXR / 3D paired scan-text datasets.
* **Key Tasks:**
  1. Build a custom `Dataset` class using **MONAI** and **PyTorch**.
  2. Implement spatial transforms: Resampling to isotropic voxel spacing, intensity normalization (Z-score / min-max scaling), spatial cropping into sub-volumes (e.g., $96 \times 96 \times 96$ voxels).
  3. Validate tensor shapes and slicing visualizations locally on your GTX 1070.

---

### Phase 1: Self-Supervised Pre-Training (The 3D Vision Encoder)
* **Goal:** Train a 3D Vision Transformer (ViT) or 3D UNETR encoder to reconstruct unannotated 3D scans.
* **Methodology:**
  1. Divide 3D image volumes into spatial patches.
  2. Randomly mask out **75% of the 3D patches**.
  3. Pass unmasked patches through the 3D ViT encoder.
  4. Use a lightweight decoder to reconstruct the original missing 3D voxels.
  5. Train using Mean Squared Error (MSE) loss on masked patch reconstruction.

---

### Branch A: Low-Data Segmentation Proof
* **Goal:** Prove the value of self-supervised learning under real-world clinical data scarcity.
* **Methodology:**
  1. **Control Group (Model A):** Train a standard 3D U-Net from scratch (random initialization) using only **5 labeled training scans**.
  2. **Experimental Group (Model B):** Initialize the 3D U-Net / Swin UNETR with your SSL pre-trained weights from Phase 1, fine-tuning on the exact same **5 labeled training scans**.
  3. **Evaluation:** Compare **Dice Similarity Coefficient (DSC)** scores on an unseen test set.
* **Expected Result:** Model B demonstrates significantly faster convergence and a higher Dice score despite data scarcity.

---

### Phase 2: Multimodal Integration (3D Vision-Language Generator)
* **Goal:** Bridge the pre-trained 3D visual encoder with an open-source Large Language Model to auto-draft structured radiology reports.
* **Architecture Components:**
  1. **Visual Encoder:** Freeze (or partially unfreeze) your 3D SSL encoder from Phase 1 to extract 3D volumetric feature embeddings.
  2. **Projection Layer:** Train a 2-layer Multi-Layer Perceptron (MLP) to map 3D visual feature dimensions (e.g., 1024-d) to the LLM's text embedding dimension (e.g., 4096-d).
  3. **Language Model:** Utilize a lightweight instruction-tuned LLM (e.g., **Llama-3-8B-Instruct**, **Phi-3-Mini**, or **Qwen-2.5**).
  4. **Parameter-Efficient Fine-Tuning (PEFT):** Apply **LoRA (Low-Rank Adaptation)** to fine-tune the LLM weights efficiently on GPU VRAM.

---

### Phase 3: Clinical Evaluation & Metrics
Standard NLP evaluation metrics (BLEU, ROUGE) are inadequate for medical text because they measure generic word overlap rather than clinical semantic correctness.

* **Primary Metric — RadGraph F1:**
  * Uses a specialized information extraction pipeline to parse generated reports into medical entities (e.g., *anatomical site*, *pathology*, *severity*) and direct relations.
  * Measures Precision, Recall, and F1-score specifically on correct medical findings.
* **Secondary Metric — BLEU-4 / ROUGE-L:** Included for baseline comparison with traditional NLP methods.

---

## Project Repository Structure

```
medical-3d-vlm/
├── config/
│   └── train_config.yaml          # Hyperparameters, data paths, model settings
├── data/
│   ├── dataset.py                 # Custom MONAI/PyTorch DataLoaders
│   └── transforms.py              # Spatial 3D intensity & normalization pipeline
├── models/
│   ├── ssl_encoder_3d.py          # 3D Masked Autoencoder / ViT definition
│   ├── projection_layer.py        # Linear/MLP projection mapping module
│   └── vlm_pipeline.py            # End-to-end Multimodal Model wrapper
├── evaluation/
│   ├── metrics_segmentation.py    # 3D Dice Coefficient & Hausdorff Distance
│   └── metrics_clinical.py        # RadGraph F1 & clinical entity extraction
├── notebooks/
│   ├── 01_local_data_check.ipynb   # Local testing & 3D slice visualization
│   └── 02_colab_training.ipynb    # Training execution notebook
├── scripts/
│   ├── train_ssl.py               # Pre-training script
│   ├── train_segmentation.py      # Downstream fine-tuning script
│   └── train_vlm.py               # Multimodal report generation script
├── README.md                      # Comprehensive project documentation
└── requirements.txt               # Dependencies (PyTorch, MONAI, HuggingFace, etc.)
```

---

## Expected Portfolio Deliverables & Results
1. **GitHub Repository:** Clean, modular, modularized Python codebase with setup instructions.
2. **Interactive Visualizations:** Slices showing original 3D scans, masked inputs, and 3D reconstructions from Phase 1.
3. **Benchmarking Graph:** A bar chart comparing Dice Scores: *Scratch Model (5 scans)* vs. *SSL Pre-trained Model (5 scans)* vs. *Full Data Baseline*.
4. **Sample Clinical Outputs:** Side-by-side comparison of ground-truth radiologist reports vs. AI-generated structured drafts, complete with RadGraph evaluation scores.