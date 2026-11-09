# GeoScope

**Explainable land-use classification of satellite imagery.**  
GeoScope is an end-to-end explainable deep-learning system that classifies Sentinel-2 satellite patches into 10 land-use classes using the [EuroSAT](https://github.com/phelber/EuroSAT) dataset. It benchmarks three distinct vision architectures (ResNet-18, EfficientNet-B0, and Vision Transformer ViT-B/16) in PyTorch, producing comprehensive empirical evaluations alongside Grad-CAM visual heatmaps that explain *which regions of the satellite patch drove the model's decision*.

> Course Project --- Track A (Application Development). Team of 2.

---

## Project Status

| Phase | Description | Status | Deliverable Details |
|-------|-------------|--------|---------------------|
| 1 | Problem definition & domain research | **Done** | See `report/` |
| 2 | Data acquisition, cleaning, preprocessing & EDA | **Done** | See `data_scripts/` & `figures/` |
| 3 | Model architecture implementation & hyperparameter tuning | **Done** | See `models/` & `GeoScope_results/` |
| 4 | Empirical metric evaluation, bias check & Grad-CAM explainability | **Done** | See `models/gradcam.py` & `figures/evaluation_results/` |
| 5 | Web application deployment | **Planned** | Track A deliverable |

---

## Model Checkpoints (External Weights Access)

In strict compliance with repository size guidelines, large binary model files ($>100\text{ MB}$) are excluded from Git version control (the `models/checkpoints/` directory is git-ignored). 

All pre-trained PyTorch model weight checkpoints (`resnet18_best.pth`, `efficientnet_best.pth`, and `vit_best.pth`) are hosted externally on Google Drive:

👉 **[Download GeoScope Model Checkpoints on Google Drive](https://drive.google.com/drive/folders/1t420bNansMSE7BrX4Qnnc8burWkLmg0m?usp=sharing)**

---

## Benchmark Performance Summary

All models were trained on $18{,}900$ images ($70\%$), validated on $4{,}050$ images ($15\%$), and evaluated on a held-out test split of $4{,}050$ unseen patches ($15\%$).

| Model Architecture | Parameters | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | AUC-ROC (OvR) | Latency (ms/img) | Checkpoint Size |
|--------------------|-----------:|--------------:|----------------:|-------------:|---------:|--------------:|----------------:|----------------:|
| **ResNet-18** | 11.2M | 98.37% | 0.9834 | 0.9828 | 0.9831 | 0.9998 | **0.85 ms** | 44.8 MB |
| **EfficientNet-B0** | **4.0M** | 98.64% | 0.9867 | 0.9852 | 0.9858 | 0.9998 | 1.08 ms | **16.4 MB** |
| **ViT-B/16** | 85.8M | **98.84%** | **0.9886** | **0.9879** | **0.9882** | **0.9999** | 10.74 ms | 343.3 MB |

---

## Dataset Overview

- **Name:** EuroSAT (RGB edition)
- **Source:** Sentinel-2 satellite imagery across 34 European countries (Helber et al., IEEE JSTARS, 2019).
- **Volume:** 27,000 labeled patches ($64\times64$ pixels), 2,000--3,000 patches per class.
- **Land-Use Classes (10):** `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`.

*Note: Raw satellite imagery is **not** committed to Git (`data/` is git-ignored). Run the automated collection script detailed below to fetch it.*

---

## Repository Structure

```text
GeoScope/
├── data_scripts/
│   ├── common.py           # Paths, constants, and dataset helper functions
│   ├── download_data.py    # Automated dataset download via torchvision + validation logging
│   ├── clean_data.py       # Audits corrupt files, duplicates, degenerate patches & outliers
│   ├── preprocess.py       # Label encoding, 70/15/15 stratified split, normalization
│   └── eda.py              # Exploratory data analysis charts -> figures/
├── models/
│   ├── architectures.py    # PyTorch implementations for ResNet-18, EfficientNet-B0, ViT-B/16
│   ├── data_utils.py       # PyTorch Dataset wrappers & Sentinel-2 transforms
│   ├── metrics_utils.py    # Metric computations, confusion matrices, bias check & TeX exporters
│   ├── train.py            # Model training loop with Cosine Annealing & AdamW optimizer
│   ├── tune.py             # 18-run grid search hyperparameter tuning engine
│   ├── evaluate.py         # Test set evaluation, metrics, ROC curves, confusion matrices
│   └── gradcam.py          # Grad-CAM explainability engine for CNNs and Vision Transformers
├── figures/
│   ├── evaluation_results/ # Test evaluation charts (ROC curves, confusion matrices, error grids, Grad-CAM maps)
│   └── eda/                # Data distribution & channel histograms from Phase 2
├── results/                # Metrics outputs, per-class CSV summaries, and TeX benchmark tables
├── report/                 # Full technical report source in LaTeX
├── requirements.txt        # Exact Python dependencies for environment reproduction
└── README.md
```

---

## Reproduction Pipeline (Phases 2, 3 & 4)

### 1. Environment Setup & Data Preparation (Phase 2)
```bash
# Install dependencies
pip install -r requirements.txt

# Execute data collection, cleaning, preprocessing, and EDA
python data_scripts/download_data.py   # 1. Download EuroSAT into data/
python data_scripts/clean_data.py      # 2. Audit images, generate data/clean_manifest.csv
python data_scripts/preprocess.py      # 3. Label encode, generate 70/15/15 stratified split
python data_scripts/eda.py             # 4. Save EDA plots to figures/
```

### 2. Hyperparameter Search & Final Training (Phase 3)
```bash
# Hyperparameter grid search runs (18 total documented experiments)
python models/tune.py --model resnet18 --lrs 1e-3 1e-4 1e-5 --dropouts 0.0 0.3 --epochs 3
python models/tune.py --model efficientnet --lrs 1e-3 1e-4 1e-5 --epochs 3
python models/tune.py --model vit --lrs 1e-4 3e-5 1e-5 --epochs 3

# Final model training with best selected configurations
python models/train.py --model resnet18 --epochs 10 --lr 1e-4 --batch_size 64
python models/train.py --model efficientnet --epochs 10 --lr 1e-4 --batch_size 32
python models/train.py --model vit --epochs 8 --lr 1e-5 --batch_size 32 --dropout 0.3
```

### 3. Empirical Evaluation & Grad-CAM Heatmaps (Phase 4)
```bash
# Evaluate all models on the held-out test split (4,050 unseen patches)
python models/evaluate.py --runs resnet18 efficientnet vit

# Generate Grad-CAM explainability heatmaps across models
python models/gradcam.py --runs resnet18 efficientnet vit
```

### 4. Technical Report Compilation
```bash
cd report && pdflatex report.tex && pdflatex report.tex
```

---

## Core Engineering & Cleaning Policies

### Data Cleaning Policy
Raw image files are never modified in place. `clean_data.py` inspects every image and writes an audit manifest:

| Quality Issue | Detection Criteria | Action Taken |
|---------------|-------------------|--------------|
| Corrupt / Unreadable file | PIL `verify()` + image decode test | File removed |
| Invalid dimension / mode | Header inspect (Must be $64\times64$ RGB) | File removed |
| Exact Duplicate | Pixel MD5 hash match | Keep first instance, remove redundant copies |
| Cross-Class Label Conflict | Identical MD5 across different folders | Remove all conflicting copies |
| Degenerate / Blank Patch | Pure black/white flat pixel array | File removed |
| Near-Duplicate | 64-bit dHash (Hamming distance $\le 3$) | Flagged in manifest |
| Statistical Outlier | $3\times\text{IQR}$ per class on brightness/contrast | Flagged in manifest |

### Feature & Preprocessing Strategy
- **Resizing:** Upscaled from $64\times64 \rightarrow 224\times224$ pixels.
- **Normalization:** Standardized to ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Encoding:** Class names mapped to discrete integer IDs (`data/label_map.json`).
- **Splitting:** Stratified $70/15/15$ split ($18{,}900$ train, $4{,}050$ val, $4{,}050$ test) with seed 42 (`data/splits.csv`).
- **Augmentation (Training Only):** Random horizontal and vertical flips.

---

## Professional Git & Collaboration Policy

In strict accordance with project policies, **no manual file uploads were performed via the GitHub web UI**. All branches, scripts, and commits were managed and pushed exclusively using the **Git Command Line Interface (CLI)**.

### Module Ownership & Contribution Mapping

| Group Member | Roll Number | Git Commit Ownership & Implementation Contributions |
|--------------|------------|-----------------------------------------------------|
| **Syeda Faryal Zehra** | `24l-2556` | Data collection & preprocessing scripts; Grad-CAM explainability module (`models/gradcam.py`); training, hyperparameter search, and evaluation pipelines for ResNet-18 & EfficientNet-B0; subgroup bias check. |
| **Ibrahim Qaiser** | `24l-2513` | Model architecture definitions (`models/architectures.py`); training, hyperparameter search, and evaluation pipeline for Vision Transformer (ViT-B/16); project structure setup, environment configuration (`requirements.txt`), and LaTeX report compilation. |