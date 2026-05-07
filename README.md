# Data Refinement Research — Chest X-Ray Classification with Uncertainty-Aware Cleaning

This repository contains the code, experiments, and research paper for a **CET3006** project investigating how systematic data refinement (noise detection, uncertainty-aware sample removal, and retraining) affects the performance of a deep learning model for **pneumonia detection** on chest X-ray images.

---

## Project Summary

The project explores whether deliberately cleaning a noisy medical image dataset — using model uncertainty and error-detection heuristics — improves classification accuracy, calibration, and generalisation compared to training on the raw uncleaned data.

Key contributions:

- A modular data-cleaning pipeline with configurable noise-removal thresholds (10%, 20%, 30%)
- Monte Carlo Dropout for predictive uncertainty estimation
- Comparison of baseline vs. cleaned model performance (accuracy, AUC, calibration)
- Visualisation of confusion matrices and per-class metrics across cleaning regimes

---

## Repository Structure

```
.
├── chest_xray_project/          # Core packaged pipeline
│   ├── config.py                # Hyperparameters and path configuration
│   ├── data_loader.py           # Dataset loading and augmentation
│   ├── model.py                 # CNN model definition
│   ├── train.py                 # Training loop
│   ├── detection.py             # Error / noise sample detection
│   ├── evaluation.py            # Metrics, calibration, confusion matrix
│   ├── uncertainty.py           # MC-Dropout uncertainty estimation
│   ├── main.py                  # Full pipeline entry point
│   └── CET3006_Research_Paper.docx  # Research paper document
│
├── chest_xray_cleaning.py       # Standalone: data cleaning script
├── chest_xray_compare.py        # Standalone: compare baseline vs cleaned
├── chest_xray_confusion_matrix.py
├── chest_xray_error_detection.py
├── chest_xray_evaluate.py
├── chest_xray_mc_dropout.py     # MC-Dropout inference
├── chest_xray_model.py          # Standalone model definition
├── chest_xray_noise.py          # Noise injection utilities
├── chest_xray_pipeline.py       # End-to-end pipeline runner
├── chest_xray_retrain.py        # Retrain after cleaning
├── chest_xray_train.py          # Standalone training script
├── chest_xray_visualize.py      # Plot generation
│
├── generate_diagrams.py         # Architecture and flowchart diagram generator
├── diagrams_reference.md        # Diagram descriptions and figure notes
│
├── run_pipeline.py              # Quick-start runner
├── step1_diagnose.py            # Environment diagnostics
├── verify_env.py                # Dependency verification
│
├── results/
│   ├── artifacts/
│   │   ├── histories/           # Training history JSONs per cleaning regime
│   │   ├── metrics/             # Final metrics and confusion matrix JSONs
│   │   ├── uncertainty/         # MC-Dropout audit results
│   │   └── paper_summary.txt
│   ├── charts/                  # Generated metric plots
│   └── diagrams/                # Architecture diagrams
│
├── plots/                       # Additional visualisation outputs
│   ├── plot1_loss_curves.png
│   ├── plot2_accuracy_comparison.png
│   ├── plot3_uncertainty_histogram.png
│   ├── plot4_threshold_f1.png
│   ├── plot5_multi_metrics.png
│   ├── plot6_val_accuracy_curves.png
│   ├── plot7_roc_pr_curves.png
│   └── plot8_uncertainty_scatter.png
│
├── figure1_flowchart.png        # Pipeline flowchart (paper figure 1)
├── figure2_arch_final.png       # Model architecture diagram (paper figure 2)
├── PIPELINE_AUDIT_AND_RUN_GUIDE.md
└── README.md
```

> **Note:** The `chest_xray/` dataset directory (2.4 GB) and trained model weights (`*.pth`) are excluded from this repository. See the Dataset section below.

---

## Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:

- Source: [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Classes: `NORMAL`, `PNEUMONIA`
- Expected directory structure after download:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Methods

### Model

A ResNet-based CNN (defined in `chest_xray_project/model.py`) pretrained on ImageNet and fine-tuned on the chest X-ray dataset.

### Data Cleaning Pipeline

1. **Train baseline model** on the raw training set
2. **Detect noisy/misclassified samples** using high-loss and high-uncertainty heuristics (`detection.py`, `chest_xray_mc_dropout.py`)
3. **Remove** the noisiest N% of training samples (N = 10, 20, 30)
4. **Retrain** on the cleaned subset and evaluate on the held-out test set
5. **Compare** baseline vs. cleaned performance across accuracy, AUC, ECE calibration error

### Uncertainty Estimation

Monte Carlo Dropout is applied at inference time to estimate per-sample predictive uncertainty. High-uncertainty samples that are also misclassified are flagged for removal.

---

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install torch torchvision scikit-learn matplotlib numpy pillow tqdm
```

---

## Usage

### Run the full pipeline

```bash
python run_pipeline.py
```

### Run the core packaged pipeline

```bash
cd chest_xray_project
python main.py
```

### Step-by-step

```bash
# 1. Verify environment
python verify_env.py

# 2. Train baseline
python chest_xray_train.py

# 3. Detect noisy samples
python chest_xray_error_detection.py

# 4. Clean and retrain
python chest_xray_retrain.py

# 5. Compare models
python chest_xray_compare.py

# 6. Generate visualisations
python chest_xray_visualize.py
```

---

## Results

Key result files are under `results/` and `plots/`:

- `results/charts/` — accuracy, AUC, and F1 plots across cleaning thresholds
- `results/artifacts/` — saved predictions and probability outputs
- `results/diagrams/` — architecture and pipeline diagrams
- `figure1_flowchart.png` — full pipeline diagram
- `figure2_arch_final.png` — model architecture

---

## Research Paper

The accompanying CET3006 research paper is available in:

- `chest_xray_project/CET3006_Research_Paper.docx`

---

## Academic Integrity

This repository accompanies a university assessment. Do not submit this work as your own. Cite all external sources and follow your institution's academic integrity policy.

---

## Citation

> Abhay Kalathil Shinil (2026). *Data Refinement and Uncertainty-Aware Cleaning for Chest X-Ray Pneumonia Classification*. CET3006 Research Project, University of Sunderland.
