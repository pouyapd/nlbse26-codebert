# NLBSE'26 Code Comment Classification

Improved classifier for the [NLBSE'26 Tool Competition](https://nlbse2026.github.io/tools/) on code comment classification.

## Quick Start

### Docker

```bash
docker build -t nlbse26-classifier .
docker run -v $(pwd)/models:/app/models nlbse26-classifier python train.py
docker run -v $(pwd)/models:/app/models nlbse26-classifier python evaluate.py
```

### Local

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```
## What I Changed

1. **Replaced the baseline encoder:**  
   SetFit (`paraphrase-MiniLM-L6-v2`) → CodeBERT (`microsoft/codebert-base`)

2. **Added weighted BCE loss:**  
   Per-label `pos_weight = neg_count / pos_count`, capped at 10×, to better handle class imbalance.

3. **Kept the evaluation setup comparable:**  
   Same train/test splits, same evaluation protocol, and the same combo-field preprocessing.
   
## Approach

**Model:** `microsoft/codebert-base` fine-tuned with multi-label binary cross-entropy loss.

**Why CodeBERT over the SetFit baseline?**  
CodeBERT is pre-trained on both code and natural language (GitHub code + NL pairs), making it a better fit for code comments than a general-purpose sentence transformer (`paraphrase-MiniLM-L6-v2`).

**Key improvement — weighted loss:**  
The dataset is heavily imbalanced (e.g., `Collaborators` in Pharo has very few positive examples). We compute per-label `pos_weight = neg_count / pos_count`, capped at 10×, and pass it to `BCEWithLogitsLoss`. This prevents the model from ignoring rare categories.

**Configuration:**
- Model: `microsoft/codebert-base`
- Epochs: 7
- Batch size: 16
- Learning rate: 2e-5
- Loss: `BCEWithLogitsLoss` with per-label `pos_weight` (capped at 10×)

## Results

| Lang | Category | Baseline F1 | Ours F1 | Δ |
|------|----------|-------------|---------|---|
| java | summary | 0.879 | 0.889 | +0.010 |
| java | Ownership | 1.000 | 1.000 | 0.000 |
| java | Expand | 0.374 | 0.440 | +0.066 |
| java | usage | 0.867 | 0.868 | +0.001 |
| java | Pointer | 0.861 | 0.819 | -0.042 |
| java | deprecation | 0.778 | 0.842 | +0.064 |
| java | rational | 0.356 | 0.366 | +0.010 |
| python | Usage | 0.674 | 0.758 | +0.084 |
| python | Parameters | 0.719 | 0.834 | +0.115 |
| python | DevelopmentNotes | 0.305 | 0.424 | +0.119 |
| python | Expand | 0.540 | 0.565 | +0.025 |
| python | Summary | 0.672 | 0.703 | +0.031 |
| pharo | Keyimplementationpoints | 0.600 | 0.523 | -0.077 |
| pharo | Example | 0.881 | 0.859 | -0.022 |
| pharo | Responsibilities | 0.681 | 0.611 | -0.070 |
| pharo | Intent | 0.783 | 0.766 | -0.017 |
| pharo | Keymessages | 0.579 | 0.735 | +0.156 |
| pharo | Collaborators | 0.167 | 0.313 | +0.146 |

**Overall avg F1 — Baseline: 0.6509 → Ours: 0.6841 (+0.033)**

## Efficiency Tradeoff

While F1 improved over the baseline, CodeBERT is larger and computationally heavier than MiniLM.

As a result, runtime and computational cost increased, creating a performance–efficiency tradeoff.

## Compute Note

The project can run on CPU, but transformer training is significantly slower.  
For practical training and faster experimentation, Google Colab with a T4 GPU is recommended.

## Google Colab

Open `nlbse26_codebert_classification.ipynb` in Google Colab.  
The notebook is self-contained and includes outputs from our run.

## Repository structure

```text
├── Dockerfile
├── requirements.txt
├── train.py
├── evaluate.py
├── nlbse26_codebert_classification.ipynb
├── results.csv
└── models/
```
