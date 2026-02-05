# Multi-modal TCGA-BRCA

This repository reproduces the TCGA-BRCA survival analysis with Shapley interaction indices for multi-modal DeepHit model.

## Overview

**Pipeline:**
1. **Data Preprocessing** (`01_data_preprocessing.ipynb`): Download TCGA-BRCA from Kaggle, extract UNI2-h features
2. **Training** (`02_training.ipynb`): Train MIL survival model with DeepHit loss
3. **SurvSHAP-IQ** (`03_survshapiq.ipynb`): Compute Shapley interactions for survival predictions

**Model:** Gated Attention MIL + DeepHit for discrete-time survival prediction

**Players:**
- 6 top-attention patches (image modality)
- 8 clinical features (tabular modality)

## Setup

Install [uv](https://docs.astral.sh/uv/) (if not already installed)

### 2. Create environment and install dependencies

```bash
uv sync
```

### 3. Configure Kaggle API

The dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/jmalagontorres/tcga-brca-survival-analysis). You need a Kaggle account and API key.


### 4. Configure HuggingFace (for UNI2-h)

The UNI2-h encoder requires HuggingFace access:

1. Create an account at https://huggingface.co
2. Request access to the [UNI2-h model](https://huggingface.co/MahmoodLab/UNI2-h))
3. Create an access token and replace `YOUR_HF_TOKEN` in `01_data_preprocessing.ipynb`

## Running the Notebooks

```bash
uv run jupyter lab
```

Then run the notebooks in order:
1. `01_data_preprocessing.ipynb` - Downloads data, extracts features (~5h)
2. `02_training.ipynb` - Trains the model (~5 min)
3. `03_survshapiq.ipynb` - Computes Shapley interactions (~1 min)

## Pre-trained Checkpoint

A pre-trained model checkpoint is included in `checkpoints/final_model.pt`. You can skip `02_training.ipynb` and directly run the interpretability analysis. However, the feature-extracted dataset is required.

### Note on Reproducibility

Due to running on a GPU, extracting the features and re-training the model may produce slightly different results.

Expected performance:
- **Test C-index:** ~0.738
- **Test IBS:** ~0.151

## Project Structure

```
real-data-TCGA-BRCA/
├── 01_data_preprocessing.ipynb  # Data download & feature extraction
├── 02_training.ipynb            # Model training
├── 03_survshapiq.ipynb          # SurvSHAPIQ analysis
├── utils.py                     # Helper functions
├── pyproject.toml               # Dependencies
├── README.md                    # This file
├── checkpoints/
│   └── final_model.pt           # Pre-trained model
└── data/                        # Created by notebooks
    ├── processed_images/        # Resized patches (224x224)
    ├── extracted_features/      # UNI2-h features (.npy)
    └── clinical_data_split.csv  # Clinical data with splits
```
