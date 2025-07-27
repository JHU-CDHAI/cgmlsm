# A Large Sensor Foundation Model Pretrained on Continuous Glucose Monitor Data for Diabetes Management

A novel transformer-based language model approach for continuous glucose monitoring (CGM) data analysis and glucose prediction. This repository implements CGM-LSM, which treats glucose values as tokens in a sequence and applies natural language processing techniques to achieve state-of-the-art performance on glucose forecasting tasks.

## Overview

CGM-LSM introduces a paradigm shift in glucose prediction by treating CGM time series as sequences of discrete tokens rather than continuous numerical values. This approach leverages the power of transformer architectures and pre-training to learn rich representations of glucose patterns, achieving superior performance compared to traditional time series forecasting methods.


## Repository Structure

```
cgmlsm/
├── 1-CGMLSM/                    # Main CGM-LSM implementation and experiments
│   ├── run_cgmlsm_pretrain.py   # Pre-training script for CGM-LSM model
│   └── Notebook/                # Jupyter notebooks for development workflow
│       ├── a-ConvertDataToHF.ipynb        # Data conversion to HuggingFace format
│       ├── b-Pretrain-CGMLSM.ipynb        # Model pre-training
│       ├── c-Prediction-Visualization.ipynb # Performance evaluation and visualization
│       ├── c-Visualize-Model.ipynb        # Model representation analysis
│       └── d-MakePrediction.ipynb         # Inference pipeline
├── 2-OhioT1DM/                  # OhioT1DM dataset processing and baseline comparisons
│   ├── 1_run_record.py          # Raw data processing
│   ├── 2_run_case_tag.py        # Case tagging and labeling
│   ├── 3_run_ds_case_split.py   # Dataset splitting
│   ├── 4_run_case_ohio.py       # Ohio-specific case processing
│   ├── 5_run_ohio_hfds.py       # HuggingFace dataset creation
│   ├── run_glucopred_models.py  # Baseline model evaluation
│   └── Notebook/                # Analysis notebooks
│       ├── 1-Split-OhioT1D.ipynb          # Dataset splitting and preparation
│       ├── 2-OhioT1DM-HFDS.ipynb          # HuggingFace dataset conversion
│       ├── 3-Nixtla-Process-1ph.ipynb     # Nixtla model data preparation
│       └── 4-Model-Results.ipynb          # Comprehensive model comparison
└── code/                        # Core implementation
    └── pipeline/                # Data processing and model pipeline
        ├── nn/cgmlsm/           # CGM-LSM model implementation
        │   ├── configuration_cgmlsm.py    # Model configuration
        │   ├── modeling_cgmlsm.py         # Core model architecture
        │   ├── inference_cgmlsm.py        # Inference utilities
        │   └── instance_cgmlsm.py         # Model instantiation
        ├── fn/                  # Feature engineering functions
        ├── recfldtkn/          # Record, field, and token processing utilities
        └── config/             # Configuration files
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- HuggingFace Datasets
- pandas, numpy, scikit-learn


### Basic Usage

1. **Data Preparation**: Convert your CGM data to HuggingFace format
```python
# See 1-CGMLSM/Notebook/a-ConvertDataToHF.ipynb for detailed example
```

2. **Model Training**: Pre-train CGM-LSM on your dataset
```bash
cd 1-CGMLSM
python run_cgmlsm_pretrain.py
```

3. **Evaluation**: Analyze model performance
```python
# See 1-CGMLSM/Notebook/c-Prediction-Visualization.ipynb for evaluation examples
```

## Model Architecture

CGM-LSM is based on a custom transformer architecture specifically designed for glucose sequence modeling:

- **Vocabulary**: 
  - CGM values: 394 tokens (10-401 mg/dL)
  - Time tokens: 288 tokens (5-minute intervals over 24 hours)
  - Special tokens: PAD, UNKNOWN, MASK
- **Architecture**: 12-layer transformer with 768 hidden dimensions
- **Training Objective**: Causal language modeling (next-token prediction)
- **Context Length**: Up to 1024 tokens (supporting ~85 hours of 5-minute CGM data)

## Datasets

The repository supports multiple diabetes datasets:

### WellDoc Dataset
- Multi-cohort diabetes management dataset
- Both Type 1 and Type 2 diabetes patients
- Used for pre-training and primary evaluation

### OhioT1DM Dataset
- 12 patients with Type 1 diabetes
- 8-week continuous monitoring
- Used for out-of-domain evaluation and baseline comparisons

