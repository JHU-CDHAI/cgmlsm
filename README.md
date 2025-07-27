# A Large Sensor Foundation Model Pretrained on Continuous Glucose Monitor Data for Diabetes Management

A novel transformer-based language model approach for continuous glucose monitoring (CGM) data analysis and glucose prediction. This repository implements CGM-LSM, which treats glucose values as tokens in a sequence and applies natural language processing techniques to achieve state-of-the-art performance on glucose forecasting tasks.

## Overview

CGM-LSM introduces a paradigm shift in glucose prediction by treating CGM time series as sequences of discrete tokens rather than continuous numerical values. This approach leverages the power of transformer architectures and pre-training to learn rich representations of glucose patterns, achieving superior performance compared to traditional time series forecasting methods.


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

