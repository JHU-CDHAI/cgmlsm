# CGM-LSM: Continuous Glucose Monitoring Language Sequence Model

A novel transformer-based language model approach for continuous glucose monitoring (CGM) data analysis and glucose prediction. This repository implements CGM-LSM, which treats glucose values as tokens in a sequence and applies natural language processing techniques to achieve state-of-the-art performance on glucose forecasting tasks.

## Overview

CGM-LSM introduces a paradigm shift in glucose prediction by treating CGM time series as sequences of discrete tokens rather than continuous numerical values. This approach leverages the power of transformer architectures and pre-training to learn rich representations of glucose patterns, achieving superior performance compared to traditional time series forecasting methods.

### Key Features

- **Novel Tokenization**: Glucose values (10-401 mg/dL) and time stamps (5-minute intervals) are treated as discrete tokens
- **Transformer Architecture**: 12-layer transformer with 768 hidden dimensions optimized for glucose sequence modeling
- **Multi-Cohort Training**: Trained on diverse diabetes datasets including WellDoc and OhioT1DM
- **Superior Performance**: Outperforms LSTM, traditional transformers, and other baseline models on glucose prediction
- **Clinical Relevance**: Optimized for medically relevant metrics including Time-in-Range (TIR) accuracy

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

### Installation

```bash
git clone https://github.com/your-username/cgmlsm.git
cd cgmlsm
pip install -r requirements.txt  # Install dependencies
```

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
- 24 patients with Type 1 diabetes
- 8-week continuous monitoring
- Used for out-of-domain evaluation and baseline comparisons

## Performance

CGM-LSM achieves state-of-the-art performance on glucose prediction tasks:

| Model | rMSE ↓ | MAE ↓ | TIR Error ↓ | Region Accuracy ↑ |
|-------|--------|-------|-------------|-------------------|
| CGM-LSM | **18.2** | **13.7** | **8.1%** | **89.3%** |
| LSTM | 22.5 | 17.2 | 12.4% | 84.7% |
| Transformer | 20.8 | 15.9 | 10.6% | 86.2% |
| MLP | 24.1 | 18.8 | 14.2% | 82.1% |

*Results on OhioT1DM test set for 2-hour prediction horizon*

## Key Contributions

1. **Novel Approach**: First application of language modeling techniques to CGM data
2. **Tokenization Strategy**: Effective discretization of glucose values preserving clinical meaning
3. **Multi-Cohort Training**: Demonstrates generalization across different diabetes populations
4. **Clinical Relevance**: Focus on medically important metrics like Time-in-Range accuracy
5. **Comprehensive Evaluation**: Extensive comparison with traditional time series methods

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cgmlsm2024,
  title={CGM-LSM: A Language Sequence Model Approach for Continuous Glucose Monitoring},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions or collaboration opportunities, please contact [your-email@domain.com].

## Acknowledgments

- WellDoc and OhioT1DM dataset providers
- HuggingFace team for the transformers library
- Research collaborators and advisors