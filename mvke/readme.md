# MVKE Implementation for Multi-Objective Recommendation

This is a PyTorch implementation of the paper "Mixture of Virtual-Kernel Experts for Multi-Objective User Profile Modeling" (SIGKDD 2022).

## Overview

MVKE (Mixture of Virtual-Kernel Experts) is a novel multi-task learning framework for recommendation systems that addresses multi-objective user profile modeling. It uses virtual kernels to capture different user behavior patterns and dynamically selects relevant experts based on item characteristics.

## Features

- **MVKE Model**: Full implementation with virtual kernel attention mechanism
- **Baseline Models**: MMoE and Shared Bottom architectures for comparison
- **Multi-Objective Learning**: Simultaneous optimization of CTR and CVR
- **MovieLens Integration**: Adapted for MovieLens 100K dataset with synthetic labels
- **Comprehensive Evaluation**: Detailed metrics and visualization tools

## Requirements

- Python 3.8+
- PyTorch 2.1.0
- CUDA-capable GPU (RTX 4070 recommended)
- 8GB+ RAM

## Quick Start

### Option 1: Using the Quick Start Script

```bash
chmod +x quickstart.sh
./quickstart.sh
```

This script will:
1. Set up the virtual environment
2. Install dependencies
3. Download and preprocess data
4. Train all models
5. Evaluate and compare results

### Option 2: Manual Setup

1. **Create virtual environment:**
```bash
python -m venv mvke_env
source mvke_env/bin/activate  # On Windows: mvke_env\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download MovieLens data:**
```bash
mkdir -p data
cd data
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
cd ..
```

4. **Preprocess data:**
```bash
python preprocess_data.py
```

5. **Train MVKE model:**
```bash
python train.py --model mvke --epochs 50 --batch_size 256
```

6. **Evaluate model:**
```bash
python evaluate.py --checkpoint_path checkpoints/mvke_default/best_model.pth
```

## Model Architecture

### MVKE Components

1. **Virtual Kernels**: Learnable embeddings representing different user behavior patterns
2. **MVKE Bottom Layer**: Attention mechanism between features and virtual kernels
3. **Expert Networks**: Task-specific or shared networks for each virtual kernel
4. **VKG Layer**: Virtual Kernel Gating to select relevant experts based on items

### Key Parameters

- `num_vk`: Number of virtual kernels (default: 5)
- `task_config`: VK assignment for tasks (CTR: 0-2, CVR: 1-4)
- `embedding_dim`: Feature embedding dimension (default: 64)
- `hidden_dims`: Hidden layer dimensions (default: [256, 128])

## Training

### Basic Training

```bash
python train.py --model mvke --epochs 50 --batch_size 256
```

### Advanced Options

```bash
python train.py \
    --model mvke \
    --epochs 50 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --embedding_dim 64 \
    --hidden_dim 256 \
    --num_vk 5 \
    --dropout 0.1 \
    --experiment_name my_experiment
```

### Training Baseline Models

```bash
# MMoE baseline
python train.py --model mmoe --num_experts 10

# Shared Bottom baseline
python train.py --model shared_bottom
```

## Evaluation

### Single Model Evaluation

```bash
python evaluate.py --checkpoint_path checkpoints/mvke_default/best_model.pth
```

### Model Comparison

```bash
python compare_models.py
```

This generates:
- `model_comparison.csv`: Metrics table
- `model_comparison.png`: Performance comparison plot

## Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/
```

View at: http://localhost:6006

## Results

Expected performance on MovieLens 100K:

| Model | CTR AUC | CVR AUC |
|-------|---------|---------|
| MVKE | ~0.78 | ~0.73 |
| MMoE | ~0.76 | ~0.71 |
| Shared Bottom | ~0.74 | ~0.69 |

## Project Structure

```
mvke_implementation/
├── core/               # Core modules
│   ├── embedding_ops.py    # Embedding layers
│   ├── mvke_ops.py        # MVKE operations
│   └── model_ops.py       # Model utilities
├── models/             # Model implementations
│   └── mvke_model.py      # MVKE, MMoE, Shared Bottom
├── utils/              # Utilities
│   └── data_loader.py     # Data loading and processing
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── preprocess_data.py  # Data preprocessing
└── compare_models.py   # Model comparison
```

## Customization

### Using Custom Datasets

1. Modify `preprocess_data.py` to load your data
2. Update feature configurations in `utils/data_loader.py`
3. Adjust model parameters based on your feature dimensions

### Adding New Tasks

1. Update task configuration in model initialization:
```python
task_config = {
    'task1': (0, 2),    # Use VK 0-2
    'task2': (1, 4),    # Use VK 1-4
    'task3': (3, 5)     # Use VK 3-5
}
```

2. Modify loss computation in `BaseMultiTaskModel.compute_loss()`

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` (try 128 or 64)
- Reduce `hidden_dim` or `embedding_dim`
- Use gradient accumulation

### Poor Performance
- Increase `epochs` (try 100)
- Tune `learning_rate` (try 0.0001)
- Adjust `num_vk` based on task complexity
- Check data preprocessing and feature engineering

### Slow Training
- Ensure CUDA is available: `torch.cuda.is_available()`
- Reduce `num_workers` if CPU-bound
- Use mixed precision training (add to train.py)

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{mvke2022,
  title={Mixture of Virtual-Kernel Experts for Multi-Objective User Profile Modeling},
  author={Authors},
  booktitle={SIGKDD},
  year={2022}
}
```

## License

This implementation is for research purposes. Please refer to the original paper for details.

## Acknowledgments

- Original MVKE paper authors
- MovieLens dataset from GroupLens Research