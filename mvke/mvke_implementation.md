# MVKE Implementation Guide

## Project Structure

```
mvke_implementation/
├── data/
│   └── ml-100k/         # MovieLens dataset
├── core/
│   ├── __init__.py
│   ├── model_ops.py
│   ├── mvke_ops.py
│   ├── embedding_ops.py
│   └── layers.py
├── models/
│   ├── __init__.py
│   └── mvke_model.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   └── metrics.py
├── train.py
├── evaluate.py
├── preprocess_data.py
└── requirements.txt
```

## Implementation Steps

### 1. Environment Setup

First, create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv mvke_env
source mvke_env/bin/activate  # On Windows: mvke_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Data Preparation

Download the MovieLens 100K dataset:
```bash
mkdir -p data
cd data
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
cd ..
```

### 3. Preprocess Data

Run the preprocessing script to prepare the data:
```bash
python preprocess_data.py
```

This will:
- Load MovieLens data
- Create user and item features
- Generate synthetic CTR and CVR labels based on ratings
- Split data into train/validation/test sets

### 4. Training

Train the MVKE model:
```bash
# Train MVKE model
python train.py --model mvke --epochs 50 --batch_size 256

# Train baseline models for comparison
python train.py --model mmoe --epochs 50 --batch_size 256
python train.py --model shared_bottom --epochs 50 --batch_size 256
```

### 5. Evaluation

Evaluate the trained models:
```bash
python evaluate.py --model mvke --checkpoint_path checkpoints/mvke_best.pth
```

## Model Architecture

### MVKE (Mixture of Virtual-Kernel Experts)
- **Virtual Kernels**: 5 kernels that capture different user behavior patterns
- **Task-specific Selection**: CTR uses kernels 0-2, CVR uses kernels 1-4
- **Attention Mechanism**: Item embeddings attend to virtual kernels to select relevant experts

### Baseline Models
1. **MMoE**: Multi-gate Mixture of Experts with 10 experts
2. **Shared Bottom**: Hard parameter sharing with task-specific heads

## Key Features

1. **Multi-objective Learning**: Simultaneously optimizes CTR and CVR
2. **Virtual Kernel Attention**: Dynamic expert selection based on item characteristics
3. **Feature Engineering**: Rich user and item features including:
   - User demographics (age, gender, occupation)
   - Item metadata (genres, release year)
   - Interaction history features

## Expected Results

On MovieLens 100K dataset:
- **CTR Task**: AUC ~0.75-0.80
- **CVR Task**: AUC ~0.70-0.75
- **Training Time**: ~5-10 minutes per epoch on RTX 4070

## GPU Memory Usage

With batch size 256:
- MVKE: ~2-3 GB
- MMoE: ~2-3 GB
- Shared Bottom: ~1-2 GB

Your RTX 4070 (12GB) should handle these models comfortably.

## Hyperparameter Tuning

Key hyperparameters to tune:
- `learning_rate`: [0.001, 0.0001]
- `embedding_dim`: [32, 64, 128]
- `num_virtual_kernels`: [3, 5, 8]
- `hidden_dims`: [128, 256]
- `dropout_rate`: [0.1, 0.3, 0.5]

## Monitoring Training

The training script will:
- Log metrics to TensorBoard
- Save best checkpoints based on validation AUC
- Display progress bars with current metrics

To view TensorBoard:
```bash
tensorboard --logdir logs/
```