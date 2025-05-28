#!/bin/bash

# MVKE Quick Start Script

echo "==================================="
echo "MVKE Implementation Quick Start"
echo "==================================="

# Check if virtual environment exists
if [ ! -d "mvke_env" ]; then
    echo "Creating virtual environment..."
    python -m venv mvke_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source mvke_env/Scripts/activate
else
    source mvke_env/bin/activate
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/ml-100k
mkdir -p data/processed
mkdir -p logs
mkdir -p checkpoints
mkdir -p evaluation_results

# Download MovieLens dataset
echo "Downloading MovieLens 100K dataset..."
cd data
if [ ! -f "ml-100k.zip" ]; then
    wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
    unzip ml-100k.zip
fi
cd ..

# Preprocess data
echo "Preprocessing data..."
python preprocess_data.py

# Train models
echo "Training models..."
echo "1. Training MVKE model..."
python train.py --model mvke --epochs 30 --batch_size 256 --experiment_name quickstart

echo "2. Training MMoE baseline..."
python train.py --model mmoe --epochs 30 --batch_size 256 --experiment_name quickstart

echo "3. Training Shared Bottom baseline..."
python train.py --model shared_bottom --epochs 30 --batch_size 256 --experiment_name quickstart

# Evaluate models
echo "Evaluating models..."
python evaluate.py --checkpoint_path checkpoints/mvke_quickstart/best_model.pth
python evaluate.py --checkpoint_path checkpoints/mmoe_quickstart/best_model.pth
python evaluate.py --checkpoint_path checkpoints/shared_bottom_quickstart/best_model.pth

# Compare results
echo "Comparing models..."
python compare_models.py

echo "==================================="
echo "Quick start completed!"
echo "Check the following outputs:"
echo "- TensorBoard logs: tensorboard --logdir logs/"
echo "- Model comparison: model_comparison.png"
echo "- Evaluation plots: evaluation_results/"
echo "===================================="