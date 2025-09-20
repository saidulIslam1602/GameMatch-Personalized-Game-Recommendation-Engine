#!/bin/bash

echo "Setting up GameMatch: Personalized Game Recommendation Engine"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust based on your system)
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Setup pre-commit hooks
pre-commit install

# Create necessary directories
mkdir -p data/{raw,processed,models,embeddings}
mkdir -p logs
mkdir -p models/{fine_tuned,embeddings}
mkdir -p config
mkdir -p tests
mkdir -p notebooks
mkdir -p api
mkdir -p src/{data,models,utils,agents}

echo "Setup complete! Activate your environment with: source venv/bin/activate" 