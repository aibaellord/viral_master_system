#!/bin/bash
# install_dependencies.sh

# System dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv nodejs npm cuda-toolkit-11-4

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install numpy pandas scikit-learn tensorflow torch torchvision torchaudio transformers \
          matplotlib seaborn flask fastapi uvicorn requests aiohttp \
          pytest pytest-cov mypy black isort

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm init -y
npm install typescript ts-node @types/node express axios winston \
          jest supertest eslint prettier

