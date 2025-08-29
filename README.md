# AMGCNet
PyTorch implementation of AMGCNet — a lightweight convolutional neural network for image classification and beyond.


# AMGCNet

**AMGCNet** is a deep learning model implemented in **PyTorch**, designed as a modular and efficient convolutional neural network.  
This repository provides:
- Model definition (`amgcnet/models/amgcnet.py`)
- Training pipeline (`scripts/train.py`)
- Inference script (`scripts/infer.py`)
- Configuration management (`configs/default.yaml`)
- Unit tests (`tests/`)

## ✨ Features
- 🧩 Clean, modular PyTorch model implementation  
- ⚡ Training loop with mixed precision (AMP) support  
- 🔧 Config-driven hyperparameters (YAML-based)  
- 📦 Git LFS integration for checkpoints (`*.pt`, `*.pth`)  
- ✅ Continuous integration with GitHub Actions & PyTest  


# Run inference
python scripts/infer.py --config configs/default.yaml --checkpoint outputs/checkpoints/amgcnet_best.pt --image path/to/sample.jpg
