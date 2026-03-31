# 🫁 Chest X-Ray Disease Classifier - Instruction Context

This document provides essential context and instructions for AI agents (like Gemini) working on the Chest X-Ray Disease Classifier project.

## Project Overview

A deep learning project for classifying diseases (e.g., Pneumonia, Tuberculosis) from chest X-ray images using PyTorch.

- **Architecture**: Supports ResNet (18, 34, 50, 101) and DenseNet (121, 169, 201) backbones with custom classification heads.
- **Frameworks**: PyTorch, torchvision, Gradio (for UI), Hugging Face (for deployment).
- **Core Package**: `src/classifier/` contains the modular logic.
- **Infrastructure**: Uses `uv` for lightning-fast dependency management and reproducible environments.

## Directory Structure

- `src/classifier/`: Core logic
    - `data/dataset.py`: `ChestXRayDataset` and DataLoader utilities.
    - `models/model.py`: Model architectures and factory functions.
    - `utils/training.py`: Training loops, evaluation metrics, and visualization tools.
- `train.py`: Main entry point for model training.
- `predict.py`: CLI for running inference on single images or directories.
- `app.py`: Gradio web application for deployment.
- `deploy.py`: Automation script for multi-platform deployment (GitHub + HF).
- `data/raw/`: Target directory for dataset organization (class-named folders).
- `models/`: Default output directory for checkpoints and training artifacts.

## Building and Running

### Setup
The project uses `uv`. Ensure it's installed before proceeding.
```bash
# Sync dependencies and create virtual environment
uv sync
```

### Training
```bash
# Basic training with default parameters (ResNet50)
uv run python train.py --data_dir data/raw --epochs 50

# Custom architecture
uv run python train.py --model_type densenet --model_name densenet121 --batch_size 16
```

### Inference
```bash
# Single image prediction
uv run python predict.py --model_path models/best_model.pth --image_path sample.jpg
```

### Development/UI
```bash
# Run local Gradio demo
uv run python app.py
```

## Development Conventions

1. **Modular Architecture**: Keep core logic in `src/classifier/`. Avoid bloating `train.py` or `predict.py` with implementation details.
2. **Type Hinting**: All new functions and classes should include Python type hints for clarity and safety.
3. **Training Artifacts**: Training runs should always save a `config.json` and a `metrics.json` in the output directory for reproducibility and tracking.
4. **Device Agnostic**: Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")` to ensure code runs on both local machines and servers.
5. **Data Augmentation**: Training transforms are defined in `src/classifier/data/dataset.py:get_transforms`. Always include `Normalize` with ImageNet stats.
6. **Class Imbalance**: The `ChestXRayDataset` class includes a `get_class_weights()` method. Use it with `nn.CrossEntropyLoss(weight=...)` when training on unbalanced datasets.

## Deployment Instructions

- **GitHub**: Use standard git flow.
- **Hugging Face**: The project is designed to be pushed to Hugging Face Spaces.
    - Use `deploy.py` for automated syncing.
    - `requirements_hf.txt` is specifically for the Hugging Face environment.
    - `README_HF.md` contains the Space's metadata (YAML header).
