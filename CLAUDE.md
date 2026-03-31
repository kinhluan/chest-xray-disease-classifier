# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chest X-ray disease classifier using PyTorch with transfer learning (ResNet/DenseNet backbones). Deploys to Hugging Face Spaces via Streamlit. Python 3.10, managed with `uv`.

## Commands

```bash
# Install dependencies
uv sync

# Train a model
uv run python train.py --data_dir data/raw --epochs 50

# Train with specific architecture
uv run python train.py --data_dir data/raw --model_type densenet --model_name densenet121 --epochs 100

# Run inference on a single image
uv run python predict.py --model_path models/best_model.pth --image_path path/to/image.jpg

# Run inference on a directory
uv run python predict.py --model_path models/best_model.pth --image_path path/to/images/ --output_path predictions.json

# Run Streamlit app locally
uv run streamlit run streamlit_app.py

# Run Gradio app locally (HF Spaces version)
uv run python app.py

# Download dataset from Kaggle
uv run python download_dataset.py --dataset pneumonia

# Deploy to both GitHub and HF Spaces
uv run python deploy.py --commit-message "Deploy: Latest update"

# Deploy to HF Spaces only
uv run python deploy.py --hf-only --commit-message "Update Space"

# Push model checkpoint to HF Hub
uv run python push_to_hf.py --repo_id kinhluan/chest-xray-classifier --model_dir models/experiment_name
```

## Architecture

### Library code (`src/classifier/`)

- **`models/model.py`** — Two model classes: `ChestXRayClassifier` (ResNet) and `DenseNetClassifier`. Both use ImageNet-pretrained backbones with a custom 2-layer classification head (Dropout → Linear → ReLU → Dropout → Linear). Factory function `create_model()` dispatches by `model_type`.
- **`data/dataset.py`** — `ChestXRayDataset` loads images from class-named subdirectories. `create_dataloaders()` handles train/val splitting (stratified via sklearn), wrapping subsets with different augmentation transforms via `SubsetTransformed`.
- **`utils/training.py`** — `train_epoch()` and `validate()` functions with AMP support. Also `calculate_metrics()` (sklearn-based), and matplotlib plotting helpers.

### Entry points (project root)

- **`train.py`** — Full training loop with argparse CLI. Saves best/latest checkpoints, config, metrics, confusion matrix, and training curves to `models/<experiment_name>/`.
- **`predict.py`** — `ChestXRayPredictor` class for single image, batch, or directory inference.
- **`app.py`** — Gradio-based HF Spaces app. Duplicates the ResNet50 model class inline (not imported from `src/`).
- **`streamlit_app.py`** — Streamlit-based HF Spaces app. Also duplicates the model class inline.
- **`deploy.py`** — Pushes to GitHub and/or HF Spaces via `huggingface_hub` API.

### Key design notes

- `app.py` and `streamlit_app.py` intentionally duplicate the model class rather than importing from `src/classifier/` — this is for HF Spaces deployment where the package may not be installed.
- Model checkpoints (`.pth`) store: `model_state_dict`, `class_names`, `args`, `optimizer_state_dict`, `epoch`, `val_acc`, `val_loss`.
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) is used consistently across training, inference, and app code.
- Dataset expects `data/raw/<ClassName>/image.jpg` directory structure. Class names are inferred from directory names.

## Deployment

- HF Space ID: `kinhluan/chest-xray-disease-classifier`
- HF Space uses Streamlit (switched from Gradio to avoid `huggingface_hub` version conflicts)
- Dockerfile builds with pinned versions for HF Spaces compatibility
- `MODEL_PATH` env var controls where the app looks for the checkpoint (default: `models/best_model.pth`)
