# 🫁 Chest X-Ray Disease Classifier

A deep learning project for classifying diseases from chest X-ray images using PyTorch.

## Features

- 🏗️ **Multiple Architectures**: Support for ResNet (18/34/50/101) and DenseNet (121/169/201)
- 📊 **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- 🚀 **Mixed Precision Training**: Faster training with AMP
- 📈 **Visualization**: Training curves and confusion matrix plots
- 🌐 **Hugging Face Ready**: Deploy to Hugging Face Spaces with Gradio
- 🎯 **Class Imbalance Handling**: Automatic class weight calculation
- 🔄 **One-Click Deploy**: Push to GitHub and HF Spaces simultaneously

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/kinhluan/chest-xray-disease-classifier.git
cd chest-xray-disease-classifier

# Install dependencies
uv sync
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

Organize your dataset in the following structure:

```
data/
└── raw/
    ├── class_1/          # e.g., Normal
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class_2/          # e.g., Pneumonia
    │   ├── image3.jpg
    │   ├── image4.jpg
    │   └── ...
    ├── class_3/          # e.g., Tuberculosis
    │   └── ...
    └── ...
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

## Usage

### Training

```bash
# Basic training
uv run python train.py --data_dir data/raw --epochs 50

# With custom parameters
uv run python train.py \
    --data_dir data/raw \
    --model_type resnet \
    --model_name resnet50 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --img_size 224 \
    --output_dir models
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/raw` | Path to dataset directory |
| `--model_type` | `resnet` | Model type: `resnet` or `densenet` |
| `--model_name` | `resnet50` | Model variant |
| `--batch_size` | `32` | Batch size |
| `--epochs` | `50` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--img_size` | `224` | Image size for training |
| `--dropout_rate` | `0.5` | Dropout rate |
| `--freeze_backbone` | `False` | Freeze backbone weights |
| `--scheduler` | `cosine` | LR scheduler: `reduce`, `cosine`, `none` |
| `--val_split` | `0.2` | Validation split ratio |

### Inference

```bash
# Single image
uv run python predict.py \
    --model_path models/best_model.pth \
    --image_path path/to/chest_xray.jpg

# Directory of images
uv run python predict.py \
    --model_path models/best_model.pth \
    --image_path path/to/images/ \
    --output_path predictions.json
```

### Inference Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to model checkpoint (required) |
| `--image_path` | Path to image or directory |
| `--output_path` | Path to save predictions (JSON) |
| `--device` | Device to use: `cuda` or `cpu` |

## Project Structure

```
chest-xray-disease-classifier/
├── src/
│   └── classifier/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset.py      # Dataset and DataLoader utilities
│       ├── models/
│       │   ├── __init__.py
│       │   └── model.py        # Model architectures
│       └── utils/
│           ├── __init__.py
│           └── training.py     # Training utilities and metrics
├── data/
│   ├── raw/                # Raw dataset
│   └── processed/          # Processed dataset
├── models/                  # Saved models and checkpoints
├── notebooks/               # Jupyter notebooks
├── configs/                 # Configuration files
├── train.py                 # Training script
├── predict.py               # Inference script
├── app.py                   # Gradio app for Hugging Face
├── push_to_hf.py           # Push to Hugging Face script
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

## Output Files

After training, the `models/` directory will contain:

```
models/
└── experiment_name/
    ├── best_model.pth          # Best model checkpoint
    ├── latest_model.pth        # Latest model checkpoint
    ├── config.json             # Training configuration
    ├── metrics.json            # Evaluation metrics
    ├── history.json            # Training history
    ├── classification_report.txt  # Detailed classification report
    ├── confusion_matrix.png    # Confusion matrix plot
    └── training_history.png    # Training curves
```

## Deployment to Hugging Face

### Option 1: One-Click Deploy (Recommended)

Deploy to both GitHub and Hugging Face Spaces with a single command:

```bash
# Login to Hugging Face first
huggingface-cli login

# Add GitHub remote if not already added
git remote add origin https://github.com/kinhluan/chest-xray-disease-classifier.git

# Deploy to both platforms
uv run python deploy.py --commit-message "Deploy: Latest update"
```

Or use the bash script:
```bash
./deploy.sh
```

### Option 2: Deploy to GitHub Only

```bash
git add -A
git commit -m "Your commit message"
git push origin master
```

### Option 3: Deploy to Hugging Face Spaces Only

```bash
uv run python deploy.py --hf-only --commit-message "Update Space"
```

### Option 4: Push Model Checkpoint to HF Hub

```bash
# Login to Hugging Face
huggingface-cli login

# Push model
uv run python push_to_hf.py \
    --repo_id kinhluan/chest-xray-classifier \
    --model_dir models/experiment_name \
    --private
```

### Manual Space Update

1. Go to your Space: https://huggingface.co/spaces/kinhluan/chest-xray-disease-classifier
2. Click "Files" → "Add file" → "Upload files"
3. Upload `app.py`, `requirements_hf.txt`, and your model
4. Copy `README_HF.md` content to `README.md` in the Space

## Example Notebook

See `notebooks/` directory for example Jupyter notebooks demonstrating:
- Data exploration
- Model training
- Evaluation and visualization
- Inference examples

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision 0.15+
- CUDA-compatible GPU (optional, for faster training)

## License

MIT License

## Disclaimer

⚠️ **This project is for educational and research purposes only.** It should NOT be used for medical diagnosis. Always consult qualified healthcare professionals for medical advice.

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Hugging Face](https://huggingface.co/)
- [Gradio](https://gradio.app/)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{chest-xray-classifier,
  author = {Your Name},
  title = {Chest X-Ray Disease Classifier},
  year = {2024},
  url = {https://github.com/kinhluan/chest-xray-disease-classifier}
}
```
