#!/bin/bash
# Download and train chest X-ray classification model

set -e

echo "========================================"
echo "Chest X-Ray Disease Classifier"
echo "Download & Train Script"
echo "========================================"
echo ""

# Check Kaggle credentials
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "❌ Kaggle credentials not set!"
    echo ""
    echo "Please set environment variables:"
    echo "  export KAGGLE_USERNAME=your_username"
    echo "  export KAGGLE_KEY=your_api_key"
    echo ""
    echo "Get API key from: https://www.kaggle.com/account"
    exit 1
fi

echo "✓ Kaggle credentials found"
echo ""

# Download dataset
echo "Downloading dataset..."
echo "Dataset: Chest X-Ray (Pneumonia, Covid-19, Tuberculosis)"
echo "URL: https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis"
echo ""

mkdir -p data/raw

# Download using Kaggle CLI
kaggle datasets download -d jtiptj/chest-xray-pneumoniacovid19tuberculosis -p data/raw --unzip

echo ""
echo "✓ Dataset downloaded"
echo ""

# Organize dataset
echo "Organizing dataset structure..."
python -c "
import os
from pathlib import Path
import shutil

raw_dir = Path('data/raw')

# Find train folder
train_dir = raw_dir / 'train'
if train_dir.exists():
    # Move class folders to root
    for class_folder in train_dir.iterdir():
        if class_folder.is_dir():
            target = raw_dir / class_folder.name
            if not target.exists():
                shutil.move(str(class_folder), str(raw_dir))
                print(f'  Moved {class_folder.name} to data/raw/')
    
    # Cleanup
    if train_dir.exists():
        shutil.rmtree(train_dir)

# Remove test/val if exists
for folder in ['test', 'val', '__MACOSX']:
    folder_path = raw_dir / folder
    if folder_path.exists():
        shutil.rmtree(folder_path)
        print(f'  Removed {folder}/')

print('✓ Dataset organized')
"

echo ""
echo "========================================"
echo "Dataset ready!"
echo "========================================"
echo ""

# Show dataset info
echo "Dataset structure:"
ls -la data/raw/
echo ""

# Count images
echo "Image count per class:"
for dir in data/raw/*/; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" 2>/dev/null | wc -l)
        class=$(basename "$dir")
        echo "  $class: $count images"
    fi
done
echo ""

# Train
echo "========================================"
echo "Starting training..."
echo "========================================"
echo ""

uv run python train.py \
    --data_dir data/raw \
    --model_type resnet \
    --model_name resnet50 \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-4 \
    --img_size 224 \
    --output_dir models

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo ""
echo "Model saved to: models/best_model.pth"
echo ""
echo "To upload to Hugging Face Space:"
echo "  1. Go to: https://huggingface.co/spaces/kinhluan/chest-xray-disease-classifier"
echo "  2. Click 'Files' → 'Add file' → 'Upload file'"
echo "  3. Upload models/best_model.pth"
echo ""
