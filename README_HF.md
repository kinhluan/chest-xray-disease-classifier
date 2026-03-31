---
title: Chest X-Ray Disease Classifier
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: start.sh
pinned: false
license: mit
tags:
  - medical
  - xray
  - classification
  - pytorch
  - computer-vision
---

# Chest X-Ray Disease Classifier

This Space demonstrates a deep learning model for classifying diseases from chest X-ray images.

## How to Use

1. Upload a chest X-ray image
2. Click "Classify"
3. View the predicted diseases with probabilities

## Model Information

- **Architecture:** ResNet50 / DenseNet121
- **Framework:** PyTorch
- **Input:** Chest X-ray images (RGB)
- **Output:** Disease classification with probabilities

## Training

To train your own model:

```bash
# Install dependencies
uv sync

# Train the model
uv run python train.py --data_dir data/raw --epochs 50
```

## Dataset

This model is trained on chest X-ray images. For the best results, use images similar to the training data.

## Disclaimer

⚠️ **This tool is for educational and research purposes only.** It should NOT be used for medical diagnosis. Always consult qualified healthcare professionals for medical advice.

## Repository

[GitHub Repository](https://github.com/kinhluan/chest-xray-disease-classifier)
