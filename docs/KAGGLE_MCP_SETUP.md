---
tags: [kaggle, mcp, setup, training]
status: planned
created: 2026-03-31
related: [[ATTENTION_TODO]], [[EXPERIMENT_PROTOCOL]]
---

# Kaggle MCP Setup Guide

## 🎯 Overview

**Kaggle MCP Server** cho phép code local và đẩy lên Kaggle để train với GPU free (P100).

**Workflow:**
```
Local Development (VS Code)
    ↓
Kaggle MCP Server
    ↓
Kaggle Notebooks (GPU P100)
    ↓
Auto-save Checkpoints
    ↓
Download Results
```

---

## 📋 Prerequisites

### 1. Kaggle Account

```bash
# Sign up: https://www.kaggle.com
# Verify phone number: Settings → Phone
# Enable GPU access
```

### 2. Kaggle API Credentials

```bash
# Get API token: https://www.kaggle.com/settings
# Download kaggle.json
```

### 3. Setup Local Credentials

```bash
# Create ~/.kaggle directory
mkdir -p ~/.kaggle

# Move kaggle.json
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json

# Verify
cat ~/.kaggle/kaggle.json
# Should show: {"username":"your_username","key":"your_api_key"}
```

---

## 🔧 MCP Server Setup

### Option 1: Claude Desktop (Mac)

```json
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "kaggle": {
      "command": "uvx",
      "args": ["mcp-server-kaggle"],
      "env": {
        "KAGGLE_USERNAME": "your_username",
        "KAGGLE_KEY": "your_api_key"
      }
    }
  }
}
```

### Option 2: Qwen Code

```json
// .qwen/settings.json (project level)
{
  "mcpServers": {
    "kaggle": {
      "command": "npx",
      "args": ["-y", "@kaggle/mcp-server"],
      "env": {
        "KAGGLE_USERNAME": "your_username",
        "KAGGLE_KEY": "your_api_key"
      }
    }
  }
}
```

### Option 3: Manual Installation

```bash
# Install Kaggle MCP server
pip install kaggle-mcp-server

# Or using uv
uv pip install kaggle-mcp-server

# Run MCP server
kaggle-mcp-server
```

---

## 🛠️ Available Kaggle MCP Tools

### Notebook Management

```python
# Create notebook
kaggle.create_notebook(
    title="Chest X-Ray Attention Training",
    dataset="jtiptj/chest-xray-pneumoniacovid19tuberculosis",
    accelerator="GPU",  # P100
    timeout_hours=12
)

# List notebooks
kaggle.list_notebooks()

# Get notebook status
kaggle.get_notebook_status(notebook_id)

# Download notebook output
kaggle.download_notebook_output(notebook_id, output_path)
```

### Dataset Management

```python
# Upload dataset
kaggle.upload_dataset(
    dataset_dir="data/raw",
    title="Chest X-Ray Diseases",
    is_public=False
)

# Download dataset
kaggle.download_dataset(
    dataset_id="jtiptj/chest-xray-pneumoniacovid19tuberculosis",
    output_dir="data/raw"
)

# List datasets
kaggle.list_datasets()
```

### Model/Output Management

```python
# Upload model
kaggle.upload_model(
    model_dir="results/models",
    title="Chest X-Ray Attention Models",
    is_public=False
)

# Download results
kaggle.download_output(
    notebook_id="your_notebook_id",
    output_path="results/"
)
```

---

## 🚀 Workflow: Local Code → Kaggle Train

### Step 1: Prepare Project Locally

```bash
# Project structure
chest-xray-disease-classifier/
├── src/
├── train.py
├── configs/
└── data/
```

### Step 2: Create Kaggle Notebook via MCP

```python
# Using MCP tools
notebook = kaggle.create_notebook(
    title="Attention CNN Training - ResNet50-CBAM",
    source="kaggle_notebook_template.ipynb",
    dataset="jtiptj/chest-xray-pneumoniacovid19tuberculosis",
    accelerator="GPU",
    internet=True,
    timeout_hours=12
)

print(f"Created notebook: {notebook.id}")
print(f"URL: {notebook.url}")
```

### Step 3: Push Code to Kaggle

```python
# Push training code
kaggle.push_code(
    notebook_id=notebook.id,
    files=[
        "src/classifier/models/attention.py",
        "src/classifier/models/attention_models.py",
        "src/classifier/data/dataset.py",
        "src/classifier/utils/training.py",
        "train.py",
        "pyproject.toml"
    ]
)
```

### Step 4: Start Training

```python
# Run notebook
kaggle.run_notebook(notebook_id=notebook.id)

# Monitor progress
status = kaggle.get_notebook_status(notebook_id)
print(f"Status: {status.state}")
print(f"GPU: {status.accelerator}")
print(f"Runtime: {status.runtime_seconds}s")
```

### Step 5: Monitor Training

```python
# Check logs
logs = kaggle.get_notebook_logs(notebook_id)
print(logs)

# Or tail logs in real-time
kaggle.tail_logs(notebook_id, callback=print)
```

### Step 6: Download Results

```python
# Training complete
if kaggle.is_notebook_complete(notebook_id):
    # Download checkpoints
    kaggle.download_output(
        notebook_id=notebook_id,
        output_path="results/models/"
    )
    
    # Download metrics
    kaggle.download_output(
        notebook_id=notebook_id,
        output_path="results/metrics/"
    )
```

---

## 📝 Kaggle Notebook Template

```python
# kaggle_notebook_template.ipynb

# Cell 1: Setup
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers scikit-learn matplotlib seaborn tqdm opencv-python-headless

# Cell 2: Clone repo
!git clone https://github.com/kinhluan/chest-xray-disease-classifier.git
%cd chest-xray-disease-classifier

# Cell 3: Install as package
!pip install -e .

# Cell 4: Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 5: Train Model
!python train.py \
  --data_dir data/raw \
  --model_type resnet \
  --model_name resnet50 \
  --attention cbam \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_dir results/models/resnet50_cbam

# Cell 6: Save Results
import zipfile
!zip -r results.zip results/
from kaggle_secrets import UserSecretsClient
# Download automatically via MCP
```

---

## 🔄 Complete Training Pipeline

### Script: `scripts/train_on_kaggle.py`

```python
#!/usr/bin/env python3
"""
Train all attention models on Kaggle using MCP
"""

from kaggle_mcp import KaggleMCP
import time

def main():
    # Initialize MCP
    kaggle = KaggleMCP()
    
    # Models to train
    models = [
        ("resnet50", None, "resnet50_baseline"),
        ("resnet50", "se", "resnet50_se"),
        ("resnet50", "cbam", "resnet50_cbam"),
        ("resnet50", "eca", "resnet50_eca"),
        ("densenet121", None, "densenet121_baseline"),
        ("densenet121", "se", "densenet121_se"),
        ("densenet121", "cbam", "densenet121_cbam"),
    ]
    
    # Create notebook
    notebook = kaggle.create_notebook(
        title="Chest X-Ray Attention Models - Full Training",
        accelerator="GPU",
        timeout_hours=12,
        internet=True
    )
    
    print(f"Created notebook: {notebook.url}")
    
    # Push code
    kaggle.push_code(
        notebook_id=notebook.id,
        files=[
            "src/",
            "train.py",
            "pyproject.toml",
            "configs/"
        ]
    )
    
    # Training script for all models
    training_script = """
#!/bin/bash
set -e

cd /kaggle/working/chest-xray-disease-classifier

# Download dataset
kaggle datasets download -d jtiptj/chest-xray-pneumoniacovid19tuberculosis -p data/raw --unzip

# Train all models
MODELS=(
    "resnet50:none:resnet50_baseline"
    "resnet50:se:resnet50_se"
    "resnet50:cbam:resnet50_cbam"
    "resnet50:eca:resnet50_eca"
    "densenet121:none:densenet121_baseline"
    "densenet121:se:densenet121_se"
    "densenet121:cbam:densenet121_cbam"
)

for config in "${MODELS[@]}"; do
    IFS=':' read -r backbone attention name <<< "$config"
    
    echo "========================================"
    echo "Training $name"
    echo "========================================"
    
    python train.py \\
        --data_dir data/raw \\
        --model_type $backbone \\
        ${attention:+--attention $attention} \\
        --epochs 50 \\
        --batch_size 32 \\
        --lr 1e-4 \\
        --output_dir results/models/$name
    
    echo "✅ Completed $name"
done

# Zip results
cd /kaggle/working
zip -r results.zip chest-xray-disease-classifier/results/
echo "🎉 All models trained!"
"""
    
    # Run training
    kaggle.run_notebook(notebook_id=notebook.id, script=training_script)
    
    # Monitor progress
    while True:
        status = kaggle.get_notebook_status(notebook_id=notebook.id)
        print(f"Status: {status.state} ({status.runtime_seconds}s)")
        
        if status.is_complete:
            break
        elif status.is_error:
            print(f"Error: {status.error_message}")
            break
        
        time.sleep(60)
    
    # Download results
    if status.is_success:
        kaggle.download_output(
            notebook_id=notebook.id,
            output_path="results/"
        )
        print("✅ Results downloaded!")

if __name__ == "__main__":
    main()
```

---

## 💰 Cost

| Item | Cost |
|------|------|
| Kaggle GPU (P100) | $0 (free) |
| MCP Server | $0 (open source) |
| Storage (20GB) | $0 (included) |
| **Total** | **$0** |

---

## ⏱️ Time Estimate

| Task | Time |
|------|------|
| Setup MCP | 30 minutes |
| Create notebook | 5 minutes |
| Upload code | 5 minutes |
| Train 7 models | 6-7 hours |
| Download results | 10 minutes |
| **Total** | **~8 hours** (mostly automated) |

---

## ⚠️ Troubleshooting

### Issue: MCP Server not found

```bash
# Install manually
npm install -g @kaggle/mcp-server

# Or use uvx
uvx mcp-server-kaggle
```

### Issue: Authentication failed

```bash
# Verify credentials
cat ~/.kaggle/kaggle.json

# Regenerate token if needed
# Go to: https://www.kaggle.com/settings → API → Create New Token
```

### Issue: Notebook timeout

```python
# Split into multiple notebooks
# Train 3 models per notebook (4 hours each)
# 3 notebooks total
```

### Issue: Storage limit (20GB)

```python
# Clean up during training
!rm -rf /kaggle/working/chest-xray-disease-classifier/results/models/*/latest_model.pth
!rm -rf ~/.cache/pip

# Download results frequently
kaggle.download_output(notebook_id, output_path="local/results")
```

---

## ✅ Checklist

- [ ] Kaggle account created
- [ ] Phone verified
- [ ] API credentials downloaded
- [ ] `~/.kaggle/kaggle.json` configured
- [ ] MCP server installed
- [ ] MCP configured in Qwen/Claude
- [ ] Test notebook created
- [ ] Training script ready
- [ ] Download path configured

---

## 🔗 References

- Kaggle MCP Docs: https://www.kaggle.com/docs/mcp
- Kaggle API: https://github.com/Kaggle/kaggle-api
- Model Context Protocol: https://modelcontextprotocol.io

---

**Next Step:** Install MCP server and test connection
