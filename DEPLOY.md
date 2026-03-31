# Quick Deploy Guide

## First Time Setup

### 1. Login to Hugging Face
```bash
hf auth login
```

Get your token from: https://huggingface.co/settings/tokens

### 2. Add GitHub Remote (if not already done)
```bash
git remote add origin https://github.com/kinhluan/chest-xray-disease-classifier.git
```

## Deploy Commands

### Deploy to Both GitHub & Hugging Face Spaces
```bash
# Using Python script (recommended)
uv run python deploy.py

# Using bash script
./deploy.sh
```

### Deploy to GitHub Only
```bash
git add -A
git commit -m "Your message"
git push origin master
```

### Deploy to Hugging Face Spaces Only
```bash
uv run python deploy.py --hf-only
```

### Deploy with Custom Message
```bash
uv run python deploy.py --commit-message "Fix: Update model architecture"
```

### Include Model Checkpoints (not recommended for large files)
```bash
uv run python deploy.py --include-models
```

## Deploy Script Options

| Option | Description |
|--------|-------------|
| `--space-id` | Hugging Face Space ID (default: kinhluan/chest-xray-disease-classifier) |
| `--commit-message` | Git commit message |
| `--github-only` | Only push to GitHub |
| `--hf-only` | Only push to Hugging Face |
| `--include-models` | Include .pth/.pt files in HF upload |

## Verify Deployment

### GitHub
https://github.com/kinhluan/chest-xray-disease-classifier

### Hugging Face Spaces
https://huggingface.co/spaces/kinhluan/chest-xray-disease-classifier

The Space will automatically rebuild after files are uploaded.

## Troubleshooting

### "Not logged in to Hugging Face"
```bash
hf auth login
```

### "No git remote found"
```bash
git remote add origin https://github.com/kinhluan/chest-xray-disease-classifier.git
```

### Space Build Failed
Check the Space logs at:
https://huggingface.co/spaces/kinhluan/chest-xray-disease-classifier/tree/main/logs

### Large Files
For model checkpoints >100MB, use git-lfs:
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add models/best_model.pth
git commit -m "Add model checkpoint"
git push origin master
```

## File Upload Patterns

### Included Files
- `*.py` - Python scripts
- `*.txt` - Requirements
- `*.toml` - Project config
- `*.md` - Documentation
- `*.json` - Config files

### Excluded Files
- `*.pth`, `*.pt` - Model checkpoints (unless --include-models)
- `data/*` - Dataset files
- `.venv/*` - Virtual environment
- `.git/*` - Git files
- `__pycache__/*` - Python cache
