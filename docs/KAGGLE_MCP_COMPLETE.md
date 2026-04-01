---
tags: [kaggle, mcp, setup, tutorial]
status: completed
created: 2026-03-31
author: Luân B.
---

# Kaggle MCP Setup - Completed ✅

## 🎉 Setup Complete!

Kaggle MCP đã được setup thành công với codebase này.

---

## ✅ What's Done

### 1. Credentials Configured

```bash
~/.kaggle/kaggle.json  ✅ Created
Username: luanbhk
API Key: 310dd14a64c263ab8524c21d9b417f83
```

### 2. MCP Config Created

```json
.qwen/mcp-config.json  ✅ Created
{
  "mcpServers": {
    "kaggle": {
      "command": "uvx",
      "args": ["mcp-server-kaggle"],
      "env": {
        "KAGGLE_USERNAME": "luanbhk",
        "KAGGLE_KEY": "310dd14a64c263ab8524c21d9b417f83"
      }
    }
  }
}
```

### 3. Security Protected

```gitignore
# .gitignore updated
~/.kaggle/kaggle.json  # Never commit!
.qwen/mcp-config.json  # Contains API key!
```

### 4. Test Scripts Created

- `scripts/test_kaggle_mcp.py` - Test connection
- `scripts/push_to_kaggle.py` - Push notebook
- `notebooks/kaggle_train_attention.ipynb` - Training template

---

## 🚀 How to Use

### Option 1: Via Qwen MCP (Recommended)

```
1. Restart Qwen Code
2. MCP server will auto-connect
3. Ask Qwen to:
   - "List my Kaggle datasets"
   - "Create a training notebook"
   - "Push to Kaggle and train"
```

### Option 2: Manual Upload

```bash
# 1. Test connection
uv run python scripts/test_kaggle_mcp.py

# 2. Push notebook to Kaggle
uv run python scripts/push_to_kaggle.py

# 3. Open Kaggle and run
# URL: https://www.kaggle.com/code/luanbhk/chest-xray-attention-training
```

### Option 3: Direct Kaggle CLI

```bash
# Download dataset
uv run kaggle datasets download -d jtiptj/chest-xray-pneumoniacovid19tuberculosis

# Upload notebook
uv run kaggle kernels push -p notebooks/kaggle_train_attention.ipynb
```

---

## 📋 Training Workflow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Local Development (VS Code + Qwen)                  │
│    - Write code                                        │
│    - Test locally                                      │
│    - Create notebook                                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Push to Kaggle (MCP or Manual)                      │
│    - Upload notebook                                   │
│    - Enable GPU (P100)                                 │
│    - Enable Internet                                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Train on Kaggle (Free GPU)                          │
│    - Run all cells                                     │
│    - Monitor progress                                  │
│    - Auto-save checkpoints                             │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Download Results                                     │
│    - Checkpoints (.pth)                                │
│    - Metrics (JSON)                                    │
│    - Visualizations (PNG)                              │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Quick Start Commands

### Test Connection
```bash
uv run python -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); print('✅ Connected!')"
```

### Download Dataset
```bash
uv run kaggle datasets download -d jtiptj/chest-xray-pneumoniacovid19tuberculosis -p data/raw --unzip
```

### Push Notebook
```bash
uv run python scripts/push_to_kaggle.py
```

---

## 📊 Available Resources

| Resource | Limit | Status |
|----------|-------|--------|
| GPU (P100) | 30 hours/week | ✅ Enabled |
| Storage | 20 GB | ✅ Available |
| RAM | 16 GB | ✅ Available |
| Session | 12 hours | ✅ Auto-save |

---

## ⚠️ Important Notes

### Security
- ✅ API credentials in `~/.kaggle/kaggle.json` (not committed)
- ✅ MCP config in `.qwen/mcp-config.json` (gitignored)
- ✅ Never commit these files!

### GPU Quota
- 30 hours/week reset every Monday
- Check at: https://www.kaggle.com/settings
- Phone verification required (already done ✅)

### File Limits
- Max notebook size: 20 MB
- Max output size: 10 GB
- Max file download: 20 GB/day

---

## 🔗 Useful Links

| Purpose | URL |
|---------|-----|
| Kaggle Home | https://www.kaggle.com |
| Your Profile | https://www.kaggle.com/luanbhk |
| Settings | https://www.kaggle.com/settings |
| GPU Quota | https://www.kaggle.com/settings#accelerator |
| API Docs | https://www.kaggle.com/docs/api |
| MCP Docs | https://www.kaggle.com/docs/mcp |

---

## 🐛 Troubleshooting

### "Authentication failed"
```bash
# Regenerate token
# Go to: https://www.kaggle.com/settings → API → Create New Token
# Update ~/.kaggle/kaggle.json
```

### "GPU not available"
```
1. Check phone verification
2. Go to Settings → Accelerator
3. Select GPU (P100)
4. Save settings
```

### "Notebook timeout"
```
- Split into multiple notebooks
- Save checkpoints every epoch
- Download results frequently
```

---

## ✅ Checklist

- [x] Kaggle account created
- [x] API credentials downloaded
- [x] `~/.kaggle/kaggle.json` configured
- [x] `.qwen/mcp-config.json` created
- [x] `.gitignore` updated
- [x] Test scripts created
- [x] Notebook template ready
- [ ] First training run (next step!)

---

**Ready to train! 🚀**

Next: Run `uv run python scripts/push_to_kaggle.py` to push notebook and start training.
