#!/bin/bash
# Quick push to Kaggle and start training

set -e

echo "============================================================"
echo "🚀 Quick Push to Kaggle"
echo "============================================================"
echo ""

# Check if Kaggle API is configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle API credentials not found!"
    echo ""
    echo "Please setup Kaggle API:"
    echo "  1. Go to: https://www.kaggle.com/settings"
    echo "  2. Click 'Create New Token' under API section"
    echo "  3. Download kaggle.json"
    echo "  4. Move to ~/.kaggle/kaggle.json"
    echo "  5. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "✅ Kaggle API credentials found"
echo ""

# Check notebook exists
NOTEBOOK="notebooks/kaggle_train_all_models.ipynb"
if [ ! -f "$NOTEBOOK" ]; then
    echo "❌ Notebook not found: $NOTEBOOK"
    exit 1
fi

echo "📓 Notebook: $NOTEBOOK"
echo ""

# Push to Kaggle
echo "🚀 Pushing to Kaggle..."
uv run python scripts/push_to_kaggle.py --open

echo ""
echo "============================================================"
echo "✅ Push Complete!"
echo "============================================================"
echo ""
echo "📋 Next Steps:"
echo "  1. Check browser for Kaggle notebook"
echo "  2. Verify GPU is enabled (Settings → Accelerator → GPU)"
echo "  3. Click 'Run All' to start training"
echo "  4. Wait ~2-3 hours for all 7 models"
echo ""
echo "🔗 Notebook URL: https://www.kaggle.com/code/luanbhk/chest-xray-attention-training"
echo "============================================================"
