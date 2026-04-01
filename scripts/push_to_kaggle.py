#!/usr/bin/env python3
"""
Push code to Kaggle and start training
"""

import os
import webbrowser
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    print("=" * 60)
    print("🚀 Push Training Notebook to Kaggle")
    print("=" * 60)
    
    # Initialize API
    api = KaggleApi()
    api.authenticate()
    print("✅ Kaggle API connected")
    
    # Notebook to push
    notebook_path = "notebooks/kaggle_train_attention.ipynb"
    
    if not Path(notebook_path).exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return
    
    print(f"\n📓 Notebook: {notebook_path}")
    
    # Option 1: Create new kernel via API
    print("\n📝 Creating kernel on Kaggle...")
    
    try:
        # Create kernel
        kernel = api.kernels_push(
            kernel=notebook_path,
            kernel_meta={
                "id": "luanbhk/chest-xray-attention-training",
                "title": "Chest X-Ray Attention Training - ResNet50 CBAM",
                "code_file": "kaggle_train_attention.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": False,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": []
            }
        )
        print(f"✅ Kernel created/updated!")
        print(f"   URL: https://www.kaggle.com/code/luanbhk/chest-xray-attention-training")
        
        # Open in browser
        webbrowser.open("https://www.kaggle.com/code/luanbhk/chest-xray-attention-training")
        print("\n🌐 Opening kernel in browser...")
        
    except Exception as e:
        print(f"⚠️  API push failed: {e}")
        print("\n📋 Manual upload instructions:")
        print("   1. Go to: https://www.kaggle.com/code/create")
        print(f"   2. Upload: {notebook_path}")
        print("   3. Enable GPU: Settings → Accelerator → GPU")
        print("   4. Enable Internet: Settings → Internet → On")
        print("   5. Click 'Run All'")
    
    print("\n" + "=" * 60)
    print("📋 Next Steps:")
    print("=" * 60)
    print("1. Run all cells in Kaggle notebook")
    print("2. Monitor training progress")
    print("3. Download results when complete")
    print("4. Repeat for other model variants")
    print("=" * 60)

if __name__ == "__main__":
    main()
