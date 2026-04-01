#!/usr/bin/env python3
"""
Push code to Kaggle and start training
"""

import os
import webbrowser
import argparse
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    parser = argparse.ArgumentParser(description="Push training notebook to Kaggle")
    parser.add_argument(
        "--notebook",
        type=str,
        default="kaggle_train_all_models.ipynb",
        choices=["kaggle_train_all_models.ipynb", "kaggle_train_attention.ipynb"],
        help="Notebook to push"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Chest X-Ray Attention Training - All 7 Models",
        help="Kernel title"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open kernel in browser after push"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Push Training Notebook to Kaggle")
    print("=" * 60)

    # Initialize API
    api = KaggleApi()
    api.authenticate()
    print("✅ Kaggle API connected")

    # Notebook to push
    notebook_path = f"notebooks/{args.notebook}"

    if not Path(notebook_path).exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return

    print(f"\n📓 Notebook: {notebook_path}")

    # Create kernel via API
    print("\n📝 Creating kernel on Kaggle...")

    kernel_id = "luanbhk/chest-xray-attention-training"
    
    try:
        # Create kernel
        kernel = api.kernels_push(
            kernel=notebook_path,
            kernel_meta={
                "id": kernel_id,
                "title": args.title,
                "code_file": args.notebook,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": False,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [
                    "jtiptj/chest-xray-pneumoniacovid19tuberculosis"
                ],
                "competition_sources": [],
                "kernel_sources": []
            }
        )
        print(f"✅ Kernel created/updated!")
        print(f"   URL: https://www.kaggle.com/code/{kernel_id}")

        # Open in browser
        if args.open:
            webbrowser.open(f"https://www.kaggle.com/code/{kernel_id}")
            print("\n🌐 Opening kernel in browser...")

    except Exception as e:
        print(f"⚠️  API push failed: {e}")
        print("\n📋 Manual upload instructions:")
        print("   1. Go to: https://www.kaggle.com/code/create")
        print(f"   2. Upload: {notebook_path}")
        print("   3. Enable GPU: Settings → Accelerator → GPU")
        print("   4. Enable Internet: Settings → Internet → On")
        print("   5. Add dataset: jtiptj/chest-xray-pneumoniacovid19tuberculosis")
        print("   6. Click 'Run All'")

    print("\n" + "=" * 60)
    print("📋 Next Steps:")
    print("=" * 60)
    print("1. Open kernel on Kaggle")
    print("2. Verify GPU is enabled (Settings → Accelerator → GPU)")
    print("3. Click 'Run All' to train all 7 models (~3-4 hours)")
    print("4. Monitor training progress in notebook output")
    print("5. Download results when complete")
    print("\n💡 Tip: Models will be saved to results/models/")
    print("=" * 60)

if __name__ == "__main__":
    main()
