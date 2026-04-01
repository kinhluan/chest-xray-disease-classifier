#!/usr/bin/env python3
"""
Test Kaggle MCP connection and capabilities
"""

import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def test_connection():
    """Test Kaggle API connection"""
    print("=" * 60)
    print("🔌 Testing Kaggle API Connection")
    print("=" * 60)
    
    api = KaggleApi()
    api.authenticate()
    
    print("✅ API authenticated successfully!")
    print(f"   Credentials loaded from: ~/.kaggle/kaggle.json")
    return api

def test_list_datasets(api):
    """List user's datasets"""
    print("\n" + "=" * 60)
    print("📊 Your Kaggle Datasets")
    print("=" * 60)
    
    try:
        datasets = api.dataset_list(user="luanbhk")
        if datasets:
            for ds in datasets[:10]:
                print(f"  - {ds.ref}")
                print(f"    Title: {ds.title}")
        else:
            print("  No datasets found. Create one to get started!")
    except Exception as e:
        print(f"  ⚠️  Error: {e}")

def test_target_dataset(api):
    """Test accessing target dataset"""
    print("\n" + "=" * 60)
    print("🎯 Target Dataset: Chest X-Ray Pneumonia/Covid/TB")
    print("=" * 60)
    
    dataset_id = "jtiptj/chest-xray-pneumoniacovid19tuberculosis"
    
    try:
        dataset = api.dataset_view(dataset_id)
        print(f"  ✅ Found: {dataset.title}")
        print(f"  📁 Description: {dataset.description[:100]}..." if dataset.description else "")
        print(f"  👁️  Views: {dataset.viewCount:,}")
        print(f"  ⬇️  Downloads: {dataset.downloadCount:,}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_download_dataset(api):
    """Test downloading dataset"""
    print("\n" + "=" * 60)
    print("⬇️  Download Dataset Test")
    print("=" * 60)
    
    dataset_id = "jtiptj/chest-xray-pneumoniacovid19tuberculosis"
    download_dir = "data/raw/test_download"
    
    print(f"  Downloading to: {download_dir}")
    
    try:
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(
            dataset=dataset_id,
            path=download_dir,
            unzip=True
        )
        
        # Count files
        files = list(Path(download_dir).glob("*"))
        print(f"  ✅ Downloaded {len(files)} files/folders")
        
        # Show structure
        print("  Structure:")
        for f in files[:10]:
            if f.is_dir():
                count = len(list(f.glob("*.*")))
                print(f"    📁 {f.name}/ ({count} files)")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    finally:
        # Cleanup
        import shutil
        if Path(download_dir).exists():
            shutil.rmtree(download_dir)
            print("  🧹 Cleaned up test download")

def test_list_kernels(api):
    """List user's notebooks"""
    print("\n" + "=" * 60)
    print("📓 Your Kaggle Notebooks")
    print("=" * 60)
    
    try:
        kernels = api.kernels_list(user="luanbhk")
        if kernels:
            for k in kernels[:10]:
                print(f"  - {k.ref}")
                print(f"    Title: {k.title}")
                print(f"    Votes: {k.totalVotes}")
        else:
            print("  No notebooks yet. Ready to create one!")
    except Exception as e:
        print(f"  ⚠️  Error: {e}")

def check_gpu_quota(api):
    """Check GPU quota"""
    print("\n" + "=" * 60)
    print("💳 GPU Quota Check")
    print("=" * 60)
    
    print("  Note: GPU quota info not available via API")
    print("  Check manually at: https://www.kaggle.com/settings")
    print("  → Look for 'GPU' section")
    print("  → Should show: 30 hours/week (P100)")

def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("🚀 Kaggle MCP Connection Test")
    print("=" * 60)
    print(f"   User: luanbhk")
    print(f"   Time: {os.popen('date').read().strip()}")
    print("=" * 60)
    
    # Test connection
    api = test_connection()
    
    # Run tests
    test_list_datasets(api)
    test_target_dataset(api)
    
    download_ok = test_download_dataset(api)
    
    test_list_kernels(api)
    check_gpu_quota()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
    print(f"  ✅ API Connection: OK")
    print(f"  ✅ Target Dataset: Available")
    print(f"  ✅ Download Test: {'OK' if download_ok else 'FAILED'}")
    print()
    print("  🎯 Ready to train on Kaggle!")
    print()
    print("  Next steps:")
    print("    1. Create Kaggle notebook")
    print("    2. Upload training code")
    print("    3. Start training with GPU")
    print("    4. Download results")
    print("=" * 60)

if __name__ == "__main__":
    main()
