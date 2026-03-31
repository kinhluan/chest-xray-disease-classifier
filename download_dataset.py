"""Download and prepare chest X-ray datasets."""

import argparse
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(
    dataset_id: str,
    output_dir: str = "data/raw",
    unzip: bool = True,
):
    """Download dataset from Kaggle.
    
    Args:
        dataset_id: Kaggle dataset ID (e.g., 'paultimothymooney/chest-xray-pneumonia')
        output_dir: Directory to save dataset
        unzip: Whether to unzip after download
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    print(f"Downloading {dataset_id}...")
    api.dataset_download_files(dataset_id, path=output_dir, unzip=unzip)
    print(f"✓ Downloaded to {output_path}")


def setup_pneumonia_dataset(data_dir: str = "data/raw"):
    """Setup the Pneumonia dataset from Kaggle.
    
    Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
    """
    dataset_id = "paultimothymooney/chest-xray-pneumonia"
    
    print("=" * 60)
    print("Downloading Chest X-Ray Pneumonia Dataset")
    print("=" * 60)
    print(f"Dataset: {dataset_id}")
    print(f"Classes: Normal, Pneumonia")
    print(f"Images: ~5,800")
    print()
    
    download_kaggle_dataset(dataset_id, data_dir)
    
    # Organize structure if needed
    pneumonia_dir = Path(data_dir) / "chest_xray"
    if pneumonia_dir.exists():
        print("\nOrganizing dataset structure...")
        # Move train set to raw
        train_dir = pneumonia_dir / "train"
        if train_dir.exists():
            for class_folder in train_dir.iterdir():
                if class_folder.is_dir():
                    target = Path(data_dir) / class_folder.name
                    target.mkdir(exist_ok=True)
                    # Move files
                    for img in class_folder.glob("*.jpeg"):
                        img.rename(target / img.name)
            print(f"✓ Organized training data")
    
    print("\n✓ Dataset ready!")
    print(f"Location: {Path(data_dir).absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Download chest X-ray datasets"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="pneumonia",
        choices=["pneumonia", "17diseases"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # Check Kaggle credentials
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        print("❌ Kaggle credentials not found!")
        print("\nTo download datasets, you need:")
        print("1. Create account at https://www.kaggle.com")
        print("2. Go to https://www.kaggle.com/account")
        print("3. Create API token")
        print("4. Set environment variables:")
        print("   export KAGGLE_USERNAME=your_username")
        print("   export KAGGLE_KEY=your_api_key")
        print("\nOr download manually from the links below.")
        return
    
    if args.dataset == "pneumonia":
        setup_pneumonia_dataset(args.output_dir)
    else:
        print("Dataset not implemented yet. Download manually:")
        print("- 17 Diseases: https://www.kaggle.com/datasets/trainingdatapro/chest-xray-17-diseases")


if __name__ == "__main__":
    main()
