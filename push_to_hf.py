"""Script to push model to Hugging Face Hub."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Push model to Hugging Face Hub"
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID (username/repo-name)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory containing model files",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private",
    )
    parser.add_argument(
        "--include_requirements",
        action="store_true",
        help="Include requirements_hf.txt",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            repo_type="model",
            exist_ok=True,
        )
        print(f"Repository created/verified: {args.repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Upload model directory
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return
    
    # Upload folder
    upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        token=args.token,
        repo_type="model",
        ignore_patterns=["*.gitkeep", "latest_model.pth"],
    )
    
    print(f"\n✓ Model uploaded to: https://huggingface.co/{args.repo_id}")
    
    # Upload requirements if requested
    if args.include_requirements:
        req_path = Path("requirements_hf.txt")
        if req_path.exists():
            api.upload_file(
                path_or_fileobj=str(req_path),
                path_in_repo="requirements.txt",
                repo_id=args.repo_id,
                token=args.token,
                repo_type="model",
            )
            print("✓ Requirements uploaded")


if __name__ == "__main__":
    main()
