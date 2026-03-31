"""Deploy to both GitHub and Hugging Face Spaces."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError


def run_command(command: list, cwd: str = None) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def check_hf_login() -> bool:
    """Check if logged in to Hugging Face."""
    try:
        api = HfApi()
        api.whoami()
        return True
    except Exception:
        return False


def hf_login_check() -> bool:
    """Check HF login using hf CLI command."""
    import subprocess
    try:
        result = subprocess.run(
            ["hf", "auth", "whoami"],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def push_to_github(message: str = "Deploy: Update"):
    """Push changes to GitHub."""
    print("\n" + "=" * 50)
    print("Pushing to GitHub...")
    print("=" * 50)
    
    # Check if remote exists
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print("❌ No GitHub remote found.")
        print("Please add your remote:")
        print("  git remote add origin https://github.com/kinhluan/chest-xray-disease-classifier.git")
        return False
    
    # Stage all changes
    run_command(["git", "add", "-A"])
    
    # Commit
    run_command(["git", "commit", "-m", message])
    
    # Push
    if not run_command(["git", "push", "origin", "master"]):
        print("❌ Failed to push to GitHub")
        return False
    
    print("✅ Pushed to GitHub")
    return True


def push_to_hf_space(space_id: str, include_models: bool = False):
    """Push to Hugging Face Spaces."""
    print("\n" + "=" * 50)
    print(f"Pushing to Hugging Face Spaces: {space_id}")
    print("=" * 50)
    
    # Check login
    if not hf_login_check():
        print("❌ Not logged in to Hugging Face")
        print("Please run: hf auth login")
        return False
    
    api = HfApi()
    
    # Get repo root
    repo_root = Path.cwd()
    
    # Files to upload
    include_patterns = [
        "*.py",
        "*.txt",
        "*.toml",
        "*.md",
        "*.json",
    ]
    
    # Patterns to exclude
    exclude_patterns = [
        ".git/*",
        "*.pth",
        "*.pt",
        "*.bin",
        "*.onnx",
        "data/*",
        ".venv/*",
        "__pycache__/*",
        "*.pyc",
        ".DS_Store",
    ]
    
    if not include_models:
        exclude_patterns.append("models/*")
    
    try:
        # Upload to space
        api.upload_folder(
            folder_path=str(repo_root),
            repo_id=space_id,
            repo_type="space",
            include=include_patterns,
            exclude=exclude_patterns,
        )
        print("✅ Pushed to Hugging Face Spaces")
        return True
    except HfHubHTTPError as e:
        print(f"❌ Failed to push to Hugging Face: {e}")
        return False


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(
        description="Deploy to GitHub and Hugging Face Spaces"
    )
    
    parser.add_argument(
        "--space-id",
        type=str,
        default="kinhluan/chest-xray-disease-classifier",
        help="Hugging Face Space ID",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Deploy: Update",
        help="Git commit message",
    )
    parser.add_argument(
        "--github-only",
        action="store_true",
        help="Only push to GitHub",
    )
    parser.add_argument(
        "--hf-only",
        action="store_true",
        help="Only push to Hugging Face",
    )
    parser.add_argument(
        "--include-models",
        action="store_true",
        help="Include model checkpoints in HF upload",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 50)
    print("🚀 Deployment Script")
    print("=" * 50)
    print(f"Space ID: {args.space_id}")
    print(f"Commit: {args.commit_message}")
    print()
    
    success = True
    
    # Push to GitHub
    if not args.hf_only:
        if not push_to_github(args.commit_message):
            success = False
    
    # Push to Hugging Face
    if not args.github_only:
        if not push_to_hf_space(args.space_id, args.include_models):
            success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Deployment Complete!")
        print("=" * 50)
        print("\nLinks:")
        print(f"  GitHub: https://github.com/kinhluan/chest-xray-disease-classifier")
        print(f"  Hugging Face: https://huggingface.co/spaces/{args.space_id}")
    else:
        print("❌ Deployment failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
