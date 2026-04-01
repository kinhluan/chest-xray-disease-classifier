#!/usr/bin/env python3
"""Evaluate all trained models and calculate comprehensive metrics.

This script loads all trained model checkpoints and evaluates them on the test set.
Metrics include accuracy, precision, recall, F1-score, TB sensitivity, and more.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from tqdm import tqdm

from classifier.data.dataset import create_dataloaders
from classifier.models.model import create_model
from classifier.models.attention_models import create_attention_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate all trained models"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="results/models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/metrics",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model and checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})
    
    # Determine model type from args
    model_type = args.get("model_type", "resnet")
    attention_type = args.get("attention_type", None)
    
    # Create model
    if model_type == "attention" or attention_type not in [None, "none"]:
        model = create_attention_model(
            backbone=args.get("model_name", "resnet50"),
            attention_type=attention_type or "none",
            num_classes=len(checkpoint["class_names"]),
            pretrained=False,
            dropout_rate=args.get("dropout_rate", 0.5),
        )
    else:
        model = create_model(
            model_type=model_type,
            model_name=args.get("model_name", "resnet50"),
            num_classes=len(checkpoint["class_names"]),
            pretrained=False,
            dropout_rate=args.get("dropout_rate", 0.5),
        )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # TB-specific metrics (assuming Tuberculosis is one of the classes)
    tb_idx = next((i for i, name in enumerate(class_names) if "tuberculosis" in name.lower()), None)
    if tb_idx is None:
        tb_idx = 2  # Default to index 2 if not found
    
    tb_sensitivity = recall_per_class[tb_idx] if tb_idx < len(recall_per_class) else 0.0
    tb_precision = precision_per_class[tb_idx] if tb_idx < len(precision_per_class) else 0.0
    tb_f1 = f1_per_class[tb_idx] if tb_idx < len(f1_per_class) else 0.0
    
    # Calculate specificity for TB
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape[0] > tb_idx and cm.shape[1] > tb_idx:
        # TN = sum of all elements except TB row and TB column
        tn = cm.sum() - cm[tb_idx, :].sum() - cm[:, tb_idx].sum() + cm[tb_idx, tb_idx]
        fp = cm[:, tb_idx].sum() - cm[tb_idx, tb_idx]
        tb_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        tb_specificity = 0.0
    
    # ROC-AUC (One-vs-Rest)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = 0.0
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    params_millions = num_params / 1e6
    
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "tb_sensitivity": tb_sensitivity,
        "tb_precision": tb_precision,
        "tb_f1": tb_f1,
        "tb_specificity": tb_specificity,
        "roc_auc_macro": roc_auc,
        "params_millions": params_millions,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_probs": all_probs,
        "per_class_precision": precision_per_class.tolist(),
        "per_class_recall": recall_per_class.tolist(),
        "per_class_f1": f1_per_class.tolist(),
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all model directories
    models_dir = Path(args.models_dir)
    model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
    
    if not model_dirs:
        print(f"No model directories found in {models_dir}")
        return
    
    print(f"\nFound {len(model_dirs)} models to evaluate:")
    for d in model_dirs:
        print(f"  - {d.name}")
    print()
    
    # Create test dataloader (using val_split as test for now)
    print("Loading test data...")
    _, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=0.2,  # Use validation split as test
        num_workers=4,
        random_state=42,
        test_mode=True,  # Use test transforms (no augmentation)
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {class_names}")
    print()
    
    # Evaluate all models
    results = []
    
    for model_dir in tqdm(model_dirs, desc="Evaluating models"):
        checkpoint_path = model_dir / "best_model.pth"
        
        if not checkpoint_path.exists():
            print(f"  ⚠ Skipping {model_dir.name}: No checkpoint found")
            continue
        
        print(f"\nEvaluating {model_dir.name}...")
        
        try:
            # Load model
            model, checkpoint = load_model(checkpoint_path, device)
            
            # Evaluate
            metrics = evaluate_model(model, test_loader, device, class_names)
            
            # Add model info
            metrics["model_name"] = model_dir.name
            metrics["checkpoint_epoch"] = checkpoint.get("epoch", 0)
            metrics["model_size_mb"] = checkpoint_path.stat().st_size / 1e6
            
            results.append(metrics)
            
            print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}")
            print(f"  ✓ TB Sensitivity: {metrics['tb_sensitivity']:.4f}")
            print(f"  ✓ F1-Macro: {metrics['f1_macro']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error evaluating {model_dir.name}: {e}")
            continue
    
    if not results:
        print("\nNo models were evaluated successfully!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV (without arrays)
    df_csv = df.drop(columns=["confusion_matrix", "all_preds", "all_labels", "all_probs"], errors="ignore")
    df_csv.to_csv(output_dir / f"all_metrics_{timestamp}.csv", index=False)
    df_csv.to_csv(output_dir / "all_metrics.csv", index=False)  # Latest
    
    # Save detailed JSON
    results_json = []
    for r in results:
        r_copy = r.copy()
        if "confusion_matrix" in r_copy:
            r_copy["confusion_matrix"] = r_copy["confusion_matrix"].tolist()
        results_json.append(r_copy)
    
    with open(output_dir / f"all_metrics_{timestamp}.json", "w") as f:
        json.dump(results_json, f, indent=2)
    
    with open(output_dir / f"all_metrics_latest.json", "w") as f:
        json.dump(results_json, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    # Sort by TB sensitivity (priority metric)
    df_sorted = df.sort_values("tb_sensitivity", ascending=False)
    
    print("\nModels ranked by TB Sensitivity:")
    print("-" * 80)
    for _, row in df_sorted.iterrows():
        print(f"{row['model_name']:30s} | TB Sens: {row['tb_sensitivity']:.4f} | "
              f"Acc: {row['accuracy']:.4f} | F1: {row['f1_macro']:.4f} | "
              f"Params: {row['params_millions']:.1f}M")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
