"""Main training script for chest X-ray disease classification."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from classifier.data.dataset import create_dataloaders
from classifier.models.model import create_model
from classifier.utils.training import (
    train_epoch,
    validate,
    calculate_metrics,
    plot_confusion_matrix,
    plot_training_history,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train chest X-ray disease classifier"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet",
        choices=["resnet", "densenet"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        help="Specific model variant",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone weights",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["reduce", "cosine", "none"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save models and outputs",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save arguments
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        random_state=args.seed,
    )
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_type=args.model_type,
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout_rate=args.dropout_rate,
        freeze_backbone=args.freeze_backbone,
    )
    model = model.to(device)
    
    # Calculate class weights for imbalanced dataset
    try:
        class_weights = train_loader.dataset.dataset.get_class_weights()
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights}")
    except:
        class_weights = None
    
    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    if args.scheduler == "reduce":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    else:
        scheduler = None
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_amp=args.use_amp,
        )
        
        # Validate
        val_loss, val_acc, val_labels, val_preds = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        
        # Update scheduler
        if scheduler is not None:
            if args.scheduler == "reduce":
                scheduler.step(val_loss)
            elif args.scheduler == "cosine":
                scheduler.step()
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "class_names": class_names,
                    "args": vars(args),
                },
                output_dir / "best_model.pth",
            )
            print(f"  ✓ Saved new best model (acc: {val_acc:.4f})")
        
        # Save latest model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "args": vars(args),
            },
            output_dir / "latest_model.pth",
        )
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print("=" * 50)
    
    # Load best model for final evaluation
    checkpoint = torch.load(output_dir / "best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Final validation with metrics
    _, _, final_labels, final_preds = validate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
    )
    
    metrics = calculate_metrics(final_labels, final_preds, class_names)
    
    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = metrics.copy()
        metrics_json["confusion_matrix"] = metrics["confusion_matrix"].tolist()
        del metrics_json["per_class"]
        json.dump(metrics_json, f, indent=2)
    
    # Save classification report
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(metrics["per_class"])
    
    # Plot and save confusion matrix
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        save_path=output_dir / "confusion_matrix.png",
        title="Confusion Matrix",
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=output_dir / "training_history.png",
    )
    
    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nClassification Report:")
    print(metrics["per_class"])


if __name__ == "__main__":
    main()
