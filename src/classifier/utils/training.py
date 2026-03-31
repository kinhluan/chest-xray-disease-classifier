"""Training utilities and metrics."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(accuracy, labels.size(0))
        
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss_meter.avg:.4f} Acc: {acc_meter.avg:.4f}"
            )
    
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate the model.
    
    Args:
        model: The model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Tuple of (average_loss, accuracy, all_labels, all_predictions)
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    all_labels = []
    all_predictions = []
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(accuracy, labels.size(0))
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
    
    return (
        loss_meter.avg,
        acc_meter.avg,
        np.array(all_labels),
        np.array(all_predictions),
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    
    # Per-class metrics
    metrics["per_class"] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    title: str = "Confusion Matrix",
):
    """Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = None,
):
    """Plot training history (loss and accuracy).
    
    Args:
        history: Dictionary containing train/val loss and accuracy
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history["train_acc"], label="Train Accuracy")
    ax2.plot(history["val_acc"], label="Val Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
