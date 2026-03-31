"""Dataset utilities for chest X-ray classification."""

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class ChestXRayDataset(Dataset):
    """Dataset for chest X-ray disease classification.
    
    Expected directory structure:
    data/
        raw/
            class_1/
                image1.jpg
                image2.jpg
            class_2/
                image3.jpg
                image4.jpg
            ...
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[list] = None,
    ):
        """
        Args:
            root_dir: Path to the dataset directory
            transform: Optional transform to apply to images
            class_names: Optional list of class names (will be inferred if not provided)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Get class names from directories or use provided ones
        if class_names:
            self.class_names = class_names
        else:
            self.class_names = sorted([
                d.name for d in self.root_dir.iterdir() if d.is_dir()
            ])
        
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }
        
        # Load all image paths and labels
        self._load_images()
        
    def _load_images(self):
        """Load all image paths and their corresponding labels."""
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            # Support common image formats
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.dcm']:
                for img_path in class_dir.glob(ext):
                    self.images.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self) -> list:
        """Return list of class names."""
        return self.class_names
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        class_counts = np.bincount(self.labels)
        total = len(self.labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)


def get_transforms(
    img_size: int = 224,
    is_training: bool = True,
) -> transforms.Compose:
    """Get image transforms for training or validation.
    
    Args:
        img_size: Target image size
        is_training: Whether to use training transforms (with augmentation)
    
    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    val_split: float = 0.2,
    num_workers: int = 4,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, list]:
    """Create training and validation dataloaders.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for dataloaders
        img_size: Target image size
        val_split: Fraction of data to use for validation
        num_workers: Number of workers for data loading
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # Create full dataset
    full_dataset = ChestXRayDataset(
        root_dir=data_dir,
        transform=None,  # Will apply transforms after split
    )
    
    # Split into train and validation
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        random_state=random_state,
        stratify=full_dataset.labels,
    )
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Apply transforms
    train_transform = get_transforms(img_size, is_training=True)
    val_transform = get_transforms(img_size, is_training=False)
    
    # Wrapper to apply transforms to subsets
    class SubsetTransformed(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            img = self.transform(img)
            return img, label
            
        def __len__(self):
            return len(self.subset)
    
    train_dataset = SubsetTransformed(train_dataset, train_transform)
    val_dataset = SubsetTransformed(val_dataset, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, full_dataset.get_class_names()
