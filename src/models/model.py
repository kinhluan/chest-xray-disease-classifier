"""Model architectures for chest X-ray classification."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ChestXRayClassifier(nn.Module):
    """ResNet-based classifier for chest X-ray disease classification.
    
    Uses a pretrained ResNet backbone with a custom classification head.
    Supports ResNet18, ResNet34, ResNet50, and ResNet101.
    """
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet50",
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            num_classes: Number of disease classes to classify
            model_name: ResNet variant to use
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout rate for classification head
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        
        if model_name == "resnet18":
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
        elif model_name == "resnet34":
            self.backbone = models.resnet34(weights=weights)
            in_features = self.backbone.fc.in_features
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
        elif model_name == "resnet101":
            self.backbone = models.resnet101(weights=weights)
            in_features = self.backbone.fc.in_features
        else:
            raise ValueError(
                f"Unsupported model_name: {model_name}. "
                "Choose from: resnet18, resnet34, resnet50, resnet101"
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Probabilities tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Predicted class indices of shape (batch_size,)
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


class DenseNetClassifier(nn.Module):
    """DenseNet-based classifier for chest X-ray disease classification.
    
    Uses a pretrained DenseNet backbone with a custom classification head.
    Supports DenseNet121, DenseNet169, and DenseNet201.
    """
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "densenet121",
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            num_classes: Number of disease classes to classify
            model_name: DenseNet variant to use
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout rate for classification head
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained backbone
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        
        if model_name == "densenet121":
            self.backbone = models.densenet121(weights=weights)
            in_features = self.backbone.classifier.in_features
        elif model_name == "densenet169":
            self.backbone = models.densenet169(weights=weights)
            in_features = self.backbone.classifier.in_features
        elif model_name == "densenet201":
            self.backbone = models.densenet201(weights=weights)
            in_features = self.backbone.classifier.in_features
        else:
            raise ValueError(
                f"Unsupported model_name: {model_name}. "
                "Choose from: densenet121, densenet169, densenet201"
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Probabilities tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Predicted class indices of shape (batch_size,)
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


def create_model(
    model_type: str = "resnet",
    model_name: str = "resnet50",
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Factory function to create a model.
    
    Args:
        model_type: Type of model ('resnet' or 'densenet')
        model_name: Specific model variant
        num_classes: Number of disease classes
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for classification head
        freeze_backbone: Whether to freeze backbone weights
    
    Returns:
        Initialized model
    """
    if model_type == "resnet":
        return ChestXRayClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "densenet":
        return DenseNetClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            "Choose from: resnet, densenet"
        )
