"""Model architectures with attention mechanisms for chest X-ray classification."""

import torch
import torch.nn as nn
from typing import Optional, Literal
from torchvision import models

from classifier.models.attention import SEBlock, CBAM, ECA, create_attention_module


AttentionType = Literal["se", "cbam", "eca", "none", None]


class ResNetWithAttention(nn.Module):
    """ResNet backbone with attention module integration.

    Applies attention mechanism after each residual block stage.
    Supports ResNet18, ResNet34, ResNet50, and ResNet101.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet50",
        attention_type: AttentionType = "cbam",
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
        attention_reduction: int = 16,
    ):
        """
        Args:
            num_classes: Number of disease classes to classify
            model_name: ResNet variant to use
            attention_type: Type of attention ('se', 'cbam', 'eca', None)
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout rate for classification head
            freeze_backbone: Whether to freeze backbone weights
            attention_reduction: Reduction ratio for attention modules
        """
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name
        self.attention_type = attention_type

        # Load pretrained backbone
        if pretrained:
            if model_name.startswith("resnet"):
                weights = models.ResNet50_Weights.IMAGENET1K_V2
            else:
                weights = None
        else:
            weights = None

        # Create ResNet backbone
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

        # Remove original FC layer
        self.backbone.fc = nn.Identity()

        # Get channel counts for each stage
        if model_name in ["resnet18", "resnet34"]:
            channel_counts = [64, 128, 256, 512]
        else:  # resnet50, resnet101
            channel_counts = [256, 512, 1024, 2048]

        # Create attention modules for each stage
        self.attention_modules = nn.ModuleList()
        for channels in channel_counts:
            attn_module = create_attention_module(
                attention_type=attention_type,
                channels=channels,
                reduction=attention_reduction,
            )
            self.attention_modules.append(attn_module)

        # Hook handles for applying attention
        self._setup_hooks()

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

    def _setup_hooks(self):
        """Setup forward hooks to apply attention after each layer."""
        self.layer_outputs = {}

        def create_hook(stage_idx):
            def hook(module, input, output):
                # Apply attention to the output
                return self.attention_modules[stage_idx](output)

            return hook

        # Register hooks on each layer
        self.backbone.layer1.register_forward_hook(create_hook(1))
        self.backbone.layer2.register_forward_hook(create_hook(2))
        self.backbone.layer3.register_forward_hook(create_hook(3))
        self.backbone.layer4.register_forward_hook(create_hook(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Conv1 and pooling layers (before layer1)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer1-4 with attention applied via hooks
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Global average pooling
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # Classification head
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


class DenseNetWithAttention(nn.Module):
    """DenseNet backbone with attention module integration.

    Applies attention mechanism after each dense block.
    Supports DenseNet121, DenseNet169, and DenseNet201.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "densenet121",
        attention_type: AttentionType = "cbam",
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
        attention_reduction: int = 16,
    ):
        """
        Args:
            num_classes: Number of disease classes to classify
            model_name: DenseNet variant to use
            attention_type: Type of attention ('se', 'cbam', 'eca', None)
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout rate for classification head
            freeze_backbone: Whether to freeze backbone weights
            attention_reduction: Reduction ratio for attention modules
        """
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name
        self.attention_type = attention_type

        # Load pretrained backbone
        if pretrained:
            if model_name.startswith("densenet"):
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
            else:
                weights = None
        else:
            weights = None

        # Create DenseNet backbone
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

        # Get channel counts for each dense block
        if model_name == "densenet121":
            channel_counts = [256, 512, 1024, 1024]
        elif model_name == "densenet169":
            channel_counts = [384, 768, 1280, 1664]
        else:  # densenet201
            channel_counts = [384, 768, 1792, 1920]

        # Create attention modules for each dense block
        self.attention_modules = nn.ModuleList()
        for channels in channel_counts:
            attn_module = create_attention_module(
                attention_type=attention_type,
                channels=channels,
                reduction=attention_reduction,
            )
            self.attention_modules.append(attn_module)

        # Setup hooks for dense blocks
        self._setup_hooks()

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

    def _setup_hooks(self):
        """Setup forward hooks to apply attention after each dense block."""
        def create_hook(block_idx):
            def hook(module, input, output):
                return self.attention_modules[block_idx](output)
            return hook

        # Register hooks on each dense block
        if hasattr(self.backbone, "features") and hasattr(self.backbone.features, "denseblock1"):
            self.backbone.features.denseblock1.register_forward_hook(create_hook(0))
            self.backbone.features.denseblock2.register_forward_hook(create_hook(1))
            self.backbone.features.denseblock3.register_forward_hook(create_hook(2))
            self.backbone.features.denseblock4.register_forward_hook(create_hook(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


def create_attention_model(
    backbone: str = "resnet50",
    attention_type: AttentionType = "cbam",
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False,
    attention_reduction: int = 16,
) -> nn.Module:
    """Factory function to create models with attention.

    Args:
        backbone: Backbone architecture ('resnet18/34/50/101' or 'densenet121/169/201')
        attention_type: Type of attention ('se', 'cbam', 'eca', None)
        num_classes: Number of disease classes
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for classification head
        freeze_backbone: Whether to freeze backbone weights
        attention_reduction: Reduction ratio for attention modules

    Returns:
        Initialized model with attention

    Examples:
        >>> model = create_attention_model(
        ...     backbone='resnet50',
        ...     attention_type='cbam',
        ...     num_classes=4,
        ...     pretrained=True
        ... )
    """
    if backbone.startswith("resnet"):
        return ResNetWithAttention(
            num_classes=num_classes,
            model_name=backbone,
            attention_type=attention_type,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            attention_reduction=attention_reduction,
        )
    elif backbone.startswith("densenet"):
        return DenseNetWithAttention(
            num_classes=num_classes,
            model_name=backbone,
            attention_type=attention_type,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            attention_reduction=attention_reduction,
        )
    else:
        raise ValueError(
            f"Unsupported backbone: {backbone}. "
            "Choose from: resnet18/34/50/101, densenet121/169/201"
        )
