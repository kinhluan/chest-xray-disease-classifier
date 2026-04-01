#!/usr/bin/env python3
"""Visualize attention maps using Grad-CAM.

Shows where models are focusing attention in chest X-ray images.
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image

from classifier.models.model import create_model
from classifier.models.attention_models import create_attention_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize attention maps using Grad-CAM"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/raw",
        help="Directory with test images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures/attention_maps",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="Tuberculosis",
        help="Class to visualize attention for",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=5,
        help="Number of images to visualize",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default=None,
        help="Target layer name for Grad-CAM (auto-detected if None)",
    )
    
    return parser.parse_args()


class GradCAM:
    """Grad-CAM implementation for visualizing model attention."""
    
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        # Find target layer
        if self.target_layer_name is None:
            # Auto-detect based on model type
            if hasattr(self.model, "backbone"):
                if hasattr(self.model.backbone, "layer4"):
                    self.target_layer_name = "backbone.layer4"
                elif hasattr(self.model.backbone, "features"):
                    # DenseNet
                    self.target_layer_name = "backbone.features.denseblock4"
        
        # Get the target module
        target_module = None
        if self.target_layer_name:
            parts = self.target_layer_name.split(".")
            module = self.model
            for part in parts:
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    break
            target_module = module
        
        if target_module is not None:
            target_module.register_forward_hook(self._forward_hook)
            target_module.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0]
    
    def generate(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor
            class_idx: Target class index (None for predicted class)
            
        Returns:
            Heatmap array
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        target = output[0, class_idx]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(2, 3))[0]
        
        # Weighted sum of activations
        heatmap = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[0, i, :, :]
        
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
        
        return heatmap, class_idx


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})
    
    # Determine model type
    model_type = args.get("model_type", "resnet")
    attention_type = args.get("attention_type", None)
    num_classes = len(checkpoint["class_names"])
    
    if model_type == "attention" or attention_type not in [None, "none"]:
        model = create_attention_model(
            backbone=args.get("model_name", "resnet50"),
            attention_type=attention_type or "none",
            num_classes=num_classes,
            pretrained=False,
            dropout_rate=args.get("dropout_rate", 0.5),
        )
        target_layer = "backbone.layer4" if "resnet" in args.get("model_name", "") else None
    else:
        model = create_model(
            model_type=model_type,
            model_name=args.get("model_name", "resnet50"),
            num_classes=num_classes,
            pretrained=False,
            dropout_rate=args.get("dropout_rate", 0.5),
        )
        target_layer = "backbone.layer4" if model_type == "resnet" else None
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint, target_layer


def find_images(image_dir, class_name, num_images):
    """Find sample images for a given class."""
    image_dir = Path(image_dir)
    class_dir = image_dir / class_name
    
    if not class_dir.exists():
        # Try to find similar class name
        for d in image_dir.iterdir():
            if d.is_dir() and class_name.lower() in d.name.lower():
                class_dir = d
                break
    
    if not class_dir.exists():
        print(f"  ⚠ Class directory not found: {class_name}")
        return []
    
    # Get image files
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]
    image_files = [
        f for f in class_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    # Sample images
    if len(image_files) > num_images:
        import random
        image_files = random.sample(image_files, num_images)
    
    return image_files[:num_images]


def preprocess_image(image_path, img_size=224):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    
    return image, tensor


def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlay heatmap on original image."""
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    
    # Convert to RGB heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    image_array = np.array(image) / 255.0
    overlay = (1 - alpha) * image_array + alpha * (heatmap_colored / 255.0)
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def visualize_attention(model, grad_cam, image_path, class_names, device, output_path):
    """Generate and save attention visualization."""
    # Load and preprocess image
    image, tensor = preprocess_image(image_path)
    tensor = tensor.to(device)
    
    # Generate Grad-CAM
    heatmap, pred_idx = grad_cam.generate(tensor)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image\nPredicted: {class_names[pred_idx]}", fontsize=12)
    axes[0].axis("off")
    
    # Heatmap only
    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    overlay = overlay_heatmap(image, heatmap)
    axes[2].imshow(overlay)
    axes[2].set_title(f"Attention Overlay\nTarget: {class_names[pred_idx]}", fontsize=12)
    axes[2].axis("off")
    
    plt.suptitle(f"Grad-CAM Visualization - {image_path.name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return pred_idx, heatmap


def main():
    """Main visualization function."""
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, checkpoint, target_layer = load_checkpoint(args.checkpoint, device)
    class_names = checkpoint["class_names"]
    
    print(f"  Model: {checkpoint['args'].get('model_name', 'unknown')}")
    print(f"  Classes: {class_names}")
    print(f"  Target layer: {target_layer}")
    
    # Create Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Find images
    print(f"\nFinding {args.num_images} images of class: {args.class_name}")
    image_files = find_images(args.image_dir, args.class_name, args.num_images)
    
    if not image_files:
        print("  ⚠ No images found")
        return
    
    print(f"  ✓ Found {len(image_files)} images")
    
    # Visualize each image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nGenerating attention visualizations...")
    print("-" * 50)
    
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {image_path.name}")
        
        output_path = output_dir / f"gradcam_{timestamp}_{i:03d}.png"
        
        try:
            pred_idx, heatmap = visualize_attention(
                model, grad_cam, image_path, class_names, device, output_path
            )
            print(f"  ✓ Saved: {output_path}")
            print(f"    Prediction: {class_names[pred_idx]}")
            print(f"    Heatmap max: {heatmap.max():.4f}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("-" * 50)
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
