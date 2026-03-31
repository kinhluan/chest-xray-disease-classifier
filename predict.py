"""Inference script for chest X-ray disease classification."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.models.model import ChestXRayClassifier, DenseNetClassifier


class ChestXRayPredictor:
    """Predictor for chest X-ray disease classification."""
    
    def __init__(
        self,
        model_path: str,
        device: str = None,
        img_size: int = 224,
    ):
        """
        Args:
            model_path: Path to saved model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
            img_size: Image size for inference
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        
        # Load checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model info
        self.class_names = self.checkpoint["class_names"]
        self.num_classes = len(self.class_names)
        self.args = self.checkpoint.get("args", {})
        
        # Initialize model
        model_type = self.args.get("model_type", "resnet")
        model_name = self.args.get("model_name", "resnet50")
        
        if model_type == "resnet":
            self.model = ChestXRayClassifier(
                num_classes=self.num_classes,
                model_name=model_name,
                pretrained=False,
            )
        elif model_type == "densenet":
            self.model = DenseNetClassifier(
                num_classes=self.num_classes,
                model_name=model_name,
                pretrained=False,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        print(f"Loaded model from {model_path}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")
    
    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Preprocess a single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed tensor
        """
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)
    
    @torch.no_grad()
    def predict(
        self,
        image_path: Union[str, Path],
        top_k: int = 3,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Predict disease for a single image.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
        
        Returns:
            Tuple of (predicted_class, list of (class, probability) tuples)
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Get predictions
        outputs = self.model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        
        predictions = [
            (self.class_names[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return predictions[0][0], predictions
    
    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
    ) -> List[Tuple[str, List[Tuple[str, float]]]]:
        """Predict diseases for multiple images.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for inference
        
        Returns:
            List of tuples (predicted_class, list of (class, probability) tuples)
        """
        results = []
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = [
                self.preprocess_image(path) for path in batch_paths
            ]
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Get predictions
            outputs = self.model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Process each prediction
            for j in range(len(batch_paths)):
                top_probs, top_indices = torch.topk(probs[j], 3)
                predictions = [
                    (self.class_names[idx.item()], prob.item())
                    for idx, prob in zip(top_indices, top_probs)
                ]
                results.append((predictions[0][0], predictions))
        
        return results
    
    def predict_directory(
        self,
        dir_path: Union[str, Path],
        extensions: List[str] = None,
    ) -> dict:
        """Predict diseases for all images in a directory.
        
        Args:
            dir_path: Path to directory
            extensions: List of file extensions to process
        
        Returns:
            Dictionary mapping image paths to predictions
        """
        dir_path = Path(dir_path)
        
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        
        # Collect all image paths
        image_paths = []
        for ext in extensions:
            image_paths.extend(dir_path.glob(f"*{ext}"))
            image_paths.extend(dir_path.glob(f"*{ext.upper()}"))
        
        image_paths = sorted(image_paths)
        
        if len(image_paths) == 0:
            print(f"No images found in {dir_path}")
            return {}
        
        # Make predictions
        predictions = self.predict_batch(image_paths)
        
        # Create results dictionary
        results = {
            str(path): pred for path, pred in zip(image_paths, predictions)
        }
        
        return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Predict chest X-ray diseases"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to single image or directory of images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save predictions (JSON)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size",
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ChestXRayPredictor(
        model_path=args.model_path,
        device=args.device,
        img_size=args.img_size,
    )
    
    # Make predictions
    if args.image_path is None:
        print("Please provide --image_path")
        return
    
    image_path = Path(args.image_path)
    
    if image_path.is_file():
        # Single image
        pred_class, predictions = predictor.predict(image_path)
        
        print(f"\nImage: {image_path}")
        print(f"Predicted: {pred_class}")
        print("Top predictions:")
        for cls, prob in predictions:
            print(f"  {cls}: {prob:.4f}")
        
        results = {str(image_path): {"prediction": pred_class, "probabilities": predictions}}
    
    elif image_path.is_dir():
        # Directory
        results = predictor.predict_directory(image_path)
        
        print(f"\nProcessed {len(results)} images")
        for path, (pred_class, predictions) in list(results.items())[:5]:
            print(f"\n{path}: {pred_class}")
            for cls, prob in predictions:
                print(f"  {cls}: {prob:.4f}")
        if len(results) > 5:
            print(f"... and {len(results) - 5} more")
    
    else:
        print(f"Invalid path: {image_path}")
        return
    
    # Save results
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
