"""Hugging Face Gradio app for chest X-ray disease classification."""

import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import gradio as gr

# Model classes
class ChestXRayClassifier(nn.Module):
    def __init__(self, num_classes=4, model_name="resnet50", pretrained=True):
        super().__init__()
        from torchvision import models
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

# Default class names (will be updated when model is loaded)
DEFAULT_CLASSES = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model
model = None
class_names = DEFAULT_CLASSES
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pth")

def load_model():
    global model, class_names
    if Path(MODEL_PATH).exists():
        try:
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
            class_names = checkpoint.get("class_names", DEFAULT_CLASSES)
            model = ChestXRayClassifier(num_classes=len(class_names))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            print(f"✓ Loaded model from {MODEL_PATH}")
            print(f"  Classes: {class_names}")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            model = ChestXRayClassifier()
    else:
        print(f"⚠️ No model found at {MODEL_PATH}, using random weights")
        model = ChestXRayClassifier()

def predict_disease(image):
    """Predict disease from uploaded image."""
    global model, class_names
    
    if model is None:
        load_model()
    
    # Preprocess
    image_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        probs = model.predict_proba(image_tensor)[0]
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probs, min(5, len(class_names)))
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "Disease": class_names[idx.item()],
            "Probability": f"{prob.item():.2%}"
        })
    
    return results

def create_demo():
    """Create Gradio demo interface."""
    with gr.Blocks(title="Chest X-Ray Disease Classifier") as demo:
        gr.Markdown(
            """
            # 🫁 Chest X-Ray Disease Classifier
            
            Upload a chest X-ray image to classify potential diseases.
            
            **Disclaimer:** For research/educational purposes only. 
            NOT for medical diagnosis.
            """
        )
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Chest X-Ray", type="pil")
            
            with gr.Column():
                output_df = gr.Dataframe(
                    label="Predictions",
                    headers=["Disease", "Probability"],
                    row_count=5,
                    wrap=True,
                )
        
        submit_btn = gr.Button("Classify", variant="primary")
        submit_btn.click(
            fn=predict_disease,
            inputs=input_image,
            outputs=output_df,
        )
        
        gr.Markdown(
            """
            ---
            ### Model Info
            - **Architecture:** ResNet50
            - **Framework:** PyTorch
            - **Input:** 224x224 RGB images
            
            **⚠️ Disclaimer:** Educational purposes only. 
            Consult healthcare professionals for diagnosis.
            """
        )
    
    return demo

if __name__ == "__main__":
    # Load model on startup
    load_model()
    
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
