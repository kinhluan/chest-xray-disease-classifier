"""Streamlit app for chest X-ray disease classification using pretrained model from Hugging Face."""

import os
from pathlib import Path

import torch
from PIL import Image
import streamlit as st
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

# Page config
st.set_page_config(
    page_title="Chest X-Ray Classifier",
    page_icon="🫁",
    layout="centered",
)

# Title
st.title("🫁 Chest X-Ray Disease Classifier")
st.markdown("""
Upload a chest X-ray image to classify potential diseases.

**⚠️ Disclaimer:** For educational/research purposes only. 
**NOT for medical diagnosis.** Always consult healthcare professionals.
""")

# Load model from Hugging Face
@st.cache_resource
def load_model():
    """Load pretrained model from Hugging Face."""
    
    # Try to load local trained model first
    model_path = os.environ.get("MODEL_PATH", "models/best_model.pth")
    if Path(model_path).exists():
        try:
            # Local PyTorch model
            from torchvision import transforms
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            class_names = checkpoint.get("class_names", [])
            
            # Load custom model
            import torch.nn as nn
            from torchvision import models
            
            class ChestXRayClassifier(nn.Module):
                def __init__(self, num_classes=4):
                    super().__init__()
                    self.backbone = models.resnet50(weights=None)
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
            
            model = ChestXRayClassifier(num_classes=len(class_names))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            st.sidebar.success("✓ Loaded trained model from checkpoint")
            return {"type": "pytorch", "model": model, "transform": transform, "classes": class_names}
            
        except Exception as e:
            st.sidebar.warning(f"Could not load checkpoint: {e}")
    
    # Fallback to Hugging Face pretrained model
    model_name = "DIMA806/CHEST_XRAY_PNEUMONIA_DETECTION"
    
    try:
        classifier = pipeline(
            "image-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        st.sidebar.info(f"ℹ️ Using HF pretrained model: {model_name}")
        return {"type": "hf", "model": classifier, "classes": None}
    except Exception as e:
        st.sidebar.error(f"Could not load HF model: {e}")
        return {"type": "none", "model": None, "classes": None}

# Sidebar
st.sidebar.header("Model Info")
st.sidebar.markdown("""
- **Framework:** PyTorch / Transformers
- **Source:** Hugging Face Hub
- **Device:** CPU/CUDA Auto
""")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is not None:
    # Display image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model_info = load_model()
    
    if model_info["type"] == "none":
        st.error("❌ No model available. Please train a model first.")
        st.markdown("""
        ### How to train:
        ```bash
        # Download dataset
        export KAGGLE_USERNAME=your_username
        export KAGGLE_KEY=your_api_key
        ./download_and_train.sh
        
        # Or use custom dataset
        uv run python train.py --data_dir data/raw
        ```
        """)
    else:
        # Predict
        with st.spinner("Analyzing image..."):
            if model_info["type"] == "hf":
                # Hugging Face pipeline
                results = model_info["model"](image)
                predictions = [
                    {"Disease": r["label"], "Probability": f"{r['score']*100:.1f}%"}
                    for r in results[:5]
                ]
            else:
                # PyTorch model
                image_tensor = model_info["transform"](image).unsqueeze(0)
                with torch.no_grad():
                    probs = model_info["model"].predict_proba(image_tensor)[0]
                
                top_probs, top_indices = torch.topk(probs, min(5, len(model_info["classes"])))
                predictions = [
                    {"Disease": model_info["classes"][idx.item()], "Probability": f"{prob.item()*100:.1f}%"}
                    for prob, idx in zip(top_probs, top_indices)
                ]
        
        with col2:
            st.subheader("Predictions")
            st.dataframe(predictions, hide_index=True, use_container_width=True)
        
        # Show confidence bar chart
        st.subheader("Confidence Distribution")
        chart_data = {
            "Disease": [p["Disease"] for p in predictions],
            "Probability": [float(p["Probability"].replace("%", "")) for p in predictions],
        }
        st.bar_chart(chart_data, x="Disease", y="Probability")

else:
    st.info("👆 Please upload a chest X-ray image to begin.")
    
    # Show example classes
    st.markdown("""
    ### What this model can detect:
    - ✅ **Normal** - Healthy chest X-ray
    - 🦠 **Pneumonia** - Lung infection
    - 🦠 **Tuberculosis (TB)** - Bacterial infection  
    - 🦠 **COVID-19** - Coronavirus infection
    
    ### Using Pretrained Model
    This Space uses **DIMA806/CHEST_XRAY_PNEUMONIA_DETECTION** from Hugging Face Hub.
    
    For custom 4-class classification (Normal, Pneumonia, TB, COVID), train your own model:
    ```bash
    ./download_and_train.sh
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>

**Chest X-Ray Disease Classifier** | Powered by PyTorch & Streamlit

[GitHub](https://github.com/kinhluan/chest-xray-disease-classifier) | 
[Hugging Face](https://huggingface.co/spaces/kinhluan/chest-xray-disease-classifier)

</div>
""", unsafe_allow_html=True)
