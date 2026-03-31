"""Streamlit app for chest X-ray disease classification."""

import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import streamlit as st

# Model class
class ChestXRayClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
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

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model
@st.cache_resource
def load_model():
    model_path = os.environ.get("MODEL_PATH", "models/best_model.pth")
    DEFAULT_CLASSES = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]
    
    if Path(model_path).exists():
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            class_names = checkpoint.get("class_names", DEFAULT_CLASSES)
            model = ChestXRayClassifier(num_classes=len(class_names))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            return model, class_names
        except Exception as e:
            st.warning(f"Could not load model: {e}")
    
    # Return untrained model
    return ChestXRayClassifier(), DEFAULT_CLASSES

# Sidebar
st.sidebar.header("Model Info")
st.sidebar.info("""
- **Architecture:** ResNet50
- **Framework:** PyTorch
- **Input Size:** 224x224
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
    
    # Load model and predict
    with st.spinner("Loading model..."):
        model, class_names = load_model()
    
    # Predict
    with st.spinner("Analyzing..."):
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            probs = model.predict_proba(image_tensor)[0]
        
        # Get predictions
        top_probs, top_indices = torch.topk(probs, min(5, len(class_names)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                "Disease": class_names[idx.item()],
                "Probability": f"{prob.item()*100:.1f}%",
            })
    
    with col2:
        st.subheader("Predictions")
        st.dataframe(results, hide_index=True, use_container_width=True)
    
    # Show confidence bar chart
    st.subheader("Confidence Distribution")
    chart_data = {
        "Disease": [r["Disease"] for r in results],
        "Probability": [float(r["Probability"].replace("%", "")) for r in results],
    }
    st.bar_chart(chart_data, x="Disease", y="Probability")

else:
    st.info("👆 Please upload a chest X-ray image to begin.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>

**Chest X-Ray Disease Classifier** | Powered by PyTorch & Streamlit

[GitHub](https://github.com/kinhluan/chest-xray-disease-classifier) | 
[Hugging Face](https://huggingface.co/spaces/kinhluan/chest-xray-disease-classifier)

</div>
""", unsafe_allow_html=True)
