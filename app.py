"""Hugging Face Gradio app for chest X-ray disease classification."""

import os
import tempfile
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from predict import ChestXRayPredictor

# Get model path from environment or use default
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pth")

# Check if model exists
if not Path(MODEL_PATH).exists():
    MODEL_PATH = None

# Initialize predictor if model is available
predictor = None
if MODEL_PATH:
    try:
        predictor = ChestXRayPredictor(
            model_path=MODEL_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        predictor = None


def predict_disease(image: Image.Image) -> list:
    """Predict disease from uploaded image.
    
    Args:
        image: PIL Image of chest X-ray
    
    Returns:
        List of dictionaries with class names and probabilities
    """
    if predictor is None:
        return [{"Error": "Model not loaded. Please upload a model checkpoint."}]
    
    # Save image temporarily
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Get predictions
        _, predictions = predictor.predict(tmp_path, top_k=5)
        
        # Format for Gradio
        result = [
            {"Disease": cls, "Probability": f"{prob:.2%}"}
            for cls, prob in predictions
        ]
        
        return result
    finally:
        # Clean up
        os.unlink(tmp_path)


def create_demo() -> gr.Blocks:
    """Create Gradio demo interface."""
    with gr.Blocks(title="Chest X-Ray Disease Classifier") as demo:
        gr.Markdown(
            """
            # 🫁 Chest X-Ray Disease Classifier
            
            Upload a chest X-ray image to classify potential diseases.
            This model uses a deep learning architecture (ResNet/DenseNet) 
            trained on chest X-ray images to detect various diseases.
            
            **Note:** This is for research/educational purposes only and 
            should NOT be used for medical diagnosis.
            """
        )
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Chest X-Ray",
                    type="pil",
                )
            
            with gr.Column():
                output_df = gr.Dataframe(
                    label="Predictions",
                    headers=["Disease", "Probability"],
                    row_count=5,
                    col_count=2,
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
            ### About
            This model is trained on chest X-ray images for disease classification.
            
            **Supported diseases:** Depends on the training dataset
            
            **Model Architecture:** ResNet/DenseNet with custom classification head
            
            **Disclaimer:** This tool is for educational and research purposes only.
            Always consult qualified medical professionals for diagnosis.
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
