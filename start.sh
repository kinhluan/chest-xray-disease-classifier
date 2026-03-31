#!/bin/bash
# Startup script for Hugging Face Space

# Install the package in editable mode
pip install -e . --quiet

# Start the Gradio app
python app.py
