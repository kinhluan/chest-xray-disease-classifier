FROM python:3.10.11

# Install streamlit instead of gradio - no huggingface_hub oauth dependency
RUN pip install --no-cache-dir \
    streamlit==1.35.0 \
    torch==2.3.1 \
    torchvision==0.18.1 \
    pillow==10.3.0 \
    opencv-python-headless==4.9.0.80

COPY . /app
WORKDIR /app

# Install project
RUN pip install -e . --no-deps
RUN pip install --no-cache-dir \
    scikit-learn \
    matplotlib \
    pandas \
    tqdm

CMD ["streamlit", "run", "streamlit_app.py"]
