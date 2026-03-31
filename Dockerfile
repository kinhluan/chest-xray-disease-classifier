FROM python:3.10.11

# Install dependencies
RUN pip install --no-cache-dir \
    transformers>=4.30.0 \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    pillow>=9.0.0 \
    opencv-python-headless>=4.0.0 \
    tqdm \
    scikit-learn \
    pandas

COPY . /app
WORKDIR /app

CMD ["streamlit", "run", "streamlit_app.py"]
