# Streamlit entry point
import streamlit as st
from streamlit_app import load_model, predict_disease, transform

# Re-run the streamlit app
exec(open("streamlit_app.py").read())
