import os
import pickle
import streamlit as st

st.write("App is running ✅")

@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.getcwd(), "model.pkl")
        return pickle.load(open(model_path, "rb"))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
