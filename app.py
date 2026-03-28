import streamlit as st
import numpy as np
import pickle

st.title("🚀 Satellite Brightness Predictor")

st.write("App is running ✅")

@st.cache_resource
def load_model():
    try:
        return pickle.load(open("model.pkl", "rb"))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is not None:
    range_km = st.number_input("Range (km)", value=500.0)
    range_rate = st.number_input("Range Rate (km/s)", value=1.0)
    lat = st.number_input("Latitude", value=20.0)
    lon = st.number_input("Longitude", value=77.0)
    declination = st.number_input("Declination", value=10.0)
    ra = st.number_input("Right Ascension", value=50.0)
    lim_mag = st.number_input("Limiting Magnitude", value=6.0)

    if st.button("Predict"):
        input_data = np.array([[range_km, range_rate, lat, lon, declination, ra, lim_mag]])
        prediction = model.predict(input_data)
        st.success(f"Predicted Brightness: {prediction[0]:.2f}")
