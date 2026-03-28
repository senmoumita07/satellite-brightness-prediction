import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Satellite Dashboard", layout="wide")

st.title("🚀 Satellite Brightness Prediction Dashboard")
st.markdown("Interactive ML App with Real Data Visualization")

# =========================
# DEBUG / FILE CHECK
# =========================
st.sidebar.header("System Info")
st.sidebar.write("Files:", os.listdir())

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model.pkl")
        return pickle.load(open(model_path, "rb"))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("satellite_observations_all.csv")

df = load_data()

# =========================
# SESSION STATE
# =========================
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Input Parameters")

range_km = st.sidebar.number_input("Range (km)", value=500.0)
range_rate = st.sidebar.number_input("Range Rate (km/s)", value=1.0)
lat = st.sidebar.number_input("Latitude", value=20.0)
lon = st.sidebar.number_input("Longitude", value=77.0)
declination = st.sidebar.number_input("Declination", value=10.0)
ra = st.sidebar.number_input("Right Ascension", value=50.0)
lim_mag = st.sidebar.slider("Limiting Magnitude", 0.0, 10.0, 6.0)

# =========================
# MAIN DASHBOARD
# =========================
col1, col2 = st.columns(2)

# =========================
# PREDICTION SECTION
# =========================
with col1:
    st.subheader("🔮 Prediction")

    if model is not None:
        if st.button("Predict Brightness"):
            input_data = np.array([[range_km, range_rate, lat, lon, declination, ra, lim_mag]])
            prediction = model.predict(input_data)[0]

            st.success(f"🌟 Predicted Brightness: {prediction:.2f}")

            # Save history
            st.session_state.predictions.append(prediction)

            # Interpretation
            if prediction < 3:
                st.info("Very Bright 🌟")
            elif prediction < 6:
                st.info("Moderately Visible 👀")
            else:
                st.warning("Dim Satellite 🌑")

    else:
        st.error("Model not loaded ❌")

# =========================
# PREDICTION HISTORY GRAPH
# =========================
with col2:
    st.subheader("📈 Prediction Trend")

    if len(st.session_state.predictions) > 0:
        fig, ax = plt.subplots()
        ax.plot(st.session_state.predictions, marker='o')
        ax.set_xlabel("Attempt")
        ax.set_ylabel("Brightness")
        ax.set_title("Prediction History")

        st.pyplot(fig)
    else:
        st.write("No predictions yet")

# =========================
# RESET BUTTON
# =========================
if st.button("Reset Predictions"):
    st.session_state.predictions = []

# =========================
# DATASET SECTION
# =========================
st.subheader("📊 Dataset Overview")

st.write(df.head())

# =========================
# REAL DATA VISUALIZATION
# =========================
col3, col4 = st.columns(2)

with col3:
    st.subheader("📊 Distance vs Brightness")

    fig2, ax2 = plt.subplots()
    ax2.scatter(df['range_to_satellite_km'], df['apparent_magnitude'])
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Brightness")
    ax2.set_title("Distance vs Brightness")

    st.pyplot(fig2)

with col4:
    st.subheader("📊 Brightness Distribution")

    fig3, ax3 = plt.subplots()
    ax3.hist(df['apparent_magnitude'], bins=30)
    ax3.set_xlabel("Brightness")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Brightness Distribution")

    st.pyplot(fig3)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Built by Moumita Sen | Data Science Project 🚀")
