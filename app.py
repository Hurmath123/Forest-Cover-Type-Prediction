import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model and scaler
with open('best_forest_model.pkl', 'rb') as file:
    model, scaler = pickle.load(file)

# Forest cover type labels
cover_type_map = {
    1: 'Spruce/Fir 🌲',
    2: 'Lodgepole Pine 🌲',
    3: 'Ponderosa Pine 🌲',
    4: 'Cottonwood/Willow 🌳',
    5: 'Aspen 🍂',
    6: 'Douglas-fir 🌲',
    7: 'Krummholz 🌿'
}

# UI setup
st.set_page_config(page_title="Forest Cover Predictor (LGBM)", page_icon="🌲")
st.title("Forest Cover Type Prediction App (LGBM)")
st.markdown("""
Enter details about the terrain and environment to predict the most likely forest cover type.
""")

# --- Terrain Section ---
st.header("Terrain Features")
elevation = st.slider("Elevation (m)", 1800, 4000, 2500)
aspect = st.slider("Aspect (°)", 0, 360, 100)
slope = st.slider("Slope (°)", 0, 60, 15)

# --- Distance Features ---
st.header("Distance to Features")
h_distance_hydro = st.number_input("Horizontal Distance to Hydrology (m)", value=100)
v_distance_hydro = st.number_input("Vertical Distance to Hydrology (m)", value=30)
h_distance_road = st.number_input("Horizontal Distance to Roadways (m)", value=200)
h_distance_fire = st.number_input("Horizontal Distance to Fire Points (m)", value=300)

# --- Sunlight/Hillshade ---
st.header("Sunlight Exposure (Hillshade)")
hillshade_9am = st.slider("Hillshade at 9am", 0, 255, 180)
hillshade_noon = st.slider("Hillshade at Noon", 0, 255, 200)
hillshade_3pm = st.slider("Hillshade at 3pm", 0, 255, 150)

# --- Wilderness Area ---
st.header("Wilderness Area")
wilderness_option = st.selectbox("Select Wilderness Area", ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"])
wilderness_map = {
    "Rawah": [1, 0, 0, 0],
    "Neota": [0, 1, 0, 0],
    "Comanche Peak": [0, 0, 1, 0],
    "Cache la Poudre": [0, 0, 0, 1]
}
wilderness_encoded = wilderness_map[wilderness_option]

# --- Soil Type ---
st.header("Soil Type")
soil_type = st.slider("Soil Type Index (0–39)", 0, 39, 10)
soil_encoded = [0] * 40
soil_encoded[soil_type] = 1

# --- Feature Engineering ---
elevation_hydro_diff = elevation - v_distance_hydro

# --- Final Feature Vector ---
features = [
    elevation, aspect, slope,
    h_distance_hydro, v_distance_hydro, h_distance_road, h_distance_fire,
    hillshade_9am, hillshade_noon, hillshade_3pm,
    elevation_hydro_diff
] + wilderness_encoded + soil_encoded

features_np = np.array([features])
features_scaled = scaler.transform(features_np)

# --- Prediction ---
if st.button("🔍 Predict Forest Cover Type"):
    prediction = model.predict(features_scaled)[0] + 1
    result = cover_type_map.get(prediction, "Unknown")

    st.success(f"Predicted Cover Type: **{result}**")

    # Explanation (simplified)
    st.markdown(f"**Why?** The model considered elevation ({elevation} m), slope ({slope}°), and sunlight (noon hillshade {hillshade_noon}) to suggest this forest type.")

    # Chart: Your elevation vs. average elevations
    st.subheader("📊 Elevation Comparison")
    average_elevations = {
        'Spruce/Fir 🌲': 3500,
        'Lodgepole Pine 🌲': 2800,
        'Ponderosa Pine 🌲': 2500,
        'Cottonwood/Willow 🌳': 1900,
        'Aspen 🍂': 2700,
        'Douglas-fir 🌲': 3100,
        'Krummholz 🌿': 3750
    }
    labels = list(average_elevations.keys())
    values = list(average_elevations.values())

    fig, ax = plt.subplots()
    ax.barh(labels, values, color='lightgray')
    ax.axvline(elevation, color='green', label='Your Elevation')
    ax.set_xlabel("Elevation (m)")
    ax.set_title("Your Elevation vs. Typical Forest Types")
    ax.legend()
    st.pyplot(fig)
