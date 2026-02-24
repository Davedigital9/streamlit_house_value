import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# --- Model Loading from Google Drive ---

# Google Drive link to your best_model.pkl file
# IMPORTANT: Replace this with your actual shareable link
google_drive_url = 'https://drive.google.com/file/d/1CEoVeR6moN0Mv8EUmSiEhmfqVx9sJCV8/view?usp=drive_link'

# Define a path to save the downloaded model locally
model_local_path = 'best_model.pkl'

# Function to download the model from Google Drive
# This function will be cached by Streamlit to avoid re-downloading on every rerun
@st.cache_resource
def load_model_from_gdrive(url, filename):
    if not os.path.exists(filename):
        st.info(f"Downloading model from Google Drive to {filename}...")
        try:
            # gdown simplifies downloading from Google Drive links
            # The 'fuzzy=True' option makes it more robust to different Google Drive link formats
            gdown.download(url, filename, quiet=False, fuzzy=True)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            st.stop() # Stop the app if download fails
    else:
        st.info("Model file already exists locally. Skipping download.")

    # Load the model using joblib
    st.write("Loading model...")
    model = joblib.load(filename)
    st.success("Model loaded successfully!")
    return model

# Load the model using the cached function
model = load_model_from_gdrive(google_drive_url, model_local_path)

# --- Streamlit App Layout and Inputs ---
st.title('Housing Price Predictor')
st.write('Enter the details of the property to get a predicted median house value.')

# Input features for your model
# Referencing the columns in X_train: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income

st.subheader('Location Information')
longitude = st.number_input('Longitude', value=-122.0, min_value=-125.0, max_value=-114.0, format="%.4f")
latitude = st.number_input('Latitude', value=37.0, min_value=32.0, max_value=42.0, format="%.4f")

st.subheader('Property Details')
housing_median_age = st.slider('Housing Median Age', min_value=1, max_value=52, value=30)
total_rooms = st.number_input('Total Rooms', value=2000.0, min_value=1.0, max_value=40000.0)
total_bedrooms = st.number_input('Total Bedrooms', value=400.0, min_value=1.0, max_value=7000.0)

st.subheader('Demographic Information')
population = st.number_input('Population', value=1000.0, min_value=3.0, max_value=40000.0)
households = st.number_input('Households', value=350.0, min_value=1.0, max_value=7000.0)
median_income = st.number_input('Median Income (in 10k USD)', value=3.5, min_value=0.5, max_value=15.0, format="%.4f")

# Create a DataFrame from user inputs
input_df = pd.DataFrame([[
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income
]],
    columns=[
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income'
    ])

if st.button('Predict Median House Value'):
    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f'Predicted Median House Value: ${prediction:,.2f}')
