import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# --- 1. SETUP ---
st.set_page_config(page_title="Berlin Airbnb Price Predictor", page_icon="🏠")

sys.path.append(os.getcwd())

@st.cache_resource
def load_pipeline():
    try:
        # Load all three components
        engineer = joblib.load('models/engineer.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        model = joblib.load('models/rf_airbnb_final.pkl')
        return engineer, preprocessor, model
    except ModuleNotFoundError as e:
        st.error(f"❌ Python couldn't find your custom code: {e}")
        st.error("Make sure you have uploaded the 'src' folder to your GitHub repository!")
        st.stop()
    except FileNotFoundError:
        st.error("❌ Model files not found. Check the 'models/' folder.")
        st.stop()

engineer, preprocessor, model = load_pipeline()

# --- 2. USER INTERFACE ---
st.title("Berlin Airbnb Price Predictor 🏠")
st.markdown("Enter the raw listing details below.")

# Dropdown Options (These should match the raw values in your dataset)
prop_options = [
    'Entire rental unit', 'Private room in rental unit', 'Entire condo', 'Private room in condo', 
    'Entire home', 'Private room in home', 'Entire loft', 'Private room in loft', 'Entire serviced apartment',
    'Room in boutique hotel', 'Room in hotel', 'Entire townhouse', 'Other'
]
room_options = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
neighborhood_options = ['Mitte', 'Pankow', 'Tempelhof - Schöneberg', 'Friedrichshain-Kreuzberg', 'Neukölln', 'Charlottenburg-Wilm.', 'Lichtenberg', 'Marzahn - Hellersdorf', 'Reinickendorf', 'Spandau', 'Steglitz - Zehlendorf', 'Treptow - Köpenick']

# Sidebar Inputs
st.sidebar.header("Listing Details")
prop_type = st.sidebar.selectbox("Property Type", prop_options)
room_type = st.sidebar.selectbox("Room Type", room_options)
neighborhood = st.sidebar.selectbox("Neighbourhood", neighborhood_options)

accommodates = st.sidebar.number_input("Accommodates", min_value=1, value=2)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, value=1)
beds = st.sidebar.number_input("Beds", min_value=1, value=1)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1.0, value=1.0)

# Optional/Advanced inputs (Adjust these based on what your Engineer expects)
with st.expander("Detailed Features"):
    min_nights = st.number_input("Minimum Nights", min_value=1, value=2)
    availability_365 = st.slider("Availability (days per year)", 0, 365, 100)
    amenities_text = st.text_area("Amenities (comma separated)", "Wifi, Kitchen, Heating")

# --- 3. PREDICTION ---
if st.button("Predict Price", type="primary"):
    
    # A. Build Raw DataFrame
    raw_data = pd.DataFrame({
        'property_type': [prop_type],
        'room_type': [room_type],
        'neighbourhood_cleansed': [neighborhood],
        'neighbourhood_group_cleansed': [neighborhood_group],
        'accommodates': [accommodates],
        'bedrooms': [bedrooms],
        'beds': [beds],
        'bathrooms_text': [bathrooms], 
        'minimum_nights': [min_nights],
        'availability_365': [availability_365],
        # Add default values for columns the model needs but user doesn't input
        'host_response_time': ['within an hour'],
        'host_is_superhost': ['f'],
        'number_of_reviews': [10],
        'amenities': [amenities_text] 
    })

    # B. Pipeline Execution
    try:
        # 1. Preprocessing
        processed_data = preprocessor.transform(raw_data)
        
        # 2. Preprocessing
        engineered_data = engineer.transform(processed_data)
        
        # 3. Prediction
        log_price = model.predict(engineered_data)[0]
        price = np.exp(log_price)
        
        st.subheader(f"Estimated Price: €{price:.2f}")
        
    except KeyError as e:
        st.error(f"❌ Metadata Error: Your pipeline expects a column that is missing from the input.")
        st.error(f"Missing Column: {e}")
        st.write("Debug - Your Input Data:", raw_data)
    except Exception as e:
        st.error(f"An error occurred: {e}")