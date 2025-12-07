import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# --- 1. PATH CONFIGURATION---
current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

src_dir = os.path.join(parent_dir, 'src')

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.data_preprocessing import AirbnbPreprocessor
from src.feature_engineering import AirbnbEngineer
from src.modelling import AirbnbModel

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="Berlin Airbnb Price Predictor", page_icon="🏠")

st.title("Berlin Airbnb Price Predictor 🏠")
st.markdown("Enter the raw listing details below.")

# --- 3. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    models_dir = os.path.join(parent_dir, 'models')
    
    # Paths to files
    model_path = os.path.join(models_dir, 'rf_airbnb_final.pkl')
    engineer_path = os.path.join(models_dir, 'engineer.pkl')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')

    # Verify existence
    if not os.path.exists(model_path):
        st.error(f"❌ Error: Model file not found at: {model_path}")
        st.stop()
        
    # Load files
    try:
        model = joblib.load(model_path)
        engineer = joblib.load(engineer_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, engineer, preprocessor
    except Exception as e:
        st.error(f"❌ Error loading files: {e}")
        st.stop()

model, engineer_pipeline, preprocessor = load_resources()

# --- 4. CONSTANTS ---
PROPERTY_TYPES = ['Entire rental unit', 'Entire loft', 'Entire condo','Private room in rental unit', 'Private room in condo',
       'Entire guest suite', 'Entire home', 'Private room in loft','Entire townhouse', 'Private room in home','Private room in hostel', 
       'Shared room in hostel', 'Entire place', 'Entire guesthouse', 'Shared room in rental unit', 'Private room','Entire bungalow', 'Boat', 
       'Room in aparthotel', 'Entire cottage','Private room in townhouse', 'Entire villa','Entire serviced apartment', 'Room in serviced apartment',
       'Tiny home', 'Private room in villa', 'Private room in tipi','Room in boutique hotel', 'Houseboat', 'Earthen home',
       'Private room in guest suite', 'Private room in bed and breakfast','Room in hostel', 'Private room in guesthouse','Private room in cave', 
       'Room in hotel', 'Entire chalet','Private room in bungalow', 'Room in bed and breakfast','Private room in pension', 'Entire vacation home', 'Island',
       'Private room in serviced apartment', 'Entire cabin','Private room in boat', 'Private room in casa particular',
       'Private room in houseboat', 'Private room in vacation home','Cave', 'Shared room in serviced apartment', 'Campsite','Private room in castle', 
       'Private room in shipping container','Shared room in hotel', 'Shared room in bed and breakfast', 'Dome','Casa particular', 'Private room in chalet',
       'Shared room in condo', 'Camper/RV', 'Private room in cottage','Shared room in guesthouse', 'Shepherd’s hut', 'Hut', 'Castle']

ROOM_TYPES = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']

NEIGHBOURHOODS = ['Prenzlauer Berg Südwest', 'Prenzlauer Berg Nordwest', 'Reuterstraße', 'Brunnenstr. Süd', 'Tempelhofer Vorstadt',
       'Helmholtzplatz', 'Düsseldorfer Straße', 'Schöneberg-Nord', 'Regierungsviertel', 'südliche Luisenstadt',
       'Frankfurter Allee Süd FK', 'Neue Kantstraße','Prenzlauer Berg Süd', 'Brunnenstr. Nord', 'Prenzlauer Berg Nord',
       'Schmargendorf', 'Alexanderplatz','Blankenfelde/Niederschönhausen', 'Frankfurter Allee Nord',
       'Schöneberg-Süd', 'Südliche Friedrichstadt', 'Wiesbadener Straße','Rixdorf', 'Blankenburg/Heinersdorf/Märchenland', 'Pankow Zentrum',
       'Prenzlauer Berg Ost', 'Buckow Nord', 'Karlshorst', 'Rudow', 'nördliche Luisenstadt', 'Otto-Suhr-Allee', 'Mierendorffplatz',
       'Wedding Zentrum', 'Moabit West', 'Altglienicke', 'Moabit Ost', 'Lichtenrade', 'Westend', 'Zehlendorf  Südwest', 'Johannisthal',
       'Marienfelde', 'Friedenau', 'Tiergarten Süd', 'Heerstraße Nord','Karl-Marx-Allee-Süd', 'Baumschulenweg', 'Halensee','Neuköllner Mitte/Zentrum',
       'Schillerpromenade', 'Tempelhof','Rahnsdorf/Hessenwinkel', 'Ost 2', 'Parkviertel','Volkspark Wilmersdorf', 'Schönholz/Wilhelmsruh/Rosenthal',
       'Alt-Hohenschönhausen Nord', 'Albrechtstr.', 'Gropiusstadt','Ost 1', 'Britz', 'Oberschöneweide', 'Heerstrasse', 'Nord 1',
       'Adlershof', 'Mahlsdorf', 'Friedrichshagen', 'Barstraße',
       'Schloß Charlottenburg', 'Osloer Straße', 'Karl-Marx-Allee-Nord','Köllnische Heide', 'Pankow Süd', 'Zehlendorf  Nord',
       'Kurfürstendamm', 'Alt  Treptow', 'Rummelsburger Bucht','Schloßstr.', 'Neu Lichtenberg', 'Köpenick-Nord',
       'Frankfurter Allee Süd', 'MV 2', 'Teltower Damm', 'Biesdorf','Fennpfuhl', 'Kantstraße',
       'Schmöckwitz/Karolinenhof/Rauchfangswerder', 'Weißensee','Alt-Lichtenberg', 'Mariendorf', 'Charlottenburg Nord', 'Buch',
       'Buckow', 'Drakestr.', 'Grunewald', 'Nord 2', 'Forst Grunewald','Karow', 'Neu-Hohenschönhausen Nord', 'Buchholz',
       'Niederschöneweide', 'Falkenhagener Feld', 'West 5','Hellersdorf-Süd', 'Spandau Mitte', 'Kaulsdorf', 'Plänterwald',
       'Bohnsdorf', 'West 3', 'Ostpreußendamm', 'West 2', 'Weißensee Ost','West 4', 'Altstadt-Kietz', 'Köpenick-Süd', 'Lankwitz',
       'Friedrichsfelde Süd', 'Haselhorst', 'Alt-Hohenschönhausen Süd', 'Brunsbütteler Damm', 'Grünau', 'Friedrichsfelde Nord', 'West 1',
       'MV 1', 'Dammvorstadt', 'Müggelheim', 'Gatow / Kladow','Hakenfelde', 'Marzahn-Süd', 'Malchow, Wartenberg und Falkenberg',
       'Wilhelmstadt', 'Siemensstadt', 'Allende-Viertel', 'Marzahn-Mitte','Kölln. Vorstadt/Spindlersf.',
        'Hellersdorf-Nord','Neu-Hohenschönhausen Süd', 'Marzahn-Nord', 'Hellersdorf-Ost']

NEIGHBOURHOOD_GROUPS = ['Pankow', 'Neukölln', 'Mitte', 'Friedrichshain-Kreuzberg',
                   'Charlottenburg-Wilm.', 'Tempelhof - Schöneberg', 'Lichtenberg',
                   'Treptow - Köpenick', 'Steglitz - Zehlendorf', 'Spandau',
                   'Reinickendorf', 'Marzahn - Hellersdorf']

RESPONSE_TIMES = ['within a day', 'a few days or more', 'within an hour', 'within a few hours']

with st.form("prediction_form"):

    # 1. LOCATION & PROPERTY SPECS
    st.header("📍 Location & Property")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        neighbourhood_group = st.selectbox("Neighbourhood Group", NEIGHBOURHOOD_GROUPS)
        neighbourhood = st.selectbox("Neighbourhood", NEIGHBOURHOODS)
        
    with col2:
        property_type = st.selectbox("Property Type", PROPERTY_TYPES)
        room_type = st.selectbox("Room Type", ROOM_TYPES)
        
    with col3:
        accommodates = st.number_input("Accommodates", 1, 16, 2)
        bathrooms_text = st.text_input("Bathroom Text", "1 bath")
        
    with col4:
        bedrooms = st.number_input("Bedrooms", 0.0, 10.0, 1.0)
        beds = st.number_input("Beds", 0.0, 20.0, 1.0)

    st.markdown("---")

    # 2. HOST METRICS
    st.header("👤 Host Details")
    
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        host_response_time = st.selectbox("Response Time", RESPONSE_TIMES)
        host_response_rate = st.slider("Response Rate (%)", 0.0, 100.0, 100.0)
        host_acceptance_rate = st.slider("Acceptance Rate (%)", 0.0, 100.0, 90.0)
    with col_h2:
        host_listings_count = st.number_input("Total Host Listings", 1, 5000, 1)
    with col_h3:
        st.caption("Listings Breakdown")
        calc_entire = st.number_input("Entire Homes", 0, 500, 1)
        calc_private = st.number_input("Private Rooms", 0, 500, 0)
        calc_shared = st.number_input("Shared Rooms", 0, 500, 0)
        calculated_host_listings_count = calc_entire + calc_private + calc_shared

    st.markdown("---")

    # 3. AVAILABILITY & BOOKING
    st.header("📅 Booking Rules & Availability")
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        minimum_nights = st.number_input("Minimum Nights", 1, 365, 2)
        maximum_nights = st.number_input("Maximum Nights", 1, 1125, 30)
        instant_bookable = st.checkbox("Instant Bookable", value=False)
        has_availability = st.checkbox("Has Availability", value=True)

    with col_a2:
        st.caption("Availability (Days)")
        avail_30 = st.number_input("Next 30 Days", 0, 30, 15)
        avail_60 = st.number_input("Next 60 Days", 0, 60, 30)
        avail_90 = st.number_input("Next 90 Days", 0, 90, 45)
        avail_365 = st.number_input("Next 365 Days", 0, 365, 180)

    st.markdown("---")

    # 4. REVIEWS & PERFORMANCE
    st.header("⭐ Reviews & Finances")
    
    with st.expander("Detailed Review Scores (Click to Expand)", expanded=False):
        c_r1, c_r2, c_r3 = st.columns(3)
        with c_r1:
            score_rating = st.slider("Overall Rating", 0.0, 5.0, 4.8)
            score_accuracy = st.slider("Accuracy", 0.0, 5.0, 4.8)
        with c_r2:
            score_cleanliness = st.slider("Cleanliness", 0.0, 5.0, 4.8)
            score_checkin = st.slider("Check-in", 0.0, 5.0, 4.9)
        with c_r3:
            score_comm = st.slider("Communication", 0.0, 5.0, 4.9)
            score_loc = st.slider("Location", 0.0, 5.0, 4.8)
            score_val = st.slider("Value", 0.0, 5.0, 4.7)

    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        number_of_reviews = st.number_input("Total Reviews", 0, 2000, 50)
        reviews_per_month = st.number_input("Reviews/Month", 0.0, 50.0, 1.5)
        
    with col_p2:
        reviews_ltm = st.number_input("Reviews (Last 12m)", 0, 500, 10)
        reviews_l30d = st.number_input("Reviews (Last 30d)", 0, 50, 1)

    with col_p3:
        est_occupancy = st.number_input("Est. Occupancy (Days/Yr)", 0, 365, 200)
        est_revenue = st.number_input("Est. Revenue (Yearly)", 0.0, 500000.0, 25000.0)

    # 5. AMENITIES (Special Handling)
    st.markdown("---")
    st.header("✨ Amenities")
    amenities_input = st.text_area("Enter Amenities", "['Wifi', 'Kitchen', 'Heating']")    
    
    # 6. SUBMIT
    submit_button = st.form_submit_button("Predict Price")

# --- PREDICTION LOGIC ---
if submit_button:
    # 1. Create Dataframe
    input_data = pd.DataFrame({
        'host_response_time': [host_response_time],
        'host_response_rate': [f"{int(host_response_rate)}%"],
        'host_acceptance_rate': [f"{int(host_acceptance_rate)}%"],
        'host_listings_count': [host_listings_count],
        'neighbourhood_cleansed': [neighbourhood],
        'neighbourhood_group_cleansed': [neighbourhood_group],
        'property_type': [property_type],
        'room_type': [room_type],
        'accommodates': [accommodates],
        'bathrooms_text': [bathrooms_text],
        'bedrooms': [bedrooms],
        'beds': [beds],
        'amenities': [amenities_input],
        'minimum_nights': [minimum_nights],
        'maximum_nights': [maximum_nights],
        'has_availability': ['t' if has_availability else 'f'],
        'availability_30': [avail_30],
        'availability_60': [avail_60],
        'availability_90': [avail_90],
        'availability_365': [avail_365],
        'number_of_reviews': [number_of_reviews],
        'number_of_reviews_ltm': [reviews_ltm],
        'number_of_reviews_l30d': [reviews_l30d],
        'estimated_occupancy_l365d': [est_occupancy],
        'estimated_revenue_l365d': [est_revenue],
        'review_scores_rating': [score_rating],
        'review_scores_accuracy': [score_accuracy],
        'review_scores_cleanliness': [score_cleanliness],
        'review_scores_checkin': [score_checkin],
        'review_scores_communication': [score_comm],
        'review_scores_location': [score_loc],
        'review_scores_value': [score_val],
        'instant_bookable': ['t' if instant_bookable else 'f'],
        'calculated_host_listings_count': [calculated_host_listings_count],
        'calculated_host_listings_count_entire_homes': [calc_entire],
        'calculated_host_listings_count_private_rooms': [calc_private],
        'calculated_host_listings_count_shared_rooms': [calc_shared],
        'reviews_per_month': [reviews_per_month]
    })
    
    input_data['price'] = 0 

    st.info("Processing data...")

    try:
        # 2. Preprocess (Clean)
        processed_data = preprocessor.transform(input_data)
        
        # 3. Engineer (Transform)
        if hasattr(engineer_pipeline, 'transform_inference'):
             engineered_data = engineer_pipeline.transform_inference(processed_data)
        else:
             engineer_pipeline.df = processed_data.copy()
             engineer_pipeline.transform()
             engineered_data = engineer_pipeline.df
        
        # 4. Align Columns (Fix Shape Mismatch)
        # We need the model's feature names to align the columns
        if hasattr(model, "feature_names_in_"):
            model_cols = model.feature_names_in_
            # Reindex ensures the input has exactly the columns the model expects
            X_final = engineered_data.reindex(columns=model_cols, fill_value=0)
        else:
            st.warning("Model does not have feature_names_in_. Passing data directly.")
            X_final = engineered_data

        # 5. Predict
        log_price = model.predict(X_final)[0]
        price = np.expm1(log_price) # Inverse of log1p

        st.success(f"💰 Estimated Price: €{price:,.2f}")
        
        # Debug view
        with st.expander("See processed data"):
            st.dataframe(X_final)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

