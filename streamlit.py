import streamlit as st
import json
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# Load model and data 
def load_saved_artifacts():
    with open("./columns.json", 'r') as f:
        data_columns = json.load(f)['data_columns']
        locations = data_columns[3:]

    with open("./banglore_home_prices_model.pickle", 'rb') as f:
        model = pickle.load(f)

    return data_columns, locations, model

# Prediction function
def get_estimated_price(location, sqft, bhk, bath, data_columns, model):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = bhk
    x[1] = sqft
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1
    return round(model.predict([x])[0], 2)

# Load model and data
data_columns, locations, model = load_saved_artifacts()

# Streamlit UI
st.title("🏠 Bangalore Home Price Prediction")

location = st.selectbox("Select Location", sorted(locations))
sqft = st.number_input("Enter Total Square Feet Area", value=1000)
bhk = st.slider("Select Number of Bedrooms (BHK)", 1, 10, 3)
bath = st.slider("Select Number of Bathrooms", 1, 10, 2)

if st.button("Estimate Price"):
    price = get_estimated_price(location, sqft, bhk, bath, data_columns, model)
    st.success(f"Estimated Price: ₹ {price} Lakhs")
