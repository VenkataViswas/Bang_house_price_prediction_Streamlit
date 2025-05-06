import streamlit as st

# Streamlit config must be first
st.set_page_config(
    page_title="Bangalore Home Price Prediction",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Now import other libraries
import json
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load model and data
@st.cache_resource
def load_saved_artifacts():
    with open("columns.json", 'r') as f:
        data_columns = json.load(f)['data_columns']
        locations = data_columns[3:]  # Location names start from index 3

    with open("banglore_home_prices_model.pickle", 'rb') as f:
        model = pickle.load(f)

    return data_columns, locations, model

# Prediction function
def get_estimated_price(location, sqft, bhk, bath, data_columns, model):
    try:
        loc_index = data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = bhk
    x[1] = sqft
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1

    prediction = model.predict([x])[0]
    return round(prediction, 2)

# Load model and data
data_columns, locations, model = load_saved_artifacts()

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3e8e41;
            transform: scale(1.02);
        }
        .stNumberInput, .stSelectbox, .stSlider {
            padding-bottom: 1rem;
        }
        .price-result {
            background-color: #e8f5e9;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .section {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .footer {
            text-align: center;
            padding: 1rem;
            color: #666;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #2E8B57; margin-bottom: 0;'>Bangalore Home Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; margin-top: 0;'>Get accurate price estimates for properties in Bangalore</p>", unsafe_allow_html=True)

# Main content
with st.container():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🏡 About This Tool")
        st.markdown("""
        This predictive tool helps you estimate property prices in Bangalore based on:
        - Location
        - Square footage
        - Number of bedrooms (BHK)
        - Number of bathrooms
        
        Our machine learning model analyzes historical data to provide reliable price estimates.
        """)
        
        st.markdown("### ℹ️ How to Use")
        st.markdown("""
        1. Select the property location
        2. Enter the total square footage
        3. Set the number of bedrooms
        4. Set the number of bathrooms
        5. Click 'Estimate Price' to see the result
        """)
    
    with col2:
        with st.form("price_prediction_form"):
            st.markdown("### 📝 Enter Property Details")
            
            location = st.selectbox(
                "Location",
                sorted(locations),
                help="Select the neighborhood where the property is located"
            )
            
            sqft = st.number_input(
                "Total Square Feet",
                min_value=300,
                max_value=10000,
                value=1000,
                step=50,
                help="Enter the total area of the property in square feet"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                bhk = st.slider(
                    "Bedrooms (BHK)",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Select number of bedrooms"
                )
            with col_b:
                bath = st.slider(
                    "Bathrooms",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Select number of bathrooms"
                )
            
            submitted = st.form_submit_button("Estimate Price", type="primary")
            
            if submitted:
                price = get_estimated_price(location, sqft, bhk, bath, data_columns, model)
                st.markdown(
                    f"""
                    <div class="price-result">
                        <h3 style='color:#2E8B57; margin-bottom: 0.5rem;'>Estimated Property Value</h3>
                        <h2 style='color:#1E90FF; margin-top: 0;'>₹ {price:,} Lakhs</h2>
                        <p style='color:#666; font-size: 0.9rem;'>Based on current market trends in {location}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Additional Information
with st.expander("📊 About the Model & Data"):
    st.markdown("""
    ### Model Details
    - **Algorithm Used**: Random Forest Regression
    - **Data Source**: Property listings from Bangalore (2017-2019)
    - **Features Considered**: Location, Square Footage, BHK, Bathrooms
    
    ### Accuracy Metrics
    - **R² Score**: 0.86
    - **Mean Absolute Error**: ₹8.5 Lakhs
    
    Note: These estimates are for reference only. Actual prices may vary based on market conditions and property-specific factors.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Developed with ❤️ by Venkata Viswas | Data Science Project</p>
    <p style="font-size: 0.8rem;">© 2023 Bangalore Real Estate Analytics</p>
</div>
""", unsafe_allow_html=True)
