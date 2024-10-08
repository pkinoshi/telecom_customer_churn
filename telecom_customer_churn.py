import numpy as np
import pandas as pd
import sklearn
from xgboost import XGBClassifier
import pickle
import streamlit as st


# Load the pre-trained model
with open('best_XGB_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set a modern app layout
st.set_page_config(page_title="Telecom Customer Churn Prediction", page_icon="📊", layout="wide")

# Apply custom CSS for a white theme with blue sliders
def apply_custom_css():
    st.markdown(
        """
        <style>
        /* Set global background and text colors */
        .reportview-container {
            background-color: white;  /* Change to white background */
            color: black;  /* Change text color to black */
        }
        .sidebar .sidebar-content {
            background-color: #f7f7f7;  /* Light gray sidebar background */
        }
        .main-header {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #007BFF;  /* Adjusted header color to blue */
        }
        .stButton button {
            background-color: #007BFF;  /* Blue button */
            color: white;
            border-radius: 8px;
            padding: 0.8rem 1.5rem;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #0056b3;  /* Darker blue on hover */
        }
        .stMarkdown h3 {
            color: #333;  /* Dark gray for subheadings */
            font-size: 22px;
        }
        .info-text {
            color: #666;  /* Medium gray for info text */
            font-size: 14px;
            margin-bottom: 10px;
        }
        .section-header {
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
            color: #007BFF;  /* Blue section headers */
        }
        /* Slider Styles */
        .stSlider > div:first-child {
            background: linear-gradient(90deg, #007BFF 0%, #00BFFF 100%);
        }
        .stSlider .slider-handle {
            background-color: #007BFF !important;  /* Blue slider handle */
            border-color: #0056b3 !important;  /* Darker blue border */
        }
        input, select {
            color: black !important;  /* Change input text color to black */
            background-color: #f0f0f0 !important;  /* Light gray for inputs */
            border: 1px solid #ccc !important;  /* Light gray border */
        }
        .stNumberInput input, .stTextInput input {
            background-color: #fff !important;  /* White input background */
            color: black !important;  /* Black text color */
            border-radius: 5px !important;
            border: 1px solid #ccc !important;  /* Light gray border */
        }
        .stSelectbox select, .stSelectbox {
            background-color: #fff !important;  /* White background */
            color: black !important;  /* Black text color */
        }
        .stTextArea textarea {
            background-color: #fff !important;  /* White background */
            color: black !important;  /* Black text color */
        }
        .stMarkdown div, p, span {
            color: #333 !important;  /* Dark gray for markdown text */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_css()

# Main Header
st.markdown("<div class='main-header'>Telecom Customer Churn Prediction</div>", unsafe_allow_html=True)

# Introduction
st.write("""
This application predicts the likelihood of a customer churning based on specific features such as usage, service plans, and account history. 
Fill in the required fields to analyze churn probability.
""")

# Sidebar for input features
st.sidebar.header("Call Details")
st.sidebar.markdown("Use the sliders and inputs below to define customer details:")

# Sliders for usage and charges with blue slider background and button
local_calls = st.sidebar.slider("Local Calls Made", min_value=0, max_value=2000, value=100)
local_mins = st.sidebar.slider("Total Local Call Minutes", min_value=0.0, max_value=5000.0, value=1000.0)
intl_mins = st.sidebar.slider("Total International Call Minutes", min_value=0.0, max_value=2000.0, value=100.0)
extra_intl_charges = st.sidebar.slider("Extra International Charges ($)", min_value=0.0, max_value=1000.0, value=50.0)
monthly_charge = st.sidebar.slider("Monthly Charge ($)", min_value=0.0, max_value=90.0, value=45.0)
total_charges = st.sidebar.slider("Total Charges ($)", min_value=0.0, max_value=8000.0, value=1000.0)

# Main Inputs
st.markdown("<div class='section-header'>Account Information</div>", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    account_length = st.number_input("Account Age (in months)", min_value=1, max_value=100, value=12)
    intl_calls = st.number_input("International Calls Made", min_value=0, value=10)
    customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=2)
    avg_monthly_gb_download = st.number_input("Average Monthly GB Download", min_value=0.0, value=10.0)

with col2:
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
    num_customers_in_group = st.number_input("Number of Customers in Group", min_value=1, value=1)
    extra_data_charges = st.number_input("Extra Data Charges ($)", min_value=0.0, value=5.0)

# Dropdowns for categorical inputs
st.markdown("<div class='section-header'>Plan and Service Options</div>", unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    intl_active = st.selectbox("Are International Calls Active?", options=["Yes", "No"], help="Specify if international calls are active.")
    intl_plan = st.selectbox("Is there a Premium International Plan?", options=["Yes", "No"], help="Does the customer have a premium international plan?")
    unlimited_data_plan = st.selectbox("Unlimited Data Plan?", options=["Yes", "No"], help="Is the customer on an unlimited data plan?")

with col4:
    group = st.selectbox("Does the Customer Belong to a Group?", options=["Yes", "No"], help="Is the customer part of a group plan?")
    device_protection_online_backup = st.selectbox("Device Protection & Online Backup?", options=["Yes", "No"], help="Does the customer have device protection?")
    contract_type = st.selectbox("Contract Type", options=['Month-to-Month', 'One Year', 'Two Year'])
    payment_method = st.selectbox("Payment Method", options=['Paper Check', 'Credit Card', 'Direct Debit'])

# Prepare input data for the model
input_data = pd.DataFrame({
    'account length (in months)': [account_length],
    'local calls': [local_calls],
    'local mins': [local_mins],
    'intl calls': [intl_calls],
    'intl mins': [intl_mins],
    'intl active': [1 if intl_active == "Yes" else 0],
    'intl plan': [1 if intl_plan == "Yes" else 0],
    'extra international charges': [extra_intl_charges],
    'customer service calls': [customer_service_calls],
    'avg monthly gb download': [avg_monthly_gb_download],
    'unlimited data plan': [1 if unlimited_data_plan == "Yes" else 0],
    'extra data charges': [extra_data_charges],
    'age': [age],
    'group': [1 if group == "Yes" else 0],
    'number of customers in group': [num_customers_in_group],
    'device protection & online backup': [1 if device_protection_online_backup == "Yes" else 0],
    'contract type': [0 if contract_type == "Month-to-Month" else (1 if contract_type == "One Year" else 2)],
    'payment method': [0 if payment_method == "Paper Check" else (1 if payment_method == "Credit Card" else 2)],
    'monthly charge': [monthly_charge],
    'total charges': [total_charges]
})

# Prediction button
if st.button("Predict Churn Probability"):
    prediction = model.predict_proba(input_data)[0][1]  # Get churn probability
    st.success(f"The customer is likely to churn with a probability of: **{prediction:.2%}**")

# Explanation section
st.markdown("---")
st.markdown("""
The result shown above reflects the likelihood of the customer leaving the service based on the current data. 
Use this information to take proactive steps and reduce churn rates.
""")
