import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the Assets ---
@st.cache_resource 
def load_assets():
    model = joblib.load('xgboost_credit_model.pkl')
    scaler = joblib.load('credit_scaler.pkl')
    cols = joblib.load('model_columns.pkl')
    return model, scaler, cols

model, scaler, cols = load_assets()

# --- 2. Custom CSS for Green Accents ---
st.markdown("""
    <style>
    /* Injecting the green accent for headers to contrast the Navy/Purple */
    h1, h2, h3 { color: #00FF7F !important; }
    hr { border-color: #00FF7F; }
    </style>
""", unsafe_allow_html=True)

# --- 3. The App UI ---
st.title("Credit Risk AI Prediction Engine")
st.markdown("Enter applicant details below to calculate default probability.")

# Sidebar for Inputs (Strict Number Inputs)
st.sidebar.header("Applicant Profile")
int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.0, step=0.1)
dti = st.sidebar.number_input("Debt-to-Income (DTI)", min_value=0.0, max_value=50.0, value=20.0, step=0.5)
annual_inc = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=1000)
loan_amnt = st.sidebar.number_input("Loan Amount Requested ($)", min_value=1000, max_value=40000, value=10000, step=500)
term = st.sidebar.selectbox("Loan Term", ["36 months", "60 months"])

# --- 4. Data Processing ---
if st.button("Calculate Risk"):
    with st.spinner("Analyzing profile..."):
        
        # Create a baseline applicant
        input_data = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
        
        # Overwrite with specific user inputs
        input_data['int_rate'] = int_rate
        input_data['dti'] = dti
        input_data['annual_inc'] = annual_inc
        input_data['loan_amnt'] = loan_amnt
        
        # Handle the encoded Term column
        if term == "60 months" and 'term_ 60 months' in cols:
            input_data['term_ 60 months'] = 1
            
        # Scale the data using the fitted scaler
        scaled_data = scaler.transform(input_data.values)
        
        # --- 5. Make the Prediction ---
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1] 
        
        # --- 6. Display Results ---
        st.divider()
        if prediction == 1:
            st.error(f" **HIGH RISK: REJECT**")
            st.markdown(f"**Probability of Default:** {probability * 100:.2f}%")
        else:
            st.success(f" **LOW RISK: APPROVE**")
            st.markdown(f"**Probability of Default:** {probability * 100:.2f}%")