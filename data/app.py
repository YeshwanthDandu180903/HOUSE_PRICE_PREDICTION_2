import streamlit as st # type: ignore
import joblib
import pandas as pd
import json

# Load the model and preprocessing objects
best_model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')
with open('encoded_columns.json', 'r') as f:
    encoded_columns = json.load(f)

# Streamlit app
st.title('House Price Prediction')

# Input fields
st.sidebar.header('Input Features')
house_type = st.sidebar.selectbox('House Type', ['A', 'B', 'C'])
no_of_rooms = st.sidebar.number_input('Number of Rooms', min_value=1, max_value=10, value=3)
location = st.sidebar.selectbox('Location', ['X', 'Y', 'Z'])
age = st.sidebar.number_input('Age of House (Years)', min_value=1, max_value=100, value=20)

# Create a DataFrame from the input
input_data = pd.DataFrame({
    'House_type': [house_type],
    'no_of_rooms': [no_of_rooms],
    'location': [location],
    'age': [age]
})

# Display the input data
st.write('Input Data:')
st.write(input_data)

# Preprocess the input data
def preprocess_input(input_data, imputer, scaler, encoded_columns):
    """
    Preprocess the input data for prediction.
    """
    # Ensure all encoded columns are present (add missing columns with 0s)
    for col in encoded_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Apply imputation (if used)
    input_data_imputed = pd.DataFrame(imputer.transform(input_data), columns=input_data.columns)
    
    # Apply scaling
    input_data_scaled = scaler.transform(input_data_imputed)
    
    return input_data_scaled

# Preprocess the input data
input_data_preprocessed = preprocess_input(input_data, imputer, scaler, encoded_columns)

# Make a prediction
if st.button('Predict'):
    prediction = best_model.predict(input_data_preprocessed)
    st.write(f'Predicted House Price: ${prediction[0]:,.2f}')