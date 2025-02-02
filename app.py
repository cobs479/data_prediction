import pandas as pd
import streamlit as st
from algorithm_random_forest import predict_random_forest
from algorithm_xgboost import predict_xgboost

# Title of the app
st.title("Weather Prediction")

# Radio button
selected_algorithm = st.radio("Select a Algorithm:", [
    "Random Forest", "XGBoost"], key="radio_algorithm")

# Dropdown (selectbox) to choose fields
columns = ["Please select field", "PM10",  "PM25", "SO2", "NO2", "O3",
           "CO", "Humidity", "Temperature", "Wind Direction", "Wind Speed", "Global Radiation", "Usage"]
selected_field = st.selectbox(
    "Select field to predict:", columns, key="dropdown_field")

# Date picker to select the date
selected_date = st.date_input(
    "Select the date for prediction", pd.to_datetime("today"))

predict_btn = st.button("Predict")
if predict_btn:
    # Validation: Check if the user has selected a valid column (not the default)
    if selected_field == "Please select field":
        st.error("Please select a field to proceed with prediction.")
    # Validation: Check if a date has been selected
    elif selected_date is None:
        st.error("Please select a valid date to proceed with prediction.")
    # Validation: Check if an algorithm is selected
    elif not selected_algorithm:
        st.error("Please select an algorithm to proceed with prediction.")
    else:
        st.success(f"Prediction started with Algorithm: {selected_algorithm}, Field: {
                   selected_field}, and Date: {selected_date}")
        # Call the function for prediction
        selected_field = selected_field.replace(" ", "")
        print("Field : ", selected_field)
        if selected_algorithm == 'Random Forest':
            predict_random_forest(selected_field, selected_date)
        elif selected_algorithm == 'XGBoost':
            predict_xgboost(selected_field, selected_date)
