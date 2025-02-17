import pandas as pd
import streamlit as st
from algorithm_random_forest import predict_random_forest
from algorithm_xgboost import predict_xgboost

# Title of the app
st.title("Weather Prediction")

# Initialize session state for form submission
if "prediction_started" not in st.session_state:
    st.session_state.prediction_started = False

# Create a form for input selection
with st.form("prediction_form"):
    # Radio button for algorithm selection
    selected_algorithm = st.radio("Select an Algorithm:", [
        "Random Forest", "XGBoost"], key="radio_algorithm")

    # Dropdown (selectbox) to choose fields
    columns = ["Please select field", "PM10", "PM25", "SO2", "NO2", "O3",
               "CO", "Humidity", "Temperature", "Wind Direction", "Wind Speed", "Global Radiation", "Usage"]
    selected_field = st.selectbox(
        "Select field to predict:", columns, key="dropdown_field")

    # Date pickers for date range selection
    selected_start_date = st.date_input(
        "Select 'From' Date", pd.to_datetime("today"), key="start_date")
    selected_end_date = st.date_input(
        "Select 'To' Date", pd.to_datetime("today"), key="end_date")

    # Location selector
    locations = ["Please select location", "Batu Muda", "Petaling Jaya", "Cheras"]
    selected_location = st.selectbox(
        "Select location:", locations, key="dropdown_field_2")

    # Submit button inside the form
    submitted = st.form_submit_button("Predict")

# Process the form submission
if submitted:
    # Validation: Check if the user has selected a valid column (not the default)
    if selected_field == "Please select field":
        st.error("Please select a field to proceed with prediction.")
    # Validation: Ensure start_date is before or equal to end_date
    elif selected_start_date > selected_end_date:
        st.error("The 'From' date cannot be later than the 'To' date.")
    # Validation: Ensure start_date and end_date are in the same year
    elif selected_start_date.year != selected_end_date.year:
        st.error("Start Date and End Date must be in the same year.")
    # Validation: Check if an algorithm is selected
    elif not selected_algorithm:
        st.error("Please select an algorithm to proceed with prediction.")
    # Validation: Check if the user has selected a valid column (not the default)
    if selected_field_2 == "Please select location":
        st.error("Please select a location to proceed with prediction.")
    else:
        # **Set session state to prevent reset**
        st.session_state.prediction_started = True
        st.session_state.selected_field = selected_field
        st.session_state.selected_algorithm = selected_algorithm
        st.session_state.selected_start_date = selected_start_date
        st.session_state.selected_end_date = selected_end_date
        st.session_state.selected_field_2 = selected_field_2

        st.success(f"Prediction started with Algorithm: {selected_algorithm}, Field: {
            selected_field}, From Date: {selected_start_date} and End Date: {selected_end_date}, Location: {selected_field_2}")

        # Call the prediction function based on the selected algorithm
        selected_field = selected_field.replace(" ", "")
        selected_field_2 = selected_field_2.replace(" ", "")
        print("Field:", selected_field)
        print("Start Date:", selected_start_date)
        print("End Date:", selected_end_date)
        print("Location:", selected_field_2)

        if selected_algorithm == 'Random Forest':
            predict_random_forest(
                selected_field, selected_start_date, selected_end_date, selected_field_2)
        elif selected_algorithm == 'XGBoost':
            predict_xgboost(selected_field, selected_start_date, selected_end_date, selected_field_2)

# Display results if prediction was started
if st.session_state.prediction_started:
    print(f"### Showing results for:")
    print(f"- **Algorithm:** {st.session_state.selected_algorithm}")
    print(f"- **Field:** {st.session_state.selected_field}")
    print(f"- **Date Range:** {st.session_state.selected_start_date} to {
        st.session_state.selected_end_date}")
