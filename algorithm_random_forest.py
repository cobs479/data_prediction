import pandas as pd
import joblib
import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def predict_random_forest(field, start_date, end_date, location_select):
    train_file_path = 'data/weather_2021.csv'
    predict_file_path = f'data/weather_{end_date.year}.csv'

    # Load training data
    X = pd.read_csv(train_file_path)

    # Check if the field exists
    if field not in X.columns:
        raise ValueError(f"Field '{field}' not found in the dataset!")

    # Drop rows where target field is missing
    X.dropna(axis=0, subset=[field], inplace=True)
    y = X[field]
    X.drop([field], axis=1, inplace=True)

    # Split the dataset
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Identify categorical and numerical columns
    categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 20 and
                        X_train_full[cname].dtype == "object"]
    numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    cols = categorical_cols + numerical_cols
    X_train = X_train_full[cols].copy()
    X_valid = X_valid_full[cols].copy()

    # Define model path
    model_save_path = f'saved_model/RF_{field}.joblib'

    # Define Preprocessor
    numerical_transformer = SimpleImputer(strategy='most_frequent')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define Model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Load or Train Model
    if os.path.exists(model_save_path):
        print("Loading pre-trained model...")
        pipeline = joblib.load(model_save_path)
    else:
        print("Training new model...")
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_save_path)
        print("Model saved.")

    # Handle Future Predictions
    try:
        X_predict = pd.read_csv(predict_file_path)
        st.success(f"Loaded existing prediction file: {predict_file_path}")
    except FileNotFoundError:
        st.warning(f"Prediction file for {end_date.year} not found. Generating future dataset...")
        X_predict = generate_future_data(X, start_date, end_date, location_select)
    
    if X_predict is None or X_predict.empty:
        st.error("No valid data available for prediction. Please check your input parameters.")
        return

    if X_predict is not None:
        preds = pipeline.predict(X_predict)

        # Ensure the target field exists in X_predict before calculating error
        if field in X_predict:
            actual_values = X_predict[field]
        else:
            actual_values = np.full_like(preds, y.mean())  # Create an array of same shape as preds
        
        # Now, calculate error metrics
        mae_score = mean_absolute_error(preds, actual_values)
        mse_score = mean_squared_error(preds, actual_values)
        rmse_score = np.sqrt(mse_score)

        print('MAE:', mae_score)
        print('MSE:', mse_score)
        print('RMSE:', rmse_score)

        display_graph(X_predict, preds, start_date, end_date, location_select)
        display_table(X_predict, preds, start_date, end_date, location_select)

        st.success("Prediction completed successfully!")
    else:
        st.error("No valid data available for prediction.")


def generate_future_data(X, start_date, end_date, location_select):
    """Generate synthetic data for future predictions."""
    
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Ensure Year, Month, and Day are extracted correctly
    future_data = pd.DataFrame({
        'Year': future_dates.year,
        'Month': future_dates.month,
        'Day': future_dates.day
    })

    # Generate hourly values
    future_data['Hour'] = np.tile(range(0, 2400, 100), len(future_data) // 24 + 1)[:len(future_data)]

    # Ensure location mapping exists
    location_mapping = {"Batu Muda": 1, "Petaling Jaya": 2, "Cheras": 3}
    future_data['LocationInNum'] = location_mapping.get(location_select, 1)

    # Convert to integers
    future_data[['Year', 'Month', 'Day', 'Hour']] = future_data[['Year', 'Month', 'Day', 'Hour']].astype(int)

    # Ensure DateTime column is created
    try:
        future_data['DateTime'] = pd.to_datetime(future_data[['Year', 'Month', 'Day']])
    except Exception as e:
        st.error(f"Error creating DateTime column: {e}")

    return future_data


def display_table(X_predict, preds, start_date, end_date, location_select):
    """Display predictions in a table."""
    
    # Ensure required columns exist
    required_cols = ['Year', 'Month', 'Day', 'Hour']
    for col in required_cols:
        if col not in X_predict.columns:
            st.error(f"Missing column '{col}' in X_predict.")
            return

    # Convert date components to integers
    X_predict[required_cols] = X_predict[required_cols].fillna(0).astype(int)

    # Convert Hour to formatted time (HH:MM)
    X_predict['Hour'] = X_predict['Hour'].astype(str).str.zfill(4)
    X_predict['Formatted Hour'] = X_predict['Hour'].str[:2] + ":" + X_predict['Hour'].str[2:]

    # Convert date columns into a full datetime format
    try:
        X_predict['DateTime'] = pd.to_datetime(
            X_predict[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1) +
            ' ' + X_predict['Formatted Hour']
        )
    except Exception as e:
        st.error(f"Error converting DateTime: {e}")
        return

    # Mapping numerical locations to readable names
    location_mapping = {1: "Batu Muda", 2: "Petaling Jaya", 3: "Cheras"}
    X_predict['Location'] = X_predict['LocationInNum'].map(location_mapping)

    # Create DataFrame for display
    results_df = pd.DataFrame({
        'Date-Time': X_predict['DateTime'],
        'Location': X_predict['Location'],
        'Predicted Value': preds
    })

    # Convert start_date and end_date to datetime format
    start_datetime = pd.to_datetime(start_date.strftime('%Y-%m-%d') + ' 00:00')
    end_datetime = pd.to_datetime(end_date.strftime('%Y-%m-%d') + ' 23:00')

    # Filter data within the date range
    results_df = results_df[
        (results_df['Date-Time'] >= start_datetime) &
        (results_df['Date-Time'] <= end_datetime) &
        (results_df['Location'] == location_select)
    ]

    if results_df.empty:
        st.error(f"No predictions found for {location_select} from {start_date} to {end_date}")
    else:
        st.dataframe(results_df, hide_index=True)


def display_graph(X_predict, preds, start_date, end_date, location_select):
    """Plot prediction results."""

    # Ensure the required columns exist
    if not all(col in X_predict.columns for col in ['Year', 'Month', 'Day']):
        st.error("Missing date-related columns in prediction data. Cannot plot graph.")
        return

    # Convert Year, Month, Day to integers and drop NaNs
    X_predict = X_predict.dropna(subset=['Year', 'Month', 'Day']).copy()
    X_predict[['Year', 'Month', 'Day']] = X_predict[['Year', 'Month', 'Day']].astype(int)

    # Ensure DateTime column is created
    if 'DateTime' not in X_predict.columns:
        X_predict['DateTime'] = pd.to_datetime(X_predict[['Year', 'Month', 'Day']], errors='coerce')

    # Drop NaNs after conversion
    X_predict = X_predict.dropna(subset=['DateTime'])

    # Convert start_date and end_date to pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Ensure location mapping is correct
    location_mapping = {1: "Batu Muda", 2: "Petaling Jaya", 3: "Cheras"}
    X_predict['Location'] = X_predict['LocationInNum'].map(location_mapping)

    # Filter based on selected date range and location
    mask = (X_predict['DateTime'] >= start_date) & (X_predict['DateTime'] <= end_date) & (X_predict['Location'] == location_select)
    filtered_data = X_predict[mask]
    filtered_preds = preds[mask]

    if filtered_data.empty:
        st.error(f"No predictions found for {location_select} from {start_date.date()} to {end_date.date()}")
        return

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data['DateTime'], filtered_preds, marker='o', label=location_select)
    plt.title(f'Predictions from {start_date.date()} to {end_date.date()}')
    plt.xlabel('Date')
    plt.ylabel('Predicted Value')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)
