import pandas as pd
import joblib  # For saving and loading model
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


def predict_xgboost(field, start_date, end_date):
    train_file_path = 'data/weather_2021.csv'
    predict_file_path = f'data/weather_{end_date.year}.csv'

    # Check if the prediction file exists
    if not os.path.exists(predict_file_path):
        st.error(f"Prediction file for year {
            end_date.year} not found: {predict_file_path}")
        return

    X = pd.read_csv(train_file_path)
    X_predict = pd.read_csv(predict_file_path)

    # Check if the field is valid
    if field not in X.columns:
        raise ValueError(f"Field '{field}' not found in the data columns!")

    # Change target
    X.dropna(axis=0, subset=[field], inplace=True)
    y = X[field]
    X.drop([field], axis=1, inplace=True)
    # Only if need to validate the predicted data
    y_predict = X_predict[field]
    X_predict.drop([field], axis=1, inplace=True)

    X_train_full, X_valid_full, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0)

    # Change "cardinality" number
    categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 20 and
                        X_train_full[cname].dtype == "object"]

    numerical_cols = [
        cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    cols = categorical_cols + numerical_cols
    X_train = X_train_full[cols].copy()
    X_valid = X_valid_full[cols].copy()
    X_predict_full = X_predict[cols].copy()
    X_predict = X_predict[cols].copy()

    # Define the path for the saved model
    model_save_path = 'saved_model/XGB_' + field + '.joblib'

    # Change preprocessor settings / strategies
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

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    X_train = pipeline.fit_transform(X_train)
    X_valid = pipeline.transform(X_valid)
    X_predict = pipeline.transform(X_predict)

    # Check if there is an existing saved model
    if os.path.exists(model_save_path):
        print("Loading model from previous save...")
        model = joblib.load(model_save_path)
    else:
        print("Building and training a new model...")
        # Change Define model settings
        model = XGBRegressor(random_state=0, n_estimators=1000,
                             learning_rate=0.05, early_stopping_rounds=5)
        # Train the model
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  verbose=False)

        # Save the model to file
        joblib.dump(model, model_save_path)
        print("Model saved successfully.")

    # Do predictions and calculate error
    # Change X set
    preds = model.predict(X_predict)  # Use X_valid for validation if needed
    print(preds)

    # Calculate mean absolute error
    score = mean_absolute_error(preds, y_predict)  # Use y_valid for validation
    print('MAE:', score)

    # Calculate mean squared error (MSE)
    mse_score = mean_squared_error(preds, y_predict)
    print('MSE:', mse_score)

    # Calculate root mean squared error (RMSE)
    rmse_score = np.sqrt(mse_score)
    print('RMSE:', rmse_score)

    display_graph(X_predict_full, preds, start_date, end_date)
    display_table(X_predict_full, preds, start_date, end_date)


def display_table(X_predict, preds, start_date, end_date):
    # Ensure Hour column is properly formatted as HH:MM
    X_predict['Hour'] = X_predict['Hour'].astype(str).str.zfill(4)
    X_predict['Formatted Hour'] = X_predict['Hour'].str[:2] + \
        ":" + X_predict['Hour'].str[2:]

    # Convert date columns into a full datetime format (YYYY-MM-DD HH:MM)
    X_predict['DateTime'] = pd.to_datetime(
        X_predict[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1) +
        ' ' + X_predict['Formatted Hour']
    )

    # Mapping numerical locations to readable names
    location_mapping = {
        1: "Batu Muda",
        2: "Petaling Jaya",
        3: "Cheras"
    }
    X_predict['Location'] = X_predict['LocationInNum'].map(location_mapping)

    # Create a DataFrame for display
    results_df = pd.DataFrame({
        'Date-Time': X_predict['DateTime'],
        'Location': X_predict['Location'],
        'Predicted Value': preds
    })

    # Convert start_date and end_date to datetime format
    start_datetime = pd.to_datetime(start_date.strftime('%Y-%m-%d') + ' 00:00')
    end_datetime = pd.to_datetime(end_date.strftime('%Y-%m-%d') + ' 23:00')

    # Filter data within the date range
    results_df = results_df[(results_df['Date-Time'] >= start_datetime) &
                            (results_df['Date-Time'] <= end_datetime)]

    # Display filtered table
    if results_df.empty:
        st.error(f"No predictions found for selected locations from {
                 start_date} to {end_date}")
    else:
        st.write(f"### Predictions from {start_date} to {end_date}")
        st.dataframe(results_df, hide_index=True)


def display_graph(X_predict, preds, start_date, end_date):
    # Convert date inputs to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Convert date columns into full datetime format
    X_predict['DateTime'] = pd.to_datetime(X_predict[['Year', 'Month', 'Day']])

    # **Mapping numerical locations to readable names (Same as display_table)**
    location_mapping = {
        1: "Batu Muda",
        2: "Petaling Jaya",
        3: "Cheras"
    }
    X_predict['Location'] = X_predict['LocationInNum'].map(location_mapping)

    # Convert selected date range into formatted strings for display
    start_date_str = start_date.strftime('%d/%m/%Y')
    end_date_str = end_date.strftime('%d/%m/%Y')

    # Filter data within the selected date range
    mask = (X_predict['DateTime'] >= start_date) & (
        X_predict['DateTime'] <= end_date)
    filtered_data = X_predict[mask]
    filtered_preds = preds[mask]

    # If no data is found, display an error message
    if filtered_data.empty:
        st.error(f"No predictions found between {
                 start_date_str} and {end_date_str}.")
        return

    # Extract unique locations
    unique_locations = filtered_data['Location'].unique()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot predictions for each location
    for loc in unique_locations:
        loc_data = filtered_data[filtered_data['Location'] == loc]
        loc_preds = filtered_preds[filtered_data['Location'] == loc]

        ax.plot(loc_data['DateTime'], loc_preds, label=loc, marker='o')

    # Set graph labels and title
    ax.set_title(f'Predicted Target from {start_date_str} to {end_date_str}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Target')
    ax.legend(title="Location")
    plt.xticks(rotation=45)

    # Display the plot in Streamlit
    st.pyplot(fig)
