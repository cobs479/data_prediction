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
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def predict_random_forest(field, date):
    train_file_path = 'data/weather_2021.csv'
    predict_file_path = 'data/weather_2020.csv'

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
    X_predict = X_predict[cols].copy()

    # Define the path for the saved model
    model_save_path = 'saved_model/RF_' + field + '.joblib'

    # Preprocessor settings / strategies
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

    # Change Model settings
    model = RandomForestRegressor(n_estimators=100, random_state=0)

    # Pipeline with preprocessor and model
    pipeline = Pipeline(
        steps=[('preprocessor', preprocessor), ('model', model)])

    # Check if there is an existing saved model
    if os.path.exists(model_save_path):
        print("Loading model from previous save...")
        pipeline = joblib.load(model_save_path)
    else:
        print("Building and training a new model...")
        # Train the model
        pipeline.fit(X_train, y_train)

        # Save the model pipeline to file
        joblib.dump(pipeline, model_save_path)
        print("Model saved successfully.")

    # Change X set and do predictions
    preds = pipeline.predict(X_predict)  # Use X_valid for validation if needed
    print(preds)

    # Calculate mean absolute error
    mae_score = mean_absolute_error(
        preds, y_predict)  # Use y_valid for validation
    print('MAE:', mae_score)

    # Calculate mean squared error (MSE)
    mse_score = mean_squared_error(preds, y_predict)
    print('MSE:', mse_score)

    # Calculate root mean squared error (RMSE)
    rmse_score = np.sqrt(mse_score)
    print('RMSE:', rmse_score)

    display_graph(X_predict, preds, date)


def display_graph(X_predict, preds, date):
    # Convert the selected date to a formatted string (dd/mm/yyyy)
    date_str = date.strftime('%d/%m/%Y')

    day = X_predict['Day']
    month = X_predict['Month']
    year = X_predict['Year']
    location = X_predict['Location']

    # Convert date columns into formatted date strings
    dates = pd.to_datetime(X_predict[['Year', 'Month', 'Day']])
    dates_formatted = dates.dt.strftime('%d/%m/%Y').values

    dates_array = dates_formatted
    locations_array = location.values

    # Filter rows where the date matches the selected date
    filtered_indices = dates_formatted == date_str
    filtered_dates = dates_array[filtered_indices]
    filtered_preds = preds[filtered_indices]
    filtered_locations = locations_array[filtered_indices]

    # If no data found for the selected date, show an error
    if len(filtered_preds) == 0:
        st.error(f"No predictions found for the date {date_str}")
        return

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot predictions for each unique location on the selected date
    for loc in set(filtered_locations):  # Using set to avoid duplicate locations
        loc_indices = filtered_locations == loc
        loc_dates = filtered_dates[loc_indices]
        loc_target = filtered_preds[loc_indices]

        ax.plot(loc_dates, loc_target, label=loc, marker='o')

    ax.set_title(f'Predicted Target on {date_str}')
    ax.set_xlabel('Date (dd/mm/yyyy)')
    ax.set_ylabel('Predicted Target')
    ax.set_xticklabels(filtered_dates, rotation=45)  # Rotate x labels

    # Display the plot using Streamlit
    st.pyplot(fig)
