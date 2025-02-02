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


def predict_xgboost(field, date):
    train_file_path = 'data/weather_2021.csv'
    predict_file_path = 'data/weather_2020.csv'

    X = pd.read_csv(train_file_path)
    X_predict = pd.read_csv(predict_file_path)

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

    display_graph(X_predict_full, preds, date)


def display_graph(X_predict_full, preds, date):
    # Convert the selected date to a formatted string (dd/mm/yyyy)
    date_str = date.strftime('%d/%m/%Y')

    day = X_predict_full['Day']
    month = X_predict_full['Month']
    year = X_predict_full['Year']
    location = X_predict_full['Location']

    # Convert date columns into formatted date strings
    dates = pd.to_datetime(X_predict_full[['Year', 'Month', 'Day']])
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
