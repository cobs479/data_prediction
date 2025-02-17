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

def load_all_data(data_folder='data'):
    """Load all available weather data CSV files for training."""
    all_files = [f for f in os.listdir(data_folder) if f.startswith('weather_') and f.endswith('.csv')]
    all_files.sort()
    
    data_frames = []
    for file in all_files:
        df = pd.read_csv(os.path.join(data_folder, file))
        df['Year'] = int(file.split('_')[1].split('.')[0])  # Extract year from filename
        data_frames.append(df)
    
    return pd.concat(data_frames, ignore_index=True)

def predict_random_forest(field, start_date, end_date, location_select):
    data_folder = 'data'
    
    # Load all available data
    X = load_all_data(data_folder)
    
    # Validate field
    if field not in X.columns:
        raise ValueError(f"Field '{field}' not found in the data columns!")
    
    # Drop missing target values
    X.dropna(axis=0, subset=[field], inplace=True)
    y = X[field]
    X.drop(columns=[field], inplace=True)
    
    # Filter based on location
    if 'Location' in X.columns:
        X = X[X['Location'] == location_select]
        y = y.loc[X.index]  # Ensure y is filtered the same way as X
    
    # Check if data is available after filtering
    if X.empty:
        st.error(f"No data available for the selected location: {location_select}")
        return
    
    # Train/test split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0)
    
    # Identify categorical and numerical columns
    categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 20 and X_train[cname].dtype == "object"]
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
    
    cols = categorical_cols + numerical_cols
    X_train = X_train[cols].copy()
    X_valid = X_valid[cols].copy()
    
    # Model file path
    model_save_path = f'saved_model/RF_{field}.joblib'
    
    # Preprocessing pipeline
    numerical_transformer = SimpleImputer(strategy='most_frequent')
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    
    # Train model if not found
    if os.path.exists(model_save_path):
        print("Loading saved model...")
        pipeline = joblib.load(model_save_path)
    else:
        print("Training new model...")
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_save_path)
        print("Model saved.")
    
    # Predict future values (2024 and beyond or 2016 and below)
    future_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    future_data = pd.DataFrame({'Year': future_dates.year, 'Month': future_dates.month, 'Day': future_dates.day})
    
    # Add missing numerical and categorical columns
    for col in cols:
        if col not in future_data:
            future_data[col] = X[col].mode()[0]  # Fill with mode of training data
    
    preds = pipeline.predict(future_data[cols])
    
    # Display results
    results_df = pd.DataFrame({'Date': future_dates, 'Predicted Value': preds})
    st.dataframe(results_df)
    st.success(f"Prediction complete for {start_date} to {end_date} at {location_select}")
