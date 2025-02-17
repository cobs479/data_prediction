import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Function to load and merge multiple CSVs from 2017 to 2023
def load_weather_data(start_year=2017, end_year=2023):
    dataframes = []
    
    for year in range(start_year, end_year + 1):
        file_path = f'data/weather_{year}.csv'
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_cols = ['Year', 'Month', 'Day', 'Hour']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing column '{col}' in {file_path}")

            # Ensure Year, Month, Day, Hour are numeric (convert if necessary)
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows where any of the datetime components are missing
            df.dropna(subset=required_cols, inplace=True)

            # Convert Hour to two-digit format
            df['Hour'] = df['Hour'].astype(int).astype(str).str.zfill(2)
            df['Month'] = df['Month'].astype(int).astype(str).str.zfill(2)
            df['Day'] = df['Day'].astype(int).astype(str).str.zfill(2)

            # Create datetime column
            df['datetime'] = pd.to_datetime(
                df['Year'].astype(str) + '-' +
                df['Month'].astype(str) + '-' +
                df['Day'].astype(str) + ' ' +
                df['Hour'].astype(str) + ':00',
                format='%Y-%m-%d %H:%M',
                errors='coerce'
            )

            # Drop any rows where datetime conversion failed
            df.dropna(subset=['datetime'], inplace=True)

            # Drop unnecessary columns
            df.drop(columns=['Year', 'Month', 'Day', 'Hour'], inplace=True)

            dataframes.append(df)

    # Merge all data
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return None


def train_random_forest(field):
    # Load full dataset (2017-2023)
    data = load_weather_data()

    if data is None:
        st.error("No training data found from 2017 to 2023!")
        return None

    # Ensure target field exists
    if field not in data.columns:
        raise ValueError(f"Field '{field}' not found in dataset!")

    # Drop rows where target field is NaN
    data.dropna(subset=[field], inplace=True)

    # Define features (excluding target)
    X = data.drop(columns=[field])
    y = data[field]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Feature selection
    categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 20 and X_train[cname].dtype == "object"]
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

    # Define preprocessing
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

    # Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Full pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Train model
    pipeline.fit(X_train, y_train)

    # Save model
    model_path = f'saved_model/RF_{field}.joblib'
    joblib.dump(pipeline, model_path)

    return pipeline


def predict_random_forest(field, start_date, end_date, location_select):
    # Load trained model
    model_path = f'saved_model/RF_{field}.joblib'
    
    if not os.path.exists(model_path):
        st.error(f"Model for '{field}' not found. Training new model...")
        model = train_random_forest(field)
        if model is None:
            return
    else:
        model = joblib.load(model_path)

    # Load past weather data (2017-2023)
    data = load_weather_data()
    
    if data is None:
        st.error("No historical weather data found!")
        return

    # Create a dataframe for future predictions
    future_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    future_data = pd.DataFrame({'datetime': future_dates})

    # Add location selection
    future_data['location'] = location_select

    # Ensure features match training data
    feature_cols = model.named_steps['preprocessor'].transformers[0][2] + model.named_steps['preprocessor'].transformers[1][2]
    for col in feature_cols:
        if col not in future_data.columns:
            future_data[col] = np.nan  # Placeholder for missing columns

    # Predict values
    preds = model.predict(future_data)

    # Store predictions
    future_data['Predicted'] = preds

    # Display results
    display_table(future_data, preds, start_date, end_date, location_select)
    display_graph(future_data, preds, start_date, end_date, location_select)

    st.success("Prediction complete")


def display_table(future_data, preds, start_date, end_date, location_select):
    future_data['DateTime'] = pd.to_datetime(future_data['datetime'])

    location_mapping = {1: "Batu Muda", 2: "Petaling Jaya", 3: "Cheras"}
    future_data['Location'] = location_mapping.get(location_select, "Unknown")

    results_df = pd.DataFrame({
        'Date-Time': future_data['DateTime'],  # Ensure this column is in datetime format
        'Location': future_data['Location'],
        'Predicted Value': preds
    })

    # 🔥 Convert Date-Time column to datetime (Fix)
    results_df['Date-Time'] = pd.to_datetime(results_df['Date-Time'])

    # Ensure start_date and end_date are also datetime objects
    start_date = pd.to_datetime(str(start_date) + ' 00:00')
    end_date = pd.to_datetime(str(end_date) + ' 23:00')

    # Apply filtering (Comparison will now work correctly)
    results_df = results_df[(results_df['Date-Time'] >= start_date) &
                            (results_df['Date-Time'] <= end_date) & 
                            (results_df['Location'] == location_mapping.get(location_select, "Unknown"))]

    if results_df.empty:
        st.error(f"No predictions found for location ({location_select}) from {start_date} to {end_date}")
    else:
        st.dataframe(results_df)


def display_graph(future_data, preds, start_date, end_date, location_select):
    future_data['DateTime'] = future_data['datetime']

    location_mapping = {1: "Batu Muda", 2: "Petaling Jaya", 3: "Cheras"}
    future_data['Location'] = location_mapping.get(location_select, "Unknown")

    start_date_str = start_date.strftime('%d/%m/%Y')
    end_date_str = end_date.strftime('%d/%m/%Y')

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(future_data['DateTime'], preds, label=future_data['Location'][0], marker='o')

    ax.set_title(f'Predicted {field} from {start_date_str} to {end_date_str}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Value')
    ax.legend(title="Location")
    plt.xticks(rotation=45)

    st.pyplot(fig)
