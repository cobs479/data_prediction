import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 🔹 Function to Load & Train Model
def predict_random_forest(field, start_date, end_date, location_select):
    train_file_path = 'data/weather_2021.csv'
    predict_file_path = f'data/weather_{end_date.year}.csv'

    # Load training data
    X = pd.read_csv(train_file_path)

    # Check if the field exists in training data
    if field not in X.columns:
        st.error(f"⚠️ Field '{field}' not found in the dataset!")
        return

    # Prepare the dataset
    X.dropna(axis=0, subset=[field], inplace=True)
    y = X[field]
    X.drop([field], axis=1, inplace=True)

    # Train-test split
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
        st.success("✅ Loaded pre-trained model")
        pipeline = joblib.load(model_save_path)
    else:
        st.warning("⚠️ Training new model...")
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_save_path)
        st.success("✅ Model trained and saved")

    # Handle Future Predictions
    if os.path.exists(predict_file_path):
        X_predict = pd.read_csv(predict_file_path)
    else:
        st.warning(f"⚠️ Prediction file for {end_date.year} not found. Generating future dataset...")
        X_predict = generate_future_data(X, start_date, end_date)

    if X_predict is not None:
        preds = pipeline.predict(X_predict)

        # Calculate error metrics
        mae_score = mean_absolute_error(preds, np.full_like(preds, y.mean()))
        mse_score = mean_squared_error(preds, np.full_like(preds, y.mean()))
        rmse_score = np.sqrt(mse_score)

        st.write(f"📊 **Model Performance**")
        st.write(f"🟢 MAE: {mae_score}")
        st.write(f"🟢 MSE: {mse_score}")
        st.write(f"🟢 RMSE: {rmse_score}")

        # Display results
        display_graph(X_predict, preds, start_date, end_date, location_select)
        display_table(X_predict, preds, start_date, end_date, location_select)
    else:
        st.error("⚠️ No valid data available for prediction")


# 🔹 Function to Generate Future Data
def generate_future_data(X, start_date, end_date):
    """Generate synthetic future dataset for all locations"""

    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate locations
    location_mapping = {1: "Batu Muda", 2: "Petaling Jaya", 3: "Cheras"}
    location_nums = list(location_mapping.keys())  # [1, 2, 3]

    # Create future data with all locations
    future_data = pd.DataFrame({
        'Year': np.repeat(future_dates.year, len(location_nums)),
        'Month': np.repeat(future_dates.month, len(location_nums)),
        'Day': np.repeat(future_dates.day, len(location_nums)),
        'LocationInNum': np.tile(location_nums, len(future_dates))
    })

    # DEBUGGING: Check if Year, Month, Day exist
    st.write("📌 Future Data Preview (Before DateTime):")
    st.dataframe(future_data.head())

    # Generate hourly values
    future_data['Hour'] = np.tile(range(0, 2400, 100), len(future_data) // 24 + 1)[:len(future_data)]

    # Ensure missing categorical and numerical columns are filled
    for col in X.columns:
        if col not in future_data.columns:
            if X[col].dtype == "object":
                future_data[col] = X[col].mode()[0]  # Fill categorical with mode
            else:
                future_data[col] = X[col].mean()  # Fill numerical with mean

    # DEBUGGING: Check if Year, Month, Day are still correct
    st.write("📌 Future Data (After Filling Columns):")
    st.dataframe(future_data.head())

    # Convert DateTime column correctly
    try:
        future_data['DateTime'] = pd.to_datetime(
            future_data[['Year', 'Month', 'Day']], errors='coerce'
        )
    except Exception as e:
        st.error(f"⚠️ Error creating DateTime column: {e}")

    # DEBUGGING: Confirm DateTime exists
    st.write("📌 Future Data with DateTime:")
    st.dataframe(future_data.head())

    return future_data


# 🔹 Function to Display Table
def display_table(X_predict, preds, start_date, end_date, location_select):
    """Display predictions in a table"""
    X_predict['Location'] = X_predict['LocationInNum'].map({1: "Batu Muda", 2: "Petaling Jaya", 3: "Cheras"})

    results_df = pd.DataFrame({
        'DateTime': X_predict['DateTime'],
        'Location': X_predict['Location'],
        'Predicted Value': preds
    })

    # Filter by location and date
    results_df = results_df[(results_df['DateTime'] >= start_date) & (results_df['DateTime'] <= end_date) & (results_df['Location'] == location_select)]

    if results_df.empty:
        st.error("⚠️ No predictions found for the selected location and date range")
    else:
        st.dataframe(results_df)


# 🔹 Function to Display Graph
def display_graph(X_predict, preds, start_date, end_date, location_select):
    """Plot prediction results"""

    # Ensure the DateTime column exists
    if 'DateTime' not in X_predict.columns:
        st.error("⚠️ Missing 'DateTime' column in X_predict!")
        st.write("Columns in X_predict:", X_predict.columns)
        return

    # Convert DateTime to datetime64 if necessary
    X_predict['DateTime'] = pd.to_datetime(X_predict['DateTime'], errors='coerce')

    # Ensure locations are mapped correctly
    location_mapping = {1: "Batu Muda", 2: "Petaling Jaya", 3: "Cheras"}
    X_predict['Location'] = X_predict['LocationInNum'].map(location_mapping)

    # Print debug info
    st.write("Unique Locations in X_predict:", X_predict['Location'].unique())

    # Apply filtering
    mask = (X_predict['DateTime'] >= start_date) & (X_predict['DateTime'] <= end_date) & (X_predict['Location'] == location_select)
    filtered_data = X_predict[mask]

    if filtered_data.empty:
        st.error(f"⚠️ No predictions found for {location_select} from {start_date.date()} to {end_date.date()}")
        return

    # Plot the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data['DateTime'], preds[mask], marker='o', label=location_select)
    plt.title(f'Predictions for {location_select}')
    plt.xlabel('Date')
    plt.ylabel('Predicted Value')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)
