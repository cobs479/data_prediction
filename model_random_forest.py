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


def load_all_data(data_folder='data'):
    all_files = [f for f in os.listdir(data_folder) if f.startswith('weather_') and f.endswith('.csv')]
    all_files.sort()
    
    data_frames = []
    for file in all_files:
        df = pd.read_csv(os.path.join(data_folder, file))
        df['Year'] = int(file.split('_')[1].split('.')[0])  # Extract year from filename
        data_frames.append(df)

    st.warning("Loading all data")
    all_data = pd.concat(data_frames, ignore_index=True)
    st.dataframe(all_data)
    
    return all_data


def predict_random_forest(field, start_date, end_date, location_select):

    data_folder = 'data'
    
    X = load_all_data(data_folder)

    # Check if the field is valid
    if field not in X.columns:
        raise ValueError(f"Field '{field}' not found in the data columns!")

    # Change target
    X.dropna(axis=0, subset=[field], inplace=True)
    y = X[field]
    X.drop([field], axis=1, inplace=True)

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

    st.warning(f"Columns in X_train_full: {X_train_full.columns}")
    st.warning(f"Columns in X_valid_full: {X_valid_full.columns}")
    st.warning(f"Columns in X_train: {X_train.columns}")
    st.warning(f"Columns in X_valid: {X_valid.columns}")

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
    preds = pipeline.predict(X_valid)  # Use X_valid for validation if needed
    print(preds)

    # Calculate mean absolute error
    mae_score = mean_absolute_error(
        preds, y_valid)  # Use y_valid for validation
    print('MAE:', mae_score)

    # Calculate mean squared error (MSE)
    mse_score = mean_squared_error(preds, y_valid)
    print('MSE:', mse_score)

    # Calculate root mean squared error (RMSE)
    rmse_score = np.sqrt(mse_score)
    print('RMSE:', rmse_score)

    st.dataframe(X_valid)

    display_graph(X_valid, preds, start_date, end_date, location_select)
    display_table(X_valid, preds, start_date, end_date, location_select)

    st.success(f"Prediction ended")


def display_table(X, preds, start_date, end_date, location_select):
    # Ensure Hour column is properly formatted as HH:MM
    X['Hour'] = X['Hour'].astype(str).str.zfill(4)
    X['Formatted Hour'] = X['Hour'].str[:2] + \
        ":" + X['Hour'].str[2:]

    # Convert date columns into a full datetime format (YYYY-MM-DD HH:MM)
    X['DateTime'] = pd.to_datetime(
        X[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1) +
        ' ' + X['Formatted Hour']
    )

    # Mapping numerical locations to readable names
    location_mapping = {
        1: "Batu Muda",
        2: "Petaling Jaya",
        3: "Cheras"
    }
    X['Location'] = X['LocationInNum'].map(location_mapping)

    # Create a DataFrame for display
    results_df = pd.DataFrame({
        'Date-Time': X['DateTime'],
        'Location': X['Location'],
        'Predicted Value': preds
    })

    # Convert start_date and end_date to datetime format
    start_datetime = pd.to_datetime(start_date.strftime('%Y-%m-%d') + ' 00:00')
    end_datetime = pd.to_datetime(end_date.strftime('%Y-%m-%d') + ' 23:00')

    # Filter data within the date range
    results_df = results_df[(results_df['Date-Time'] >= start_datetime) &
                            (results_df['Date-Time'] <= end_datetime) & 
                            (results_df['Location'] == location_select)]

    # Display filtered table
    if results_df.empty:
        st.error(f"No predictions found for selected location ({location_select}) from {
                 start_date} to {end_date}")
    else:
        # st.write(f"### Predictions from {start_date} to {end_date}")
        st.dataframe(results_df, hide_index=True)


def display_graph(X, preds, start_date, end_date, location_select):
    # Convert date inputs to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Convert date columns into full datetime format
    X['DateTime'] = pd.to_datetime(X[['Year', 'Month', 'Day']])

    # **Mapping numerical locations to readable names (Same as display_table)**
    location_mapping = {
        1: "Batu Muda",
        2: "Petaling Jaya",
        3: "Cheras"
    }
    X['Location'] = X['LocationInNum'].map(location_mapping)

    # Convert selected date range into formatted strings for display
    start_date_str = start_date.strftime('%d/%m/%Y')
    end_date_str = end_date.strftime('%d/%m/%Y')

    # Filter data within the selected date range
    mask = (X['DateTime'] >= start_date) & (
        X['DateTime'] <= end_date) & (X['Location'] == location_select)
    filtered_data = X[mask]
    filtered_preds = preds[mask]

    # If no data is found, display an error message
    if filtered_data.empty:
        st.error(f"No predictions found for selected location ({location_select}) from {start_date_str} to {end_date_str}")
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
