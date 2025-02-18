import pandas as pd
import joblib
import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.interpolate import interp1d
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def interpolate_data(weather_data):

    weather_data['Datetime'] = pd.to_datetime(weather_data['Datetime'])
    weather_data = weather_data.sort_values(by='Datetime').reset_index(drop=True)
    
    start_date = "2024-01-01 00:00"  # Example start date (modify as needed)
    end_date = "2024-01-31 23:00"    # Example end date (modify as needed)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')

    st.success(f"Original weather data")
    st.dataframe(weather_data)
    
    complete_weather_data = pd.DataFrame({'Datetime': date_range})
    weather_data = pd.merge(complete_weather_data, weather_data, on='Datetime', how='left')

    st.success(f"After merge weather data")
    st.dataframe(weather_data)
    
    exclude_columns = ['Datetime', 'Location', 'LocationInNum', 'LocationInNum.1']
    numeric_columns = [col for col in weather_data.columns if col not in exclude_columns]
    
    weather_data['Timestamp'] = weather_data['Datetime'].astype(np.int64) // 10**9

    # Step 8: Interpolate & extrapolate missing values based on known data (2017-2023)
    for col in numeric_columns:
        known_data = weather_data.dropna(subset=[col])  # Get only known values
        if known_data.empty:
            print(f"Skipping {col} - No known values for interpolation")
            continue  # Skip if no known values exist

        X = known_data['Timestamp'].values  # Convert datetime to numerical format
        y = known_data[col].values

        # Create interpolation function using linear extrapolation
        interp_func = interp1d(X, y, kind='linear', fill_value='extrapolate')

        # Apply interpolation to missing values
        missing_indices = weather_data[weather_data[col].isna()].index
        weather_data.loc[missing_indices, col] = interp_func(weather_data.loc[missing_indices, 'Timestamp'].values)

    st.success(f"Interpolated weather data")
    st.dataframe(weather_data)


def load_all_data(data_folder='data'):
    all_files = [f for f in os.listdir(data_folder) if f.startswith('weather_') and f.endswith('.csv')]
    all_files.sort()
    
    data_frames = []
    for file in all_files:
        df = pd.read_csv(os.path.join(data_folder, file), parse_dates=[['Year', 'Month', 'Day', 'Hour']], dayfirst=True)
        df['Year'] = int(file.split('_')[1].split('.')[0])
        df.rename(columns={'Year_Month_Day_Hour': 'Datetime'}, inplace=True)
        data_frames.append(df)

    all_data = pd.concat(data_frames, ignore_index=True)
    #st.success("Combined data")
    #st.dataframe(all_data)
    
    return all_data


def predict_random_forest(field, start_date, end_date, location_select):

    data_folder = 'data'
    
    X = load_all_data(data_folder)
    interpolate_data(X)

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

    cols = ['Datetime'] + categorical_cols + numerical_cols
    X_train = X_train_full[cols].copy()
    X_valid = X_valid_full[cols].copy()

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

        #Must force rebuild model if the columns are different
        #print("Building and training a new model...")
        #pipeline.fit(X_train, y_train)
        #joblib.dump(pipeline, model_save_path)
        #print("Model saved successfully.")
    else:
        print("Building and training a new model...")
        pipeline.fit(X_train, y_train)
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

    #st.dataframe(X_valid)

    display_graph(X_valid, preds, start_date, end_date, location_select)
    display_table(X_valid, preds, start_date, end_date, location_select)

    st.success(f"Prediction ended")


def display_table(X, preds, start_date, end_date, location_select):
    # Convert date columns into a full datetime format (YYYY-MM-DD HH:MM)
    X['Datetime'] = pd.to_datetime(X['Datetime'])

    # Mapping numerical locations to readable names
    location_mapping = {
        1: "Batu Muda",
        2: "Petaling Jaya",
        3: "Cheras"
    }
    X['Location'] = X['LocationInNum'].map(location_mapping)

    # Create a DataFrame for display
    results_df = pd.DataFrame({
        'Date-Time': X['Datetime'],
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

    results_df = results_df.sort_values(by='Date-Time')
    
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
    X['Datetime'] = pd.to_datetime(X['Datetime'])

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
    mask = (X['Datetime'] >= start_date) & (
        X['Datetime'] <= end_date) & (X['Location'] == location_select)
    filtered_data = X[mask]
    filtered_preds = preds[mask]

    # If no data is found, display an error message
    if filtered_data.empty:
        st.error(f"No predictions found for selected location ({location_select}) from {start_date_str} to {end_date_str}")
        return

    filtered_data = filtered_data.sort_values(by='Datetime')

    # Extract unique locations
    unique_locations = filtered_data['Location'].unique()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot predictions for each location
    for loc in unique_locations:
        loc_data = filtered_data[filtered_data['Location'] == loc]
        loc_preds = filtered_preds[filtered_data['Location'] == loc]

        ax.plot(loc_data['Datetime'], loc_preds, label=loc, marker='o')

    # Set graph labels and title
    ax.set_title(f'Predicted Target from {start_date_str} to {end_date_str}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Target')
    ax.legend(title="Location")
    plt.xticks(rotation=45)

    # Display the plot in Streamlit
    st.pyplot(fig)
