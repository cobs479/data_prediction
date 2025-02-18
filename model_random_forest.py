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


def interpolate_data(weather_data, start_date, end_date):

    weather_data['Datetime'] = pd.to_datetime(weather_data['Datetime'])
    weather_data = weather_data.sort_values(by='Datetime').reset_index(drop=True)
    
    start_date = start_date.strftime('%Y-%m-%d') + " 00:00"  # Example start date (modify as needed)
    end_date = end_date.strftime('%Y-%m-%d') + " 23:00"    # Example end date (modify as needed)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    complete_weather_data = pd.DataFrame({'Datetime': date_range})
    weather_data = pd.merge(weather_data, complete_weather_data, on='Datetime', how='outer', indicator=True)
    
    exclude_columns = ['Datetime', 'Location', 'LocationInNum', 'LocationInNum.1', '_merge']
    numeric_columns = [col for col in weather_data.columns if col not in exclude_columns]
    
    weather_data['Timestamp'] = weather_data['Datetime'].astype(np.int64) // 10**9

    air_quality_vars = ["PM10", "PM25", "CO", "SO2", "NO2", "O3"]  # Air quality variables
    #other_vars = ["Temperature", "Humidity", "WindSpeed", "WindDirection", "Evaporation", "GlobalRadiation", "SolarRadiation", "Usage", "MaxDemand", "Bill"]  # Other climate data

    for col in numeric_columns:
        known_data = weather_data.dropna(subset=[col])  # Get only known values
        if known_data.empty:
            print(f"Skipping {col} - No known values for interpolation")
            continue  
    
        known_data[col] = pd.to_numeric(known_data[col], errors='coerce')
        known_data = known_data.dropna(subset=[col])
    
        if len(known_data) < 3:  # At least 3 points needed for interpolation
            print(f"Skipping {col} - Not enough data points for interpolation")
            continue
    
        known_data = known_data.drop_duplicates(subset=['Timestamp'])
    
        X = known_data['Timestamp'].values
        y = known_data[col].values.astype(np.float64)
    
        try:
            if col in air_quality_vars:
                if len(known_data) >= 4:
                    smooth_values = lowess(y, X, frac=0.1, return_sorted=False)
                    interp_func = interp1d(X, smooth_values, kind='linear', fill_value='extrapolate', bounds_error=False)
                else:
                    interp_func = interp1d(X, y, kind='linear', fill_value='extrapolate', bounds_error=False)
    
            else:
                if len(known_data) >= 4:
                    interp_func = CubicSpline(X, y, extrapolate=True)
                else:
                    interp_func = interp1d(X, y, kind='linear', fill_value='extrapolate', bounds_error=False)
    
            missing_indices = weather_data[weather_data[col].isna()].index
            interpolated_values = interp_func(weather_data.loc[missing_indices, 'Timestamp'].values)
    
            q1, q3 = np.percentile(y, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            interpolated_values = np.clip(interpolated_values, lower_bound, upper_bound)
    
            weather_data.loc[missing_indices, col] = interpolated_values
    
        except Exception as e:
            print(f"Skipping {col} due to interpolation error: {e}")
            continue
    
    weather_data[numeric_columns] = weather_data[numeric_columns].interpolate(method='linear', limit_direction='both')
    weather_data[numeric_columns] = weather_data[numeric_columns].fillna(method='ffill').fillna(method='bfill')

    weather_data = weather_data[weather_data['_merge'] == 'right_only'].drop(columns=['_merge'])
    
    return weather_data

    """
    # Check for interpolation
    st.success("Interpolation Completed")
    st.dataframe(weather_data)

    weather_data['Datetime'] = pd.to_datetime(weather_data['Datetime'])

    exclude_columns = ['Datetime', 'Location', 'LocationInNum', 'LocationInNum.1', 'Date', 'DateTemp', 'Year', 'Timestamp']
    weather_variables = [col for col in weather_data.columns if col not in exclude_columns]
    
    known_data = weather_data[weather_data['Datetime'] < '2024-01-01'].copy()  # Actual (2017-2023)
    interpolated_data = weather_data[weather_data['Datetime'] >= '2024-01-01'].copy()  # Interpolated (2024)
    
    for variable in weather_variables:
        known_data[variable] = pd.to_numeric(known_data[variable], errors='coerce')
        interpolated_data[variable] = pd.to_numeric(interpolated_data[variable], errors='coerce')
    
    for variable in weather_variables:
        num_known = known_data[variable].count()
        num_interpolated = interpolated_data[variable].count()
    
    num_vars = len(weather_variables)
    fig, axes = plt.subplots(nrows=num_vars, ncols=1, figsize=(12, 4 * num_vars), sharex=True)
    
    if num_vars == 1:
        axes = [axes]
    
    plotted_any = False  # Track if any graph is plotted
    for i, variable in enumerate(weather_variables):
        ax = axes[i]
    
        known_plot = known_data[['Datetime', variable]].dropna()
        interp_plot = interpolated_data[['Datetime', variable]].dropna()
    
        if known_plot.empty and interp_plot.empty:
            st.write(f"Skipping {variable} - No valid data to plot")
            continue
    
        ax.plot(known_plot['Datetime'], known_plot[variable], 'b-', label="Actual Data (2017-2023)")
        ax.plot(interp_plot['Datetime'], interp_plot[variable], 'r--', label="Interpolated Data (2024)")
        
        ax.set_ylabel(variable)
        ax.legend()
        ax.grid()
        plotted_any = True  # At least one plot was drawn
    
    if plotted_any:
        plt.xlabel("Date")
        plt.suptitle("Interpolation of Weather Variables from 2017 to 2024")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)  # ✅ Streamlit way to display Matplotlib figures
    else:
        st.write("No valid data available for any variable. No graph drawn.")
    """
    

def load_all_data(data_folder='data'):
    all_files = [f for f in os.listdir(data_folder) if f.startswith('weather_') and f.endswith('.csv')]
    all_files.sort()

    first_year = int(all_files[0].split('_')[1].split('.')[0])
    last_year = int(all_files[-1].split('_')[1].split('.')[0])
    
    data_frames = []
    for file in all_files:
        df = pd.read_csv(os.path.join(data_folder, file), parse_dates=[['Year', 'Month', 'Day', 'Hour']], dayfirst=True)
        df['Year'] = int(file.split('_')[1].split('.')[0])
        df.rename(columns={'Year_Month_Day_Hour': 'Datetime'}, inplace=True)
        data_frames.append(df)

    all_data = pd.concat(data_frames, ignore_index=True)
    
    return all_data, first_year, last_year


def predict_random_forest(field, start_date, end_date, location_select):

    data_folder = 'data'
    
    X, first_year, last_year = load_all_data(data_folder)
    
    if first_year <= start_date.year <= last_year and first_year <= end_date.year <= last_year:
        X_predict = pd.read_csv(f'data/weather_{end_date.year}.csv', parse_dates=[['Year', 'Month', 'Day', 'Hour']], dayfirst=True)
        X_predict['Year'] = int(end_date.year)
        st.warning("✅ The date range is within the year range.")
    else:
        X_predict = interpolate_data(X, start_date, end_date)
        st.warning("❌ The date range is NOT within the year range.")

    X_predict.rename(columns={'Year_Month_Day_Hour': 'Datetime'}, inplace=True)
    X_predict['LocationInNum.1'] = ""

    if field not in X.columns:
        raise ValueError(f"Field '{field}' not found in the data columns!")

    X.dropna(axis=0, subset=[field], inplace=True)
    y = X[field]
    X.drop([field], axis=1, inplace=True)

    #X['CO'] = pd.to_numeric(X['CO'], errors='coerce')
    #X['CO'] = X['CO'].fillna(0)

    X.fillna(0, inplace=True)

    y_predict = X_predict[field]
    X_predict.drop([field], axis=1, inplace=True)

    X_predict.fillna(0, inplace=True)

    X_train_full, X_valid_full, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0)

    categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 20 and
                        X_train_full[cname].dtype == "object"]
    numerical_cols = [
        cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    cols = ['Datetime'] + categorical_cols + numerical_cols
    X_train = X_train_full[cols].copy()
    X_valid = X_valid_full[cols].copy()

    location_in_num_mapping = {
            "Batu Muda": 1,
            "Petaling Jaya": 2,
            "Cheras": 3
        }

    location_mapping = {
            "Batu Muda": "KL",
            "Petaling Jaya": "PJ",
            "Cheras": "KL"
        }
    
    if first_year <= start_date.year <= last_year and first_year <= end_date.year <= last_year:
        X_predict = X_predict[cols].copy()
    else:
        X_predict.drop(['DateTemp', 'Date', 'Location', 'LocationInNum', 'Timestamp'], axis=1, inplace=True)
        X_predict['Location'] = location_mapping[location_select]
        X_predict['LocationInNum'] = location_in_num_mapping[location_select]
        X_predict['Year'] = start_date.year

    X_predict = X_predict.replace("", np.nan)

    model_save_path = 'saved_model/RF_' + field + '.joblib'

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

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    pipeline = Pipeline(
        steps=[('preprocessor', preprocessor), ('model', model)])

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
    
    preds = pipeline.predict(X_predict)  # Use X_valid for validation if needed
    print(preds)

    mae_score = mean_absolute_error(
        preds, y_predict)  # Use y_valid for validation
    print('MAE:', mae_score)

    mse_score = mean_squared_error(preds, y_predict)
    print('MSE:', mse_score)

    rmse_score = np.sqrt(mse_score)
    print('RMSE:', rmse_score)

    st.dataframe(X_valid)
    st.dataframe(X_predict)

    display_graph(X_predict, preds, start_date, end_date, location_select)
    display_table(X_predict, preds, start_date, end_date, location_select)

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
