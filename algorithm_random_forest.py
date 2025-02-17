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
    if not os.path.exists(train_file_path):
        st.error("Training file not found! Model cannot be trained.")
        return
    
    X = pd.read_csv(train_file_path)

    st.warning(f"Running")
    
    # Try loading the prediction file, if unavailable, generate new data based on historical data
    if os.path.exists(predict_file_path):
        X_predict = pd.read_csv(predict_file_path)
    else:
        st.warning(f"Prediction file for {end_date.year} not found. Generating synthetic data based on historical trends.")
        X_predict = generate_synthetic_data(X, start_date, end_date)
    
    if field not in X.columns:
        st.error(f"Field '{field}' not found in the dataset!")
        return
    
    X.dropna(axis=0, subset=[field], inplace=True)
    y = X[field]
    X.drop([field], axis=1, inplace=True)
    
    y_predict = X_predict[field] if field in X_predict.columns else None
    X_predict.drop([field], axis=1, inplace=True, errors='ignore')
    
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0)
    
    categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 20 and X_train_full[cname].dtype == "object"]
    numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
    
    cols = categorical_cols + numerical_cols
    X_train = X_train_full[cols].copy()
    X_valid = X_valid_full[cols].copy()
    X_predict = X_predict[cols].copy()
    
    model_save_path = f'saved_model/RF_{field}.joblib'
    
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
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    if os.path.exists(model_save_path):
        pipeline = joblib.load(model_save_path)
    else:
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_save_path)
    
    preds = pipeline.predict(X_predict)
    
    if y_predict is not None:
        mae_score = mean_absolute_error(preds, y_predict)
        mse_score = mean_squared_error(preds, y_predict)
        rmse_score = np.sqrt(mse_score)
        print('MAE:', mae_score, 'MSE:', mse_score, 'RMSE:', rmse_score)
    
    st.dataframe(X_predict)
    display_graph(X_predict, preds, start_date, end_date, location_select)
    display_table(X_predict, preds, start_date, end_date, location_select)
    st.success("Prediction completed")

def generate_synthetic_data(X, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    synthetic_data = pd.DataFrame({
        'Year': date_range.year,
        'Month': date_range.month,
        'Day': date_range.day,
        'Hour': date_range.hour,
        'LocationInNum': np.random.choice([1, 2, 3], size=len(date_range))
    })
    return synthetic_data

def display_table(X_predict, preds, start_date, end_date, location_select):
    X_predict['Hour'] = X_predict['Hour'].astype(str).str.zfill(2) + ':00'
    X_predict['DateTime'] = pd.to_datetime(
        X_predict[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1) + ' ' + X_predict['Hour'])
    
    location_mapping = {1: "Batu Muda", 2: "Petaling Jaya", 3: "Cheras"}
    X_predict['Location'] = X_predict['LocationInNum'].map(location_mapping)
    
    results_df = pd.DataFrame({
        'Date-Time': X_predict['DateTime'],
        'Location': X_predict['Location'],
        'Predicted Value': preds
    })
    
    start_datetime = pd.to_datetime(start_date.strftime('%Y-%m-%d') + ' 00:00')
    end_datetime = pd.to_datetime(end_date.strftime('%Y-%m-%d') + ' 23:00')
    
    results_df = results_df[(results_df['Date-Time'] >= start_datetime) &
                            (results_df['Date-Time'] <= end_datetime) & 
                            (results_df['Location'] == location_select)]
    
    if results_df.empty:
        st.error(f"No predictions found for {location_select} from {start_date} to {end_date}")
    else:
        st.dataframe(results_df, hide_index=True)

def display_graph(X_predict, preds, start_date, end_date, location_select):
    X_predict['DateTime'] = pd.to_datetime(X_predict[['Year', 'Month', 'Day']])
    location_mapping = {1: "Batu Muda", 2: "Petaling Jaya", 3: "Cheras"}
    X_predict['Location'] = X_predict['LocationInNum'].map(location_mapping)
    
    mask = (X_predict['DateTime'] >= start_date) & (X_predict['DateTime'] <= end_date) & (X_predict['Location'] == location_select)
    filtered_data = X_predict[mask]
    filtered_preds = preds[mask]
    
    if filtered_data.empty:
        st.error(f"No predictions found for {location_select} from {start_date} to {end_date}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered_data['DateTime'], filtered_preds, label=location_select, marker='o')
    ax.set_title(f'Predicted Data from {start_date} to {end_date}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Value')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
