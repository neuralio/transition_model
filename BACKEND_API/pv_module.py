import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pvlib import solarposition

# Additional plotting and analysis imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from scipy.interpolate import griddata
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

#############################################
# Utility Functions
#############################################
def ensure_directory(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

#############################################
# Directories & Constants
#############################################
# In this past module the main directory is set as in your notebook.
# You can modify MAIN_DIR if needed.
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(MAIN_DIR, "INPUT")
CLIMATE_PROJ_DIR = os.path.join(DATA_DIR, "ERA5_LAND")
SLOPE_DIR = os.path.join(DATA_DIR, "SLOPE")
TRAINING_DIR = os.path.join(MAIN_DIR, "TRAINING")
PREDICTION_DIR = os.path.join(MAIN_DIR, "PREDICTIONS/CSVs")
PLOTS_DIR = os.path.join(MAIN_DIR, "PLOTS")

# Ensure that output directories exist
ensure_directory(PREDICTION_DIR)
ensure_directory(PLOTS_DIR)

# Define constants for the module
AREA = 2.4                # m² (panel area)
EFFICIENCY = 18.5 / 100    # 18.5% efficiency
ELECTRICITY_RATE = 0.12   # €/kWh
CLASS_LABELS = ['0', '1', '2', '3', '4', '5']
LUSA_SCORES = [0, 40, 60, 85, 95, 100]

# Path to input JSON file 
INPUT_JSON_PATH = os.path.join(MAIN_DIR, "input_mongo_past.json")

#############################################
# Training and Model Functions
#############################################
# def load_config(config_path):
#     print(f"Loading configuration from: {config_path}")
#     with open(config_path, 'r') as f:
#         return json.load(f)

def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config, config.get("task_id"), config.get("user_id")


def load_training_data(training_file):
    print(f"Loading training data from: {training_file}")
    df = pd.read_csv(training_file, sep=';')
    X = df.iloc[:, 1:12]
    y = np.floor(df.iloc[:, 0] / 20).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, num_features=5):
    print("Training the Feedforward Neural Network...")
    # Build the model architecture
    input_layer = tf.keras.layers.Input(shape=(num_features,))
    dense1 = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
    dense3 = tf.keras.layers.Dense(32, activation='relu')(dense2)
    # Use sigmoid (or softmax) for 6 classes
    output_layer = tf.keras.layers.Dense(6, activation='sigmoid')(dense3)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)
    
    # Save training history for later analysis
    history_file = os.path.join(TRAINING_DIR, "PAST_history_train.txt")
    with open(history_file, 'w') as f:
        f.write(str(history.history))
    print(f"Training metrics saved in: {history_file}")
    return model, history

#############################################
# Data Loading Functions for Past Module
#############################################
def load_slope_data():
    """Load slope data from NetCDF and return a DataFrame."""
    slope_file = os.path.join(SLOPE_DIR, "SLOPE.nc")
    ds = xr.open_dataset(slope_file)
    slope_data = ds['slope'].values
    lon = ds['lon'].values
    lat = ds['lat'].values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    df_slope = pd.DataFrame({
        'lon': lon_grid.flatten(),
        'lat': lat_grid.flatten(),
        'value': slope_data.flatten()
    })
    df_slope['parameter'] = 'slope'
    df_slope['date'] = '0000-00-00'
    df_slope = df_slope[['parameter', 'date', 'lon', 'lat', 'value']]
    print("Loaded slope data.")
    return df_slope

def load_rsds_data():
    """Load ERA5-Land solar radiation data and return a DataFrame with a datetime index and computed year."""
    rsds_file = os.path.join(CLIMATE_PROJ_DIR, "input_ERA5_LAND_SOLAR.nc")
    ds = xr.open_dataset(rsds_file)
    # In your notebook the variable was 'var169'
    rsds_data = ds['var169'].values
    lon = ds['lon'].values
    lat = ds['lat'].values
    time = ds['time'].values
    dates = pd.to_datetime(time)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    df_rsds = pd.DataFrame({
        'lon': np.tile(lon_grid.flatten(), len(dates)),
        'lat': np.tile(lat_grid.flatten(), len(dates)),
        'value': rsds_data.flatten(),
        'date': np.repeat(dates, len(lon) * len(lat))
    })
    df_rsds['parameter'] = 'solar_radiation'
    df_rsds['year'] = df_rsds['date'].dt.year
    df_rsds = df_rsds[['parameter', 'date', 'lon', 'lat', 'value', 'year']]
    df_rsds = df_rsds.sort_values(by=['lon', 'lat', 'year', 'date']).reset_index(drop=True)
    print("Loaded solar radiation (RSDS) data.")
    return df_rsds

def load_temperature_data():
    """Load ERA5-Land air temperature data and return a DataFrame with a datetime index and year."""
    tas_file = os.path.join(CLIMATE_PROJ_DIR, "input_ERA5_LAND.nc")
    ds = xr.open_dataset(tas_file)
    # In your notebook the temperature variable is 'temperature'
    tas_data = ds['temperature'].values
    lon = ds['lon'].values
    lat = ds['lat'].values
    time = ds['time'].values
    dates = pd.to_datetime(time)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    df_tas = pd.DataFrame({
        'lon': np.tile(lon_grid.flatten(), len(dates)),
        'lat': np.tile(lat_grid.flatten(), len(dates)),
        'value': tas_data.flatten(),
        'date': np.repeat(dates, len(lon) * len(lat))
    })
    df_tas['parameter'] = 'temperature'
    df_tas['year'] = df_tas['date'].dt.year
    df_tas = df_tas[['parameter', 'date', 'lon', 'lat', 'value', 'year']]
    df_tas = df_tas.sort_values(by=['lon', 'lat', 'year', 'date']).reset_index(drop=True)
    print("Loaded air temperature data.")
    return df_tas

def compute_yearly_stats(df, agg_func):
    """Group the DataFrame by year, lat, lon and compute an aggregate of 'value'."""
    df_yearly = df.groupby(['year', 'lat', 'lon'], as_index=False)['value'].agg(agg_func)
    return df_yearly

def round_coordinates(dfs, decimals=2):
    """Round 'lat' and 'lon' columns in a list of DataFrames."""
    for df in dfs:
        df['lat'] = df['lat'].round(decimals)
        df['lon'] = df['lon'].round(decimals)
    return dfs

#############################################
# Prediction and Post-Processing Functions
#############################################
def predict_pv_suitability_past(model, df_rsds_max, df_rsds_min, df_tas_mean, df_slope):
    """Generate PV suitability predictions and save them in a task-specific directory."""
    config, task_id, user_id = load_config(INPUT_JSON_PATH)

    # Create task-specific directory
    task_prediction_dir = os.path.join(PREDICTION_DIR, f"{task_id}_{user_id}")
    ensure_directory(task_prediction_dir)  # Ensure the directory exists

    YEARS, LUSA, LATS, LONS = [], [], [], []

    # Loop over groups defined by lat and lon from the RSDS max dataset
    for (lat, lon), group in df_rsds_max.groupby(['lat', 'lon']):
        for i in range(len(group)):
            year = group['year'].values[i]
            try:
                rsds_max = df_rsds_max[(df_rsds_max['lat'] == lat) & 
                                       (df_rsds_max['lon'] == lon) & 
                                       (df_rsds_max['year'] == year)]['value'].values[0]
                rsds_min = df_rsds_min[(df_rsds_min['lat'] == lat) & 
                                       (df_rsds_min['lon'] == lon) & 
                                       (df_rsds_min['year'] == year)]['value'].values[0]
                tas_mean = df_tas_mean[(df_tas_mean['lat'] == lat) & 
                                       (df_tas_mean['lon'] == lon) & 
                                       (df_tas_mean['year'] == year)]['value'].values[0]
                slope_val = df_slope[(df_slope['lat'] == lat) & (df_slope['lon'] == lon)]['value'].values[0]
            except IndexError:
                continue  # Skip if any data is missing
            
            X_test = pd.DataFrame([[rsds_max, rsds_min, tas_mean, tas_mean, slope_val]])
            prediction = np.argmax(model.predict(X_test), axis=1)

            print(f"Prediction for lat: {lat}, lon: {lon}, year: {year} -> {LUSA_SCORES[prediction[0]]}")

            YEARS.append(year)
            LUSA.append(LUSA_SCORES[prediction[0]])
            LATS.append(lat)
            LONS.append(lon)
    
    df_predictions = pd.DataFrame({
        'year': YEARS,
        'score': LUSA,
        'lat': LATS,
        'lon': LONS
    })

    # Save inside the correct folder
    prediction_file = os.path.join(task_prediction_dir, "PAST_PV_SUITABILITY_PREDICTIONS.csv")
    df_predictions.to_csv(prediction_file, index=False)
    print(f"Predictions saved in: {prediction_file}")

    return df_predictions, task_prediction_dir  # Return correct path


def apply_conditions_to_predictions(pred_file, task_prediction_dir):
    """Adjust scores based on config.json and save updated predictions in task-specific directory."""
    
    # Read the predictions CSV
    df = pd.read_csv(pred_file)

    config, task_id, user_id = load_config(INPUT_JSON_PATH)

    proximity_to_powerlines = config["model_parameters"].get("proximity_to_powerlines", 10)
    road_network_accessibility = config["model_parameters"].get("road_network_accessibility", 10)

    for index, row in df.iterrows():
        score = float(row['score'])
        if proximity_to_powerlines < 1 and road_network_accessibility < 1:
            score *= 0.95
        elif 1 < proximity_to_powerlines < 2 and 1 < road_network_accessibility < 2:
            score *= 0.7
        elif 2 < proximity_to_powerlines < 10 and 2 < road_network_accessibility < 3:
            score *= 0.5
        elif proximity_to_powerlines > 10 and road_network_accessibility > 4:
            score = 0
        df.at[index, 'score'] = score

    # Save in the correct task-specific directory
    updated_file = os.path.join(task_prediction_dir, "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    df.to_csv(updated_file, index=False)
    print(f"Updated predictions saved in: {updated_file}")

    return df


def calculate_pv_yield_past():
    """
    Calculate the annual electricity savings from PV for each prediction (for years 2010-2020)
    using ERA5-Land solar data.
    """
    config, task_id, user_id = load_config(INPUT_JSON_PATH)

    # Read parameters from input_mongo_past.json
    area = config["model_parameters"].get("area", 2.4)  # Default 2.4 m²
    electricity_rate = config["model_parameters"].get("electricity_rate", 0.12)  # Default €/kWh

    rsds_file = os.path.join(CLIMATE_PROJ_DIR, "input_ERA5_LAND_SOLAR.nc")
    ds_rsds = xr.open_dataset(rsds_file)

    # Task-specific directory
    task_prediction_dir = os.path.join(PREDICTION_DIR, f"{task_id}_{user_id}")
    ensure_directory(task_prediction_dir)

    pred_file = os.path.join(task_prediction_dir, "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    df_pred = pd.read_csv(pred_file)
    df_pred = df_pred[df_pred['year'].between(2010, 2020)]

    daylight_cache = {}
    results = []
    
    def get_daylight_hours(lat, lon, date):
        times = pd.date_range(date, date + timedelta(days=1), freq='H', tz='UTC')
        solar_pos = solarposition.get_solarposition(times, lat, lon)
        return (solar_pos['apparent_elevation'] > 0).sum()
    
    def daily_energy(rsds, hours):
        return rsds * area * (EFFICIENCY) * hours  # in Wh

    for _, row in df_pred.iterrows():
        year = int(row['year'])
        lat = row['lat']
        lon = row['lon']
        annual_savings = 0
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        rsds_slice = ds_rsds.sel(time=slice(start_date, end_date))
        rsds_values = rsds_slice.sel(lat=lat, lon=lon, method="nearest")['var169'].values
        
        for day_offset in range((end_date - start_date).days + 1):
            current_date = start_date + timedelta(days=day_offset)
            try:
                rsds_value = rsds_values[day_offset]
            except IndexError:
                continue
            day_of_year = current_date.timetuple().tm_yday
            key = (lat, day_of_year)
            if key in daylight_cache:
                hours = daylight_cache[key]
            else:
                hours = get_daylight_hours(lat, lon, current_date)
                daylight_cache[key] = hours
            daily_output = daily_energy(rsds_value, hours)
            annual_savings += daily_output * electricity_rate
        results.append({
            'year': year,
            'lat': lat,
            'lon': lon,
            'annual_electricity_savings_€': round(annual_savings, 2)
        })

    df_yield = pd.DataFrame(results)

    # Save inside the task-specific folder
    yield_file = os.path.join(task_prediction_dir, "PAST_PV_YIELD.csv")
    df_yield.to_csv(yield_file, index=False)
    print(f"PV Yield data saved to: {yield_file}")

    return df_yield, yield_file

