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
# Define Directories and Constants
#############################################
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(MAIN_DIR, "INPUT")
CLIMATE_PROJ_DIR = os.path.join(DATA_DIR, "CLIMATE_PROJECTIONS")
SLOPE_DIR = os.path.join(DATA_DIR, "SLOPE")
TRAINING_DIR = os.path.join(MAIN_DIR, "TRAINING")
PREDICTION_DIR = os.path.join(MAIN_DIR, "PREDICTIONS/CSVs")
PLOTS_DIR = os.path.join(MAIN_DIR, "PLOTS")

# Ensure output directories exist
ensure_directory(PREDICTION_DIR)
ensure_directory(PLOTS_DIR)

# Constants for PV calculations
AREA = 2.4  # m² (panel size)
EFFICIENCY = 18.5 / 100  # 18.5% efficiency
ELECTRICITY_RATE = 0.12  # €/kWh
CLASS_LABELS = ['0', '1', '2', '3', '4', '5']
LUSA_SCORES = [0, 40, 60, 85, 95, 100]

# Path to configuration file
INPUT_JSON_PATH = os.path.join(MAIN_DIR, "input_mongo_pv_future.json")

#############################################
# Configuration Loader
#############################################
def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config, config.get("task_id"), config.get("user_id")

#############################################
# Training and Model Functions
#############################################
def load_training_data(training_file):
    """Load training data from CSV file."""
    print(f"Loading training data from: {training_file}")
    df = pd.read_csv(training_file, sep=';')
    X = df.iloc[:, 1:12]
    y = np.floor(df.iloc[:, 0] / 20).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, config, num_features=5):
    """Train the Feedforward Neural Network and save training history with scenario-based filename."""
    if "scenario" not in config["model_parameters"]:
        raise KeyError("The 'scenario' key is missing from 'model_parameters' in the JSON file.")

    scenario = config["model_parameters"]["scenario"]

    print(f"Training the Feedforward Neural Network for Scenario {scenario}...")
    # Build the model architecture
    input_layer = tf.keras.layers.Input(shape=(num_features,))
    dense1 = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
    dense3 = tf.keras.layers.Dense(32, activation='relu')(dense2)
    output_layer = tf.keras.layers.Dense(6, activation='sigmoid')(dense3)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)
    
    # Generate the filename based on the scenario
    history_file = os.path.join(TRAINING_DIR, f"{scenario}_history_train.txt")
    with open(history_file, 'w') as f:
        f.write(str(history.history))
    
    print(f"Training metrics saved in: {history_file}")
    return model, history



#############################################
# Data Loading Functions
#############################################
def load_netcdf(file_path, variable):
    """Load NetCDF file and extract the specified variable along with lat, lon, and time."""
    print(f"Loading NetCDF file: {file_path} for variable: {variable}")
    ds = xr.open_dataset(file_path)
    data = ds[variable].values
    lat = ds['lat'].values
    lon = ds['lon'].values
    time = ds['time'].values if 'time' in ds else None
    return data, lat, lon, time

def process_climate_data(data_dir, variable, scenario):
    """Process NetCDF climate data and return a structured DataFrame."""
    
    # Use fixed filename for slope data (scenario is not relevant for slope)
    if variable.lower() == "slope":
        file_path = os.path.join(data_dir, "SLOPE.nc")
    else:
        file_path = os.path.join(data_dir, f"{variable}_rcp{scenario}_2021_2100.nc")

    print(f"Processing climate data: {file_path}")
    data, lat, lon, time = load_netcdf(file_path, variable)

    # Handle missing time dimension by using a default placeholder date
    dates = pd.to_datetime(time) if time is not None else pd.to_datetime(['1900-01-01'])
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    df = pd.DataFrame({
        'lon': np.tile(lon_grid.flatten(), len(dates)),
        'lat': np.tile(lat_grid.flatten(), len(dates)),
        'value': data.flatten(),
        'date': np.repeat(dates, len(lon) * len(lat))
    })

    df['parameter'] = variable
    df['year'] = df['date'].dt.year
    return df.sort_values(by=['lon', 'lat', 'year', 'date']).reset_index(drop=True)

#############################################
# Prediction and Post-Processing Functions
#############################################
def predict_pv_suitability_future(model, df_rsds_max, df_rsds_min, df_tas_max, df_tas_min, df_slope, config_path):
    """Generate PV suitability predictions and save them in a task-specific directory."""
    
    # Load configuration and extract required metadata
    config, task_id, user_id = load_config(config_path)

    # Ensure scenario is present in the JSON
    if "scenario" not in config["model_parameters"]:
        raise KeyError("The 'scenario' key is missing from 'model_parameters' in the JSON file.")

    scenario = config["model_parameters"]["scenario"]

    # Create task-specific directory
    task_prediction_dir = os.path.join(PREDICTION_DIR, f"{task_id}_{user_id}")
    ensure_directory(task_prediction_dir)

    print(f"Running PV Suitability Predictions for Scenario RCP{scenario}...")

    results = {'year': [], 'score': [], 'lat': [], 'lon': []}

    # Loop over each unique (lat, lon) location
    for (lat, lon), point in df_rsds_max.groupby(['lat', 'lon']):
        for i in range(len(point)):
            year = point['year'].values[i]
            try:
                rsds_max = df_rsds_max.loc[(df_rsds_max['lat'] == lat) & (df_rsds_max['lon'] == lon) & (df_rsds_max['year'] == year), 'value'].values[0]
                rsds_min = df_rsds_min.loc[(df_rsds_min['lat'] == lat) & (df_rsds_min['lon'] == lon) & (df_rsds_min['year'] == year), 'value'].values[0]
                tas_max = df_tas_max.loc[(df_tas_max['lat'] == lat) & (df_tas_max['lon'] == lon) & (df_tas_max['year'] == year), 'value'].values[0]
                tas_min = df_tas_min.loc[(df_tas_min['lat'] == lat) & (df_tas_min['lon'] == lon) & (df_tas_min['year'] == year), 'value'].values[0]
                slope = df_slope.loc[(df_slope['lat'] == lat) & (df_slope['lon'] == lon), 'value'].values[0]
            except IndexError:
                continue  # Skip if any required data is missing

            # Construct input feature vector
            X_test = pd.DataFrame([[rsds_max, rsds_min, tas_max, tas_min, slope]])
            prediction = np.argmax(model.predict(X_test), axis=1)

            print(f"Prediction for lat: {lat}, lon: {lon}, year: {year} -> {LUSA_SCORES[prediction[0]]}")

            results['year'].append(year)
            results['score'].append(LUSA_SCORES[prediction[0]])
            results['lat'].append(lat)
            results['lon'].append(lon)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Save results in the task-specific folder with scenario-specific naming
    output_path = os.path.join(task_prediction_dir, f"RCP{scenario}_PV_SUITABILITY_PREDICTIONS.csv")
    df_results.to_csv(output_path, index=False)
    
    print(f"PV Suitability Predictions saved to: {output_path}")
    
    return df_results, task_prediction_dir


def apply_conditions_to_predictions(task_prediction_dir, config_path):
    """Adjust scores based on config.json and save updated predictions in a task-specific directory."""
    
    # Load configuration and extract necessary parameters
    config, task_id, user_id = load_config(config_path)

    # Ensure scenario is present in the JSON
    if "scenario" not in config["model_parameters"]:
        raise KeyError("The 'scenario' key is missing from 'model_parameters' in the JSON file.")

    scenario = config["model_parameters"]["scenario"]
    proximity_to_powerlines = config["model_parameters"].get("proximity_to_powerlines", 0.1)
    road_network_accessibility = config["model_parameters"].get("road_network_accessibility", 0.1)

    # Create task-specific directory
    task_prediction_dir = os.path.join(PREDICTION_DIR, f"{task_id}_{user_id}")
    ensure_directory(task_prediction_dir)

    # Define the predictions CSV file (before applying conditions)
    pred_file = os.path.join(task_prediction_dir, f"RCP{scenario}_PV_SUITABILITY_PREDICTIONS.csv")

    # Ensure the predictions file exists before processing
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    # Read the predictions CSV
    df = pd.read_csv(pred_file)

    # Apply conditions to adjust the suitability score
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

    # Save the updated predictions inside the correct folder with scenario-specific naming
    updated_file = os.path.join(task_prediction_dir, f"RCP{scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    df.to_csv(updated_file, index=False)
    
    print(f"Updated predictions saved in: {updated_file}")

    return df

def calculate_pv_yield_future(config_path):
    """
    Calculate the annual electricity savings from PV for each prediction using projected climate data.
    """
    # Load configuration and extract necessary parameters
    config, task_id, user_id = load_config(config_path)

    # Ensure scenario is present in the JSON
    if "scenario" not in config["model_parameters"]:
        raise KeyError("The 'scenario' key is missing from 'model_parameters' in the JSON file.")

    scenario = config["model_parameters"]["scenario"]

    # Read parameters from input_mongo_pv_future.json
    area = config["model_parameters"].get("area", 2.4)  # Default 2.4 m²
    electricity_rate = config["model_parameters"].get("electricity_rate", 0.12)  # Default €/kWh

    # Construct the file path for the RSDS NetCDF dataset based on the scenario
    rsds_file = os.path.join(CLIMATE_PROJ_DIR, f"rsds_rcp{scenario}_2021_2100.nc")
    ds_rsds = xr.open_dataset(rsds_file)

    # Task-specific directory
    task_prediction_dir = os.path.join(PREDICTION_DIR, f"{task_id}_{user_id}")
    ensure_directory(task_prediction_dir)

    # Define the predictions file that contains PV suitability scores
    pred_file = os.path.join(task_prediction_dir, f"RCP{scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    # Load predictions
    df_pred = pd.read_csv(pred_file)

    # Process only predictions for the future years (2021-2100)
    df_pred = df_pred[df_pred['year'].between(2021, 2100)]

    # Caching daylight hours calculations
    daylight_cache = {}
    results = []
    
    def get_daylight_hours(lat, lon, date):
        times = pd.date_range(date, date + timedelta(days=1), freq='H', tz='UTC')
        solar_pos = solarposition.get_solarposition(times, lat, lon)
        return (solar_pos['apparent_elevation'] > 0).sum()
    
    def daily_energy(rsds, hours):
        return rsds * area * (EFFICIENCY) * hours  # in Wh

    # Iterate over predictions
    for _, row in df_pred.iterrows():
        year = int(row['year'])
        lat = row['lat']
        lon = row['lon']
        annual_savings = 0

        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)

        # Extract RSDS values from the dataset for the given location and year
        rsds_slice = ds_rsds.sel(time=slice(start_date, end_date))
        rsds_values = rsds_slice.sel(lat=lat, lon=lon, method="nearest")['rsds'].values  # Ensure 'rsds' is the correct variable name in the dataset

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
            annual_savings += daily_output * electricity_rate  # Convert Wh to monetary savings

        results.append({
            'year': year,
            'lat': lat,
            'lon': lon,
            'annual_electricity_savings_€': round(annual_savings, 2)
        })

    df_yield = pd.DataFrame(results)

    # Save inside the task-specific folder with scenario-specific naming
    yield_file = os.path.join(task_prediction_dir, f"RCP{scenario}_PV_YIELD.csv")
    df_yield.to_csv(yield_file, index=False)
    print(f"PV Yield data saved to: {yield_file}")

    return df_yield, yield_file

