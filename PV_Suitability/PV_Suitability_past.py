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

#############################################
# Training and Model Functions
#############################################
def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)

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
    history_file = os.path.join(TRAINING_DIR, "RCP26_history_train.txt")
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
    """
    For each unique (lat,lon) and each year in the RSDS max DataFrame,
    extract rsds max, rsds min, and temperature (mean) along with slope;
    then run the model prediction and save results.
    """
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
                continue  # skip if any data is missing
            
            # Construct input vector (order: rsds_max, rsds_min, tas_mean, tas_mean, slope)
            # Here we use tas_mean for both lower and upper since your notebook used the same value.
            X_test = pd.DataFrame([[rsds_max, rsds_min, tas_mean, tas_mean, slope_val]])
            prediction = np.argmax(model.predict(X_test), axis=1)

            print(f"Prediction for lat: {lat}, lon: {lon}, year: {year} -> {LUSA_SCORES[prediction[0]]}")
            # Adjust the predicted class to index into LUSA_SCORES.
            # (Assumes classes 0-5 map directly to LUSA_SCORES.)
            YEARS.append(year)
            LUSA.append(LUSA_SCORES[prediction[0]])
            LATS.append(lat)
            LONS.append(lon)
    
    # Create a DataFrame for predictions and save to CSV.
    df_predictions = pd.DataFrame({
        'year': YEARS,
        'score': LUSA,
        'lat': LATS,
        'lon': LONS
    })
    output_file = os.path.join(PREDICTION_DIR, "PAST_PV_SUITABILITY_PREDICTIONS.csv")
    df_predictions.to_csv(output_file, index=False)
    print(f"Past PV Suitability Predictions saved to: {output_file}")
    return df_predictions

def apply_conditions_to_predictions(pred_file):
    """
    Read the predictions CSV, adjust the 'score' based on proximity and road conditions,
    and save an updated CSV.
    """
    df = pd.read_csv(pred_file)
    for index, row in df.iterrows():
        score = float(row['score'])
        # Apply conditions as in your notebook:
        if proximity_to_powerlines < 1 and road_network_accessibility < 1:
            score *= 0.95
        elif 1 < proximity_to_powerlines < 2 and 1 < road_network_accessibility < 2:
            score *= 0.7
        elif 2 < proximity_to_powerlines < 10 and 2 < road_network_accessibility < 3:
            score *= 0.5
        elif proximity_to_powerlines > 10 and road_network_accessibility > 4:
            score = 0
        df.at[index, 'score'] = score
    updated_file = os.path.join(PREDICTION_DIR, "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    df.to_csv(updated_file, index=False)
    print(f"Updated PV Suitability Predictions saved to: {updated_file}")
    return df

#############################################
# Plotting Functions
#############################################
def plot_predictions_time_series(main_dir, plots_dir, filename="PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv", crop_type="PV"):
    """Plot time series for each unique location using Plotly."""
    pred_file = os.path.join(main_dir, "PREDICTIONS/CSVs", filename)
    print(f"Loading predictions from: {pred_file}")
    data = pd.read_csv(pred_file)
    # Group by lat and lon
    grouped = data.groupby(['lat', 'lon'])
    for (lat, lon), group in grouped:
        years = group['year'].values
        scores = group['score'].values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=scores, mode="lines+markers",
                                 marker_color='mediumspringgreen', name='Land Suitability'))
        fig.update_layout(title={
                            'text': f'Land Suitability Analysis for {crop_type}<br>Location: {lat}, {lon}',
                            'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                          yaxis_title='Suitability Score (%)',
                          xaxis_title='Year',
                          paper_bgcolor='rgb(243,243,243)',
                          plot_bgcolor='rgb(243,243,243)')
        # Save each figure to file
        save_path = os.path.join(plots_dir, f'prediction_{lat}_{lon}.png')
        fig.write_image(save_path)
        

def plot_training_metrics(history, plots_dir):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train'], loc='upper left')
    acc_path = os.path.join(plots_dir, 'training_accuracy.png')
    plt.savefig(acc_path)
    plt.show()
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train'], loc='upper left')
    loss_path = os.path.join(plots_dir, 'training_loss.png')
    plt.savefig(loss_path)


def plot_confusion_matrix(model, X_test, y_test, plots_dir):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    conf_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(conf_path)
    print(classification_report(y_test, y_pred_classes, labels=np.arange(len(CLASS_LABELS)),
                                target_names=CLASS_LABELS, zero_division=0))

def plot_high_variance_point(df_predictions, df_rsds_yearly_min, df_rsds_yearly_max, df_tas_yearly_min, df_tas_yearly_max, plots_dir):
    def find_closest(df, lat, lon):
        df['lat_diff'] = np.abs(df['lat'] - lat)
        df['lon_diff'] = np.abs(df['lon'] - lon)
        df['distance'] = df['lat_diff'] + df['lon_diff']
        closest = df.loc[df['distance'].idxmin()]
        return df[(df['lat'] == closest['lat']) & (df['lon'] == closest['lon'])]
    
    df_predictions['score'] = df_predictions['score'].astype(float)
    df_clean = df_predictions.dropna(subset=['score'])
    variance_df = df_clean.groupby(['lat', 'lon'])['score'].var().reset_index().dropna(subset=['score'])
    if variance_df.empty:
        raise ValueError("No valid variance found.")
    max_var = variance_df.loc[variance_df['score'].idxmax()]
    max_lat, max_lon = max_var['lat'], max_var['lon']
    print(f"Point with highest variance: (lat: {max_lat}, lon: {max_lon})")
    
    df_point_rsds_min = find_closest(df_rsds_yearly_min, max_lat, max_lon)
    df_point_rsds_max = find_closest(df_rsds_yearly_max, max_lat, max_lon)
    df_point_tasmin = find_closest(df_tas_yearly_min, max_lat, max_lon)
    df_point_tasmax = find_closest(df_tas_yearly_max, max_lat, max_lon)
    df_point_suit = df_clean[(df_clean['lat'] == max_lat) & (df_clean['lon'] == max_lon)]
    
    # Merge data on 'year'
    df_combined = pd.merge(df_point_rsds_min[['year','value']], df_point_rsds_max[['year','value']], on='year', suffixes=('_rsds_min','_rsds_max'))
    df_combined = pd.merge(df_combined, df_point_tasmin[['year','value']], on='year')
    df_combined = pd.merge(df_combined, df_point_tasmax[['year','value']], on='year')
    df_combined = pd.merge(df_combined, df_point_suit[['year','score']], on='year')
    df_combined.columns = ['year', 'rsds_min', 'rsds_max', 'tasmin', 'tasmax', 'suitability']
    print("Combined data for high variance point:")
    print(df_combined)
    
    fig, axs = plt.subplots(5, 1, figsize=(10,18), sharex=True)
    axs[0].plot(df_combined['year'], df_combined['rsds_min'], color='blue')
    axs[0].set_ylabel('RSDS Min (W/m²)')
    axs[0].set_title(f'RSDS Min at (lat: {max_lat}, lon: {max_lon})')
    axs[0].grid(True)
    axs[1].plot(df_combined['year'], df_combined['rsds_max'], color='purple')
    axs[1].set_ylabel('RSDS Max (W/m²)')
    axs[1].set_title(f'RSDS Max at (lat: {max_lat}, lon: {max_lon})')
    axs[1].grid(True)
    axs[2].plot(df_combined['year'], df_combined['tasmin'], color='green')
    axs[2].set_ylabel('Temp Min (°C)')
    axs[2].set_title(f'Temperature Min at (lat: {max_lat}, lon: {max_lon})')
    axs[2].grid(True)
    axs[3].plot(df_combined['year'], df_combined['tasmax'], color='red')
    axs[3].set_ylabel('Temp Max (°C)')
    axs[3].setTitle(f'Temperature Max at (lat: {max_lat}, lon: {max_lon})')
    axs[3].grid(True)
    axs[4].plot(df_combined['year'], df_combined['suitability'], color='orange', linestyle='--')
    axs[4].set_ylabel('Suitability')
    axs[4].set_title(f'Suitability at (lat: {max_lat}, lon: {max_lon})')
    axs[4].grid(True)
    
    ticks = np.arange(df_combined['year'].min(), df_combined['year'].max()+1, 5)
    for ax in axs:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45)
    axs[4].set_xlabel('Year')
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, 'high_variance_point.png')
    plt.savefig(save_path)

def plot_continuous_spatial(main_dir, plots_dir):
    # Load updated predictions
    csv_file = os.path.join(main_dir, "PREDICTIONS/CSVs", "PAST_PV_SUITABILITY_PREDICTIONS.csv")
    df_pred = pd.read_csv(csv_file)
    df_pred['score'] = pd.to_numeric(df_pred['score'], errors='coerce')
    yearly_std = df_pred.groupby('year')['score'].std()
    year_max = yearly_std.idxmax()
    print(f"Year with maximum fluctuation: {year_max}")
    df_year = df_pred[df_pred['year'] == year_max]
    gdf_points = gpd.GeoDataFrame(df_year, geometry=[Point(xy) for xy in zip(df_year.lon, df_year.lat)], crs="EPSG:4326")
    geojson_path = os.path.join(MAIN_DIR, "PILOTS", "geoBoundaries-CYP-ADM0.geojson")
    gdf_boundary = gpd.read_file(geojson_path)
    gdf_clipped = gpd.sjoin(gdf_points, gdf_boundary, predicate='within')
    df_clipped = pd.DataFrame(gdf_clipped.drop(columns="geometry"))
    
    lon_min, lon_max = 32.1, 34.7
    lat_min, lat_max = 34.4, 35.8
    grid_lon, grid_lat = np.mgrid[lon_min:lon_max:400j, lat_min:lat_max:400j]
    
    grid_scores = griddata(
        (df_clipped['lon'], df_clipped['lat']),
        df_clipped['score'],
        (grid_lon, grid_lat),
        method='linear'
    )
    grid_scores = np.where(np.isnan(grid_scores),
                           griddata((df_clipped['lon'], df_clipped['lat']),
                                    df_clipped['score'],
                                    (grid_lon, grid_lat),
                                    method='nearest'),
                           grid_scores)
    
    grid_points = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in zip(grid_lon.flatten(), grid_lat.flatten())], crs="EPSG:4326")
    clipped_points = gpd.clip(grid_points, gdf_boundary)
    clipped_scores = grid_scores.flatten()[clipped_points.index]
    
    study_area_path = os.path.join(MAIN_DIR, "PILOTS", "Study_area_Cyprus.geojson")
    gdf_study_area = gpd.read_file(study_area_path)
    
    fig, ax = plt.subplots(figsize=(12,8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'))
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0)
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0)
    
    sc = ax.scatter(clipped_points.geometry.x, clipped_points.geometry.y, c=clipped_scores, cmap='viridis',
                    s=15, alpha=0.7, vmin=0, vmax=100, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical')
    cbar.set_label('Suitability Score')
    ax.add_geometries(gdf_study_area['geometry'], crs=ccrs.PlateCarree(),
                      edgecolor='red', facecolor='none', linewidth=2, alpha=0.7)
    study_area_patch = Patch(edgecolor='red', facecolor='none', label='Study Area', linewidth=2)
    ax.legend(handles=[study_area_patch], loc='lower right')
    plt.title(f'Continuous Spatial Suitability Map in {year_max}')
    map_path = os.path.join(plots_dir, 'continuous_map.png')
    plt.savefig(map_path)


#############################################
# PV Yield Calculation Function (Past)
#############################################
def calculate_pv_yield_past():
    """
    Calculate the annual electricity savings from PV for each prediction (for years 2010-2020)
    using ERA5-Land solar data.
    """
    rsds_file = os.path.join(CLIMATE_PROJ_DIR, "input_ERA5_LAND_SOLAR.nc")
    ds_rsds = xr.open_dataset(rsds_file)
    pred_file = os.path.join(PREDICTION_DIR, "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    df_pred = pd.read_csv(pred_file)
    df_pred = df_pred[df_pred['year'].between(2010, 2020)]
    daylight_cache = {}
    results = []
    
    def get_daylight_hours(lat, lon, date):
        times = pd.date_range(date, date + timedelta(days=1), freq='H', tz='UTC')
        solar_pos = solarposition.get_solarposition(times, lat, lon)
        return (solar_pos['apparent_elevation'] > 0).sum()
    
    def daily_energy(rsds, hours):
        return rsds * AREA * EFFICIENCY * hours  # in Wh

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
            annual_savings += daily_output * ELECTRICITY_RATE
        results.append({
            'year': year,
            'lat': lat,
            'lon': lon,
            'annual_electricity_savings_€': round(annual_savings, 2)
        })
    df_yield = pd.DataFrame(results)
    yield_file = os.path.join(PREDICTION_DIR, "PAST_PV_YIELD.csv")
    df_yield.to_csv(yield_file, index=False)
    print(f"PV Yield data saved to: {yield_file}")

#############################################
# Main Script
#############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration JSON file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    proximity_to_powerlines = config.get("proximity_to_powerlines", 0)
    road_network_accessibility = config.get("road_network_accessibility", 0)
    
    print("Running unified PV Suitability Past Module...")
    
    training_file = os.path.join(TRAINING_DIR, "PV_suitability_data_TRAIN.csv")
    X_train, X_test, y_train, y_test = load_training_data(training_file)
    model, history = train_model(X_train, y_train)
    
    df_slope = load_slope_data()
    df_rsds = load_rsds_data()
    df_tas = load_temperature_data()
    
    df_rsds_yearly_min = compute_yearly_stats(df_rsds, np.min)
    df_rsds_yearly_max = compute_yearly_stats(df_rsds, np.max)
    df_tas_yearly_mean = compute_yearly_stats(df_tas, np.mean)
    df_tas_yearly_min = compute_yearly_stats(df_tas, np.min)
    df_tas_yearly_max = compute_yearly_stats(df_tas, np.max)
    
    round_coordinates([df_rsds_yearly_min, df_rsds_yearly_max, df_tas_yearly_mean, df_tas_yearly_min, df_tas_yearly_max, df_slope])
    
    if proximity_to_powerlines == 0 and road_network_accessibility == 0:
        print("PV suitability cannot be calculated due to zero proximity parameters.")
        sys.exit(1)
    else:
        df_predictions = predict_pv_suitability_past(model, df_rsds_yearly_max, df_rsds_yearly_min, df_tas_yearly_mean, df_slope)
    
    pred_file = os.path.join(PREDICTION_DIR, "PAST_PV_SUITABILITY_PREDICTIONS.csv")
    df_predictions_updated = apply_conditions_to_predictions(pred_file)
    
    calculate_pv_yield_past()
    
    print("Generating plots...")
    plot_predictions_time_series(MAIN_DIR, PLOTS_DIR, filename="PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv", crop_type="PV")
    plot_training_metrics(history, PLOTS_DIR)
    plot_confusion_matrix(model, X_test, y_test, PLOTS_DIR)
    plot_high_variance_point(df_predictions_updated, df_rsds_yearly_min, df_rsds_yearly_max, df_tas_yearly_min, df_tas_yearly_max, PLOTS_DIR)
    plot_continuous_spatial(MAIN_DIR, PLOTS_DIR)
    
    print("Unified PV Suitability Past Module execution complete.")
