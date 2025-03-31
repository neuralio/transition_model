import os
import json
import argparse
import xarray as xr
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.model_selection import train_test_split
from pvlib import solarposition

# Additional imports for plotting and analysis
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Patch


#############################################
# Utility Functions
#############################################
def ensure_directory(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

#############################################
# Define Directories
#############################################
# Define MAIN_DIR as the script's directory
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# Define subdirectories
DATA_DIR = os.path.join(MAIN_DIR, "INPUT")
CLIMATE_PROJ_DIR = os.path.join(DATA_DIR, "CLIMATE_PROJECTIONS")
SLOPE_DIR = os.path.join(DATA_DIR, "SLOPE")
TRAINING_DIR = os.path.join(MAIN_DIR, "TRAINING")
PREDICTION_DIR = os.path.join(MAIN_DIR, "PREDICTIONS/CSVs")

# Define directory for plots and ensure it exists
PLOTS_DIR = os.path.join(MAIN_DIR, "PLOTS")
ensure_directory(PLOTS_DIR)


#############################################
# Constants
#############################################
AREA = 2.4  # m² (panel size)
EFFICIENCY = 18.5 / 100  # 18.5% efficiency
ELECTRICITY_RATE = 0.12  # €/kWh
CLASS_LABELS = ['0', '1', '2', '3', '4', '5']
LUSA_SCORES = [0, 40, 60, 85, 95, 100]

#############################################
# Core Functions (Data loading, training, etc.)
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
    print("Training the Neural Network Model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(num_features,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(6, activation='sigmoid')
    ])
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)
    print("Model Training Complete!")
    # Return both model and training history (needed for plotting metrics)
    return model, history

def load_netcdf(file_path, variable):
    print(f"Loading NetCDF file: {file_path} for variable: {variable}")
    ds = xr.open_dataset(file_path)
    data = ds[variable].values
    lat = ds['lat'].values
    lon = ds['lon'].values
    time = ds['time'].values if 'time' in ds else None
    return data, lat, lon, time

# def process_climate_data(data_dir, variable, scenario):
#     # For the slope data, ignore the scenario and use the fixed file name.
#     if variable.lower() == "slope":
#         file_path = os.path.join(data_dir, "SLOPE.nc")
#     else:
#         file_path = os.path.join(data_dir, f"{variable}_rcp{scenario}_2021_2100.nc")
#     print(f"Processing climate data: {file_path}")
#     data, lat, lon, time = load_netcdf(file_path, variable)
    
#     lon_grid, lat_grid = np.meshgrid(lon, lat)
#     dates = pd.to_datetime(time)
    
#     df = pd.DataFrame({
#         'lon': np.tile(lon_grid.flatten(), len(dates)),
#         'lat': np.tile(lat_grid.flatten(), len(dates)),
#         'value': data.flatten(),
#         'date': np.repeat(dates, len(lon) * len(lat))
#     })
    
#     df['parameter'] = variable
#     df['year'] = df['date'].dt.year
#     return df.sort_values(by=['lon', 'lat', 'year', 'date']).reset_index(drop=True)

def process_climate_data(data_dir, variable, scenario):
    # For slope, ignore scenario and use the fixed file name.
    if variable.lower() == "slope":
        file_path = os.path.join(data_dir, "SLOPE.nc")
    else:
        file_path = os.path.join(data_dir, f"{variable}_rcp{scenario}_2021_2100.nc")
    print(f"Processing climate data: {file_path}")
    data, lat, lon, time = load_netcdf(file_path, variable)
    
    # If the dataset has no time dimension, create a dummy time array.
    if time is None:
        dates = pd.to_datetime(['1900-01-01'])
    else:
        dates = pd.to_datetime(time)
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    df = pd.DataFrame({
        'lon': np.tile(lon_grid.flatten(), len(dates)),
        'lat': np.tile(lat_grid.flatten(), len(dates)),
        'value': data.flatten(),
        'date': np.repeat(dates, len(lon) * len(lat))
    })
    
    df['parameter'] = variable
    df['year'] = pd.to_datetime(df['date']).dt.year
    return df.sort_values(by=['lon', 'lat', 'year', 'date']).reset_index(drop=True)


def apply_constraints(df, proximity, road_access):
    for index, row in df.iterrows():
        score = float(row['score'])
        if proximity < 1 and road_access < 1:
            score *= 0.95
        elif 1 < proximity < 2 and 1 < road_access < 2:
            score *= 0.7
        elif 2 < proximity < 10 and 2 < road_access < 3:
            score *= 0.5
        elif proximity > 10 and road_access > 4:
            score = 0
        df.at[index, 'score'] = score
    return df

def predict_pv_suitability(model, df_rsds_max, df_rsds_min, df_tas_max, df_tas_min, df_slope, scenario, output_dir):
    print("Running PV Suitability Predictions...")
    results = {'year': [], 'score': [], 'lat': [], 'lon': []}
    
    for (lat, lon), point in df_rsds_max.groupby(['lat', 'lon']):
        for i in range(len(point)):
            year = point['year'].values[i]
            
            rsds_max = df_rsds_max[(df_rsds_max['lat'] == lat) & 
                                   (df_rsds_max['lon'] == lon) & 
                                   (df_rsds_max['year'] == year)]['value'].values[0]
            rsds_min = df_rsds_min[(df_rsds_min['lat'] == lat) & 
                                   (df_rsds_min['lon'] == lon) & 
                                   (df_rsds_min['year'] == year)]['value'].values[0]
            tas_max = df_tas_max[(df_tas_max['lat'] == lat) & 
                                 (df_tas_max['lon'] == lon) & 
                                 (df_tas_max['year'] == year)]['value'].values[0]
            tas_min = df_tas_min[(df_tas_min['lat'] == lat) & 
                                 (df_tas_min['lon'] == lon) & 
                                 (df_tas_min['year'] == year)]['value'].values[0]
            slope = df_slope[(df_slope['lat'] == lat) & (df_slope['lon'] == lon)]['value'].values[0]
            
            X_test = pd.DataFrame([[rsds_max, rsds_min, tas_max, tas_min, slope]])
            prediction = np.argmax(model.predict(X_test), axis=1)

            print(f"Prediction for lat: {lat}, lon: {lon}, year: {year} -> {LUSA_SCORES[prediction[0]]}")
            
            results['year'].append(year)
            results['score'].append(LUSA_SCORES[prediction[0]])
            results['lat'].append(lat)
            results['lon'].append(lon)
    
    df_results = pd.DataFrame(results)
    output_path = f"{output_dir}/RCP{scenario}_PV_SUITABILITY_PREDICTIONS.csv"
    df_results.to_csv(output_path, index=False)
    print(f"PV Suitability Predictions saved to: {output_path}")
    return df_results

def calculate_pv_yield(df_predictions, rsds_file, output_file):
    print("Calculating PV Yield and Electricity Savings...")
    ds_rsds = xr.open_dataset(rsds_file)
    
    def get_daylight_hours(latitude, longitude, date):
        times = pd.date_range(date, date + timedelta(days=1), freq='H', tz='UTC')
        solar_pos = solarposition.get_solarposition(times, latitude, longitude)
        return solar_pos['apparent_elevation'].gt(0).sum()
    
    def calculate_daily_energy(rsds, daylight_hours):
        return rsds * AREA * EFFICIENCY * daylight_hours  # in Wh

    df_predictions = df_predictions[df_predictions['year'].between(2021, 2030)]
    daylight_hours_cache = {}
    results = []

    for _, row in df_predictions.iterrows():
        year, lat, lon = int(row['year']), row['lat'], row['lon']
        annual_savings = 0

        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        rsds_daily_values = ds_rsds.sel(time=slice(start_date, end_date)).sel(lat=lat, lon=lon, method="nearest")['rsds'].values

        for day_offset in range((end_date - start_date).days + 1):
            current_date = start_date + timedelta(days=day_offset)
            rsds_value = rsds_daily_values[day_offset]

            day_of_year = current_date.timetuple().tm_yday
            if (lat, day_of_year) not in daylight_hours_cache:
                daylight_hours_cache[(lat, day_of_year)] = get_daylight_hours(lat, lon, current_date)

            daily_energy = calculate_daily_energy(rsds_value, daylight_hours_cache[(lat, day_of_year)])
            annual_savings += daily_energy * ELECTRICITY_RATE

        results.append({'year': year, 'lat': lat, 'lon': lon, 'annual_electricity_savings_€': round(annual_savings, 2)})

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Annual PV Yield Data saved to: {output_file}")

##########################################
# Plotting Functions (Integrated from Notebook)
##########################################

def plot_predictions(main_dir, crop_type="PV"):
    # Adjust the file name if needed (here we use scenario 26 as an example)
    predictions_file = os.path.join(main_dir, "PREDICTIONS/CSVs", "RCP26_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    print(f"Loading predictions from: {predictions_file}")
    data = pd.read_csv(predictions_file)
    
    # Group data by unique latitude and longitude
    grouped_data = data.groupby(['lat', 'lon'])
    
    # Loop through each unique location and create a Plotly figure
    for (lat, lon), group in grouped_data:
        years = group['year'].values
        scores = group['score'].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=scores, x=years,
                                 name='Land Suitability',
                                 marker_color='mediumspringgreen',
                                 mode='lines+markers'))

        fig.update_layout(title={
                             'text': f'Land Suitability Analysis for {crop_type} <br> Location: {lat}, {lon}',
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor':'top' },
                          yaxis_title='Land Suitability Score (%)', 
                          xaxis_title='Year',
                          autosize=True,
                          paper_bgcolor='rgb(243, 243, 243)',
                          plot_bgcolor='rgb(243, 243, 243)',
                          showlegend=True)
        fig.show()
        predictions_plot_path = os.path.join(plots_dir, 'predictions.png')
        plt.savefig(predictions_plot_path)

def plot_training_metrics(history):
    # Plot training accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.title('Model Accuracy Over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    # Plot training loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    loss_plot_path = os.path.join(plots_dir, 'training_loss.png')
    plt.savefig(loss_plot_path)

def plot_confusion_matrix(model, X_test, y_test):
    # Predict on test data and compute confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    confusion_plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(confusion_plot_path)
    
    print(classification_report(y_test, y_pred_classes, labels=np.arange(len(CLASS_LABELS)),
                                target_names=CLASS_LABELS, zero_division=0))

def plot_high_variance_point(df_predictions, df_rsds_yearly_min, df_rsds_yearly_max, df_tas_yearly_min, df_tas_yearly_max):
    # Helper to find the closest point in the dataset
    def find_closest(df, lat, lon):
        df['lat_diff'] = np.abs(df['lat'] - lat)
        df['lon_diff'] = np.abs(df['lon'] - lon)
        df['distance'] = df['lat_diff'] + df['lon_diff']
        closest_point = df.loc[df['distance'].idxmin()]
        return df[(df['lat'] == closest_point['lat']) & (df['lon'] == closest_point['lon'])]
    
    # Compute variance of suitability scores by point
    df_predictions['score'] = df_predictions['score'].astype(float)
    df_predictions_clean = df_predictions.dropna(subset=['score'])
    variance_per_point = df_predictions_clean.groupby(['lat', 'lon'])['score'].var().reset_index()
    variance_per_point = variance_per_point.dropna(subset=['score'])
    
    if variance_per_point.empty:
        raise ValueError("No points with valid variance found.")
    
    max_variance_point = variance_per_point.loc[variance_per_point['score'].idxmax()]
    max_lat = max_variance_point['lat']
    max_lon = max_variance_point['lon']
    
    print(f"Point with highest variance in suitability: (lat: {max_lat}, lon: {max_lon})")
    
    # Extract time series for the closest points in each dataset
    df_point_rsds_min = find_closest(df_rsds_yearly_min, max_lat, max_lon)
    df_point_rsds_max = find_closest(df_rsds_yearly_max, max_lat, max_lon)
    df_point_tasmin = find_closest(df_tas_yearly_min, max_lat, max_lon)
    df_point_tasmax = find_closest(df_tas_yearly_max, max_lat, max_lon)
    df_point_suitability = df_predictions_clean[(df_predictions_clean['lat'] == max_lat) & (df_predictions_clean['lon'] == max_lon)]
    
    # Merge the data for plotting
    df_combined = pd.merge(df_point_rsds_min[['year', 'value']],
                           df_point_rsds_max[['year', 'value']],
                           on='year', suffixes=('_rsds_min', '_rsds_max'))
    df_combined = pd.merge(df_combined, df_point_tasmin[['year', 'value']], on='year')
    df_combined = pd.merge(df_combined, df_point_tasmax[['year', 'value']], on='year')
    df_combined = pd.merge(df_combined, df_point_suitability[['year', 'score']], on='year')
    df_combined.columns = ['year', 'rsds_min', 'rsds_max', 'tasmin', 'tasmax', 'suitability']
    
    print("\nCombined Data for Point with Highest Variance:")
    print(df_combined)
    
    # Create a subplot with five panels
    fig, axs = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
    
    axs[0].plot(df_combined['year'], df_combined['rsds_min'], label='RSDS Min (W/m²)', color='blue')
    axs[0].set_ylabel('Solar radiation (W/m²)')
    axs[0].set_title(f'Solar radiation Min at (lat: {max_lat}, lon: {max_lon})')
    axs[0].grid(True)
    
    axs[1].plot(df_combined['year'], df_combined['rsds_max'], label='RSDS Max (W/m²)', color='purple')
    axs[1].set_ylabel('Solar radiation (W/m²)')
    axs[1].set_title(f'Solar radiation Max at (lat: {max_lat}, lon: {max_lon})')
    axs[1].grid(True)
    
    axs[2].plot(df_combined['year'], df_combined['tasmin'], label='Tasmin (°C)', color='green')
    axs[2].set_ylabel('Temperature (°C)')
    axs[2].set_title(f'Temperature Min at (lat: {max_lat}, lon: {max_lon})')
    axs[2].grid(True)
    
    axs[3].plot(df_combined['year'], df_combined['tasmax'], label='Tasmax (°C)', color='red')
    axs[3].set_ylabel('Temperature (°C)')
    axs[3].set_title(f'Temperature Max at (lat: {max_lat}, lon: {max_lon})')
    axs[3].grid(True)
    
    axs[4].plot(df_combined['year'], df_combined['suitability'], label='Suitability Score', color='orange', linestyle='--')
    axs[4].set_ylabel('Suitability Score')
    axs[4].set_title(f'Suitability Score at (lat: {max_lat}, lon: {max_lon})')
    axs[4].grid(True)
    
    ticks = np.arange(df_combined['year'].min(), df_combined['year'].max()+1, 10)
    for ax in axs:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45, ha="right")
    axs[4].set_xlabel('Year')
    
    plt.tight_layout()
    point_plot_path = os.path.join(plots_dir, 'high_variance_point.png')
    plt.savefig(point_plot_path)


def plot_continuous_spatial(main_dir):
    # Load the predictions CSV
    csv_path = os.path.join(main_dir, "PREDICTIONS/CSVs", "RCP26_PV_SUITABILITY_PREDICTIONS.csv")
    df_predictions = pd.read_csv(csv_path)
    df_predictions['score'] = pd.to_numeric(df_predictions['score'], errors='coerce')
    
    # Determine the year with the largest suitability fluctuation
    yearly_fluctuations = df_predictions.groupby('year')['score'].std()
    year_with_max_fluctuation = yearly_fluctuations.idxmax()
    print(f"The year with the largest fluctuation in suitability is: {year_with_max_fluctuation}")
    
    df_selected_year = df_predictions[df_predictions['year'] == year_with_max_fluctuation]
    gdf_points = gpd.GeoDataFrame(df_selected_year, 
                                  geometry=[Point(xy) for xy in zip(df_selected_year.lon, df_selected_year.lat)], 
                                  crs="EPSG:4326")
    
    # Adjust the GeoJSON file path as needed
    geojson_path = os.path.join(main_dir, "PILOTS", "geoBoundaries-CYP-ADM0.geojson")
    gdf_boundary = gpd.read_file(geojson_path)
    
    gdf_clipped = gpd.sjoin(gdf_points, gdf_boundary, predicate='within')
    df_clipped = pd.DataFrame(gdf_clipped.drop(columns="geometry"))
    
    # Create a grid for interpolation
    lon_min, lon_max = 32.1, 34.7
    lat_min, lat_max = 34.4, 35.8
    grid_lon, grid_lat = np.mgrid[lon_min:lon_max:400j, lat_min:lat_max:400j]
    
    grid_scores = griddata(
        (df_clipped['lon'], df_clipped['lat']),
        df_clipped['score'],
        (grid_lon, grid_lat),
        method='linear'
    )
    # Fill NaNs using nearest-neighbor interpolation
    grid_scores = np.where(np.isnan(grid_scores), 
                           griddata(
                               (df_clipped['lon'], df_clipped['lat']),
                               df_clipped['score'],
                               (grid_lon, grid_lat),
                               method='nearest'
                           ),
                           grid_scores)
    
    grid_points = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in zip(grid_lon.flatten(), grid_lat.flatten())],
                                   crs="EPSG:4326")
    clipped_grid_points = gpd.clip(grid_points, gdf_boundary)
    clipped_scores = grid_scores.flatten()[clipped_grid_points.index]
    
    study_area_path = os.path.join(main_dir, "PILOTS", "Study_area_Cyprus.geojson")
    gdf_study_area = gpd.read_file(study_area_path)
    
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'))
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0)
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0)
    
    sc = ax.scatter(clipped_grid_points.geometry.x, clipped_grid_points.geometry.y, 
                    c=clipped_scores, cmap='viridis', s=15, alpha=0.7,
                    vmin=0, vmax=100, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical')
    cbar.set_label('Suitability Score')
    sc.set_clim(0, 100)
    
    ax.add_geometries(gdf_study_area['geometry'], crs=ccrs.PlateCarree(), 
                      edgecolor='red', facecolor='none', linewidth=2, alpha=0.7)
    study_area_patch = Patch(edgecolor='red', facecolor='none', label='Study Area', linewidth=2)
    ax.legend(handles=[study_area_patch], loc='lower right')
    
    plt.title(f'PV Suitability Scores in {year_with_max_fluctuation}')
    map_plot_path = os.path.join(plots_dir, 'continuous_map.png')
    plt.savefig(map_plot_path)


#############################################
# Main Script: Run analysis, plots, and yield calculation
#############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--plot", action="store_true", help="Generate and save plots after processing")
    args = parser.parse_args()

    print(f"Running PV Suitability Analysis for Scenario: RCP{args.scenario}")
    
    config = load_config(args.config)
    ensure_directory(os.path.join(MAIN_DIR, "PREDICTIONS/CSVs"))

    X_train, X_test, y_train, y_test = load_training_data(os.path.join(TRAINING_DIR, "PV_suitability_data_TRAIN.csv"))
    model, history = train_model(X_train, y_train)

    # Process climate data. (Adjust if you have separate datasets.)
    df_rsds_max = process_climate_data(CLIMATE_PROJ_DIR, "rsds", args.scenario)
    df_rsds_min = process_climate_data(CLIMATE_PROJ_DIR, "rsds", args.scenario)
    df_tas_max = process_climate_data(CLIMATE_PROJ_DIR, "tas", args.scenario)
    df_tas_min = process_climate_data(CLIMATE_PROJ_DIR, "tas", args.scenario)
    df_slope = process_climate_data(SLOPE_DIR, "slope", args.scenario)
    
    df_predictions = predict_pv_suitability(model, df_rsds_max, df_rsds_min,
                                            df_tas_max, df_tas_min,
                                            df_slope, args.scenario, PREDICTION_DIR)

    df_predictions_updated = apply_constraints(df_predictions, config.get("proximity_to_powerlines", 0.1), config.get("road_network_accessibility", 0.1))

    # Save updated predictions to CSV
    updated_output_file = os.path.join(PREDICTION_DIR, f"RCP{args.scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    df_predictions_updated.to_csv(updated_output_file, index=False)
    print(f"Updated PV Suitability Predictions saved to: {updated_output_file}")
    
    # Calculate PV yield and save results to CSV in PREDICTION_DIR.
    rsds_file = os.path.join(CLIMATE_PROJ_DIR, f"rsds_rcp{args.scenario}_2021_2100.nc")
    yield_output_file = os.path.join(PREDICTION_DIR, f"RCP{args.scenario}_PV_YIELD.csv")
    calculate_pv_yield(df_predictions, rsds_file, yield_output_file)
    
    # Run plotting routines if the --plot flag is set.
    if args.plot:
        plot_predictions(MAIN_DIR, crop_type="PV", plots_dir=PLOTS_DIR)
        plot_training_metrics(history, PLOTS_DIR)
        plot_confusion_matrix(model, X_test, y_test, PLOTS_DIR)
        # For high-variance point plotting, the same datasets are re-used as placeholders.
        plot_high_variance_point(df_predictions, df_rsds_max, df_rsds_max, df_tas_max, df_tas_max, plots_dir=PLOTS_DIR)
        plot_continuous_spatial(MAIN_DIR, plots_dir=PLOTS_DIR)