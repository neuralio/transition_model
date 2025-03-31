import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Argument Parser
parser = argparse.ArgumentParser(description="Run crop suitability model for a given scenario.")
parser.add_argument("--scenario", type=int, choices=[26, 45, 85], required=True, help="Climate scenario (26, 45, or 85)")
parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
args = parser.parse_args()

# Load Configuration
with open(args.config, "r") as config_file:
    config = json.load(config_file)

crop_type = config["crop_type"]
main_dir = os.path.dirname(os.path.abspath(__file__))

# Directories
input_dir = os.path.join(main_dir, "INPUT")
output_dir = os.path.join(main_dir, "PREDICTIONS", crop_type, "CSVs")
os.makedirs(output_dir, exist_ok=True)

# File Paths
slope_file = os.path.join(input_dir, "SLOPE", "SLOPE.nc")
cec_file = os.path.join(input_dir, "SOILGRIDS", "cec.nc")
phh2o_file = os.path.join(input_dir, "SOILGRIDS", "phh2o.nc")
soc_file = os.path.join(input_dir, "SOILGRIDS", "soc.nc")
rain_file = os.path.join(input_dir, "CLIMATE_PROJECTIONS", f"pr_rcp{args.scenario}_2021_2100.nc")
tasmax_file = os.path.join(input_dir, "CLIMATE_PROJECTIONS", f"tasmax_rcp{args.scenario}_2021_2100.nc")
tasmin_file = os.path.join(input_dir, "CLIMATE_PROJECTIONS", f"tasmin_rcp{args.scenario}_2021_2100.nc")
train_data_file = os.path.join(main_dir, "TRAINING", crop_type, "suitabillity_data_TRAIN.csv")

# Model Parameters
num_features = 11 if crop_type == "MAIZE" else 9
LUSA_SCORES = ['0', '40', '60', '85', '95', '100']
CLASS_LABELS = ['0', '1', '2', '3', '4', '5']

# Load Training Data
df_train = pd.read_csv(train_data_file)
X = df_train.iloc[:, 1:num_features + 1]
y = np.floor(df_train.iloc[:, 0] / 20).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Neural Network
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(num_features,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=100, batch_size=8)

# Save training history
history_file = os.path.join(output_dir, "history_train.txt")
with open(history_file, "w") as f:
    f.write(str(history.history))

# Load NetCDF Data
def load_netcdf(file_path, var_name):
    """
    Load a NetCDF file and extract data along with lat/lon coordinates.
    
    Parameters:
        file_path (str): Path to the NetCDF file.
        var_name (str): Name of the variable to extract (e.g., "pr", "tasmax").
    
    Returns:
        pd.DataFrame: DataFrame containing lon, lat, and the extracted values.
    """
    try:
        # Open NetCDF dataset
        ds = xr.open_dataset(file_path)

        if var_name not in ds:
            raise KeyError(f"Variable '{var_name}' not found in {file_path}")

        # Extract data
        data = ds[var_name].values
        lon_values = ds["lon"].values
        lat_values = ds["lat"].values

        # Check if the dataset has a time dimension
        if "time" in ds.dims:
            data = data[0, :, :]  # Select the first time slice

        # Validate data dimensions
        if data.shape != (len(lat_values), len(lon_values)):
            raise ValueError(f"Data shape mismatch in {file_path}: Expected ({len(lat_values)}, {len(lon_values)}) but got {data.shape}")

        # Create a meshgrid for coordinates
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

        # Flatten arrays for DataFrame
        df = pd.DataFrame({
            "lon": lon_grid.flatten(),
            "lat": lat_grid.flatten(),
            "value": data.flatten()
        })

        # Handle missing values (replace NaN with 0 or another suitable value)
        df["value"] = df["value"].fillna(0)  

        return df

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

df_slope = load_netcdf(slope_file, "slope").assign(parameter="slope")
df_cec = load_netcdf(cec_file, "cec").assign(parameter="cec")
df_soc = load_netcdf(soc_file, "soc").assign(parameter="soc")
df_ph = load_netcdf(phh2o_file, "phh2o").assign(parameter="phh2o")
df_rain = load_netcdf(rain_file, "pr").assign(parameter="pr")
df_tasmax = load_netcdf(tasmax_file, "tasmax").assign(parameter="tasmax")
df_tasmin = load_netcdf(tasmin_file, "tasmin").assign(parameter="tasmin")

#--- Extract Rain data

# Define the main directory where the script is located
main_dir = os.path.dirname(os.path.abspath(__file__))

# Scenario (should be passed dynamically in the actual script)
scenario = args.scenario  # Example: Replace with `args.scenario` in actual script

# Construct the file path dynamically based on the scenario
rain_file = os.path.join(main_dir, "INPUT", "CLIMATE_PROJECTIONS", f"pr_rcp{scenario}_2021_2100.nc")

# Ensure the file exists before proceeding
if not os.path.exists(rain_file):
    raise FileNotFoundError(f"Error: File {rain_file} not found!")

# Load the NetCDF file using xarray
ds_rain = xr.open_dataset(rain_file)

# Extract latitude, longitude, and precipitation data
rain_data = ds_rain['pr'].values  # Shape: (time, lat, lon)
lon_values = ds_rain['lon'].values  # Longitude values
lat_values = ds_rain['lat'].values  # Latitude values
time_values = ds_rain['time'].values  # Time values

# Convert time values to pandas datetime format
dates = pd.to_datetime(time_values)

# Create a meshgrid of lat and lon coordinates to match the rain data
lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

# Ensure data reshaping is correct
if rain_data.shape[0] == len(dates) and rain_data.shape[1:] == lon_grid.shape:
    # Flatten arrays to create a structured DataFrame
    df_rain = pd.DataFrame({
        'lon': np.tile(lon_grid.flatten(), len(dates)),
        'lat': np.tile(lat_grid.flatten(), len(dates)),
        'value': rain_data.flatten(),
        'date': np.repeat(dates, len(lon_values) * len(lat_values))
    })
else:
    raise ValueError(f"Data shape mismatch: Expected ({len(dates)}, {len(lat_values)}, {len(lon_values)}) but got {rain_data.shape}")

# Add the 'parameter' column with the value "precipitation"
df_rain['parameter'] = 'pr'

# Extract the year from the 'date' column
df_rain['year'] = df_rain['date'].dt.year

# Rearrange columns to match the required order
df_rain = df_rain[['parameter', 'date', 'lon', 'lat', 'value', 'year']]

# Sort DataFrame by lon, lat, year, and date for better readability
df_rain = df_rain.sort_values(by=['lon', 'lat', 'year', 'date']).reset_index(drop=True)

# Group by 'year', 'lat', 'lon' and sum 'value' (precipitation) for each year
df_rain_yearly = df_rain.groupby(['year', 'lat', 'lon'], as_index=False)['value'].sum()

# Display the final DataFrame
print(f"df_rain_yearly for scenario RCP{scenario} successfully created!")
print(df_rain_yearly.head())


#--- Extract Temperature Min (tasmin) data

import xarray as xr
import pandas as pd
import numpy as np
import os

# Define the main directory where the script is located
main_dir = os.path.dirname(os.path.abspath(__file__))

# Scenario variable (should be dynamically set in the actual script)
scenario = args.scenario # Example: Replace with `args.scenario` in actual script

# Construct the file path dynamically based on the scenario
tasmin_file = os.path.join(main_dir, "INPUT", "CLIMATE_PROJECTIONS", f"tasmin_rcp{scenario}_2021_2100.nc")

# Ensure the file exists before proceeding
if not os.path.exists(tasmin_file):
    raise FileNotFoundError(f"Error: File {tasmin_file} not found!")

# Load the NetCDF file using xarray
ds_tasmin = xr.open_dataset(tasmin_file)

# Extract latitude, longitude, and temperature min data
tasmin_data = ds_tasmin["tasmin"].values  # Temperature min data
lon_values = ds_tasmin["lon"].values  # Longitude values
lat_values = ds_tasmin["lat"].values  # Latitude values
time_values = ds_tasmin["time"].values if "time" in ds_tasmin.dims else None

# Convert time values to pandas datetime format
dates = pd.to_datetime(time_values) if time_values is not None else None

# Create a meshgrid of lat and lon coordinates to match the tasmin data
lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

# Ensure data reshaping is correct
if dates is not None and tasmin_data.ndim == 3:
    # Initialize an empty list to store time-step data
    tasmin_records = []
    
    for t in range(len(dates)):
        temp_data = tasmin_data[t, :, :]  # Extract slice for the current time step
        
        # Flatten data properly
        lon_flat = lon_grid.flatten()
        lat_flat = lat_grid.flatten()
        value_flat = temp_data.flatten()
        
        # Ensure all arrays are of the same length
        if not (len(lon_flat) == len(lat_flat) == len(value_flat)):
            raise ValueError(f"Mismatch in array lengths at timestep {t}: lon {len(lon_flat)}, lat {len(lat_flat)}, value {len(value_flat)}")

        # Create a temporary DataFrame for this time step
        temp_df = pd.DataFrame({
            "lon": lon_flat,
            "lat": lat_flat,
            "value": value_flat,
            "date": dates[t]
        })
        tasmin_records.append(temp_df)

    # Concatenate all time-step DataFrames into a single DataFrame
    df_tasmin = pd.concat(tasmin_records, ignore_index=True)

elif tasmin_data.ndim == 2:
    # If no time dimension, treat as static temperature data
    df_tasmin = pd.DataFrame({
        "lon": lon_grid.flatten(),
        "lat": lat_grid.flatten(),
        "value": tasmin_data.flatten(),
        "date": pd.Timestamp("2021-01-01")  # Assign a default date
    })

else:
    raise ValueError(f"Unexpected shape {tasmin_data.shape} for temperature min data.")

# Extract year
df_tasmin["year"] = df_tasmin["date"].dt.year

# Compute yearly min and max temperature
df_tasmin_yearly_min = df_tasmin.groupby(["year", "lat", "lon"], as_index=False)["value"].min()
df_tasmin_yearly_max = df_tasmin.groupby(["year", "lat", "lon"], as_index=False)["value"].max()

# Display the processed DataFrame
print(f"\ndf_tasmin_yearly_min for scenario RCP{scenario} successfully created!")
print(df_tasmin_yearly_min.head())

print(f"\ndf_tasmin_yearly_max for scenario RCP{scenario} successfully created!")
print(df_tasmin_yearly_max.head())


#--- Extract Temperature Max (tasmax) data
import xarray as xr
import pandas as pd
import numpy as np
import os

# Define the main directory where the script is located
main_dir = os.path.dirname(os.path.abspath(__file__))

# Scenario variable (should be dynamically set in the actual script)
scenario = args.scenario  # Example: Replace with `args.scenario` in actual script

# Construct the file path dynamically based on the scenario
tasmax_file = os.path.join(main_dir, "INPUT", "CLIMATE_PROJECTIONS", f"tasmax_rcp{scenario}_2021_2100.nc")

# Ensure the file exists before proceeding
if not os.path.exists(tasmax_file):
    raise FileNotFoundError(f"Error: File {tasmax_file} not found!")

# Load the NetCDF file using xarray
ds_tasmax = xr.open_dataset(tasmax_file)

# Extract latitude, longitude, and temperature max data
tasmax_data = ds_tasmax["tasmax"].values  # Temperature max data
lon_values = ds_tasmax["lon"].values  # Longitude values
lat_values = ds_tasmax["lat"].values  # Latitude values
time_values = ds_tasmax["time"].values if "time" in ds_tasmax.dims else None

# Convert time values to pandas datetime format
dates = pd.to_datetime(time_values) if time_values is not None else None

# Create a meshgrid of lat and lon coordinates to match the tasmax data
lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

# Ensure data reshaping is correct
if dates is not None and tasmax_data.ndim == 3:
    # Initialize an empty list to store time-step data
    tasmax_records = []
    
    for t in range(len(dates)):
        temp_data = tasmax_data[t, :, :]  # Extract slice for the current time step
        
        # Flatten data properly
        lon_flat = lon_grid.flatten()
        lat_flat = lat_grid.flatten()
        value_flat = temp_data.flatten()
        
        # Ensure all arrays are of the same length
        if not (len(lon_flat) == len(lat_flat) == len(value_flat)):
            raise ValueError(f"Mismatch in array lengths at timestep {t}: lon {len(lon_flat)}, lat {len(lat_flat)}, value {len(value_flat)}")

        # Create a temporary DataFrame for this time step
        temp_df = pd.DataFrame({
            "lon": lon_flat,
            "lat": lat_flat,
            "value": value_flat,
            "date": dates[t]
        })
        tasmax_records.append(temp_df)

    # Concatenate all time-step DataFrames into a single DataFrame
    df_tasmax = pd.concat(tasmax_records, ignore_index=True)

elif tasmax_data.ndim == 2:
    # If no time dimension, treat as static temperature data
    df_tasmax = pd.DataFrame({
        "lon": lon_grid.flatten(),
        "lat": lat_grid.flatten(),
        "value": tasmax_data.flatten(),
        "date": pd.Timestamp("2021-01-01")  # Assign a default date
    })

else:
    raise ValueError(f"Unexpected shape {tasmax_data.shape} for temperature max data.")

# Extract year
df_tasmax["year"] = df_tasmax["date"].dt.year

# Compute yearly min and max temperature
df_tasmax_yearly_min = df_tasmax.groupby(["year", "lat", "lon"], as_index=False)["value"].min()
df_tasmax_yearly_max = df_tasmax.groupby(["year", "lat", "lon"], as_index=False)["value"].max()

# Display the processed DataFrame
print(f"\ndf_tasmax_yearly_min for scenario RCP{scenario} successfully created!")
print(df_tasmax_yearly_min.head())

print(f"\ndf_tasmax_yearly_max for scenario RCP{scenario} successfully created!")
print(df_tasmax_yearly_max.head())




#--- Make The Land Suitability Predictions
#--- Make The Land Suitability Predictions For timestep (year)
YEARS, LUSA, LATS, LONS = [], [], [], []

print(f"\nRunning predictions for {crop_type} under RCP{args.scenario} scenario...\n")

# Loop over all unique (lat, lon) points
for (lat, lon), point in df_rain_yearly.groupby(['lat', 'lon']):
    # Extract unique values
    rain_values = point.set_index("year")["value"]

    for year, rain_value in rain_values.items():
        try:
            # Extract corresponding temperature values
            tasmin_lower = df_tasmin_yearly_min.loc[
                (df_tasmin_yearly_min["lat"] == lat) & (df_tasmin_yearly_min["lon"] == lon) & (df_tasmin_yearly_min["year"] == year), "value"].values[0]
            tasmin_upper = df_tasmin_yearly_max.loc[
                (df_tasmin_yearly_max["lat"] == lat) & (df_tasmin_yearly_max["lon"] == lon) & (df_tasmin_yearly_max["year"] == year), "value"].values[0]
            tasmax_lower = df_tasmax_yearly_min.loc[
                (df_tasmax_yearly_min["lat"] == lat) & (df_tasmax_yearly_min["lon"] == lon) & (df_tasmax_yearly_min["year"] == year), "value"].values[0]
            tasmax_upper = df_tasmax_yearly_max.loc[
                (df_tasmax_yearly_max["lat"] == lat) & (df_tasmax_yearly_max["lon"] == lon) & (df_tasmax_yearly_max["year"] == year), "value"].values[0]

            # Extract soil values
            cec_value = df_cec.loc[(df_cec["lat"] == lat) & (df_cec["lon"] == lon), "value"].values[0]
            soc_value = df_soc.loc[(df_soc["lat"] == lat) & (df_soc["lon"] == lon), "value"].values[0]
            phh2o_value = df_ph.loc[(df_ph["lat"] == lat) & (df_ph["lon"] == lon), "value"].values[0]
            slope_value = df_slope.loc[(df_slope["lat"] == lat) & (df_slope["lon"] == lon), "value"].values[0]

            # Prepare input data
            if crop_type == "MAIZE":
                X_test_predict = pd.DataFrame([[cec_value, soc_value, phh2o_value, phh2o_value, slope_value,
                                                (tasmax_lower + tasmin_lower) / 2, (tasmax_upper + tasmin_upper),
                                                tasmin_lower, tasmin_upper, rain_value, rain_value]])
            elif crop_type == "WHEAT":
                X_test_predict = pd.DataFrame([[cec_value, soc_value, phh2o_value, phh2o_value, slope_value,
                                                (tasmax_lower + tasmin_lower) / 2, (tasmax_upper + tasmin_upper),
                                                rain_value, rain_value]])

            # Run the model prediction
            predictions = model.predict(X_test_predict)
            class_label = np.argmax(predictions, axis=1)[0]

            # Store results
            YEARS.append(year)
            LUSA.append(LUSA_SCORES[class_label])  # Convert model output to LUSA score
            LATS.append(lat)
            LONS.append(lon)

            print(f"Prediction for {year} at (lat: {lat}, lon: {lon}): {LUSA_SCORES[class_label]}")

        except IndexError:
            print(f"Warning: Missing data for {year} at (lat: {lat}, lon: {lon}). Skipping...")

# Save the predictions to CSV
outdir = os.path.join(main_dir, "PREDICTIONS", crop_type, "CSVs")
os.makedirs(outdir, exist_ok=True)

prediction_file = os.path.join(outdir, f"RCP{args.scenario}_LUSA_PREDICTIONS.csv")
df_predictions = pd.DataFrame({"year": YEARS, "score": LUSA, "lat": LATS, "lon": LONS})
df_predictions.to_csv(prediction_file, index=False)

print(f"\nLUSA Predictions for {crop_type} under RCP{args.scenario} saved at:\n{prediction_file}")


#--- Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, classification_report

# Define paths
training_dir = os.path.join(main_dir, "TRAINING", crop_type)
history_file = os.path.join(training_dir, "history_train.txt")

# Define the output directory for plots
plot_dir = os.path.join(main_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)  # Create directory if it doesn't exist

# Define scenario name for file naming
scenario_name = f"RCP{scenario}"

# Load Training History from `history_train.txt`
if os.path.exists(history_file):
    print(f"Loading training history from {history_file}")
    with open(history_file, "r") as f:
        history = json.loads(f.read().replace("'", "\""))  # Convert single quotes to double quotes for JSON format
else:
    raise FileNotFoundError(f"Training history file not found at {history_file}. Please train the model first.")

# Plot Training Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history['accuracy'], label="Train Accuracy", color='blue')
plt.title(f'Model Accuracy Over Epochs - {scenario_name}')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig(os.path.join(plot_dir, f"training_accuracy_{scenario_name}.png"), dpi=300)
plt.close()

# Plot Training Loss
plt.figure(figsize=(8, 6))
plt.plot(history['loss'], label="Train Loss", color='red')
plt.title(f'Model Loss Over Epochs - {scenario_name}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig(os.path.join(plot_dir, f"training_loss_{scenario_name}.png"), dpi=300)
plt.close()

print(f"Training accuracy and loss plots saved in {plot_dir}/")

# Predict on Test Data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot and Save Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
plt.title(f'Confusion Matrix - {scenario_name}')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{scenario_name}.png"), dpi=300)
plt.close()

print(f"Confusion Matrix plot saved in {plot_dir}/")

# Print and Save Classification Report
class_report = classification_report(y_test, y_pred_classes, labels=np.arange(len(CLASS_LABELS)), target_names=CLASS_LABELS, zero_division=0)
report_path = os.path.join(plot_dir, f"classification_report_{scenario_name}.txt")
with open(report_path, "w") as f:
    f.write(class_report)

print(f"Classification report saved at {report_path}")
