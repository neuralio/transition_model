import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Argument Parser
import argparse
parser = argparse.ArgumentParser(description="Run past crop suitability model using ERA5 data.")
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
era5_file = os.path.join(input_dir, "ERA5_LAND", "input_ERA5_LAND.nc")
train_data_file = os.path.join(main_dir, "TRAINING", crop_type, "suitabillity_data_TRAIN.csv")

# Load Training Data
df_train = pd.read_csv(train_data_file)
num_features = 11 if crop_type == "MAIZE" else 9
LUSA_SCORES = ['0', '40', '60', '85', '95', '100']

X = df_train.iloc[:, 1:num_features + 1]
y = np.floor(df_train.iloc[:, 0] / 20).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and Train Model
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

history = model.fit(X_train, y_train, epochs=100, batch_size=8)

# Save Training History
history_file = os.path.join(output_dir, "history_train.txt")
with open(history_file, "w") as f:
    f.write(str(history.history))

# Load NetCDF Data Function
def load_netcdf(file_path, var_name):
    try:
        ds = xr.open_dataset(file_path)
        if var_name not in ds:
            raise KeyError(f"Variable '{var_name}' not found in {file_path}")

        data = ds[var_name].values
        lon_values = ds["lon"].values
        lat_values = ds["lat"].values
        time_values = ds["time"].values if "time" in ds.dims else None

        if time_values is not None:
            dates = pd.to_datetime(time_values)
            df_records = []
            lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

            for t in range(len(dates)):
                df_temp = pd.DataFrame({
                    "lon": lon_grid.flatten(),
                    "lat": lat_grid.flatten(),
                    "value": data[t, :, :].flatten(),
                    "date": dates[t]
                })
                df_records.append(df_temp)

            df = pd.concat(df_records, ignore_index=True)
        else:
            lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
            df = pd.DataFrame({
                "lon": lon_grid.flatten(),
                "lat": lat_grid.flatten(),
                "value": data.flatten(),
                "date": pd.Timestamp("2021-01-01")
            })

        df["year"] = df["date"].dt.year
        return df

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

df_slope = load_netcdf(slope_file, "slope").assign(parameter="slope")
df_cec = load_netcdf(cec_file, "cec").assign(parameter="cec")
df_soc = load_netcdf(soc_file, "soc").assign(parameter="soc")
df_ph = load_netcdf(phh2o_file, "phh2o").assign(parameter="phh2o")
df_rain = load_netcdf(era5_file, "precipitation").assign(parameter="precipitation")
df_temp = load_netcdf(era5_file, "temperature").assign(parameter="temperature")

# Compute Temperature Min, Max, Mean for **ALL YEARS**
df_temp_yearly_min = df_temp.groupby(['year', 'lat', 'lon'], as_index=False)['value'].min()
df_temp_yearly_max = df_temp.groupby(['year', 'lat', 'lon'], as_index=False)['value'].max()
df_temp_yearly_mean = df_temp.groupby(['year', 'lat', 'lon'], as_index=False)['value'].mean()

# Round lat/lon for all DataFrames
dfs_to_round = [df_temp_yearly_min, df_temp_yearly_max, df_temp_yearly_mean, df_rain, df_cec, df_soc, df_ph, df_slope]
for df in dfs_to_round:
    df["lat"] = df["lat"].round(2)
    df["lon"] = df["lon"].round(2)

# **Predictions for All Years**
YEARS, LUSA, LATS, LONS = [], [], [], []

print(f"\nRunning predictions for {crop_type} using ERA5 past data...\n")

for (year, lat, lon), point in df_rain.groupby(['year', 'lat', 'lon']):
    rain_value = point["value"].values[0]

    tasmin_lower = df_temp_yearly_min.query(f"lat == {lat} & lon == {lon} & year == {year}")["value"].values
    tasmin_upper = df_temp_yearly_max.query(f"lat == {lat} & lon == {lon} & year == {year}")["value"].values
    tasmax_lower = df_temp_yearly_min.query(f"lat == {lat} & lon == {lon} & year == {year}")["value"].values
    tasmax_upper = df_temp_yearly_max.query(f"lat == {lat} & lon == {lon} & year == {year}")["value"].values

    cec_value = df_cec.query(f"lat == {lat} & lon == {lon}")["value"].values
    soc_value = df_soc.query(f"lat == {lat} & lon == {lon}")["value"].values
    phh2o_value = df_ph.query(f"lat == {lat} & lon == {lon}")["value"].values
    slope_value = df_slope.query(f"lat == {lat} & lon == {lon}")["value"].values

    if any(map(lambda x: x.size == 0, [tasmin_lower, tasmin_upper, tasmax_lower, tasmax_upper, cec_value, soc_value, phh2o_value, slope_value])):
        print(f"Skipping lat={lat}, lon={lon}, year={year} due to missing data.")
        continue

    X_test_predict = pd.DataFrame([[cec_value[0], soc_value[0], phh2o_value[0], phh2o_value[0], slope_value[0],
                                    (tasmax_lower[0] + tasmin_lower[0]) / 2, (tasmax_upper[0] + tasmin_upper[0]),
                                    tasmin_lower[0], tasmin_upper[0], rain_value, rain_value]])

    predictions = model.predict(X_test_predict)
    class_label = np.argmax(predictions, axis=1)[0]

    YEARS.append(year)
    LUSA.append(LUSA_SCORES[class_label])
    LATS.append(lat)
    LONS.append(lon)

    print(f"Prediction for {year} at (lat: {lat}, lon: {lon}): {LUSA_SCORES[class_label]}")

prediction_file = os.path.join(output_dir, f"{crop_type}_PAST_LUSA_PREDICTIONS.csv")
df_predictions = pd.DataFrame({"year": YEARS, "score": LUSA, "lat": LATS, "lon": LONS})
df_predictions.to_csv(prediction_file, index=False)

print(f"\nLUSA Predictions for **ALL YEARS** saved at:\n{prediction_file}")

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

