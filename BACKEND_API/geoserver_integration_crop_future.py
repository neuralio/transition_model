import requests
from requests.auth import HTTPBasicAuth
import sys
from geo.Geoserver import Geoserver
import pandas as pd
import xarray as xr
import numpy as np
import os
import json

convert_to_netcdf='yes'
create_store='yes'
publish_layer='yes'
delete_tmp_files='no'

###################
#### USER EDIT ####
###################

# Define the GeoServer URL and authentication details
geoserver_url = '<geoserver_url:port>/geoserver/rest'  # Replace with your GeoServer URL
url = '<geoserver_url:port>/geoserver'
username = '<geoserver_username>'  # Replace with your GeoServer username
password = '<geoserver_password>'  # Replace with your GeoServer password

crop_type = str(sys.argv[1])
scenario = str(sys.argv[2])
PREDICTION_DIR = f'/home/ESA-AGENTS/BACKEND_API/Demo_With_Geoserver/PREDICTIONS/{crop_type}/CSVs'

# Details for predictions
workspace_name = 'Land_Suitability'
new_namespace_uri = 'Land_Suitability'
store_name_predictions = f'Crop_Suitability_Predictions_{scenario}_{crop_type}'
file_path_predictions = f'/opt/geoserver_data/data/ESA-AGENTS/BACKEND_API/Demo_With_Geoserver/PREDICTIONS/{crop_type}/CSVs/{scenario}_LUSA_PREDICTIONS.nc'
file_path_predictions_HOST = f'/home/ESA-AGENTS/BACKEND_API/Demo_With_Geoserver/PREDICTIONS/{crop_type}/CSVs/{scenario}_LUSA_PREDICTIONS.csv'
upload_layer_name_predictions = 'score'
updated_layer_name_predictions = f'Crop_Suitability_Predictions_{scenario}_{crop_type}'


##########################
#### CONVERT 2 NETCDF ####
##########################

#--- Predictions

df_predictions = pd.read_csv(file_path_predictions_HOST)
df_predictions["time"] = pd.to_datetime(df_predictions["year"].astype(str) + "0101T00:00:00", format="%Y%m%dT%H:%M:%S")
times = np.sort(df_predictions["time"].unique())  # Properly formatted time dimension
lats = np.sort(df_predictions["lat"].unique())
lons = np.sort(df_predictions["lon"].unique())
data_predictions = np.full((len(times), len(lats), len(lons)), np.nan)

for _, row in df_predictions.iterrows():
    i = np.where(times == row["time"])[0][0]
    j = np.where(lats == row["lat"])[0][0]
    k = np.where(lons == row["lon"])[0][0]
    data_predictions[i, j, k] = row["score"]

da_predictions = xr.DataArray(
    data_predictions,
    coords={
        "time": times,  # Now uses properly formatted time
        "lat": lats,
        "lon": lons
    },
    dims=["time", "lat", "lon"],
    name="score"
)

ds_predictions = xr.Dataset({"score": da_predictions})
netcdf_file_predictions = f"{file_path_predictions_HOST[:-4]}.nc"
ds_predictions.to_netcdf(netcdf_file_predictions, mode="w")

#########################
#### STORE - One off ####
#########################

if create_store=='yes':

 #--- Predictions
 # Delete the workspace if it already exists
 geo = Geoserver(url, username=username, password=password)
 stores = geo.get_coveragestores(workspace=workspace_name)

 for existing_store in stores['coverageStores']['coverageStore']:
  if existing_store['name'] == store_name_predictions:
   print(f'Store {store_name_predictions} exists.')
   geo.delete_coveragestore(store_name_predictions, workspace=workspace_name)
   print(f'Store {store_name_predictions} was deleted')

 # Create the XML payload for the NetCDF coveragestore
 coveragestore_payload = f"""
 <coverageStore>
     <name>{store_name_predictions}</name>
     <type>NetCDF</type>
     <enabled>true</enabled>
     <workspace>{workspace_name}</workspace>
     <url>{file_path_predictions}</url>
 </coverageStore>
 """

 # Define the headers for the request
 headers = {
     'Content-Type': 'application/xml',
 }
 
 # Send the POST request to create the new NetCDF coveragestore
 response = requests.post(
     f'{geoserver_url}/workspaces/{workspace_name}/coveragestores',
     data=coveragestore_payload,
     headers=headers,
     auth=HTTPBasicAuth(username, password)
 )

 # Check the response status code
 if response.status_code == 201:
     print(f'NetCDF coveragestore "{store_name_predictions}" created successfully.')
 else:
    print(f'Failed to create NetCDF coveragestore "{store_name_predictions}". Status code: {response.status_code}')
    print(response.text)


####################
### LAYER UPLOAD ###
####################

#--- Predictions

if publish_layer=='yes':
  # Create the XML payload for the new coverage
  coverage_payload = f"""
  <coverage>
      <name>{upload_layer_name_predictions}</name>
      <nativeName>{upload_layer_name_predictions}</nativeName>
      <title>{upload_layer_name_predictions}</title>
      <store>
          <name>{store_name_predictions}</name>
      </store>
  </coverage>
  """
 
  # Define the headers for the request
  headers = {
      'Content-Type': 'application/xml',
  }

  # Send the POST request to create the new layer
  response = requests.post(
      f'{geoserver_url}/workspaces/{workspace_name}/coveragestores/{store_name_predictions}/coverages',
      data=coverage_payload,
      headers=headers,
      auth=HTTPBasicAuth(username, password)
  )

  # Check the response status code
  if response.status_code == 201:
      print(f'Layer "{upload_layer_name_predictions}" published successfully.')
  else:
      print(f'Failed to publish layer "{upload_layer_name_predictions}". Status code: {response.status_code}')
      print(response.text)

  ############################################
  #### LAYER UPDATE (srs, time domension) ####
  ############################################

  # Create the XML payload for updating the coverage
  coverage_update_payload = f"""
  <coverage>
      <name>{updated_layer_name_predictions}</name>
      <title>{updated_layer_name_predictions}</title>
      <metadata>
          <entry key="time">
              <dimensionInfo>
                  <enabled>true</enabled>
                  <presentation>LIST</presentation>
                  <nearestMatchEnabled>true</nearestMatchEnabled>
              </dimensionInfo>
          </entry>
      </metadata>
  </coverage>
  """
 
  # Define the headers for the request
  headers = {
      'Content-Type': 'application/xml',
  }
  
  # Send the PUT request to update the coverage
  response = requests.put(
      f'{geoserver_url}/workspaces/{workspace_name}/coveragestores/{store_name_predictions}/coverages/{upload_layer_name_predictions}',
      data=coverage_update_payload,
      headers=headers,
      auth=HTTPBasicAuth(username, password)
  )
  
  # Check the response status code
  if response.status_code == 200:
      print(f'Layer "{updated_layer_name_predictions}" updated successfully.')
  else:
      print(f'Failed to update layer "{updated_layer_name_predictions}". Status code: {response.status_code}')
      print(response.text)


##############################
### DELETE TEMPORARY FILES ###
##############################

if delete_tmp_files=='yes':
 # Remove the intermidiate NetCDF file
 os.remove(netcdf_file_predictions)
 os.remove(netcdf_file_yield)
