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

def upload_2_geoserver_pv_future():

    ###################
    #### USER EDIT ####
    ###################
    
    # Define the GeoServer URL and authentication details
    geoserver_url = '<geoserver_url:port>/geoserver/rest'  # Replace with your GeoServer URL
    url = '<geoserver_url:port>/geoserver'
    username = '<geoserver_username>'  # Replace with your GeoServer username
    password = '<geoserver_password>'  # Replace with your GeoServer password
    
    # Path to input JSON file
    def load_config(config_path):
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config, config.get("task_id"), config.get("user_id")
    
    MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_JSON_PATH = os.path.join(MAIN_DIR, "input_mongo_pv_future.json")
    PREDICTION_DIR = os.path.join(MAIN_DIR, "PREDICTIONS/CSVs")
    config, task_id, user_id = load_config(INPUT_JSON_PATH)
    scenario = config["model_parameters"]["scenario"]
    task_prediction_dir = os.path.join(PREDICTION_DIR, f"{task_id}_{user_id}")
    
    # Details for predictions
    workspace_name = 'Land_Suitability'
    new_namespace_uri = 'Land_Suitability'
    store_name_predictions = f'PV_Suitability_Predictions_RCP{scenario}_{task_id}_{user_id}'
    file_path_predictions = f'/opt/geoserver_data/data/ESA-AGENTS/BACKEND_API/Demo_With_Geoserver/PREDICTIONS/CSVs/{task_id}_{user_id}/RCP{scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.nc'
    file_path_predictions_HOST = f'/app/PREDICTIONS/CSVs/{task_id}_{user_id}/RCP{scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv'
    upload_layer_name_predictions = f'score'
    updated_layer_name_predictions = f'PV_Suitability_Predictions_RCP{scenario}_{task_id}_{user_id}'
    
    # Details for yield
    store_name_yield = f'PV_Suitability_Yield_RCP{scenario}_{task_id}_{user_id}'
    file_path_yield = f'/opt/geoserver_data/data/ESA-AGENTS/BACKEND_API/Demo_With_Geoserver/PREDICTIONS/CSVs/{task_id}_{user_id}/RCP{scenario}_PV_YIELD.nc'
    file_path_yield_HOST = f'/app/PREDICTIONS/CSVs/{task_id}_{user_id}/RCP{scenario}_PV_YIELD.csv'
    upload_layer_name_yield = f'annual_electricity_savings'
    updated_layer_name_yield = f'PV_Suitability_Yield_RCP{scenario}_{task_id}_{user_id}'
    
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
    
    #--- Yield
    df_yield = pd.read_csv(file_path_yield_HOST)
    df_yield.rename(columns={col: col.replace("_€", "") for col in df_yield.columns if col.startswith("annual_electricity_")}, inplace=True)
    df_yield["time"] = pd.to_datetime(df_yield["year"].astype(str) + "0101T00:00:00", format="%Y%m%dT%H:%M:%S")
    times = np.sort(df_yield["time"].unique())  # Properly formatted time dimension
    lats = np.sort(df_yield["lat"].unique())
    lons = np.sort(df_yield["lon"].unique())
    data_yield = np.full((len(times), len(lats), len(lons)), np.nan)
    
    for _, row in df_yield.iterrows():
        i = np.where(times == row["time"])[0][0]
        j = np.where(lats == row["lat"])[0][0]
        k = np.where(lons == row["lon"])[0][0]
        data_yield[i, j, k] = row["annual_electricity_savings"]
    
    da_yield = xr.DataArray(
        data_yield,
        coords={
            "time": times,  # Now uses properly formatted time
            "lat": lats,
            "lon": lons
        },
        dims=["time", "lat", "lon"],
        name="annual_electricity_savings"
    )
    
    ds_yield = xr.Dataset({"annual_electricity_savings": da_yield})
    netcdf_file_yield = f"{file_path_yield_HOST[:-4]}.nc"
    ds_yield.to_netcdf(netcdf_file_yield, mode="w")
    
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
    
     #--- Yield
     # Delete the workspace if it already exists
     geo = Geoserver(url, username=username, password=password)
     stores = geo.get_coveragestores(workspace=workspace_name)
    
     for existing_store in stores['coverageStores']['coverageStore']:
      if existing_store['name'] == store_name_yield:
       print(f'Store {store_name_yield} exists.')
       geo.delete_coveragestore(store_name_yield, workspace=workspace_name)
       print(f'Store {store_name_yield} was deleted')
    
     # Create the XML payload for the NetCDF coveragestore
     coveragestore_payload = f"""
     <coverageStore>
         <name>{store_name_yield}</name>
         <type>NetCDF</type>
         <enabled>true</enabled>
         <workspace>{workspace_name}</workspace>
         <url>{file_path_yield}</url>
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
         print(f'NetCDF coveragestore "{store_name_yield}" created successfully.')
     else:
        print(f'Failed to create NetCDF coveragestore "{store_name_yield}". Status code: {response.status_code}')
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
    
    
    #--- Yield
    
    if publish_layer=='yes':
      # Create the XML payload for the new coverage
      coverage_payload = f"""
      <coverage>
          <name>{upload_layer_name_yield}</name>
          <nativeName>{upload_layer_name_yield}</nativeName>
          <title>{upload_layer_name_yield}</title>
          <store>
              <name>{store_name_yield}</name>
          </store>
      </coverage>
      """
    
      # Define the headers for the request
      headers = {
          'Content-Type': 'application/xml',
      }
    
      # Send the POST request to create the new layer
      response = requests.post(
          f'{geoserver_url}/workspaces/{workspace_name}/coveragestores/{store_name_yield}/coverages',
          data=coverage_payload,
          headers=headers,
          auth=HTTPBasicAuth(username, password)
      )
    
      # Check the response status code
      if response.status_code == 201:
          print(f'Layer "{upload_layer_name_yield}" published successfully.')
      else:
          print(f'Failed to publish layer "{upload_layer_name_yield}". Status code: {response.status_code}')
          print(response.text)
    
      ############################################
      #### LAYER UPDATE (srs, time domension) ####
      ############################################
    
      # Create the XML payload for updating the coverage
      coverage_update_payload = f"""
      <coverage>
          <name>{updated_layer_name_yield}</name>
          <title>{updated_layer_name_yield}</title>
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
          f'{geoserver_url}/workspaces/{workspace_name}/coveragestores/{store_name_yield}/coverages/{upload_layer_name_yield}',
          data=coverage_update_payload,
          headers=headers,
          auth=HTTPBasicAuth(username, password)
      )
    
      # Check the response status code
      if response.status_code == 200:
          print(f'Layer "{updated_layer_name_yield}" updated successfully.')
      else:
          print(f'Failed to update layer "{updated_layer_name_yield}". Status code: {response.status_code}')
          print(response.text)
    
    ##############################
    ### DELETE TEMPORARY FILES ###
    ##############################
    
    if delete_tmp_files=='yes':
     # Remove the intermidiate NetCDF file
     os.remove(netcdf_file_predictions)
     os.remove(netcdf_file_yield)

    return url, workspace_name, updated_layer_name_predictions, updated_layer_name_yield
