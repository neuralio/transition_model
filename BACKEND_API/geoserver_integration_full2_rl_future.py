import requests
from requests.auth import HTTPBasicAuth
import sys
from geo.Geoserver import Geoserver
import pandas as pd
import xarray as xr
import numpy as np
import os
import json

convert_to_netcdf = 'yes'
create_store = 'yes'
publish_layer = 'yes'
delete_tmp_files = 'no'

def upload_2_geoserver_full2_rl_future(agent_data_file):
    ###################
    #### USER EDIT ####
    ###################
    
    # Define the GeoServer URL and authentication details
    geoserver_url = '<geoserver_url:port>/geoserver/rest'  # Replace with your GeoServer URL
    url = '<geoserver_url:port>/geoserver'
    username = '<geoserver_username>'  # Replace with your GeoServer username
    password = '<geoserver_password>'  # Replace with your GeoServer password
    
    # Load configuration (if needed to get task_id and scenario)
    def load_config(config_path):
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config, config.get("task_id"), config.get("user_id")
    
    MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_JSON_PATH = os.path.join(MAIN_DIR, "input_mongo_full2_future.json")
    PREDICTION_DIR = os.path.join(MAIN_DIR, "RESULTS_RL/FULL2_RL")
    config, task_id, user_id = load_config(INPUT_JSON_PATH)
    scenario = config["model_parameters"]["scenario"]
    task_prediction_dir = os.path.join(PREDICTION_DIR, f"{task_id}_{user_id}")

    # Build the prediction file path for the host using the scenario subdirectory.
    # We now expect the file in: /app/RESULTS_RL/FULL2_RL/{task_id}_{user_id}/RCP{scenario}/agent_data_over_time.csv
    file_path_predictions_HOST = agent_data_file  # Use the path passed in from the endpoint.
    
    # Details for predictions (for example)
    workspace_name = 'Land_Suitability'
    new_namespace_uri = 'Land_Suitability'
    store_name_predictions = f'FULL2_RL_RCP{scenario}_{task_id}_{user_id}'
    file_path_predictions = f'/opt/geoserver_data/data/ESA-AGENTS/BACKEND_API/Demo_With_Geoserver/RESULTS_RL/FULL2_RL/{task_id}_{user_id}/agent_data_over_time.nc'
    upload_layer_name_predictions = f'decision'
    updated_layer_name_predictions = f'FULL2_RL_RCP{scenario}_{task_id}_{user_id}'
    
    ##########################
    #### CONVERT 2 NETCDF ####
    ##########################
    
    # Convert the CSV predictions to NetCDF.
    df_predictions = pd.read_csv(file_path_predictions_HOST)
    lats = np.sort(df_predictions["lat"].unique())
    lons = np.sort(df_predictions["lon"].unique())
    data_predictions = np.full((len(lats), len(lons)), np.nan)
    
    for _, row in df_predictions.iterrows():
        j = np.where(lats == row["lat"])[0][0]
        k = np.where(lons == row["lon"])[0][0]
        data_predictions[j, k] = row["decision"]
    
    da_predictions = xr.DataArray(
        data_predictions,
        coords={"lat": lats, "lon": lons},
        dims=["lat", "lon"],
        name="decision"
    )
    
    ds_predictions = xr.Dataset({"decision": da_predictions})
    netcdf_file_predictions = f"{file_path_predictions_HOST[:-4]}.nc"
    ds_predictions.to_netcdf(netcdf_file_predictions, mode="w")
    
    #########################
    #### STORE - One off ####
    #########################
    
    if create_store == 'yes':
        # Delete the workspace if it already exists.
        geo = Geoserver(url, username=username, password=password)
        stores = geo.get_coveragestores(workspace=workspace_name)
    
        for existing_store in stores['coverageStores']['coverageStore']:
            if existing_store['name'] == store_name_predictions:
                print(f'Store {store_name_predictions} exists.')
                geo.delete_coveragestore(store_name_predictions, workspace=workspace_name)
                print(f'Store {store_name_predictions} was deleted')
    
        # Create the XML payload for the NetCDF coveragestore.
        coveragestore_payload = f"""
        <coverageStore>
            <name>{store_name_predictions}</name>
            <type>NetCDF</type>
            <enabled>true</enabled>
            <workspace>{workspace_name}</workspace>
            <url>{file_path_predictions}</url>
        </coverageStore>
        """
    
        headers = {'Content-Type': 'application/xml'}
     
        response = requests.post(
            f'{geoserver_url}/workspaces/{workspace_name}/coveragestores',
            data=coveragestore_payload,
            headers=headers,
            auth=HTTPBasicAuth(username, password)
        )
    
        if response.status_code == 201:
            print(f'NetCDF coveragestore "{store_name_predictions}" created successfully.')
        else:
            print(f'Failed to create NetCDF coveragestore "{store_name_predictions}". Status code: {response.status_code}')
            print(response.text)
    
    ####################
    ### LAYER UPLOAD ###
    ####################
    
    if publish_layer == 'yes':
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
     
        headers = {'Content-Type': 'application/xml'}
    
        response = requests.post(
            f'{geoserver_url}/workspaces/{workspace_name}/coveragestores/{store_name_predictions}/coverages',
            data=coverage_payload,
            headers=headers,
            auth=HTTPBasicAuth(username, password)
        )
    
        if response.status_code == 201:
            print(f'Layer "{upload_layer_name_predictions}" published successfully.')
        else:
            print(f'Failed to publish layer "{upload_layer_name_predictions}". Status code: {response.status_code}')
            print(response.text)
    
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
     
        headers = {'Content-Type': 'application/xml'}
      
        response = requests.put(
            f'{geoserver_url}/workspaces/{workspace_name}/coveragestores/{store_name_predictions}/coverages/{upload_layer_name_predictions}',
            data=coverage_update_payload,
            headers=headers,
            auth=HTTPBasicAuth(username, password)
        )
      
        if response.status_code == 200:
            print(f'Layer "{updated_layer_name_predictions}" updated successfully.')
        else:
            print(f'Failed to update layer "{updated_layer_name_predictions}". Status code: {response.status_code}')
            print(response.text)
    
    if delete_tmp_files == 'yes':
        os.remove(netcdf_file_predictions)
        # os.remove(netcdf_file_yield)  # if defined
    
    return url, workspace_name, updated_layer_name_predictions
