"""
========================================================================================
Crop Suitability Service, Future and Past modules, with dynamic change of crop type
========================================================================================

**Description:**

Folder "download_scripts" contains the Python scripts to download the Climate Projections, the ERA5-LAND data, the DEM and the SoilGrids data.

Folder "preprocessing_scripts" contains the scripts to preprocess the Climate Projections, the ERA5-LAND data, the DEM and the SoilGrids data.

This version of Crop Suitability Service can be run for each future climate scenario (RCP 2.6, RCP 4.5, RCP 8.5) as well as for the past (2010 - 2020).

At first, change the crop type in the config.json file. 

Install all the required packages in a conda environment: pip install -r requirements.txt

Then, execute the script as: 

python crop_suitability_v02.py --scenario 26 --config config.json

For the past module, execute the script as: 

python crop_suitability_past.py --config config.json

The predictions.csv will be saved in /PREDICTIONS/CSVs and the plots will be saved in /PLOTS. 

"""