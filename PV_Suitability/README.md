"""
========================================================================================
PV Suitability Service, Future and Past modules, with dynamic change of parameters
========================================================================================

**Description:**

Folder "download_scripts" contains the Python scripts to download the Climate Projections and the ERA5-LAND data.

Folder "preprocessing_scripts" contains the scripts to preprocess the Climate Projections and the ERA5-LAND data.

This version of PV Suitability scripts can be run for each future climate scenario (RCP 2.6, RCP 4.5, RCP 8.5) as well as for the past (2010 - 2020).

At first, change the proximity to powerlines and the road network accessibility distance in the config.json file. 

Install all the required packages in a conda environment: pip install -r requirements.txt

Then, execute the script as: 

python PV_Suitability_v02.py --scenario 26 --config config.json

For the past module, execute the script as: 

python PV_Suitability_past.py --config config.json


The predictions.csv and the PV yield calculations will be saved in /PREDICTIONS/CSVs and the plots will be saved in /PLOTS. 

"""