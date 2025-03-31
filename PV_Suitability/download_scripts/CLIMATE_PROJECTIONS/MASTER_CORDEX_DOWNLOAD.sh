#!/bin/bash
#
export SCRIPTS=/home/ESA-BIC/PV_Suitability/download_scripts/CLIMATE_PROJECTIONS

#--- Download Solar Radiation Data
python ${SCRIPTS}/solar_radiation/download_solar_RCP26.py
python ${SCRIPTS}/solar_radiation/download_solar_RCP45.py
python ${SCRIPTS}/solar_radiation/download_solar_RCP85.py
