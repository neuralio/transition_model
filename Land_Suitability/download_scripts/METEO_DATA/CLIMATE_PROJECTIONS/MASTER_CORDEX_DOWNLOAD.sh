#!/bin/bash
#
export SCRIPTS=/home/ESA-BIC/METEO_DATA/SCRIPTS/CLIMATE_PROJECTIONS

#--- Download Temperature Data
python ${SCRIPTS}/temperature/download_tas_RCP26.py
python ${SCRIPTS}/temperature/download_tas_RCP45.py
python ${SCRIPTS}/temperature/download_tas_RCP85.py
python ${SCRIPTS}/temperature/download_tasmin_RCP26.py
python ${SCRIPTS}/temperature/download_tasmin_RCP45.py
python ${SCRIPTS}/temperature/download_tasmin_RCP85.py
python ${SCRIPTS}/temperature/download_tasmax_RCP26.py
python ${SCRIPTS}/temperature/download_tasmax_RCP45.py
python ${SCRIPTS}/temperature/download_tasmax_RCP85.py

#--- Download Precipitation Data
python ${SCRIPTS}/precipitation/download_prc_RCP26.py
python ${SCRIPTS}/precipitation/download_prc_RCP45.py
python ${SCRIPTS}/precipitation/download_prc_RCP85.py
