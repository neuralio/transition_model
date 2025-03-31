#!/bin/bash

#set -x

export INPUT=/home/ESA-BIC/SOILGRIDS/DATA
export OUTPUT=/home/ESA-BIC/Land_Suitability/INPUT/SOILGRIDS
export GRID_FILE=/home/ESA-BIC/Land_Suitability/PREPROCESSING_SCRIPTS/grid_lonlat.txt
VARIABLES=( cec  phh2o  soc )

if [ ! -d ${OUTPUT} ]; then mkdir -p ${OUTPUT}; fi

for var in ${VARIABLES[@]}; do

for file in ${INPUT}/${var}/*.tif; do

filename=$(basename "$file" .tif)

gdalwarp -t_srs "EPSG:4326" -of NetCDF ${INPUT}/${var}/${filename}.tif ${INPUT}/${var}/tmp_${filename}.nc
ncrename -O -v Band1,${var} ${INPUT}/${var}/tmp_${filename}.nc ${INPUT}/${var}/tmp1_${filename}.nc
cdo -O remapnn,${GRID_FILE} ${INPUT}/${var}/tmp1_${filename}.nc ${INPUT}/${var}/${filename}.nc

rm -f ${INPUT}/${var}/tmp*

done

cdo -O -divc,10 -ensmean ${INPUT}/${var}/*.nc ${OUTPUT}/${var}.nc
rm -f ${INPUT}/${var}/*.nc

done

