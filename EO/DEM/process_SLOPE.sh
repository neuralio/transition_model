#!/bin/bash

#set -x

export INPUT=/home/ESA-BIC/DEM/DATA
export OUTPUT=/home/ESA-BIC/Land_Suitability/INPUT/SLOPE
export GRID_FILE=/home/ESA-BIC/Land_Suitability/PREPROCESSING_SCRIPTS/grid_lonlat.txt

if [ ! -d ${OUTPUT} ]; then mkdir -p ${OUTPUT}; fi

gdaldem slope -of NetCDF -p -s 111120 -alg ZevenbergenThorne ${INPUT}/DEM_AOI.nc ${OUTPUT}/tmp_SLOPE.nc
cdo -O -setunit,% -setname,slope -remapnn,${GRID_FILE} ${OUTPUT}/tmp_SLOPE.nc ${OUTPUT}/SLOPE.nc

rm -f ${OUTPUT}/tmp*





