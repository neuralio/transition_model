#!/bin/bash

set -x

export INPUT=/home/ESA-BIC/METEO_DATA/DATA/CORDEX
export OUTPUT=/home/ESA-BIC/Land_Suitability/INPUT/CLIMATE_PROJECTIONS
export GRID_FILE=/home/ESA-BIC/Land_Suitability/PREPROCESSING_SCRIPTS/grid_lonlat.txt
VARIABLES=( pr tas tasmin tasmax )
RCMS=( rcp26 rcp45 rcp85 )

if [ ! -d ${OUTPUT} ]; then mkdir -p ${OUTPUT}; fi

for var in ${VARIABLES[@]}; do

for rcm in ${RCMS[@]}; do

cdo -O mergetime ${INPUT}/*_${var}_*${rcm}*.nc ${OUTPUT}/tmp_${var}_${rcm}_2021_2100.nc
cdo -O remapnn,${GRID_FILE} ${OUTPUT}/tmp_${var}_${rcm}_2021_2100.nc ${OUTPUT}/tmp1_${var}_${rcm}_2021_2100.nc
ncks -O -C -x -v time_bnds,height ${OUTPUT}/tmp1_${var}_${rcm}_2021_2100.nc ${OUTPUT}/tmp2_${var}_${rcm}_2021_2100.nc

if [ ${var} == "pr" ]; then

cdo -O mulc,86400 ${OUTPUT}/tmp2_${var}_${rcm}_2021_2100.nc ${OUTPUT}/${var}_${rcm}_2021_2100.nc

else

cdo -O addc,-273.15 ${OUTPUT}/tmp2_${var}_${rcm}_2021_2100.nc ${OUTPUT}/${var}_${rcm}_2021_2100.nc

fi

rm -f ${OUTPUT}/tmp*.nc

done

done

