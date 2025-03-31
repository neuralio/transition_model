#!/bin/bash

set -x

export INPUT=/home/ESA-BIC/METEO_DATA/DATA/CORDEX
export OUTPUT=/home/ESA-BIC/PV_Suitability/INPUT/CLIMATE_PROJECTIONS
export GRID_FILE=/home/ESA-BIC/PV_Suitability/preprocessing_scripts/grid_lonlat.txt
VARIABLES=( rsds )
RCMS=( rcp26 rcp45 rcp85 )

if [ ! -d ${OUTPUT} ]; then mkdir -p ${OUTPUT}; fi

for var in ${VARIABLES[@]}; do

for rcm in ${RCMS[@]}; do

cdo -O mergetime ${INPUT}/*_${var}_*${rcm}*.nc ${OUTPUT}/tmp_${var}_${rcm}_2021_2100.nc
cdo -O remapnn,${GRID_FILE} ${OUTPUT}/tmp_${var}_${rcm}_2021_2100.nc ${OUTPUT}/tmp1_${var}_${rcm}_2021_2100.nc
ncks -O -C -x -v time_bnds,height ${OUTPUT}/tmp1_${var}_${rcm}_2021_2100.nc ${OUTPUT}/tmp2_${var}_${rcm}_2021_2100.nc

if [ ${var} == "rsds" ]; then

### Convert to W/m2 to kWh/m2/day

cdo -O -setunit,"kWh/m2/day" -mulc,24 -divc,1000 ${OUTPUT}/tmp2_${var}_${rcm}_2021_2100.nc ${OUTPUT}/${var}_${rcm}_2021_2100.nc

fi

rm -f ${OUTPUT}/tmp*.nc

done

done

