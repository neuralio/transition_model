#!/bin/bash

set -x

export INPUT=/home/ESA-BIC/METEO_DATA/DATA/ERA5_LAND
export OUTPUT=/home/ESA-BIC/PV_Suitability/INPUT/ERA5_LAND
export GRID_FILE=/home/ESA-BIC/PV_Suitability/preprocessing_scripts/grid_lonlat.txt

if [ ! -d ${OUTPUT} ]; then mkdir -p ${OUTPUT}; fi

for file in ${INPUT}/*SOLAR*.grib; do

filename=$(basename "$file" .grib)

### Convert J/m2 to W/m2
cdo -f nc copy  ${INPUT}/${filename}.grib ${OUTPUT}/${filename}.nc
cdo -settime,00:00:00 -divc,86400 -selhour,23 -shifttime,-1hour ${OUTPUT}/${filename}.nc ${OUTPUT}/ACC_${filename}.nc

### Convert to W/m2 to kWh/m2/day
cdo -O -setunit,"kWh/m2/day" -mulc,24 -divc,1000 ${OUTPUT}/ACC_${filename}.nc ${OUTPUT}/FINAL_${filename}.nc

done

#--- Merge all the timesteps in one file
cdo -O mergetime ${OUTPUT}/FINAL_*.nc ${OUTPUT}/tmp_input_ERA5_LAND_SOLAR.nc
cdo -O delete,timestep=1 ${OUTPUT}/tmp_input_ERA5_LAND_SOLAR.nc ${OUTPUT}/input_ERA5_LAND_SOLAR.nc

rm -f ${OUTPUT}/ERA5_LAND*.nc ${OUTPUT}/ACC_*.nc ${OUTPUT}/FINAL_*.nc ${OUTPUT}/tmp_input_ERA5_LAND_SOLAR.nc

