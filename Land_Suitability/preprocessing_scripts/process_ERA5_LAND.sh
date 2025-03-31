#!/bin/bash

set -x

export INPUT=/home/ESA-BIC/METEO_DATA/DATA/ERA5_LAND
export OUTPUT=/home/ESA-BIC/Land_Suitability/INPUT/ERA5_LAND

if [ ! -d ${OUTPUT} ]; then mkdir -p ${OUTPUT}; fi

for file in ${INPUT}/*.grib; do

filename=$(basename "$file" .grib)

# Convert to netcdf and select Temperature and Precipitation
cdo -O -f nc copy ${INPUT}/${filename}.grib ${OUTPUT}/tmp_${filename}.nc
cdo -O -addc,-273.15 -chname,var167,temperature -selname,var167 ${OUTPUT}/tmp_${filename}.nc ${OUTPUT}/TMP_${filename}.nc
cdo -O -mulc,1000 -chname,var228,precipitation -selname,var228 ${OUTPUT}/tmp_${filename}.nc ${OUTPUT}/RAIN_${filename}.nc

rm -f ${OUTPUT}/tmp*.nc

done

# Aggregate Daily & Merge all the timesteps in one file
cdo -O mergetime ${OUTPUT}/TMP_ERA5_LAND_*.nc ${OUTPUT}/tmp_input_ERA5_LAND_TMP.nc
cdo -O delete,timestep=1 ${OUTPUT}/tmp_input_ERA5_LAND_TMP.nc ${OUTPUT}/tmp1_input_ERA5_LAND_TMP.nc
cdo -O daymean ${OUTPUT}/tmp1_input_ERA5_LAND_TMP.nc ${OUTPUT}/input_ERA5_LAND_TMP.nc
cdo -O mergetime ${OUTPUT}/RAIN_ERA5_LAND_*.nc ${OUTPUT}/tmp_input_ERA5_LAND_RAIN.nc
cdo -O -setrtoc,-inf,0,0 -deltat ${OUTPUT}/tmp_input_ERA5_LAND_RAIN.nc ${OUTPUT}/tmp1_input_ERA5_LAND_RAIN.nc
cdo -O daysum ${OUTPUT}/tmp1_input_ERA5_LAND_RAIN.nc ${OUTPUT}/input_ERA5_LAND_RAIN.nc
cdo -O -settime,00:00:00 -merge ${OUTPUT}/input_ERA5_LAND_TMP.nc ${OUTPUT}/input_ERA5_LAND_RAIN.nc ${OUTPUT}/input_ERA5_LAND.nc

rm -f ${OUTPUT}/TMP_ERA5_LAND_*.nc ${OUTPUT}/RAIN_ERA5_LAND_*.nc ${OUTPUT}/input_ERA5_LAND_TMP.nc ${OUTPUT}/input_ERA5_LAND_RAIN.nc  ${OUTPUT}/tmp*

