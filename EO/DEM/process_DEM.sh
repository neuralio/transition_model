#!/bin/bash

set -x

export DATA_PATH=/home/ESA-BIC/DEM/DATA

cd ${DATA_PATH}

tif_files=*.tif

for tif_file in ${tif_files[@]}; do
 
 nc_file=${tif_file%????}.nc
 gdalwarp -of NetCDF ${tif_file} ${nc_file} 

done

gdal_merge.py -of NetCDF -o DEM_AOI.nc ${DATA_PATH}/*.nc

rm -f ${DATA_PATH}/Copernicus_DSM*.tif ${DATA_PATH}/Copernicus_DSM*.nc
