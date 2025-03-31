#!/bin/bash

set -x

##################################################################################
# This script crops the raw data from DORDEX to a user defined bounding box.     #
# Order of execution:                                                            #
# 1. MASTER_CORDEX_DOWNLOAD.sh                                                   #
# 2. MASTER_CORDEX_CROP.sh                                                       #
#                                                                                #
# Conda Environment: esa_bic                                                     #
#                                                                                #
# Authored by Theano Mamouka, 05/06/2024                                         #
##################################################################################

################################  USER EDIT  #####################################

#--- Define the directory where the raw CORDEX data have been downloaded
export RAW_DATA=/home/ESA-BIC/METEO_DATA/DATA/CORDEX

#--- Define the bounding box
export lat_min=34.4
export lat_max=35.8
export lon_min=32.1
export lon_max=34.7

############################  END OF USER EDIT  ##################################

#--- Decompress and remove

cd  ${RAW_DATA}
zip_files=*.zip 

for zip_file in ${zip_files[@]}; do

unzip ${zip_file} && rm -f ${zip_file}

done

#--- Crop for bbox

cd  ${RAW_DATA}
nc_files=*.nc

for nc_file in ${nc_files[@]}; do

cdo -O sellonlatbox,${lon_min},${lon_max},${lat_min},${lat_max} ${nc_file} CROPPED_${nc_file}
rm -f ${nc_file}

done



