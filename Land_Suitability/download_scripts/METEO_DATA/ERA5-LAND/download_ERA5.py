########################################################################################
# This script downloads reanalysis data from ERA5 for the parameters:                  #
# Mean surface net long-wave radiation flux,                                           #
# Mean surface downward long-wave radiation flux,                                      #
# Mean surface downward short-wave radiation flux                                      #
# The user needs to define USER EDIT portion of the script:                            #
# 1. The bounding box (minimum and maximum lon&lat)                                    #
# 2. The starting and ending year of the time period they are interested in            #
# 3. The directory in which they want to store the data                                # 
#                                                                                      #
# After that, the script will download in monthly files the requested data.            #
#                                                                                      #
# To execute this script:                                                              #
# 1. conda activate cdo_env                                                            #
# 2. python download_ERA5.py                                                           #
#                                                                                      #
# Authored by Theano Mamouka, 4/6/2024                                                 #
########################################################################################

import cdsapi
import os

c = cdsapi.Client()

##################### USER EDIT #####################

#--- Define the boundaries for the AOI

lon_min=32.1
lon_max=34.7
lat_min=34.4
lat_max=35.8

#--- Define the output directory to store the data
outdir='/home/ESA-BIC/METEO_DATA/DATA/ERA5'

#--- Define the starting and ending year
year_start=2010
year_end=2020

################ END OF USER EDIT ####################

#--- Create the outdir
if not os.path.exists(outdir):
  os.makedirs(outdir)


#--- Download the data per month
for year in range(year_start, year_end+1):
 for month in range(1, 13):
  month=str(month).zfill(2)

  c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': [
            'mean_surface_downward_long_wave_radiation_flux', 'mean_surface_downward_short_wave_radiation_flux', 'mean_surface_net_long_wave_radiation_flux',
        ],
        'year': str(year),
        'month': str(month),
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            lat_max, lon_min, lat_min,
            lon_max,
        ],
        'format': 'grib',
    },
    f'{outdir}/ERA5_{str(year)}{str(month)}.grib')

  print(f'DOWNLOAD COMPLETE: {outdir}/ERA5_{str(year)}{str(month)}.grib')

