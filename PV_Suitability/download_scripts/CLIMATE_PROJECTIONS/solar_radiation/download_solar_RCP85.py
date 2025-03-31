########################################################################################
# This script downloads climate projections (RCP 2.6) from CORDEX (2021-2100) for:     #
# solar radiation (W/m2)                                                               #
# The user needs to define USER EDIT portion of the script:                            #
# 1. The directory in which they want to store the data                                #
#                                                                                      #
# After that, the script will download in 5-year files the requested data.             #
#                                                                                      #
# To execute this script:                                                              #
# 1. conda activate seasonal                                                           #
# 2. python download_solar_RCP26.py                                                    #
#                                                                                      #
# Authored by Theano Mamouka, 4/6/2024                                                 #
########################################################################################

import cdsapi
import os

c = cdsapi.Client()

##################### USER EDIT #####################

#--- Define the output directory to store the data
outdir='/home/ESA-BIC/METEO_DATA/DATA/CORDEX'

################ END OF USER EDIT ####################

#--- Create the outdir
if not os.path.exists(outdir):
  os.makedirs(outdir)

#--- Download the data

### 2021-2100

import cdsapi

dataset = "projections-cordex-domains-single-levels"
request = {
    "domain": "europe",
    "experiment": "rcp_8_5",
    "horizontal_resolution": "0_11_degree_x_0_11_degree",
    "temporal_resolution": "daily_mean",
    "variable": ["surface_solar_radiation_downwards"],
    "gcm_model": "cnrm_cerfacs_cm5",
    "rcm_model": "cnrm_aladin63",
    "ensemble_member": "r1i1p1",
    "start_year": [
        "2021",
        "2026",
        "2031",
        "2036",
        "2041",
        "2046",
        "2051",
        "2056",
        "2061",
        "2066",
        "2071",
        "2076",
        "2081",
        "2086",
        "2091",
        "2096"
    ],
    "end_year": [
        "2025",
        "2030",
        "2035",
        "2040",
        "2045",
        "2050",
        "2055",
        "2060",
        "2065",
        "2070",
        "2075",
        "2080",
        "2085",
        "2090",
        "2095",
        "2100"
    ]
}

target = f'{outdir}/CORDEX_SOLAR_RCP85_tas_2021-2100.zip'
client = cdsapi.Client()
client.retrieve(dataset, request, target)


