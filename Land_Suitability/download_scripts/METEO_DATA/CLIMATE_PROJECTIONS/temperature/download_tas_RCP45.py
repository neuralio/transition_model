########################################################################################
# This script downloads climate projections (RCP 4.5) from CORDEX (2021-2100) for:     #
# 2m air temperature (K)                                                               #
# The user needs to define USER EDIT portion of the script:                            #
# 1. The directory in which they want to store the data                                #
#                                                                                      #
# After that, the script will download in 5-year files the requested data.             #
#                                                                                      #
# To execute this script:                                                              #
# 1. conda activate cdo_env                                                            #
# 2. python download_tas_RCP45.py                                                      #
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

#--- Download the 6-month predictions

### 2021-2035
c.retrieve(
    'projections-cordex-domains-single-levels',
    {
        'format': 'zip',
        'domain': 'europe',
        'experiment': 'rcp_4_5',
        'horizontal_resolution': '0_11_degree_x_0_11_degree',
        'temporal_resolution': 'daily_mean',
        'variable': '2m_air_temperature',
        'gcm_model': 'cnrm_cerfacs_cm5',
        'rcm_model': 'cnrm_aladin63',
        'ensemble_member': 'r1i1p1',
        'start_year': [
            '2021', '2026', '2031',
        ],
        'end_year': [
            '2025', '2030', '2035',
        ],
    },
    f'{outdir}/CORDEX_RCP45_tas_2021-2035.zip')

print(f'DOWNLOAD COMPLETE: {outdir}/CORDEX_RCP45_tas_2021-2035.zip')

### 2036-2050
c.retrieve(
    'projections-cordex-domains-single-levels',
    {
        'format': 'zip',
        'domain': 'europe',
        'experiment': 'rcp_4_5',
        'horizontal_resolution': '0_11_degree_x_0_11_degree',
        'temporal_resolution': 'daily_mean',
        'variable': '2m_air_temperature',
        'gcm_model': 'cnrm_cerfacs_cm5',
        'rcm_model': 'cnrm_aladin63',
        'ensemble_member': 'r1i1p1',
        'start_year': [
            '2036', '2041', '2046',
        ],
        'end_year': [
            '2040', '2045', '2050',
        ],
    },
    f'{outdir}/CORDEX_RCP45_tas_2036-2050.zip')

print(f'DOWNLOAD COMPLETE: {outdir}/CORDEX_RCP45_tas_2036-2050.zip')

### 2051-2065
c.retrieve(
    'projections-cordex-domains-single-levels',
    {
        'format': 'zip',
        'domain': 'europe',
        'experiment': 'rcp_4_5',
        'horizontal_resolution': '0_11_degree_x_0_11_degree',
        'temporal_resolution': 'daily_mean',
        'variable': '2m_air_temperature',
        'gcm_model': 'cnrm_cerfacs_cm5',
        'rcm_model': 'cnrm_aladin63',
        'ensemble_member': 'r1i1p1',
        'start_year': [
            '2051', '2056', '2061',
        ],
        'end_year': [
            '2055', '2060', '2065',
        ],
    },
    f'{outdir}/CORDEX_RCP45_tas_2051-2065.zip')

print(f'DOWNLOAD COMPLETE: {outdir}/CORDEX_RCP45_tas_2051-2065.zip')

### 2066-2080
c.retrieve(
    'projections-cordex-domains-single-levels',
    {
        'format': 'zip',
        'domain': 'europe',
        'experiment': 'rcp_4_5',
        'horizontal_resolution': '0_11_degree_x_0_11_degree',
        'temporal_resolution': 'daily_mean',
        'variable': '2m_air_temperature',
        'gcm_model': 'cnrm_cerfacs_cm5',
        'rcm_model': 'cnrm_aladin63',
        'ensemble_member': 'r1i1p1',
        'start_year': [
            '2066', '2071', '2076',
        ],
        'end_year': [
            '2070', '2075', '2080',
        ],
    },
    f'{outdir}/CORDEX_RCP45_tas_2066-2080.zip')

print(f'DOWNLOAD COMPLETE: {outdir}/CORDEX_RCP45_tas_2066-2080.zip')

### 2081-2095
c.retrieve(
    'projections-cordex-domains-single-levels',
    {
        'format': 'zip',
        'domain': 'europe',
        'experiment': 'rcp_4_5',
        'horizontal_resolution': '0_11_degree_x_0_11_degree',
        'temporal_resolution': 'daily_mean',
        'variable': '2m_air_temperature',
        'gcm_model': 'cnrm_cerfacs_cm5',
        'rcm_model': 'cnrm_aladin63',
        'ensemble_member': 'r1i1p1',
        'start_year': [
            '2081', '2086', '2091',
        ],
        'end_year': [
            '2085', '2090', '2095',
        ],
    },
    f'{outdir}/CORDEX_RCP45_tas_2081-2095.zip')

print(f'DOWNLOAD COMPLETE: {outdir}/CORDEX_RCP45_tas_2081-2095.zip')

### 2096-2100
c.retrieve(
    'projections-cordex-domains-single-levels',
    {
        'format': 'zip',
        'domain': 'europe',
        'experiment': 'rcp_4_5',
        'horizontal_resolution': '0_11_degree_x_0_11_degree',
        'temporal_resolution': 'daily_mean',
        'variable': '2m_air_temperature',
        'gcm_model': 'cnrm_cerfacs_cm5',
        'rcm_model': 'cnrm_aladin63',
        'ensemble_member': 'r1i1p1',
        'start_year': [
            '2096',
        ],
        'end_year': [
            '2100',
        ],
    },
    f'{outdir}/CORDEX_RCP45_tas_2096-2100.zip')

print(f'DOWNLOAD COMPLETE: {outdir}/CORDEX_RCP45_tas_2096-2100.zip')
