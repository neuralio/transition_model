"""
========================================================================================
Land Surface Temperature (LST) Processing using ESA SNAP GPT
========================================================================================

**Description:**
This project provides an ESA SNAP Graph Processing Tool (GPT) XML workflow (GPT_S3_LST.xml) for computing Land Surface Temperature (LST) from Sentinel-3 data. The workflow processes LST from Sentinel-3 SLSTR products and exports the result as a GeoTIFF.

**How to execute the script:**
gpt GPT_S3_LST.xml -PInput=<INPUT_NETCDF> -Pwkt="<AOI_WKT>" -POutput_LST=<OUTPUT_TIFF>

Arguments Explanation: 
-PInput Path to Sentinel-3 NetCDF input file
-Pwkt   Well-Known Text (WKT) defining the Area of Interest (AOI)
-POutput_LST    Path to the output GeoTIFF file

"""

