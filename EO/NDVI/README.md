"""
========================================================================================
NDVI Calculation using ESA SNAP GPT & Bash Script
========================================================================================

**Description:**
This project provides a Bash script (S2_NDVI.sh) and an ESA SNAP GPT XML file (GPT_NDVI_MB.xml) for automating the computation of Normalized Difference Vegetation Index (NDVI) from Sentinel-2 Level-2A data.

The Bash script automates the batch processing of Sentinel-2 SAFE files using ESA SNAP's Graph Processing Tool (GPT). It applies the XML workflow to compute NDVI, masks clouds, and exports the result as a GeoTIFF.


**How to execute the script:**
./S2_NDVI.sh <GPT_XML_PATH> <AOI_WKT> <SOURCE_DIR> <TARGET_DIR>

Arguments Explanation: 
GPT_XML_PATH    Path to GPT_NDVI_MB.xml <br /> 
AOI_WKT Well-Known Text (WKT) defining the Area of Interest (AOI) <br /> 
SOURCE_DIR  Directory containing Sentinel-2 .SAFE folders <br /> 
TARGET_DIR  Directory to store the NDVI output files <br /> 

"""

