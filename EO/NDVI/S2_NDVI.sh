#!/bin/bash

##Command line inputs
#Path to the SNAP GPT xml file 
snapGptXmlPath="$1"
#WKT of AOI 
wkt="$2"
#Path to source products
sourceDir="$3"
#Path to target products
targetDir="$4"


#Constants
XML_HEADER="MTD_MSIL2A.xml"
OUTPUT_EXTENSION="tif"

#Main process
for file in $(ls -1d "${sourceDir}"/S2*.SAFE); do
    absoluteSafePath=$(cd ${file} && pwd -P)
    sourceFile=${absoluteSafePath}/${XML_HEADER}
    echo $file
    safeFile=$(basename $absoluteSafePath) #Extract SAFE folder name with .SAFE extension, form absolute path#
    var_substr=$(expr substr "$safeFile" 1 19)
    targetFile="${targetDir}/${var_substr}.${OUTPUT_EXTENSION}"
    targetFile1="${targetDir}/${var_substr}_NDVI.${OUTPUT_EXTENSION}"
    gpt9 "${snapGptXmlPath}" -PInput="${sourceFile}" -Pwkt="${wkt}" -POutput_NDVI="${targetFile}"
    gdal_translate -of GTiff -a_nodata 255 "${targetFile}" "${targetFile1}"
    rm -r "${targetFile}"
done 


