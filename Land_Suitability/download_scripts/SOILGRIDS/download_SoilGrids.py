import subprocess
import os
import sys
import sys

##############################################################################################################
#### This scripts downaloads soilgrids data (sand,silt,clay,organic carbon content)  properties from:
#### https://soilgrids.org/.
#### The bounding box of the desired area is user defined.
##############################################################################################################
# Input: bounding boxes for the AOI 
# Output: tif files with soil data
# How to run the script:
# 1) Activate the appropriate environment with the necessary modules (esa_bic)
# 2) Excecute the script, after editing the bounding box and the Parent_Directory, like so:
#     python download_SoilGrids.py
# For example for a bounding box enclosing Greece, the necessary edits before executing the script are:
# 1) wgs84_lons=(19.0, 30.0)
#    wgs84_lats=(34.0, 42.0)
# 2) Parent_Directory='/home/ESA-BIC/SOILGRIDS/DATA/'
# Authored: V. Vourtlioti, T. Mamouka at 10.08.2022 
#############################################################################################################

######################### 1. Define bounding boxes. ####################################

# bounding box: 
wgs84_lons=('32.1', '34.7')
wgs84_lats=('34.4', '35.8')
bbox_original=(wgs84_lons[0], wgs84_lats[0], wgs84_lons[1], wgs84_lats[1])

print('Defined Bounding Box in EPSG:4326:')
print(bbox_original)


###################### 2. Downalod the data for all depths and all AOIs ###########################

depths=(0, 5, 15, 30, 60, 100)

box=bbox_original
Parent_Directory='/home/ESA-BIC/SOILGRIDS/DATA/' # This defines the directory in which the data will be downloaded. Edit accordingly.

file_sizes=[]
loop=0

for k in range(1,4):
  ############ START CHEMICAL SOIL ###################
    if k==1:
     # Downaload Soil organic carbon content 
     name='soc'
     directory=name
     parent_dir=Parent_Directory
     path = os.path.join(parent_dir, directory)
     if not os.path.exists(path):
         os.makedirs(path)
     os.chdir(path)
    if k==2:
     # Downaload Cation exchange capacity at ph 7
     name='cec'
     directory=name
     parent_dir=Parent_Directory
     path = os.path.join(parent_dir, directory)
     if not os.path.exists(path):
         os.makedirs(path)
     os.chdir(path)
    if k==3:
     # Downaload Soil pH in H2O 
     name='phh2o'
     directory=name
     parent_dir=Parent_Directory
     path = os.path.join(parent_dir, directory)
     if not os.path.exists(path):
         os.makedirs(path)
     os.chdir(path)
    for i in range(0,len(depths)-1):
        # This is our shell command, executed in subprocess.
        print("Downloading "+name+" for depth "+ str(depths[i])+'_'+str(depths[i+1]))
        #print("curl", "https://maps.isric.org/mapserv?map=/map/"+name+".map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID="+name+"_"+str(depths[i])+"-"+str(depths[i+1])+"cm_mean&FORMAT=image/tiff&SUBSET=long("+str(wgs84_lons[0])+","+str(wgs84_lons[1])+")&SUBSET=lat("+str(wgs84_lats[0])+","+str(wgs84_lats[1])+")&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326", "-o", name+"_"+str(depths[i])+"-"+str(depths[i+1])+"cm_mean.tif")
        p = subprocess.call([
        "curl", "https://maps.isric.org/mapserv?map=/map/"+name+".map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID="+name+"_"+str(depths[i])+"-"+str(depths[i+1])+"cm_mean&FORMAT=image/tiff&SUBSET=long("+str(wgs84_lons[0])+","+str(wgs84_lons[1])+")&SUBSET=lat("+str(wgs84_lats[0])+","+str(wgs84_lats[1])+")&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326", "-o", name+"_"+str(depths[i])+"-"+str(depths[i+1])+"cm_mean.tif"
                    ], shell=False)

        #--- Check for corrupted files
        file_size=os.path.getsize(name+"_"+str(depths[i])+"-"+str(depths[i+1])+"cm_mean.tif")
        file_sizes.append(file_size)

        if loop>0:
         #print('File size of current file:', file_size, ', File Size of previous file:', file_sizes[loop-1])
         if file_size<=(file_sizes[loop-1]-500):
          p = subprocess.call([
        "curl", f"https://maps.isric.org/mapserv?map=/map/{name}.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID={name}_{str(depths[i])}-{str(depths[i+1])}cm_mean&FORMAT=image/tiff&SUBSET=long({wgs84_lons[0]},{wgs84_lons[1]})&SUBSET=lat({wgs84_lats[0]},{wgs84_lats[1]})&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326", "-o", name+"_"+str(depths[i])+"-"+str(depths[i+1])+"cm_mean.tif"
                    ], shell=False)

        loop=loop+1


  ################################## END OF SCRIPT ############################################################
