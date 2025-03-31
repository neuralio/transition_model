# TRANSITION Model

**Introduction**

This is the repository that contains the codes for the TRANSITION Model under the TRANSITION Project. 


**Instructions**
1. Before executing any of the RL scenarios, the user should first create their input data i.e., the data that are created after executing Crop Suitability and PV Suitability services (See README inside the folders).

2. In order to execute any of the RL scenarios, the user has the option to choose the --scenario parameter. 
This option refers to the corresponding climate scenario: <br /> 
26 → RCP 2.6 (Low emissions, strong mitigation) <br />  
45 → RCP 4.5 (Intermediate emissions, moderate mitigation) <br />  
85 → RCP 8.5 (High emissions, minimal mitigation) <br />  

There is also the option "past", which refers to the past configuration covering simulations for the years 2010-2020.  

**Initiate the conda environment**
Initiate the conda "mesa" environment and install all the required packages with the following commands:

conda env create --file environment.yaml

pip install -r requirements.txt  

**Base_scenario folder:**

The folder for the Base Scenario. 

Base_scenario/simple_rl_env contains the Python scripts to execute the RL. 
In order to execute the code, change the "MAIN_DIR" and the "geojson_path" in Utils/config.py file.

The structure of the code is: 

|-- main.py <br /> 
|-- model.py <br /> 
|-- gym_env.py  <br /> 
|-- agents.py <br /> 
|-- explainability.py <br /> 
|-- validation.py <br /> 
|-- utils <br /> 
|   |---data_import.py <br /> 
|   |---plots.py <br /> 
|   |---config.py <br /> 


Execute the code: <br /> 

To Check Environment Only <br /> 
python main.py --scenario 26 --check-env or python main.py --past --check-env <br /> 

To Train the Model Only: <br /> 
python main.py --scenario 26 --train or python main.py --past --train <br /> 

To Check Environment and Train:  <br /> 
python main.py --scenario 26 --check-env --train or python main.py --past--check-env --train<br />  

Plot Model Results (only if saved model is available) <br />
python main.py --scenario 26 --plot  <br /> 

Execute the Validation with Ensemble modelling and Uncertainty Quantification <br />  
python validation.py --scenario 26 --output-folder outputs <br />  

Execute the Explainability with SHAP <br />  
python explainability.py --scenario 26 --output-folder outputs <br />  


**PECS_Scenario folder**

Contains the pipeline to <br />
1. Create a dynamic model that accepts at run time PECs parameters and saves it  <br />
2. Load the trained model and inserts modified pecs in json format  <br />
See README inside the folder <br /> 


**Full_Scenario folder**

Contains the pipeline to <br />
1. Create a dynamic model that accepts at run time PECs + Green Credit Policy parameters and saves it  <br />
2. Load the trained model and inserts modified pecs in json format  <br />
See README inside the folder <br /> 

**Full_Scenario_CCA folder**
Contains the pipeline to <br />
1. Create a dynamic model that accepts at run time PECs + Green Credit Policy parameters + Climate change adaptation (climate losses) and saves it  <br />
2. Load the trained model and inserts modified pecs in json format  <br />
See README inside the folder <br /> 

**EO folder:**

Contains the pipelines to extract the NDVI and Land Surface Temperature (LST) from Sentinel-2 MSI and Sentinel-3 SLSTR data, respectively. 
It also contains the scripts to download and preprocess Copernicus GLO-30 Digital Elevation Model (DEM).




