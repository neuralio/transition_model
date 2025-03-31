import os

# Base directory for data
MAIN_DIR = "/Users/mbanti/Documents/Projects/esa_agents_local/Simple_RL/data"

# Subdirectories for different datasets
WHEAT_PAST = os.path.join(MAIN_DIR, "WHEAT_PAST")
WHEAT_FUTURE = os.path.join(MAIN_DIR, "WHEAT_FUTURE")
MAIZE_PAST = os.path.join(MAIN_DIR, "MAIZE_PAST")
MAIZE_FUTURE = os.path.join(MAIN_DIR, "MAIZE_FUTURE")
PV_PAST = os.path.join(MAIN_DIR, "PV_PAST")
PV_FUTURE = os.path.join(MAIN_DIR, "PV_FUTURE")

# For ploting purposes  
geojson_path = "/Users/mbanti/Documents/Projects/esa_agents_local/Simple_RL/data/geoBoundaries-CYP-ADM0.geojson"

# Get file paths based on the selected RCP scenario
def get_file_paths(scenario, past=False):
    if past:
        return {
            "pv_suitability_file": os.path.join(PV_PAST, "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv"),
            "pv_yield_file": os.path.join(PV_PAST, "PAST_PV_YIELD.csv"),
            "cs_wheat_file": os.path.join(WHEAT_PAST, "WHEAT_PAST_LUSA_PREDICTIONS.csv"),
            "cy_wheat_file": os.path.join(WHEAT_PAST, "AquaCrop_Results_Past.csv"),
            "cs_maize_file": os.path.join(MAIZE_PAST, "MAIZE_PAST_LUSA_PREDICTIONS.csv"),
            "cy_maize_file": os.path.join(MAIZE_PAST, "AquaCrop_Results_Past.csv"),
        }
    else:
        return {
            "pv_suitability_file": os.path.join(PV_FUTURE, f"RCP{scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv"),
            "pv_yield_file": os.path.join(PV_FUTURE, f"RCP{scenario}_PV_SUITABILITY_ANNUAL_ELECTRICITY_SAVINGS_2021_2030.csv"),
            "cs_maize_file": os.path.join(MAIZE_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv"),
            "cs_wheat_file": os.path.join(WHEAT_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv"),
            "cy_maize_file": os.path.join(MAIZE_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv"),
            "cy_wheat_file": os.path.join(WHEAT_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv"),
        }

