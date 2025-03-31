# run_with_modified_pecs_v04.py

#"""
#==================================================
#Script: run_with_modified_parameters_v04.py
#Author: Maria Banti
#Date: 2025-01-29
#Version: 1.1
#==================================================
#
#**Description:**
#This script executes a Reinforcement Learning simulation using a trained PPO (Proximal Policy Optimization) model 
#within a customized Mesa environment (`MesaEnv`). It modifies PECS (Policy, Emotion, Cognition, Social) parameters 
#based on a provided JSON configuration file or default settings, aggregates environmental data for a specified RCP 
#(Representative Concentration Pathway) scenario, and generates visualizations of agent decisions.
#
#**Features:**
#- Aggregates environmental data based on selected RCP scenarios.
#- Modifies PECS parameters from a JSON configuration file.
#- Automatically cleans up old logs and output files before execution.
#- Executes simulations for a specified number of episodes.
#- Captures and saves final agent decisions.
#- Generates heatmaps and scatter plots to visualize agent decisions.
#- Comprehensive logging for monitoring and debugging purposes.









import os
import json
import argparse
import pandas as pd
import numpy as np
import logging
import shutil 
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from models.mesa_env import MesaEnv  # Ensure this path is correct




def cleanup_files(log_file_path, output_dir_path):
    """
    Deletes the specified log file and all contents within the output directory.
    
    :param log_file_path: Path to the log file to be deleted.
    :param output_dir_path: Path to the output directory to be cleaned.
    """
    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
            print(f"Deleted old log file: {log_file_path}")
            logging.info(f"Deleted old log file: {log_file_path}")
        except Exception as e:
            print(f"Error deleting log file '{log_file_path}': {e}")
            logging.error(f"Error deleting log file '{log_file_path}': {e}")
    
    # Delete all files in the output directory
    if os.path.exists(output_dir_path):
        try:
            # Remove the entire directory
            shutil.rmtree(output_dir_path)
            print(f"Deleted old output directory: {output_dir_path}")
            logging.info(f"Deleted old output directory: {output_dir_path}")
            
            # Recreate an empty output directory
            os.makedirs(output_dir_path, exist_ok=True)
            print(f"Recreated output directory: {output_dir_path}")
            logging.info(f"Recreated output directory: {output_dir_path}")
        except Exception as e:
            print(f"Error deleting output directory '{output_dir_path}': {e}")
            logging.error(f"Error deleting output directory '{output_dir_path}': {e}")
    else:
        # If the output directory doesn't exist, create it
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"Created output directory: {output_dir_path}")
        logging.info(f"Created output directory: {output_dir_path}")

# def load_and_modify_pecs_params(original_path, modified_path=None, modifications=None):
#     """
#     Load PECS and scenario parameters from a JSON file, apply modifications, and optionally save them.

#     :param original_path: Path to the original parameters JSON file.
#     :param modified_path: Path to save the modified parameters JSON file.
#     :param modifications: A dictionary specifying the modifications.
#     :return: Modified parameters dictionary.
#     """
#     with open(original_path, 'r') as f:
#         parameters = json.load(f)

#     if modifications:
#         for category, changes in modifications.items():
#             if category in parameters:
#                 for param, value in changes.items():
#                     if param in parameters[category]:
#                         parameters[category][param] = value
#                     else:
#                         raise ValueError(f"Parameter '{param}' not found in category '{category}'.")
#             else:
#                 raise ValueError(f"Category '{category}' not found in parameters.")

#     if modified_path:
#         # Ensure the parent directory exists
#         os.makedirs(os.path.dirname(modified_path), exist_ok=True)
#         with open(modified_path, 'w') as f:
#             json.dump(parameters, f, indent=4)

#     return parameters

def load_and_modify_pecs_params(original_path, modified_path=None, modifications=None):
    with open(original_path, 'r') as f:
        parameters = json.load(f)

    if modifications:
        for category, changes in modifications.items():
            if category not in parameters:
                parameters[category] = {}  # Ensure category exists before modification
            for param, value in changes.items():
                parameters[category][param] = value

    if modified_path:
        os.makedirs(os.path.dirname(modified_path), exist_ok=True)
        with open(modified_path, 'w') as f:
            json.dump(parameters, f, indent=4)

    return parameters


# def create_aggregated_data(scenario, main_dir, output_dir):
#     """
#     Load, process, and aggregate data based on the specified RCP scenario.

#     :param scenario: RCP scenario (e.g., '26', '45', '85')
#     :param main_dir: Main directory containing data subfolders
#     :param output_dir: Directory to save the aggregated_data CSV file
#     :return: Aggregated DataFrame
#     """
#     # Define subdirectories based on main_dir
#     WHEAT_PAST   = os.path.join(main_dir, "WHEAT_PAST")
#     WHEAT_FUTURE = os.path.join(main_dir, "WHEAT_FUTURE")
#     MAIZE_PAST   = os.path.join(main_dir, "MAIZE_PAST")
#     MAIZE_FUTURE = os.path.join(main_dir, "MAIZE_FUTURE")
#     PV_PAST      = os.path.join(main_dir, "PV_PAST")
#     PV_FUTURE    = os.path.join(main_dir, "PV_FUTURE")

#     # Define file paths based on scenario
#     pv_suitability_file = os.path.join(PV_FUTURE, f"RCP{scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
#     pv_yield_file       = os.path.join(PV_FUTURE, f"RCP{scenario}_PV_SUITABILITY_ANNUAL_ELECTRICITY_SAVINGS_2021_2030.csv")
#     cs_maize_file       = os.path.join(MAIZE_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv")
#     cs_wheat_file       = os.path.join(WHEAT_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv")
#     cy_maize_file       = os.path.join(MAIZE_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv")
#     cy_wheat_file       = os.path.join(WHEAT_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv")

#     # --- PV Suitability ---
#     try:
#         PV_ = pd.read_csv(pv_suitability_file, sep=',')
#         PV_ = PV_.rename(columns={"score": "pv_suitability"})
#         PV_ = PV_[(PV_["year"] >= 2021) & (PV_["year"] <= 2030)]
#         print(f"Loaded PV Suitability for RCP {scenario}")
#     except FileNotFoundError:
#         print(f"Error: File '{pv_suitability_file}' not found.")
#         return None

#     # --- Crop Suitability for Maize and Wheat ---
#     try:
#         CS_Maize_ = pd.read_csv(cs_maize_file, sep=',')
#         CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
#         crop_suitability = pd.concat([CS_Maize_, CS_Wheat_], ignore_index=True)
#         crop_suitability = crop_suitability[(crop_suitability["year"] >= 2021) & (crop_suitability["year"] <= 2030)]
#         crop_suitability = crop_suitability.rename(columns={"score": "crop_suitability"})
#         print(f"Loaded Crop Suitability (Maize & Wheat) for RCP {scenario}")
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         return None

#     # --- PV Yield (Profit) ---
#     try:
#         PV_Yield_ = pd.read_csv(pv_yield_file, sep=',')
#         PV_Yield_ = PV_Yield_.rename(columns={'annual_electricity_savings_€': 'pv_profit'})
#         print(f"Loaded PV Yield for RCP {scenario}")
#     except FileNotFoundError:
#         print(f"Error: File '{pv_yield_file}' not found.")
#         return None

#     # --- Crop Yield (Profit) for Maize ---
#     try:
#         CY_Maize_FULL = pd.read_csv(cy_maize_file, sep=',')
#         CY_Maize = CY_Maize_FULL[['Dates', 'Profits']].copy()
#         CY_Maize['Dates'] = pd.to_datetime(CY_Maize['Dates'])
#         CY_Maize['year'] = CY_Maize['Dates'].dt.year
#         CY_Maize = CY_Maize.groupby('year').last().reset_index()
#         CY_Maize = CY_Maize.rename(columns={'Profits': 'crop_profit'})
#         print(f"Processed Crop Yield for Maize for RCP {scenario}")
#     except FileNotFoundError:
#         print(f"Error: File '{cy_maize_file}' not found.")
#         return None

#     # --- Crop Yield (Profit) for Wheat ---
#     try:
#         CY_Wheat_FULL = pd.read_csv(cy_wheat_file, sep=',')
#         CY_Wheat = CY_Wheat_FULL[['Dates', 'Profits']].copy()
#         CY_Wheat['Dates'] = pd.to_datetime(CY_Wheat['Dates'])
#         CY_Wheat['year'] = CY_Wheat['Dates'].dt.year
#         CY_Wheat = CY_Wheat.groupby('year').last().reset_index()
#         CY_Wheat = CY_Wheat.rename(columns={'Profits': 'crop_profit'})
#         print(f"Processed Crop Yield for Wheat for RCP {scenario}")
#     except FileNotFoundError:
#         print(f"Error: File '{cy_wheat_file}' not found.")
#         return None

#     # --- Combine Crop Profit for Maize and Wheat ---
#     crop_profit = pd.concat([CY_Maize, CY_Wheat], ignore_index=True)
#     # Merge with crop_suitability based on "year", "lat", and "lon"
#     try:
#         crop_profit = crop_suitability.merge(crop_profit, on="year", how="inner")[["year", "lat", "lon", "crop_profit"]]
#     except KeyError as e:
#         print(f"Error during merging crop_profit: {e}")
#         return None

#     # Convert latitude and longitude columns to float for compatibility
#     for df in [PV_, crop_suitability, PV_Yield_, crop_profit]:
#         df["lat"] = df["lat"].astype(float).round(5)
#         df["lon"] = df["lon"].astype(float).round(5)

#     # --- Merge Datasets ---
#     try:
#         env_data = crop_suitability.merge(PV_, on=["year", "lat", "lon"], how="inner")
#         env_data = env_data.merge(PV_Yield_, on=["year", "lat", "lon"], how="inner")
#         env_data = env_data.merge(crop_profit, on=["year", "lat", "lon"], how="inner")
#     except KeyError as e:
#         print(f"Error during merging datasets: {e}")
#         return None

#     print(env_data.head())
#     print("Merged environmental dataset:")

#     # Aggregate env_data by unique lat/lon pairs
#     aggregated_data = env_data.groupby(['lat', 'lon'], as_index=False).agg({
#         'crop_suitability': 'mean',  # Average crop suitability over years
#         'pv_suitability': 'mean',    # Average PV suitability over years
#         'crop_profit': 'sum',        # Total crop profit over years
#         'pv_profit': 'sum'            # Total PV profit over years
#     })

#     print(f"Number of unique lat/lon pairs: {aggregated_data.shape[0]}")
#     print(aggregated_data.head())

#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Save the aggregated data to a CSV file
#     output_file = os.path.join(output_dir, f"aggregated_data_RCP{scenario}.csv")
#     aggregated_data.to_csv(output_file, index=False)
#     print(f"Aggregated data saved to '{output_file}'.")

#     return aggregated_data

def create_aggregated_data(scenario_name, main_dir, output_dir):
    """
    Load, process, and aggregate data based on the specified RCP scenario.

    :param scenario: RCP scenario (e.g., '26', '45', '85')
    :param main_dir: Main directory containing data subfolders
    :param output_dir: Directory to save the aggregated_data CSV file
    :return: Aggregated DataFrame
    """

    # Determine whether we're working with past or future data
    if scenario_name == "PAST":
        WHEAT_DIR = os.path.join(main_dir, "WHEAT_PAST")
        PV_DIR    = os.path.join(main_dir, "PV_PAST")
        
        pv_suitability_file = os.path.join(PV_DIR, "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
        pv_yield_file       = os.path.join(PV_DIR, "PAST_PV_YIELD.csv")
        cs_wheat_file       = os.path.join(WHEAT_DIR, "PAST_LUSA_PREDICTIONS.csv")
        cy_wheat_file       = os.path.join(WHEAT_DIR, "AquaCrop_Results_PAST.csv")
    else:
        WHEAT_DIR = os.path.join(main_dir, "WHEAT_FUTURE")
        PV_DIR    = os.path.join(main_dir, "PV_FUTURE")
        
        pv_suitability_file = os.path.join(PV_DIR, f"{scenario_name}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
        pv_yield_file       = os.path.join(PV_DIR, f"{scenario_name}_PV_SUITABILITY_ANNUAL_ELECTRICITY_SAVINGS_2021_2030.csv")
        cs_wheat_file       = os.path.join(WHEAT_DIR, f"{scenario_name}_LUSA_PREDICTIONS.csv")
        cy_wheat_file       = os.path.join(WHEAT_DIR, f"AquaCrop_Results_{scenario_name}.csv")

    print(f"Running data aggregation for {scenario_name}...")


    # # Define subdirectories based on main_dir
    # WHEAT_FUTURE = os.path.join(main_dir, "WHEAT_FUTURE")
    # PV_FUTURE    = os.path.join(main_dir, "PV_FUTURE")

    # # Define file paths based on scenario
    # pv_suitability_file = os.path.join(PV_FUTURE, f"RCP{scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    # pv_yield_file       = os.path.join(PV_FUTURE, f"RCP{scenario}_PV_SUITABILITY_ANNUAL_ELECTRICITY_SAVINGS_2021_2030.csv")
    # cs_wheat_file       = os.path.join(WHEAT_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv")
    # cy_wheat_file       = os.path.join(WHEAT_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv")

    # --- PV Suitability ---
    try:
        PV_ = pd.read_csv(pv_suitability_file, sep=',')
        PV_ = PV_.rename(columns={"score": "pv_suitability"})
        PV_ = PV_[(PV_["year"] >= 2021) & (PV_["year"] <= 2030)]
        print(f"Loaded PV Suitability for {scenario_name}")
    except FileNotFoundError:
        print(f"Error: File '{pv_suitability_file}' not found.")
        return None

    # --- Crop Suitability for Wheat ---
    try:
        CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
        CS_Wheat_ = CS_Wheat_[(CS_Wheat_["year"] >= 2021) & (CS_Wheat_["year"] <= 2030)]
        CS_Wheat_ = CS_Wheat_.rename(columns={"score": "crop_suitability"})
        print(f"Loaded Crop Suitability for Wheat for {scenario_name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # --- PV Yield (Profit) ---
    try:
        PV_Yield_ = pd.read_csv(pv_yield_file, sep=',')
        PV_Yield_ = PV_Yield_.rename(columns={'annual_electricity_savings_€': 'pv_profit'})
        print(f"Loaded PV Yield for {scenario_name}")
    except FileNotFoundError:
        print(f"Error: File '{pv_yield_file}' not found.")
        return None

    # --- Crop Yield (Profit) for Wheat ---
    try:
        CY_Wheat_FULL = pd.read_csv(cy_wheat_file, sep=',')
        CY_Wheat = CY_Wheat_FULL[['Dates', 'Profits']].copy()
        CY_Wheat['Dates'] = pd.to_datetime(CY_Wheat['Dates'])
        CY_Wheat['year'] = CY_Wheat['Dates'].dt.year
        CY_Wheat = CY_Wheat.groupby('year').last().reset_index()
        CY_Wheat = CY_Wheat.rename(columns={'Profits': 'crop_profit'})
        print(f"Processed Crop Yield for Wheat for {scenario_name}")
    except FileNotFoundError:
        print(f"Error: File '{cy_wheat_file}' not found.")
        return None

    # --- Merge Crop Profit with Crop Suitability ---
    try:
        crop_profit = CS_Wheat_.merge(CY_Wheat, on="year", how="inner")[["year", "lat", "lon", "crop_profit"]]
    except KeyError as e:
        print(f"Error during merging crop_profit: {e}")
        return None

    # Convert latitude and longitude columns to float for compatibility
    for df in [PV_, CS_Wheat_, PV_Yield_, crop_profit]:
        df["lat"] = df["lat"].astype(float).round(5)
        df["lon"] = df["lon"].astype(float).round(5)

    # --- Merge Datasets ---
    try:
        env_data = CS_Wheat_.merge(PV_, on=["year", "lat", "lon"], how="inner")
        env_data = env_data.merge(PV_Yield_, on=["year", "lat", "lon"], how="inner")
        env_data = env_data.merge(crop_profit, on=["year", "lat", "lon"], how="inner")
    except KeyError as e:
        print(f"Error during merging datasets: {e}")
        return None

    print(env_data.head())
    print("Merged environmental dataset:")

    # Aggregate env_data by unique lat/lon pairs
    aggregated_data = env_data.groupby(['lat', 'lon'], as_index=False).agg({
        'crop_suitability': 'mean',  # Average crop suitability over years
        'pv_suitability': 'mean',    # Average PV suitability over years
        'crop_profit': 'sum',        # Total crop profit over years
        'pv_profit': 'sum'            # Total PV profit over years
    })

    print(f"Number of unique lat/lon pairs: {aggregated_data.shape[0]}")
    print(aggregated_data.head())

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the aggregated data to a CSV file
    output_file = os.path.join(output_dir, f"aggregated_data_{scenario_name}.csv")
    aggregated_data.to_csv(output_file, index=False)
    print(f"Aggregated data saved to '{output_file}'.")

    return aggregated_data

def plot_decision_heatmap(decisions_df, output_dir):
    """
    Create and save a heatmap of agent decisions based on their geographical locations.

    :param decisions_df: DataFrame containing agent decisions with 'lat', 'lon', and 'decision' columns.
    :param output_dir: Directory to save the heatmap plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Pivot the DataFrame to have latitudes as rows and longitudes as columns
    pivot_table = decisions_df.pivot_table(
        index='lat',
        columns='lon',
        values='decision',
        aggfunc='mean'  # Or another aggregation method
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt=".2f")
    plt.title("Heatmap of Agent Decisions")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(os.path.join(output_dir, "agent_decisions_heatmap.png"), dpi=300)
    plt.close()
    print("Agent decisions heatmap saved as 'agent_decisions_heatmap.png'.")
    logging.info("Agent decisions heatmap saved as 'agent_decisions_heatmap.png'.")

def main():
    # ================================
    # Argument Parsing
    # ================================
    parser = argparse.ArgumentParser(description="Run the PPO model with modified PECS parameters.")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["26", "45", "85"],
        help="Specify the RCP scenario to run the simulation (options: '26', '45', '85')."
    )
    parser.add_argument(
        "--past",
        action="store_true",
        help="Use past data instead of future RCP scenarios."
    )
    parser.add_argument(
        "--main_dir",
        type=str,
        default="/Users/mbanti/Documents/Projects/esa_agents_local/PECS/data",
        help="Main directory containing data subfolders."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output",
        help="Directory to save the aggregated_data CSV file and agent decisions."
    )
    parser.add_argument(
        "--original_pecs_path",
        type=str,
        default="./trained_models/RCP85/pecs_params.json",
        help="Path to the original PECS parameters JSON file."
    )
    parser.add_argument(
        "--modified_pecs_path",
        type=str,
        default="trained_models/pecs_params_modified.json",
        help="Path to save the modified PECS parameters JSON file."
    )
    # parser.add_argument(
    #     "--model_path",
    #     type=str,
    #     default="./trained_models/RCP85/ppo_model.zip",
    #     help="Path to the trained PPO model."
    # )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run the simulation."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path to a log file."
    )
    parser.add_argument(
        "--modifications_file",
        type=str,
        default=None,
        help="Path to a JSON file specifying PECS parameter and scenario modifications."
    )
    args = parser.parse_args()

    # Ensure either --scenario or --past is specified
    if not args.past and not args.scenario:
        raise ValueError("You must specify either --scenario or --past.")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Define scenario name based on whether past data is used
    scenario_name = "PAST" if args.past else f"RCP{args.scenario}"

    # Define model path dynamically
    args.model_path = os.path.join("trained_models", scenario_name, "ppo_model.zip")

    # Define log file path dynamically
    log_file_path = os.path.join(args.output_dir, f"run_{scenario_name}.log")


    # Configure logging
    if args.log_file:
        logging.basicConfig(
            filename=args.log_file,
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s"
        )

    # Paths to the log file and output directory

    output_dir_path = args.output_dir
    
    # **Step 0: Cleanup Old Logs and Outputs**
    cleanup_files(log_file_path, output_dir_path)

    # Step 1: Create aggregated data
    logging.info(f"Starting data aggregation for {scenario_name}.")
    aggregated_data = create_aggregated_data(scenario_name, args.main_dir, args.output_dir)
    if aggregated_data is None:
        logging.error("Data aggregation failed. Exiting.")
        return
    logging.info("Data aggregation completed successfully.")

    # Path to the newly created aggregated_data CSV
    aggregated_data_path = os.path.join(args.output_dir, f"aggregated_data_{scenario_name}.csv")

    # Step 2: Load and modify parameters
    # Load modifications from file if provided
    if args.modifications_file:
        try:
            with open(args.modifications_file, 'r') as f:
                modifications = json.load(f)
            logging.info("Loaded modifications from file.")
        except FileNotFoundError:
            logging.error(f"Modifications file '{args.modifications_file}' not found.")
            return
        except json.JSONDecodeError:
            logging.error(f"Modifications file '{args.modifications_file}' contains invalid JSON.")
            return
    else:
        # Define your desired modifications here or set default modifications
        modifications = {
            "physis": {
                "health_status": 0.9,
                "labor_availability": 0.7
            },
            "emotion": {
                "stress_level": 0.2,
                "satisfaction": 0.8
            },
            "cognition": {
                "policy_incentives": 0.6,
                "information_access": 0.7
            },
            "social": {
                "social_influence": 0.5,
                "community_participation": 0.6
            },
            "scenario_params": {
                "subsidy": 0.15,
                "loan_interest_rate": 0.04,
                "tax_incentive_rate": 0.1,
                "social_prestige_weight": 0.2
            }
        }

    logging.info("Modifying parameters.")
    try:
        modified_parameters = load_and_modify_pecs_params(
            original_path=args.original_pecs_path,
            modified_path=args.modified_pecs_path,
            modifications=modifications
        )
        logging.info("Parameters modified successfully.")
    except ValueError as ve:
        logging.error(f"Parameter Modification Error: {ve}")
        return
    except FileNotFoundError:
        logging.error(f"Error: '{args.original_pecs_path}' not found.")
        return

    # Extract PECS and scenario parameters
    pecs_params = {k: v for k, v in modified_parameters.items() if k != "scenario_params"}
    scenario_params = modified_parameters.get("scenario_params", {
        "subsidy": 0.0,
        "loan_interest_rate": 0.05,
        "tax_incentive_rate": 0.0,
        "social_prestige_weight": 0.0
    })

    # Step 3: Instantiate the environment with modified parameters
    logging.info("Instantiating the environment with modified parameters.")
    width = 10
    height = 10
    max_steps = 10

    env = MesaEnv(
        env_data=aggregated_data,
        max_steps=max_steps,
        pecs_params=pecs_params,
        width=width,
        height=height,
        scenario_params=scenario_params
    )

    # Step 4: Verify environment compliance with Gym API
    logging.info("Checking environment compliance with Gym API.")
    try:
        check_env(env)
        logging.info("Environment passed the Gym API check.")
    except Exception as e:
        logging.error(f"Environment check failed: {e}")
        return

    # Step 5: Load the trained PPO model with the new environment
    logging.info("Loading the trained PPO model.")
    try:
        model = PPO.load(args.model_path, env=env)
        logging.info("PPO model loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Error: '{args.model_path}' not found.")
        return
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    # Step 6: Run the model for a specified number of episodes
    logging.info(f"Running the simulation for {args.num_episodes} episodes.")
    all_final_decisions = []

    for episode in range(args.num_episodes):
        print(f"Starting episode {episode + 1}")  # Debugging: Track episodes
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step = 0
        while not done and not truncated:
            print(f"Step {step}: Observation = {obs}") 
            action, _states = model.predict(obs, deterministic=True)
            print(f"Step {step}: Action = {action}")  # Debugging: Check action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
        print(f"Episode {episode + 1} completed with total reward: {total_reward}")

        # After the episode ends, collect final decisions
        episode_decisions = []
        for agent in env.model.schedule.agents:
            print(f"Agent {agent.unique_id}: Decision = {getattr(agent, 'decision', 'Not Set')}")
            try:
                episode_decisions.append({
                    'episode': episode + 1,
                    'agent_id': agent.unique_id,
                    'decision': agent.decision
                })
            except AttributeError as e:
                print(f"Agent {agent.unique_id} is missing 'decision' attribute: {e}")
                logging.error(f"Agent {agent.unique_id} is missing 'decision' attribute: {e}")
        all_final_decisions.extend(episode_decisions)

        logging.info(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Steps = {step}")
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Steps = {step}")

    # Step 7: Save the final decisions to a CSV file
    if all_final_decisions:
        print(f"Collected {len(all_final_decisions)} decisions.")
        decisions_df = pd.DataFrame(all_final_decisions)
        print(decisions_df.head())  # Debugging: Check the collected agent decisions
        decisions_output_file = os.path.join(args.output_dir, "final_agent_decisions.csv")
        decisions_df.to_csv(decisions_output_file, index=False)
        print(f"Final agent decisions saved to '{decisions_output_file}'.")
        logging.info(f"Final agent decisions saved to '{decisions_output_file}'.")
    else:
        print("Warning: No agent decisions were collected.")
        logging.warning("No agent decisions were collected.")

if __name__ == "__main__":
    main()

