#!/usr/bin/env python
# run_with_modified_parameters_v05.py

#==================================================
# Script: run_with_modified_parameters_v05.py
# Author: Maria Banti
# Date: 2025-01-29
# Version: 1.4
#==================================================
#
# **Description:**
# This script executes a Reinforcement Learning simulation using a trained PPO (Proximal Policy Optimization) model 
# within a customized Mesa environment (MesaEnv). It modifies PECS parameters based on a provided JSON configuration file 
# (default: modifications.json), aggregates environmental data for a specified RCP scenario or for past data (if --past is provided), 
# and automatically reads climate losses from "climate_losses.json" and the initial subsidy fund from "initial_fund.json". 
# It then implements a saturation fund mechanism for PV installations—each time an agent adopts PV, the effective subsidy is computed 
# based on the ratio (current_fund/initial_fund) and the remaining fund is decremented.
#
# **Features:**
# - Aggregates environmental data based on selected RCP scenarios or past data.
# - Reads and applies modifications from modifications.json.
# - Automatically reads climate losses from climate_losses.json and the initial subsidy fund from initial_fund.json.
# - Implements a saturation fund mechanism for PV installations.
# - Automatically cleans up old logs and output files before execution.
# - Executes simulations for a specified number of episodes.
# - Captures and saves final agent decisions (including lat/lon) to a CSV file.
# - Generates heatmaps of agent decisions.
# - Comprehensive logging for monitoring and debugging.

import os
import json
import argparse
import pandas as pd
import numpy as np
import logging
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from models.mesa_env import MesaEnv  # Ensure your models/mesa_env.py (and associated model/agent files) are updated.
import random

def cleanup_files(log_file_path, output_dir_path):
    if os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
            print(f"Deleted old log file: {log_file_path}")
            logging.info(f"Deleted old log file: {log_file_path}")
        except Exception as e:
            print(f"Error deleting log file '{log_file_path}': {e}")
            logging.error(f"Error deleting log file '{log_file_path}': {e}")
    if os.path.exists(output_dir_path):
        try:
            shutil.rmtree(output_dir_path)
            print(f"Deleted old output directory: {output_dir_path}")
            logging.info(f"Deleted old output directory: {output_dir_path}")
            os.makedirs(output_dir_path, exist_ok=True)
            print(f"Recreated output directory: {output_dir_path}")
            logging.info(f"Recreated output directory: {output_dir_path}")
        except Exception as e:
            print(f"Error deleting output directory '{output_dir_path}': {e}")
            logging.error(f"Error deleting output directory '{output_dir_path}': {e}")
    else:
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"Created output directory: {output_dir_path}")
        logging.info(f"Created output directory: {output_dir_path}")

def load_and_modify_pecs_params(original_path, modified_path=None, modifications=None):
    with open(original_path, 'r') as f:
        parameters = json.load(f)
    if modifications:
        for category, changes in modifications.items():
            if category not in parameters:
                parameters[category] = {}
            for param, value in changes.items():
                parameters[category][param] = value
    if modified_path:
        os.makedirs(os.path.dirname(modified_path), exist_ok=True)
        with open(modified_path, 'w') as f:
            json.dump(parameters, f, indent=4)
    return parameters

def create_aggregated_data(mode, scenario, main_dir, output_dir):
    """
    mode: "PAST" or "RCP"
    scenario: if mode=="RCP", the RCP scenario string ("26", "45", or "85"). Otherwise ignored.
    """
    # Define paths to data folders
    WHEAT_PAST   = os.path.join(main_dir, "WHEAT_PAST")
    WHEAT_FUTURE = os.path.join(main_dir, "WHEAT_FUTURE")
    MAIZE_PAST   = os.path.join(main_dir, "MAIZE_PAST")
    MAIZE_FUTURE = os.path.join(main_dir, "MAIZE_FUTURE")
    PV_PAST      = os.path.join(main_dir, "PV_PAST")
    PV_FUTURE    = os.path.join(main_dir, "PV_FUTURE")
    
    if mode == "PAST":
        scenario_name = "PAST"
        pv_suitability_file = os.path.join(PV_PAST, "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
        pv_yield_file       = os.path.join(PV_PAST, "PAST_PV_YIELD.csv")
        cs_wheat_file       = os.path.join(WHEAT_PAST, "WHEAT_PAST_LUSA_PREDICTIONS.csv")
        cs_maize_file       = os.path.join(MAIZE_PAST, "MAIZE_PAST_LUSA_PREDICTIONS.csv")
        cy_wheat_file       = os.path.join(WHEAT_PAST, "AquaCrop_Results_PAST.csv")
        cy_maize_file       = os.path.join(MAIZE_PAST, "AquaCrop_Results_PAST.csv")
        print("Running simulation with past data...")
    else:  # RCP mode
        scenario_name = f"RCP{scenario}"
        pv_suitability_file = os.path.join(PV_FUTURE, f"RCP{scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
        pv_yield_file       = os.path.join(PV_FUTURE, f"RCP{scenario}_PV_YIELD.csv")
        cs_maize_file       = os.path.join(MAIZE_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv")
        cs_wheat_file       = os.path.join(WHEAT_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv")
        cy_maize_file       = os.path.join(MAIZE_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv")
        cy_wheat_file       = os.path.join(WHEAT_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv")
        print(f"Running simulation for scenario {scenario_name}...")
    
    try:
        PV_ = pd.read_csv(pv_suitability_file, sep=',')
        PV_ = PV_.rename(columns={"score": "pv_suitability"})
        PV_ = PV_[(PV_["year"] >= 2021) & (PV_["year"] <= 2030)]
        print(f"Loaded PV Suitability for {scenario_name}")
    except FileNotFoundError:
        print(f"Error: File '{pv_suitability_file}' not found.")
        return None
    
    try:
        if mode == "PAST":
            CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
            CS_Maize_ = pd.read_csv(cs_maize_file, sep=',')
            crop_suitability = pd.concat([CS_Wheat_, CS_Maize_], ignore_index=True)
        else:
            CS_Maize_ = pd.read_csv(cs_maize_file, sep=',')
            CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
            crop_suitability = pd.concat([CS_Maize_, CS_Wheat_], ignore_index=True)
        crop_suitability = crop_suitability[(crop_suitability["year"] >= 2021) & (crop_suitability["year"] <= 2030)]
        crop_suitability = crop_suitability.rename(columns={"score": "crop_suitability"})
        print(f"Loaded Crop Suitability (Maize & Wheat) for {scenario_name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    
    try:
        PV_Yield_ = pd.read_csv(pv_yield_file, sep=',')
        PV_Yield_ = PV_Yield_.rename(columns={'annual_electricity_savings_€': 'pv_profit'})
        print(f"Loaded PV Yield for {scenario_name}")
    except FileNotFoundError:
        print(f"Error: File '{pv_yield_file}' not found.")
        return None
    
    try:
        CY_Maize_FULL = pd.read_csv(cy_maize_file, sep=',')
        CY_Maize_ = CY_Maize_FULL[['Dates', 'Profits']].copy()
        CY_Maize_['Dates'] = pd.to_datetime(CY_Maize_['Dates'])
        CY_Maize_['year'] = CY_Maize_['Dates'].dt.year
        CY_Maize_ = CY_Maize_.groupby('year').last().reset_index()
        CY_Maize_ = CY_Maize_.rename(columns={'Profits': 'crop_profit'})
        print(f"Processed Crop Yield for Maize for {scenario_name}")
    except FileNotFoundError:
        print(f"Error: File '{cy_maize_file}' not found.")
        return None
    
    try:
        CY_Wheat_FULL = pd.read_csv(cy_wheat_file, sep=',')
        CY_Wheat_ = CY_Wheat_FULL[['Dates', 'Profits']].copy()
        CY_Wheat_['Dates'] = pd.to_datetime(CY_Wheat_['Dates'])
        CY_Wheat_['year'] = CY_Wheat_['Dates'].dt.year
        CY_Wheat_ = CY_Wheat_.groupby('year').last().reset_index()
        CY_Wheat_ = CY_Wheat_.rename(columns={'Profits': 'crop_profit'})
        print(f"Processed Crop Yield for Wheat for {scenario_name}")
    except FileNotFoundError:
        print(f"Error: File '{cy_wheat_file}' not found.")
        return None
    
    crop_profit = pd.concat([CY_Maize_, CY_Wheat_], ignore_index=True)
    try:
        crop_profit = crop_suitability.merge(crop_profit, on="year", how="inner")[["year", "lat", "lon", "crop_profit"]]
    except KeyError as e:
        print(f"Error during merging crop_profit: {e}")
        return None
    
    for df in [PV_, crop_suitability, PV_Yield_, crop_profit]:
        df["lat"] = df["lat"].astype(float).round(5)
        df["lon"] = df["lon"].astype(float).round(5)
    
    try:
        env_data = crop_suitability.merge(PV_, on=["year", "lat", "lon"], how="inner")
        env_data = env_data.merge(PV_Yield_, on=["year", "lat", "lon"], how="inner")
        env_data = env_data.merge(crop_profit, on=["year", "lat", "lon"], how="inner")
    except KeyError as e:
        print(f"Error during merging datasets: {e}")
        return None
    
    print(env_data.head())
    print("Merged environmental dataset:")
    
    aggregated_data = env_data.groupby(['lat', 'lon'], as_index=False).agg({
        'crop_suitability': 'mean',
        'pv_suitability': 'mean',
        'crop_profit': 'sum',
        'pv_profit': 'sum'
    })
    
    print(f"Number of unique lat/lon pairs: {aggregated_data.shape[0]}")
    print(aggregated_data.head())
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"aggregated_data_{scenario_name}.csv")
    aggregated_data.to_csv(output_file, index=False)
    print(f"Aggregated data saved to '{output_file}'.")
    return aggregated_data

def plot_decision_heatmap(decisions_df, output_dir):
    import seaborn as sns
    import matplotlib.pyplot as plt
    pivot_table = decisions_df.pivot_table(
        index='lat',
        columns='lon',
        values='decision',
        aggfunc='mean'
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
    parser = argparse.ArgumentParser(description="Run the PPO model with modified PECS parameters.")
    # For RCP mode the scenario argument is needed; if --past is set, scenario is not required.
    parser.add_argument("--scenario", type=str, choices=["26", "45", "85"],
                        help="Specify the RCP scenario to run the simulation (options: '26', '45', '85'). Ignored if --past is provided.")
    parser.add_argument("--past", action="store_true",
                        help="Run simulation with past data instead of future (RCP) data.")
    parser.add_argument("--main_dir", type=str, default="/Users/mbanti/Documents/Projects/esa_agents/BACKEND_API/Demo_With_Geoserver/INPUT_RL",
                        help="Main directory containing data subfolders.")
    parser.add_argument("--output_dir", type=str, default="data/output",
                        help="Directory to save the aggregated_data CSV file and agent decisions.")
    parser.add_argument("--original_pecs_path", type=str, default="./trained_models/RCP26/pecs_params.json",
                        help="Path to the original PECS parameters JSON file.")
    parser.add_argument("--modified_pecs_path", type=str, default="trained_models/pecs_params_modified.json",
                        help="Path to save the modified PECS parameters JSON file.")
    parser.add_argument("--model_path", type=str, default="./trained_models/RCP26/ppo_model.zip",
                        help="Path to the trained PPO model. Automatically changed if --past is provided.")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of episodes to run the simulation.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional path to a log file.")
    parser.add_argument("--modifications_file", type=str, default="modifications.json",
                        help="Path to a JSON file specifying PECS parameter and scenario modifications.")
    parser.add_argument("--initial_fund", type=str, default="initial_fund.json",
                        help="Path to the JSON file containing the initial subsidy fund for PV installations.")
    parser.add_argument("--climate_losses", type=str, default="climate_losses.json",
                        help="Path to JSON file containing climate-related economic losses.")
    parser.add_argument("--subsidy", type=float, default=0.0,
                        help="Base subsidy for PV installations (default: 0.0).")
    parser.add_argument("--loan_interest_rate", type=float, default=0.05,
                        help="Loan interest rate for PV installations (default: 0.05).")
    parser.add_argument("--tax_incentive_rate", type=float, default=0.0,
                        help="Tax incentive rate for renewable energy production (default: 0.0).")
    parser.add_argument("--social_prestige_weight", type=float, default=0.0,
                        help="Weight of social prestige for early adopters (default: 0.0).")
    args = parser.parse_args()

    if not args.past and not args.scenario:
        parser.error("For RCP simulations, please provide --scenario (e.g., 26, 45, 85) or use --past for past data.")

    if args.past:
        mode = "PAST"
        scenario_for_data = None
        if args.model_path == "./trained_models/RCP26/ppo_model.zip":
            args.model_path = "./trained_models/PAST/ppo_model.zip"
        if args.original_pecs_path == "./trained_models/RCP26/pecs_params.json":
            args.original_pecs_path = "./trained_models/PAST/pecs_params.json"
    else:
        mode = "RCP"
        scenario_for_data = args.scenario

    # Create an output subfolder based on the mode.
    if mode == "PAST":
        output_subfolder = "past"
    else:
        output_subfolder = f"RCP{args.scenario}"
    new_output_dir = os.path.join(args.output_dir, output_subfolder)
    os.makedirs(new_output_dir, exist_ok=True)

    # Set default PECS parameters.
    pecs_params = {
        "physis": {"health_status": 0.8, "labor_availability": 0.6},
        "emotion": {"stress_level": 0.3, "satisfaction": 0.7},
        "cognition": {"policy_incentives": 0.5, "information_access": 0.6},
        "social": {"social_influence": 0.4, "community_participation": 0.5}
    }

    with open(args.initial_fund, "r") as f:
        fund_data = json.load(f)
    initial_subsidy_fund = fund_data.get("initial_subsidy_fund", 1.0)

    scenario_params = {
        "subsidy": args.subsidy,
        "loan_interest_rate": args.loan_interest_rate,
        "tax_incentive_rate": args.tax_incentive_rate,
        "social_prestige_weight": args.social_prestige_weight,
        "initial_subsidy_fund": initial_subsidy_fund
    }

    def validate_pecs_params(pecs_params):
        for category, params in pecs_params.items():
            for param, value in params.items():
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"PECS parameter '{category}.{param}' must be between 0 and 1. Received: {value}")
    validate_pecs_params(pecs_params)

    with open(args.climate_losses, "r") as f:
        climate_losses = json.load(f)

    print(f"Running simulation for mode: {mode}" + (f", scenario: {args.scenario}" if mode == "RCP" else ""))
    print(f"PECS Parameters: {pecs_params}")
    print(f"Scenario Parameters: {scenario_params}")
    print(f"Climate Losses: {climate_losses}")

    # Load aggregated data using the new output directory.
    aggregated_data = create_aggregated_data(mode, scenario_for_data, args.main_dir, new_output_dir)
    if aggregated_data is None:
        print("Data aggregation failed. Exiting.")
        return

    unique_lat_lon_count = aggregated_data[["lat", "lon"]].drop_duplicates().shape[0]
    print(f"Number of unique lat/lon pairs: {unique_lat_lon_count}")

    width = 10
    height = 10

    print(f"Climate Losses: {climate_losses}")

    env = MesaEnv(
        env_data=aggregated_data,
        max_steps=10,
        pecs_params=pecs_params,
        width=width,
        height=height,
        scenario_params=scenario_params,
        climate_losses=climate_losses
    )

    logging.info("Checking environment compliance with Gym API.")
    try:
        check_env(env)
        logging.info("Environment passed the Gym API check.")
    except Exception as e:
        logging.error(f"Environment check failed: {e}")
        return

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

    logging.info(f"Running the simulation for {args.num_episodes} episodes.")
    all_final_decisions = []
    for episode in range(args.num_episodes):
        print(f"Starting episode {episode + 1}")
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step = 0
        while not done and not truncated:
            print(f"Step {step}: Observation = {obs}")
            action, _states = model.predict(obs, deterministic=True)
            print(f"Step {step}: Action = {action}")
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
        print(f"Episode {episode + 1} completed with total reward: {total_reward}")

        episode_decisions = []
        for agent in env.model.schedule.agents:
            print(f"Agent {agent.unique_id}: Decision = {getattr(agent, 'decision', 'Not Set')}")
            try:
                episode_decisions.append({
                    'episode': episode + 1,
                    'agent_id': agent.unique_id,
                    'lat': agent.lat,
                    'lon': agent.lon,
                    'decision': agent.decision
                })
            except AttributeError as e:
                print(f"Agent {agent.unique_id} is missing 'decision' attribute: {e}")
                logging.error(f"Agent {agent.unique_id} is missing 'decision' attribute: {e}")
        all_final_decisions.extend(episode_decisions)
        logging.info(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Steps = {step}")
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Steps = {step}")

    if all_final_decisions:
        print(f"Collected {len(all_final_decisions)} decisions.")
        decisions_df = pd.DataFrame(all_final_decisions)
        print(decisions_df.head())
        decisions_output_file = os.path.join(new_output_dir, "final_agent_decisions.csv")
        decisions_df.to_csv(decisions_output_file, index=False)
        print(f"Final agent decisions saved to '{decisions_output_file}'.")
        logging.info(f"Final agent decisions saved to '{decisions_output_file}'.")
    else:
        print("Warning: No agent decisions were collected.")
        logging.warning("No agent decisions were collected.")

if __name__ == "__main__":
    main()

