#!/usr/bin/env python


#==================================================
# Script: run_with_modified_parameters_v05.py
# Author: Maria Banti
# Date: 2025-01-29
# Version: 1.1
#==================================================
#
# **Description:**
# This script executes a Reinforcement Learning simulation using a trained PPO (Proximal Policy Optimization) model 
# within a customized Mesa environment (MesaEnv). It modifies PECS parameters based on a provided JSON configuration file 
# (default: modifications.json), aggregates environmental data for a specified scenario, and automatically reads climate 
# losses from "climate_losses.json" and the initial subsidy fund from "initial_fund.json". It then implements a saturation 
# fund mechanism for PV installations, where the available subsidy for PV decreases as more agents adopt PV.
#
# **Features:**
# - Aggregates environmental data based on selected scenarios.
# - Reads and applies modifications from modifications.json.
# - Automatically reads climate losses from climate_losses.json and the initial subsidy fund from initial_fund.json.
# - Implements a saturation fund mechanism for PV installations.
# - Automatically cleans up old logs and output files before execution.
# - Supports running with “past” data (via --past flag) versus future (RCP) data.
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
from stable_baselines3.common.callbacks import BaseCallback
import time
import psutil
import torch
import random
import matplotlib.pyplot as plt

from models.mesa_env import MesaEnv  # Make sure your MesaEnv now does not require eu_target

# -------------------------------
# Cleanup function
# -------------------------------
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

# -------------------------------
# Parameter modification function
# -------------------------------
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

# -------------------------------
# Aggregated data creation
# -------------------------------
def create_aggregated_data(scenario, main_dir, output_dir, use_past=False):
    # Set folder paths
    WHEAT_PAST   = os.path.join(main_dir, "WHEAT_PAST")
    WHEAT_FUTURE = os.path.join(main_dir, "WHEAT_FUTURE")
    MAIZE_PAST   = os.path.join(main_dir, "MAIZE_PAST")
    MAIZE_FUTURE = os.path.join(main_dir, "MAIZE_FUTURE")
    PV_PAST      = os.path.join(main_dir, "PV_PAST")
    PV_FUTURE    = os.path.join(main_dir, "PV_FUTURE")

    if use_past:
        scenario_name = "PAST"
        print("Running simulation with past data...")
        pv_suitability_file = os.path.join(PV_PAST, "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
        pv_yield_file       = os.path.join(PV_PAST, "PAST_PV_YIELD.csv")
        cs_wheat_file       = os.path.join(WHEAT_PAST, "WHEAT_PAST_LUSA_PREDICTIONS.csv")
        cy_wheat_file       = os.path.join(WHEAT_PAST, "AquaCrop_Results_PAST.csv")
        cs_maize_file       = os.path.join(MAIZE_PAST, "MAIZE_PAST_LUSA_PREDICTIONS.csv")
        cy_maize_file       = os.path.join(MAIZE_PAST, "AquaCrop_Results_PAST.csv")
    else:
        scenario_name = f"RCP{scenario}"
        print(f"Running simulation for scenario {scenario_name}...")
        pv_suitability_file = os.path.join(PV_FUTURE, f"RCP{scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
        pv_yield_file       = os.path.join(PV_FUTURE, f"RCP{scenario}_PV_YIELD.csv")
        cs_maize_file       = os.path.join(MAIZE_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv")
        cs_wheat_file       = os.path.join(WHEAT_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv")
        cy_maize_file       = os.path.join(MAIZE_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv")
        cy_wheat_file       = os.path.join(WHEAT_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv")

    try:
        PV_ = pd.read_csv(pv_suitability_file, sep=',')
        PV_ = PV_.rename(columns={"score": "pv_suitability"})
        PV_ = PV_[(PV_["year"] >= 2021) & (PV_["year"] <= 2030)]
        print(f"Loaded PV Suitability for {scenario_name}")
    except FileNotFoundError:
        print(f"Error: File '{pv_suitability_file}' not found.")
        return None

    try:
        if use_past:
            CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
            crop_suitability = CS_Wheat_
        else:
            CS_Maize_ = pd.read_csv(cs_maize_file, sep=',')
            CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
            crop_suitability = pd.concat([CS_Maize_, CS_Wheat_], ignore_index=True)
        crop_suitability = crop_suitability[(crop_suitability["year"] >= 2021) & (crop_suitability["year"] <= 2030)]
        crop_suitability = crop_suitability.rename(columns={"score": "crop_suitability"})
        print(f"Loaded Crop Suitability for {scenario_name}")
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
        if use_past:
            CY_Maize_FULL = pd.read_csv(cy_maize_file, sep=',')
            CY_Maize = CY_Maize_FULL[['Dates', 'Profits']].copy()
        else:
            CY_Maize_FULL = pd.read_csv(cy_maize_file, sep=',')
            CY_Maize = CY_Maize_FULL[['Dates', 'Profits']].copy()
        CY_Maize['Dates'] = pd.to_datetime(CY_Maize['Dates'])
        CY_Maize['year'] = CY_Maize['Dates'].dt.year
        CY_Maize = CY_Maize.groupby('year').last().reset_index()
        CY_Maize = CY_Maize.rename(columns={'Profits': 'crop_profit'})
        print(f"Processed Crop Yield for Maize for {scenario_name}")
    except FileNotFoundError:
        print(f"Error: File '{cy_maize_file}' not found.")
        return None

    try:
        if use_past:
            CY_Wheat_FULL = pd.read_csv(cy_wheat_file, sep=',')
            CY_Wheat = CY_Wheat_FULL[['Dates', 'Profits']].copy()
        else:
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

    crop_profit = pd.concat([CY_Maize, CY_Wheat], ignore_index=True)
    try:
        crop_profit = crop_suitability.merge(crop_profit, on="year", how="inner")[["year", "lat", "lon", "crop_profit"]]
    except KeyError as e:
        print(f"Error during merging crop_profit: {e}")
        return None

    for df in [PV_, crop_suitability, PV_Yield_, crop_profit]:
        df["lat"] = df["lat"].astype(float).round(5)
        df["lon"] = df["lon"].astype(float).round(5)

    env_data = crop_suitability.merge(PV_, on=["year", "lat", "lon"], how="inner")
    env_data = env_data.merge(PV_Yield_, on=["year", "lat", "lon"], how="inner")
    env_data = env_data.merge(crop_profit, on=["year", "lat", "lon"], how="inner")

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
    parser = argparse.ArgumentParser(description="Run the land-use simulation with a specific scenario.")
    parser.add_argument("--scenario", type=str, choices=["26", "45", "85"], required=False,
                        help="Specify the RCP scenario to run the simulation (options: '26', '45', '85').")
    parser.add_argument("--past", action="store_true",
                        help="Run simulation with past data instead of future (RCP) data.")
    parser.add_argument("--subsidy", type=float, default=0.0, help="Base subsidy for PV installations (default: 0.0).")
    parser.add_argument("--loan_interest_rate", type=float, default=0.05, help="Loan interest rate for PV installations (default: 0.05).")
    parser.add_argument("--tax_incentive_rate", type=float, default=0.0, help="Tax incentive rate (default: 0.0).")
    parser.add_argument("--social_prestige_weight", type=float, default=0.0, help="Social prestige weight (default: 0.0).")
    parser.add_argument("--climate_losses", type=str, default="climate_losses.json",
                        help="Path to JSON file containing climate-related economic losses.")
    parser.add_argument("--physis_health_status", type=float, default=0.8, help="Initial health status (default: 0.8).")
    parser.add_argument("--physis_labor_availability", type=float, default=0.6, help="Initial labor availability (default: 0.6).")
    parser.add_argument("--emotion_stress_level", type=float, default=0.3, help="Initial stress level (default: 0.3).")
    parser.add_argument("--emotion_satisfaction", type=float, default=0.7, help="Initial satisfaction (default: 0.7).")
    parser.add_argument("--cognition_policy_incentives", type=float, default=0.5, help="Initial policy incentives (default: 0.5).")
    parser.add_argument("--cognition_information_access", type=float, default=0.6, help="Initial information access (default: 0.6).")
    parser.add_argument("--social_social_influence", type=float, default=0.4, help="Initial social influence (default: 0.4).")
    parser.add_argument("--social_community_participation", type=float, default=0.5, help="Initial community participation (default: 0.5).")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file for PECS parameters.")
    parser.add_argument("--initial_fund", type=str, default="initial_fund.json",
                        help="Path to the JSON file containing the initial subsidy fund for PV installations.")
    args = parser.parse_args()

    # Determine scenario name and data folders based on --past flag.
    MAIN_DIR = "/Users/mbanti/Documents/Projects/esa_agents/BACKEND_API/Demo_With_Geoserver/INPUT_RL"
    WHEAT_PAST   = os.path.join(MAIN_DIR, "WHEAT_PAST")
    WHEAT_FUTURE = os.path.join(MAIN_DIR, "WHEAT_FUTURE")
    MAIZE_PAST   = os.path.join(MAIN_DIR, "MAIZE_PAST")
    MAIZE_FUTURE = os.path.join(MAIN_DIR, "MAIZE_FUTURE")
    PV_PAST      = os.path.join(MAIN_DIR, "PV_PAST")
    PV_FUTURE    = os.path.join(MAIN_DIR, "PV_FUTURE")

    if args.past:
        scenario_name = "PAST"
        print("Running simulation with past data...")
        pv_suitability_file = os.path.join(PV_PAST, "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
        pv_yield_file       = os.path.join(PV_PAST, "PAST_PV_YIELD.csv")
        cs_wheat_file       = os.path.join(WHEAT_PAST, "WHEAT_PAST_LUSA_PREDICTIONS.csv")
        cy_wheat_file       = os.path.join(WHEAT_PAST, "AquaCrop_Results_PAST.csv")
        cs_maize_file       = os.path.join(MAIZE_PAST, "MAIZE_PAST_LUSA_PREDICTIONS.csv")
        cy_maize_file       = os.path.join(MAIZE_PAST, "AquaCrop_Results_PAST.csv")
    else:
        if not args.scenario:
            raise ValueError("For future (RCP) simulations, please provide --scenario (e.g. 26, 45, 85).")
        scenario_name = f"RCP{args.scenario}"
        print(f"Running simulation for scenario {scenario_name}...")
        pv_suitability_file = os.path.join(PV_FUTURE, f"RCP{args.scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
        pv_yield_file       = os.path.join(PV_FUTURE, f"RCP{args.scenario}_PV_YIELD.csv")
        cs_maize_file = os.path.join(MAIZE_FUTURE, f"RCP{args.scenario}_LUSA_PREDICTIONS.csv")
        cs_wheat_file = os.path.join(WHEAT_FUTURE, f"RCP{args.scenario}_LUSA_PREDICTIONS.csv")
        cy_maize_file = os.path.join(MAIZE_FUTURE, f"AquaCrop_Results_RCP{args.scenario}.csv")
        cy_wheat_file = os.path.join(WHEAT_FUTURE, f"AquaCrop_Results_RCP{args.scenario}.csv")

    # Load PECS parameters
    if args.config:
        with open(args.config, "r") as f:
            pecs_params = json.load(f)
    else:
        pecs_params = {
            "physis": {
                "health_status": args.physis_health_status,
                "labor_availability": args.physis_labor_availability
            },
            "emotion": {
                "stress_level": args.emotion_stress_level,
                "satisfaction": args.emotion_satisfaction
            },
            "cognition": {
                "policy_incentives": args.cognition_policy_incentives,
                "information_access": args.cognition_information_access
            },
            "social": {
                "social_influence": args.social_social_influence,
                "community_participation": args.social_community_participation
            }
        }

    # Load initial subsidy fund from JSON.
    with open(args.initial_fund, "r") as f:
        fund_data = json.load(f)
    initial_subsidy_fund = fund_data.get("initial_subsidy_fund", 1.0)

    # Build scenario parameters
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

    # Load climate losses
    if args.climate_losses:
        with open(args.climate_losses, "r") as f:
            climate_losses = json.load(f)
    else:
        climate_losses = {2021: 0.02, 2022: 0.05, 2023: 0.08, 2024: 0.1, 2025: 0.15}

    print(f"Running simulation for {scenario_name}")
    print(f"PECS Parameters: {pecs_params}")
    print(f"Scenario Parameters: {scenario_params}")
    print(f"Climate Losses: {climate_losses}")

    train_model = True

    # Create aggregated environmental data.
    MAIN_DIR = "/Users/mbanti/Documents/Projects/esa_agents/BACKEND_API/Demo_With_Geoserver/INPUT_RL"
    aggregated_data = create_aggregated_data(args.scenario if not args.past else None, MAIN_DIR, "data/output", use_past=args.past)
    if aggregated_data is None:
        print("Data aggregation failed. Exiting.")
        exit(1)

    width = 10
    height = 10

    # Create the environment without passing eu_target.
    env = MesaEnv(
        env_data=aggregated_data,
        max_steps=10,
        pecs_params=pecs_params,
        width=width,
        height=height,
        scenario_params=scenario_params,
        climate_losses=climate_losses
    )

    try:
        check_env(env)
        print("Environment passed the Gym API check.")
    except Exception as e:
        print(f"Environment check failed: {e}")
        exit(1)

    learning_rate = 0.0009809274356741915
    batch_size = 64
    n_epochs = 6

    if train_model:
        SAVE_DIR = "trained_models"
        os.makedirs(SAVE_DIR, exist_ok=True)
        scenario_dir = os.path.join(SAVE_DIR, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        # For training logging, re-read scenario parameters
        subsidy = scenario_params.get("subsidy", 0.0)
        loan_interest_rate = scenario_params.get("loan_interest_rate", 0.05)
        tax_incentive_rate = scenario_params.get("tax_incentive_rate", 0.0)
        social_prestige_weight = scenario_params.get("social_prestige_weight", 0.0)

        logging.basicConfig(
            filename=os.path.join(scenario_dir, "training.log"),
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s"
        )

        logging.info(f"Starting training for {scenario_name} with PECS parameters: {pecs_params}")
        logging.info(f"Subsidy: {subsidy}, Loan Interest Rate: {loan_interest_rate}, "
                     f"Tax Incentive Rate: {tax_incentive_rate}, Social Prestige Weight: {social_prestige_weight}")
        logging.info(f"Climate Losses: {climate_losses}")

        # Save the scenario parameters
        scenario_params = {
            "subsidy": subsidy,
            "loan_interest_rate": loan_interest_rate,
            "tax_incentive_rate": tax_incentive_rate,
            "social_prestige_weight": social_prestige_weight,
            "initial_subsidy_fund": scenario_params.get("initial_subsidy_fund", 1.0)
        }
        with open(os.path.join(scenario_dir, "scenario_params.json"), "w") as f:
            json.dump(scenario_params, f, indent=4)

        # Callback to log PV adoption rate (without EU target now)
        class PVAdoptionCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(PVAdoptionCallback, self).__init__(verbose)
            def _on_step(self) -> bool:
                pv_adopters = sum([agent.decision for agent in self.training_env.envs[0].model.schedule.agents])
                total_agents = len(self.training_env.envs[0].model.schedule.agents)
                pv_adoption_rate = pv_adopters / total_agents
                logging.info(f"Step {self.num_timesteps}: PV Adoption Rate = {pv_adoption_rate:.2f}")
                return True

        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            device="mps"
        )

        def get_processing_stats():
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            gpu_status = "Using Apple's MPS GPU" if torch.backends.mps.is_available() else "GPU not available"
            return cpu_usage, memory_usage, gpu_status

        start_time = time.time()
        env.reset()
        model.learn(total_timesteps=5000)
        env.close()
        end_time = time.time()
        model.save(os.path.join(scenario_dir, "ppo_model"))
        with open(os.path.join(scenario_dir, "pecs_params.json"), "w") as f:
            json.dump(pecs_params, f, indent=4)
        training_time = end_time - start_time
        cpu_usage, memory_usage, gpu_status = get_processing_stats()
        logging.info(f"Training Time: {training_time:.2f} seconds")
        logging.info(f"CPU Usage: {cpu_usage}%")
        logging.info(f"Memory Usage: {memory_usage}%")
        logging.info(f"GPU Status: {gpu_status}")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"CPU Usage: {cpu_usage}%")
        print(f"Memory Usage: {memory_usage}%")
        print(f"GPU Status: {gpu_status}")

if __name__ == "__main__":
    main()
