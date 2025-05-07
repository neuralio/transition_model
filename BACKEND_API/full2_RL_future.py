#!/usr/bin/env python
"""
full2_RL_future.py

Description:
  This script trains a PPO agent in a custom Mesa environment and then runs a simulation for a future (RCP) scenario.
  It reads all configuration (modified PECS parameters, scenario parameters, initial subsidy fund, and climate losses)
  from the JSON file '/app/input_mongo_full2_future.json'. The simulation aggregates environmental data based on the
  specified RCP scenario, trains the PPO model, and then runs several evaluation episodes to collect agent decisions.
  
Directory structure:
  MAIN_DIR = "/app/INPUT_RL"
  INPUT_JSON_PATH = "/app/input_mongo_full2_future.json"
  BASE_DIR = os.path.dirname(MAIN_DIR)
  RESULTS_DIR = os.path.join(BASE_DIR, "RESULTS_RL", "FULL2_RL")
  A directory named "task_id_user_id" will be created under RESULTS_DIR based on the input JSON.
  
Usage:
  This script is designed to be called either from the command line or by an API endpoint.
  
Note:
  The model classes (LandUseAgent, LandUseModel, and MesaEnv) have been embedded below so that no external
  module import is required.
"""

import os
import json
import time
import logging
import shutil
import pandas as pd
import numpy as np
import psutil
import torch
import random
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList

# ---------------------------
# Embedded Model Code
# ---------------------------
# BEGIN: LandUseAgent
from mesa import Agent
import random

class LandUseAgent(Agent):
    def __init__(self, unique_id, model, crop_suitability, pv_suitability, crop_profit, pv_profit, lat, lon, pecs_params, scenario_params=None, climate_losses=None):
        super().__init__(unique_id, model)
        
        # Initialize agent-specific attributes
        self.unique_id = unique_id
        self.crop_suitability = crop_suitability
        self.pv_suitability = pv_suitability
        self.crop_profit = crop_profit
        self.pv_profit = pv_profit
        self.lat = lat
        self.lon = lon
        self.decision = 0  # 0: Keep crops, 1: Convert to PV

        # Initialize PECS attributes from passed parameters
        self.physis = pecs_params["physis"].copy()
        self.emotion = pecs_params["emotion"].copy()
        self.cognition = pecs_params["cognition"].copy()
        self.social = pecs_params["social"].copy()

        # Scenario-specific parameters
        self.subsidy = scenario_params.get("subsidy", 0.0) if scenario_params else 0.0
        self.loan_interest_rate = scenario_params.get("loan_interest_rate", 0.05) if scenario_params else 0.05
        self.tax_incentive_rate = scenario_params.get("tax_incentive_rate", 0.0) if scenario_params else 0.0
        self.social_prestige_weight = scenario_params.get("social_prestige_weight", 0.0) if scenario_params else 0.0

        # Climate loss factor per year
        self.climate_losses = climate_losses or {}
        self.current_loss_factor = 0.0  # Updated dynamically during simulation

    def step(self):
        """
        Update the agent's state and make decisions based on PECS attributes.
        """
        # Update climate loss factor dynamically based on the current step (year)
        current_year = self.model.current_step + 2021
        self.current_loss_factor = self.climate_losses.get(current_year, 0.0)  # Default 0 if no data

        # Apply climate loss impact on crop profits
        self.crop_profit *= (1 - self.current_loss_factor)

        # Update PECS attributes dynamically
        self.update_physis()
        self.update_emotion()
        self.update_cognition()
        self.update_social()

        # Decision-making based on PECS and suitability
        self.make_decision()

    def update_physis(self):
        if self.decision == 0:  # Farming
            self.physis["health_status"] -= 0.01  # Farming reduces health
            self.physis["labor_availability"] -= 0.01
        elif self.decision == 1:  # PV installation
            self.physis["labor_availability"] -= 0.02  # Labor required for PV setup

    def update_emotion(self):
        if self.decision == 0:  # Farming
            self.emotion["stress_level"] += 0.01
            self.emotion["satisfaction"] += 0.02
        elif self.decision == 1:  # PV installation
            self.emotion["stress_level"] -= 0.02
            self.emotion["satisfaction"] += 0.03

    def update_cognition(self):
        if self.model.policy_incentive_active:
            self.cognition["policy_incentives"] += 0.05
        else:
            self.cognition["policy_incentives"] -= 0.01

    def update_social(self):
        if not hasattr(self, "pos") or self.pos is None:
            self.social["social_influence"] = 0  # No position assigned
            return

        # Get neighbors
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        if neighbors:
            avg_neighbor_decision = sum([neighbor.decision for neighbor in neighbors]) / len(neighbors)
        else:
            avg_neighbor_decision = 0  # No neighbors

        # Calculate social influence dynamically based on the agent's decision
        if self.decision == 0:  # Crop farming
            social_influence = 1 - avg_neighbor_decision  # Influence from PV neighbors
        elif self.decision == 1:  # PV installation
            social_influence = avg_neighbor_decision  # Influence from PV neighbors
        else:
            social_influence = 0  # Default for unexpected cases

        # Apply a dynamic weight to social influence (e.g., policy incentives)
        social_influence_weight = self.cognition["policy_incentives"]
        self.social["social_influence"] = social_influence * social_influence_weight

    def make_decision(self):
        crop_profit = self.crop_profit  # Adjusted crop profit with climate loss
        pv_profit = self.pv_profit      # PV profit remains unchanged

        # Adjust PV profit with scenario-specific parameters
        pv_profit_with_subsidy = pv_profit + self.subsidy - (pv_profit * self.loan_interest_rate)
        pv_profit_with_tax = pv_profit_with_subsidy + (pv_profit * self.tax_incentive_rate)

        # Adjust decision-making based on climate loss impact
        if self.current_loss_factor > 0.3:  # High climate loss (>30%)
            self.decision = 1  # Shift to PV due to extreme climate loss
        elif crop_profit > pv_profit_with_tax and self.physis["health_status"] > 0.5:
            self.decision = 0  # Prioritize crops
        elif pv_profit_with_tax > crop_profit and self.cognition["policy_incentives"] > 0.6:
            self.decision = 1  # Prioritize PV
        elif self.emotion["stress_level"] > 0.7:
            self.decision = 0  # Avoid risk under high stress
        elif self.social["social_influence"] > 0.5 or self.social["social_influence"] * self.social_prestige_weight > 0.6:
            self.decision = 1  # Follow social trends toward PV
        else:
            # Default to farming if no clear preference
            self.decision = 0

    def get_state(self):
        return {
            "Agent ID": self.unique_id,
            "Lat": self.lat,
            "Lon": self.lon,
            "Crop Suitability": self.crop_suitability,
            "PV Suitability": self.pv_suitability,
            "Decision": self.decision,
            "Health Status": self.physis["health_status"],
            "Stress Level": self.emotion["stress_level"],
            "Policy Incentives": self.cognition["policy_incentives"],
            "Social Influence": self.social["social_influence"],
            "Subsidy": self.subsidy,  # Added subsidy parameter
            "Loan Interest Rate": self.loan_interest_rate,  # Added loan interest rate
            "Tax Incentive Rate": self.tax_incentive_rate,  # Added tax incentive rate
            "Social Prestige Weight": self.social_prestige_weight,  # Added social prestige weight
            "Climate Loss Factor": self.current_loss_factor  # 🌍 New: Track climate losses
        }
# END: LandUseAgent

# BEGIN: LandUseModel
from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import MultiGrid

class LandUseModel(Model):
    def __init__(self, env_data, width, height, pecs_params, scenario_params=None, climate_losses=None, seed=None):
        super().__init__()
        self.env_data = env_data
        self.schedule = BaseScheduler(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.current_step = 0
        self.policy_incentive_active = False

        # Store scenario parameters and climate losses
        self.scenario_params = scenario_params or {}
        self.climate_losses = climate_losses or {}

        # Extract scenario parameters
        self.base_subsidy = self.scenario_params.get("subsidy", 0.0)  # Base subsidy for PV installations
        self.loan_interest_rate = self.scenario_params.get("loan_interest_rate", 0.05)
        self.tax_incentive_rate = self.scenario_params.get("tax_incentive_rate", 0.0)
        self.social_prestige_weight = self.scenario_params.get("social_prestige_weight", 0.0)

        # Initialize the saturation fund dedicated to PV installations.
        self.initial_fund = self.scenario_params.get("initial_subsidy_fund", 1.0)
        self.remaining_fund = self.initial_fund

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Create agents and place them on the grid.
        for agent_id, row in self.env_data.iterrows():
            crop_suitability = row["crop_suitability"]
            pv_suitability = row["pv_suitability"]
            pv_profit = row["pv_profit"]
            crop_profit = row["crop_profit"]
            lat = row["lat"]
            lon = row["lon"]

            # Create agent.
            agent = LandUseAgent(
                unique_id=agent_id,
                model=self,
                crop_suitability=crop_suitability,
                pv_suitability=pv_suitability,
                crop_profit=crop_profit,
                pv_profit=pv_profit,
                lat=lat,
                lon=lon,
                pecs_params=pecs_params,
                scenario_params=self.scenario_params,
                climate_losses=self.climate_losses
            )
            self.schedule.add(agent)

            # Normalize lat/lon to grid coordinates.
            x = int((lat - self.env_data["lat"].min()) / (self.env_data["lat"].max() - self.env_data["lat"].min()) * (width - 1))
            y = int((lon - self.env_data["lon"].min()) / (self.env_data["lon"].max() - self.env_data["lon"].min()) * (height - 1))
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            self.grid.place_agent(agent, (x, y))

    def step(self):
        current_year = self.current_step + 2021
        # Apply climate losses dynamically
        if current_year in self.climate_losses:
            loss_factor = self.climate_losses[current_year]
            for agent in self.schedule.agents:
                agent.crop_profit *= (1 - loss_factor)
                print(f"Agent {agent.unique_id}: Climate loss applied for {current_year}, new crop profit: {agent.crop_profit:.2f}")

        # Update policy incentives (toggle every 10 steps)
        if self.current_step % 10 == 0:
            self.policy_incentive_active = not self.policy_incentive_active
            for agent in self.schedule.agents:
                agent.cognition["policy_incentives"] += 0.05 if self.policy_incentive_active else -0.05
                agent.cognition["policy_incentives"] = min(max(agent.cognition["policy_incentives"], 0.0), 1.0)

        # Step all agents
        self.schedule.step()
        self.current_step += 1

        # --- Saturation Fund Mechanism ---
        ALPHA = 0.05  # Fixed deduction per PV adoption per step
        num_PV_adopters = sum(1 for agent in self.schedule.agents if agent.decision == 1)
        self.remaining_fund = max(0, self.remaining_fund - ALPHA * num_PV_adopters)
        effective_subsidy = self.base_subsidy * (self.remaining_fund / self.initial_fund)
        print(f"Remaining Fund: {self.remaining_fund:.3f} / {self.initial_fund}, Effective Subsidy: {effective_subsidy:.3f}")
        # Update each agent’s subsidy value.
        for agent in self.schedule.agents:
            agent.subsidy = effective_subsidy

    def collect_agent_data(self):
        agent_data = []
        for agent in self.schedule.agents:
            agent_data.append({
                "Agent ID": agent.unique_id,
                "Lat": agent.lat,
                "Lon": agent.lon,
                "Crop Suitability": agent.crop_suitability,
                "PV Suitability": agent.pv_suitability,
                "Crop Profit": agent.crop_profit,
                "PV Profit": agent.pv_profit,
                "Decision": agent.decision,
                "Health Status": agent.physis["health_status"],
                "Stress Level": agent.emotion["stress_level"],
                "Policy Incentives": agent.cognition["policy_incentives"],
                "Social Influence": agent.social["social_influence"],
                "Subsidy": agent.subsidy,
                "Loan Interest Rate": agent.loan_interest_rate,
                "Tax Incentive Rate": agent.tax_incentive_rate,
                "Social Prestige Weight": agent.social_prestige_weight,
                "Climate Loss Factor": self.climate_losses.get(self.current_step + 2021, 0.0)
            })
        return agent_data

    def save_agent_data(self, filename="agent_data.csv"):
        df = pd.DataFrame(self.collect_agent_data())
        df.to_csv(filename, index=False)
# END: LandUseModel

# BEGIN: MesaEnv
import gymnasium as gym
from gymnasium import spaces

class MesaEnv(gym.Env):
    def __init__(self, env_data, max_steps, pecs_params, width=5, height=5, scenario_params=None, climate_losses=None):
        super(MesaEnv, self).__init__()
        self.env_data = env_data.reset_index(drop=True)
        self.max_steps = max_steps
        self.width = width
        self.height = height

        # Store provided parameters
        self.pecs_params = pecs_params
        self.scenario_params = scenario_params or {}
        self.climate_losses = climate_losses or {}

        # Initialize the model.
        self.model = LandUseModel(
            self.env_data,
            width=self.width,
            height=self.height,
            pecs_params=self.pecs_params,
            scenario_params=self.scenario_params,
            climate_losses=self.climate_losses
        )

        # Define observation space: two suitability variables, four PECS variables, and one extra for climate_loss_factor.
        suitability_vars = ["crop_suitability", "pv_suitability"]
        pecs_vars = ["health_status", "stress_level", "policy_incentives", "social_influence"]
        min_values = np.concatenate([
            self.env_data[suitability_vars].min().values,
            np.array([0.0] * (len(pecs_vars) + 1))
        ])
        max_values = np.concatenate([
            self.env_data[suitability_vars].max().values,
            np.array([1.0] * (len(pecs_vars) + 1))
        ])
        self.observation_space = spaces.Box(low=min_values, high=max_values, dtype=np.float32)

        # Action space: Each agent chooses 0 (crop) or 1 (PV)
        num_agents = len(self.env_data)
        self.action_space = spaces.MultiDiscrete([2] * num_agents)

        self.current_step = 0
        self.state = None

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.model = LandUseModel(
            self.env_data,
            width=self.width,
            height=self.height,
            pecs_params=self.pecs_params,
            scenario_params=self.scenario_params,
            climate_losses=self.climate_losses,
            seed=seed
        )
        self.state = self._get_current_state()
        expected_shape = self.observation_space.shape
        if self.state.shape != expected_shape:
            raise ValueError(f"Mismatch: Expected {expected_shape}, but got {self.state.shape}")
        return self.state, {}

    def step(self, action):
        for idx, agent in enumerate(self.model.schedule.agents):
            if action is not None:
                agent.decision = action[idx]
        self.model.step()
        reward = self._calculate_reward()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        self.state = self._get_current_state() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        truncated = False
        return self.state, reward, terminated, truncated, {}

    def _get_current_state(self, detailed=False):
        if detailed:
            return np.array([
                {
                    "crop_suitability": agent.crop_suitability,
                    "pv_suitability": agent.pv_suitability,
                    "health_status": agent.physis["health_status"],
                    "stress_level": agent.emotion["stress_level"],
                    "policy_incentives": agent.cognition["policy_incentives"],
                    "social_influence": agent.social["social_influence"],
                    "climate_loss_factor": self.climate_losses.get(self.current_step + 2021, 0.0)
                } for agent in self.model.schedule.agents
            ], dtype=object)
        total_agents = len(self.model.schedule.agents)
        weighted_social_influence = np.sum([
            agent.social["social_influence"] * agent.cognition["policy_incentives"]
            for agent in self.model.schedule.agents
        ]) / total_agents
        aggregated_state = {
            "crop_suitability": np.mean([agent.crop_suitability for agent in self.model.schedule.agents]),
            "pv_suitability": np.mean([agent.pv_suitability for agent in self.model.schedule.agents]),
            "health_status": np.mean([agent.physis["health_status"] for agent in self.model.schedule.agents]),
            "stress_level": np.mean([agent.emotion["stress_level"] for agent in self.model.schedule.agents]),
            "policy_incentives": np.mean([agent.cognition["policy_incentives"] for agent in self.model.schedule.agents]),
            "social_influence": weighted_social_influence,
            "climate_loss_factor": np.mean([
                self.climate_losses.get(self.current_step + 2021, 0.0)
                for _ in self.model.schedule.agents
            ])
        }
        return np.array(list(aggregated_state.values()), dtype=np.float32)

    def _calculate_reward(self):
        total_reward = 0.0
        for agent in self.model.schedule.agents:
            climate_loss_factor = self.climate_losses.get(self.current_step + 2021, 0.0)
            adjusted_crop_profit = agent.crop_profit * (1 - climate_loss_factor)
            neighbors = self.model.grid.get_neighbors(agent.pos, moore=True, include_center=False)
            avg_neighbor_decision = sum([neighbor.decision for neighbor in neighbors]) / len(neighbors) if neighbors else 0
            social_influence = avg_neighbor_decision if agent.decision == 1 else (1 - avg_neighbor_decision)
            social_influence_weight = agent.cognition["policy_incentives"]
            if agent.decision == 0:
                total_reward += (
                    adjusted_crop_profit * 0.4 +
                    agent.emotion["satisfaction"] * 0.2 -
                    agent.emotion["stress_level"] * 0.2 +
                    social_influence * social_influence_weight
                )
            elif agent.decision == 1:
                total_reward += (
                    agent.pv_profit * 0.4 +
                    agent.cognition["policy_incentives"] * 0.2 -
                    agent.physis["labor_availability"] * 0.2 +
                    social_influence * social_influence_weight
                )
        return total_reward / len(self.model.schedule.agents)

    def close(self):
        pass
# END: MesaEnv

# ---------------------------
# End of Embedded Model Code
# ---------------------------

# ---------------------------
# Directory definitions for main script
# ---------------------------
# MAIN_DIR = "/app/INPUT_RL"
# INPUT_JSON_PATH = "/app/input_mongo_full2_future.json"
# BASE_DIR = os.path.dirname(MAIN_DIR)
# RESULTS_DIR = os.path.join(BASE_DIR, "RESULTS_RL", "FULL2_RL")
# os.makedirs(RESULTS_DIR, exist_ok=True)


MAIN_DIR = "/app"
INPUT_JSON_PATH = os.path.join(MAIN_DIR, "input_mongo_full2_future.json")
RESULTS_DIR = os.path.join(MAIN_DIR, "RESULTS_RL", "FULL2_RL")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------
# Default PECS parameters (used during training if none provided)
# --------------------------
DEFAULT_PECS_PARAMS = {
    "physis": {"health_status": 0.8, "labor_availability": 0.6},
    "emotion": {"stress_level": 0.3, "satisfaction": 0.7},
    "cognition": {"policy_incentives": 0.5, "information_access": 0.6},
    "social": {"social_influence": 0.4, "community_participation": 0.5}
}

DEFAULT_SCENARIO_PARAMS = {
    "subsidy": 0.0,
    "loan_interest_rate": 0.05,
    "tax_incentive_rate": 0.0,
    "social_prestige_weight": 0.0
}


# ---------------------------
# Utility functions for configuration and data aggregation
# ---------------------------
def load_config(json_path):
    """Load configuration JSON from the provided path."""
    with open(json_path, "r") as f:
        config = json.load(f)
    return config

def create_aggregated_data(scenario, main_dir, output_dir):
    """
    Create aggregated environmental data for a future scenario.
    The function reads CSV files from the future data directories, merges them,
    and aggregates the data by unique (lat, lon) pairs.
    """
    scenario_name = f"RCP{scenario}"
    # Define file paths for future data folders using the passed main_dir.
    # WHEAT_FUTURE = os.path.join(main_dir, "WHEAT_FUTURE")
    # MAIZE_FUTURE = os.path.join(main_dir, "MAIZE_FUTURE")
    # PV_FUTURE    = os.path.join(main_dir, "PV_FUTURE")

    INPUT_RL_DIR = os.path.join(MAIN_DIR, "INPUT_RL")
    WHEAT_FUTURE = os.path.join(INPUT_RL_DIR, "WHEAT_FUTURE")
    MAIZE_FUTURE = os.path.join(INPUT_RL_DIR, "MAIZE_FUTURE")
    PV_FUTURE    = os.path.join(INPUT_RL_DIR, "PV_FUTURE")

    
    pv_suitability_file = os.path.join(PV_FUTURE, f"{scenario_name}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    pv_yield_file       = os.path.join(PV_FUTURE, f"{scenario_name}_PV_YIELD.csv")
    cs_maize_file       = os.path.join(MAIZE_FUTURE, f"{scenario_name}_LUSA_PREDICTIONS.csv")
    cs_wheat_file       = os.path.join(WHEAT_FUTURE, f"{scenario_name}_LUSA_PREDICTIONS.csv")
    cy_maize_file       = os.path.join(MAIZE_FUTURE, f"AquaCrop_Results_{scenario_name}.csv")
    cy_wheat_file       = os.path.join(WHEAT_FUTURE, f"AquaCrop_Results_{scenario_name}.csv")
    
    # Read and filter the PV suitability data
    try:
        PV_ = pd.read_csv(pv_suitability_file)
        PV_ = PV_.rename(columns={"score": "pv_suitability"})
        PV_ = PV_[(PV_["year"] >= 2021) & (PV_["year"] <= 2030)]
    except FileNotFoundError:
        print(f"Error: File '{pv_suitability_file}' not found.")
        return None

    # Read crop suitability (both maize and wheat)
    try:
        CS_Maize_ = pd.read_csv(cs_maize_file)
        CS_Wheat_ = pd.read_csv(cs_wheat_file)
        crop_suitability = pd.concat([CS_Maize_, CS_Wheat_], ignore_index=True)
        crop_suitability = crop_suitability[(crop_suitability["year"] >= 2021) & (crop_suitability["year"] <= 2030)]
        crop_suitability = crop_suitability.rename(columns={"score": "crop_suitability"})
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # Read PV yield data
    try:
        PV_Yield_ = pd.read_csv(pv_yield_file)
        PV_Yield_ = PV_Yield_.rename(columns={'annual_electricity_savings_€': 'pv_profit'})
    except FileNotFoundError:
        print(f"Error: File '{pv_yield_file}' not found.")
        return None

    # Process crop yield for Maize
    try:
        CY_Maize_FULL = pd.read_csv(cy_maize_file)
        CY_Maize = CY_Maize_FULL[['Dates', 'Profits']].copy()
        CY_Maize['Dates'] = pd.to_datetime(CY_Maize['Dates'])
        CY_Maize['year'] = CY_Maize['Dates'].dt.year
        CY_Maize = CY_Maize.groupby('year').last().reset_index()
        CY_Maize = CY_Maize.rename(columns={'Profits': 'crop_profit'})
    except FileNotFoundError:
        print(f"Error: File '{cy_maize_file}' not found.")
        return None

    # Process crop yield for Wheat
    try:
        CY_Wheat_FULL = pd.read_csv(cy_wheat_file)
        CY_Wheat = CY_Wheat_FULL[['Dates', 'Profits']].copy()
        CY_Wheat['Dates'] = pd.to_datetime(CY_Wheat['Dates'])
        CY_Wheat['year'] = CY_Wheat['Dates'].dt.year
        CY_Wheat = CY_Wheat.groupby('year').last().reset_index()
        CY_Wheat = CY_Wheat.rename(columns={'Profits': 'crop_profit'})
    except FileNotFoundError:
        print(f"Error: File '{cy_wheat_file}' not found.")
        return None

    # Merge crop yield data
    crop_profit = pd.concat([CY_Maize, CY_Wheat], ignore_index=True)
    try:
        crop_profit = crop_suitability.merge(crop_profit, on="year", how="inner")[["year", "lat", "lon", "crop_profit"]]
    except KeyError as e:
        print(f"Error during merging crop_profit: {e}")
        return None

    # Ensure latitude and longitude columns are float
    for df in [PV_, crop_suitability, PV_Yield_, crop_profit]:
        df["lat"] = df["lat"].astype(float).round(5)
        df["lon"] = df["lon"].astype(float).round(5)

    # Merge all environmental datasets together
    try:
        env_data = crop_suitability.merge(PV_, on=["year", "lat", "lon"], how="inner")
        env_data = env_data.merge(PV_Yield_, on=["year", "lat", "lon"], how="inner")
        env_data = env_data.merge(crop_profit, on=["year", "lat", "lon"], how="inner")
    except KeyError as e:
        print(f"Error during merging datasets: {e}")
        return None

    # Aggregate data by (lat, lon)
    aggregated_data = env_data.groupby(['lat', 'lon'], as_index=False).agg({
        'crop_suitability': 'mean',
        'pv_suitability': 'mean',
        'crop_profit': 'sum',
        'pv_profit': 'sum'
    })

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"aggregated_data_{scenario_name}.csv")
    aggregated_data.to_csv(output_file, index=False)
    print(f"Aggregated data saved to '{output_file}'.")
    return aggregated_data

# ---------------------------
# Callback for PPO training logging
# ---------------------------
class PVAdoptionCallback(BaseCallback):
    """Callback to log PV adoption rate during training."""
    def __init__(self, verbose=0):
        super(PVAdoptionCallback, self).__init__(verbose)
    def _on_step(self) -> bool:
        pv_adopters = sum([agent.decision for agent in self.training_env.envs[0].model.schedule.agents])
        total_agents = len(self.training_env.envs[0].model.schedule.agents)
        pv_adoption_rate = pv_adopters / total_agents
        logging.info(f"Step {self.num_timesteps}: PV Adoption Rate = {pv_adoption_rate:.2f}")
        return True

def get_processing_stats():
    """Return current CPU, memory, and GPU (if available) stats."""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    gpu_status = "Using Apple's MPS GPU" if torch.backends.mps.is_available() else "GPU not available"
    return cpu_usage, memory_usage, gpu_status

# ---------------------------
# Main training and simulation function
# ---------------------------
# --------------------------
# Default PECS parameters (used during training if none provided)
# --------------------------
DEFAULT_PECS_PARAMS = {
    "physis": {"health_status": 0.8, "labor_availability": 0.6},
    "emotion": {"stress_level": 0.3, "satisfaction": 0.7},
    "cognition": {"policy_incentives": 0.5, "information_access": 0.6},
    "social": {"social_influence": 0.4, "community_participation": 0.5}
}

DEFAULT_SCENARIO_PARAMS = {
    "subsidy": 0.0,
    "loan_interest_rate": 0.05,
    "tax_incentive_rate": 0.0,
    "social_prestige_weight": 0.0
}


import os
import time
import logging
import psutil
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

# ... your other imports and DEFAULT_* definitions ...

class EarlyStoppingCallback(BaseCallback):
    """
    Stops training when the evaluation reward has not improved by at least min_delta
    for patience consecutive evaluations.
    """
    def __init__(self, eval_env, eval_freq: int, patience: int, min_delta: float = 0.0, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -float("inf")
        self.no_improvement_evals = 0

    def _on_step(self) -> bool:
        # each call is one rollout step
        if self.n_calls % self.eval_freq == 0:
            # run one deterministic episode on eval_env
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            while not done and not truncated:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                total_reward += reward

            if self.verbose > 0:
                logging.info(f"Eval at step {self.num_timesteps}: reward={total_reward:.3f}")

            # check improvement
            if total_reward > self.best_mean_reward + self.min_delta:
                self.best_mean_reward = total_reward
                self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1

            if self.no_improvement_evals >= self.patience:
                if self.verbose > 0:
                    logging.info("Stopping training early (no improvement).")
                return False  # returning False stops training

        return True  # continue training

def train_and_simulate(config):
    # ... (same config unpacking & env creation as before) ...

    # after setting up env:
    try:
        check_env(env)
        logging.info("Environment passed the Gym API check.")
    except Exception as e:
        logging.error(f"Environment check failed: {e}")
        return

    # Training parameters
    learning_rate = 0.0009809274356741915
    batch_size = 64
    n_epochs = 6
    total_timesteps = 5000
    eval_freq = 1000
    patience = 5
    min_delta = 1e-2

    # Define the PPO model
    ppo_model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )

    logging.info(f"Starting training for {scenario_name} with early stopping.")
    start_time = time.time()
    env.reset()

    # set up early stopping callback
    # you may want a fresh copy of env for evaluation to avoid interfering
    eval_env = MesaEnv(
        env_data=aggregated_data,
        max_steps=10,
        pecs_params=pecs_params,
        width=width,
        height=height,
        scenario_params=scenario_params,
        climate_losses=climate_losses
    )
    early_stop_cb = EarlyStoppingCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        patience=patience,
        min_delta=min_delta,
        verbose=1
    )
    # optional: keep your PVAdoptionCallback as well
    from full2_RL_future import PVAdoptionCallback
    callbacks = [PVAdoptionCallback(), early_stop_cb]

    ppo_model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )
    env.close()
    end_time = time.time()


def train_and_simulate(config):
    """
    Extract parameters from the configuration and run both training and simulation,
    with early stopping.
    """
    # 1) unpack config
    model_params = config.get("model_parameters", {})
    period = model_params.get("period", "future")
    if period.lower() != "future":
        logging.error("This script is configured for future scenarios only.")
        return

    scenario = model_params.get("scenario", 26)
    pecs_params = {
        "physis": model_params.get("physis", DEFAULT_PECS_PARAMS["physis"]),
        "emotion": model_params.get("emotion", DEFAULT_PECS_PARAMS["emotion"]),
        "cognition": model_params.get("cognition", DEFAULT_PECS_PARAMS["cognition"]),
        "social": model_params.get("social", DEFAULT_PECS_PARAMS["social"])
    }
    scenario_params = model_params.get("scenario_params", DEFAULT_SCENARIO_PARAMS)
    climate_losses = model_params.get("climate_losses", {})

    # validate PECS
    for cat, params in pecs_params.items():
        for key, value in params.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"PECS parameter '{cat}.{key}' must be between 0 and 1. Received: {value}")

    # prepare output dirs
    task_id = config.get("task_id", "default_task")
    user_id = config.get("user_id", "default_user")
    results_task_dir = os.path.join(RESULTS_DIR, f"{task_id}_{user_id}")
    os.makedirs(results_task_dir, exist_ok=True)
    scenario_name = f"RCP{scenario}"
    scenario_output_dir = os.path.join(results_task_dir, scenario_name)
    os.makedirs(scenario_output_dir, exist_ok=True)

    # aggregate data
    aggregated_data = create_aggregated_data(scenario, MAIN_DIR, scenario_output_dir)
    if aggregated_data is None:
        logging.error("Data aggregation failed. Exiting.")
        return

    # build env
    width, height = 10, 10
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
        logging.info("Environment passed the Gym API check.")
    except Exception as e:
        logging.error(f"Environment check failed: {e}")
        return

    # training params
    learning_rate = 0.0009809274356741915
    batch_size = 64
    n_epochs = 6
    total_timesteps = 5000
    eval_freq = 1000
    patience = 5
    min_delta = 1e-2

    # instantiate model
    ppo_model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )

    logging.info(f"Starting training for RCP{scenario} with early stopping.")
    start_time = time.time()
    env.reset()

    # prepare a fresh eval_env
    eval_env = MesaEnv(
        env_data=aggregated_data,
        max_steps=10,
        pecs_params=pecs_params,
        width=width,
        height=height,
        scenario_params=scenario_params,
        climate_losses=climate_losses
    )
    early_stop_cb = EarlyStoppingCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        patience=patience,
        min_delta=min_delta,
        verbose=1
    )

    # run training with both callbacks
    ppo_model.learn(
        total_timesteps=total_timesteps,
        callback=[PVAdoptionCallback(), early_stop_cb]
    )
    env.close()
    end_time = time.time()

    # log training stats
    training_time = end_time - start_time
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    gpu = "MPS" if torch.backends.mps.is_available() else "CPU"
    logging.info(f"Training done in {training_time:.1f}s | CPU {cpu}% | Mem {mem}% | Device {gpu}")

    # save model
    trained_dir = os.path.join(scenario_output_dir, "trained_models")
    os.makedirs(trained_dir, exist_ok=True)
    model_path = os.path.join(trained_dir, f"RCP{scenario}_ppo_mesa_model.zip")
    ppo_model.save(model_path)

    # reload for simulation
    ppo_model = PPO.load(model_path, env=env)

    # simulate and collect decisions
    all_final = []
    for ep in range(5):
        logging.info(f"Simulation episode {ep+1}")
        obs, _ = env.reset()
        done = truncated = False
        while not done and not truncated:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
        for agent in env.model.schedule.agents:
            all_final.append({
                "episode": ep+1,
                "agent_id": agent.unique_id,
                "lat": agent.lat,
                "lon": agent.lon,
                "decision": agent.decision
            })

    # save decisions
    if all_final:
        df = pd.DataFrame(all_final)
        out_csv = os.path.join(scenario_output_dir, "agent_data_over_time.csv")
        df.to_csv(out_csv, index=False)
        logging.info(f"Saved decisions to {out_csv}")
    else:
        logging.warning("No decisions collected.")

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s"
    )
    # Load configuration from JSON input file
    config = load_config(INPUT_JSON_PATH)
    train_and_simulate(config)

if __name__ == "__main__":
    main()
