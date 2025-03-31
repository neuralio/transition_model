#!/usr/bin/env python
"""
full2_RL_past.py

Description:
  This script trains a PPO agent in a custom Mesa environment and then runs a simulation for past data.
  It reads configuration (modified PECS parameters, scenario parameters, initial subsidy fund, and climate losses)
  from the JSON file '/app/input_mongo_full2_future.json'. (Make sure that the JSON is configured for past data.)
  The simulation aggregates environmental data based on the specified past scenario (here the scenario name is fixed as "PAST"),
  trains the PPO model, and then runs several evaluation episodes to collect agent decisions.
  
Directory structure:
  MAIN_DIR = "/app"
  INPUT_JSON_PATH = "/app/input_mongo_full2_future.json"
  RESULTS_DIR = os.path.join(MAIN_DIR, "RESULTS_RL", "FULL2_RL")
  A directory named "task_id_user_id" will be created under RESULTS_DIR based on the input JSON.
  
Usage:
  This script is designed to be called either from the command line or by an API endpoint.
  
Note:
  The model classes (LandUseAgent, LandUseModel, and MesaEnv) are embedded below so that no external module import is required.
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

# ---------------------------
# Embedded Model Code
# ---------------------------
# BEGIN: LandUseAgent
from mesa import Agent

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
        """Update the agent's state and make decisions based on PECS attributes."""
        current_year = self.model.current_step + 2021
        self.current_loss_factor = self.climate_losses.get(current_year, 0.0)
        self.crop_profit *= (1 - self.current_loss_factor)
        self.update_physis()
        self.update_emotion()
        self.update_cognition()
        self.update_social()
        self.make_decision()

    def update_physis(self):
        if self.decision == 0:
            self.physis["health_status"] -= 0.01
            self.physis["labor_availability"] -= 0.01
        elif self.decision == 1:
            self.physis["labor_availability"] -= 0.02

    def update_emotion(self):
        if self.decision == 0:
            self.emotion["stress_level"] += 0.01
            self.emotion["satisfaction"] += 0.02
        elif self.decision == 1:
            self.emotion["stress_level"] -= 0.02
            self.emotion["satisfaction"] += 0.03

    def update_cognition(self):
        if self.model.policy_incentive_active:
            self.cognition["policy_incentives"] += 0.05
        else:
            self.cognition["policy_incentives"] -= 0.01

    def update_social(self):
        if not hasattr(self, "pos") or self.pos is None:
            self.social["social_influence"] = 0
            return
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        avg_neighbor_decision = sum([neighbor.decision for neighbor in neighbors]) / len(neighbors) if neighbors else 0
        if self.decision == 0:
            social_influence = 1 - avg_neighbor_decision
        else:
            social_influence = avg_neighbor_decision
        social_influence_weight = self.cognition["policy_incentives"]
        self.social["social_influence"] = social_influence * social_influence_weight

    def make_decision(self):
        crop_profit = self.crop_profit
        pv_profit = self.pv_profit
        pv_profit_with_subsidy = pv_profit + self.subsidy - (pv_profit * self.loan_interest_rate)
        pv_profit_with_tax = pv_profit_with_subsidy + (pv_profit * self.tax_incentive_rate)
        if self.current_loss_factor > 0.3:
            self.decision = 1
        elif crop_profit > pv_profit_with_tax and self.physis["health_status"] > 0.5:
            self.decision = 0
        elif pv_profit_with_tax > crop_profit and self.cognition["policy_incentives"] > 0.6:
            self.decision = 1
        elif self.emotion["stress_level"] > 0.7:
            self.decision = 0
        elif self.social["social_influence"] > 0.5 or self.social["social_influence"] * self.social_prestige_weight > 0.6:
            self.decision = 1
        else:
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
            "Subsidy": self.subsidy,
            "Loan Interest Rate": self.loan_interest_rate,
            "Tax Incentive Rate": self.tax_incentive_rate,
            "Social Prestige Weight": self.social_prestige_weight,
            "Climate Loss Factor": self.current_loss_factor
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
        self.scenario_params = scenario_params or {}
        self.climate_losses = climate_losses or {}
        self.base_subsidy = self.scenario_params.get("subsidy", 0.0)
        self.loan_interest_rate = self.scenario_params.get("loan_interest_rate", 0.05)
        self.tax_incentive_rate = self.scenario_params.get("tax_incentive_rate", 0.0)
        self.social_prestige_weight = self.scenario_params.get("social_prestige_weight", 0.0)
        self.initial_fund = self.scenario_params.get("initial_subsidy_fund", 1.0)
        self.remaining_fund = self.initial_fund
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        for agent_id, row in self.env_data.iterrows():
            crop_suitability = row["crop_suitability"]
            pv_suitability = row["pv_suitability"]
            pv_profit = row["pv_profit"]
            crop_profit = row["crop_profit"]
            lat = row["lat"]
            lon = row["lon"]
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
            x = int((lat - self.env_data["lat"].min()) / (self.env_data["lat"].max() - self.env_data["lat"].min()) * (width - 1))
            y = int((lon - self.env_data["lon"].min()) / (self.env_data["lon"].max() - self.env_data["lon"].min()) * (height - 1))
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            self.grid.place_agent(agent, (x, y))
            
    def step(self):
        current_year = self.current_step + 2021
        if current_year in self.climate_losses:
            loss_factor = self.climate_losses[current_year]
            for agent in self.schedule.agents:
                agent.crop_profit *= (1 - loss_factor)
                print(f"Agent {agent.unique_id}: Climate loss applied for {current_year}, new crop profit: {agent.crop_profit:.2f}")
        if self.current_step % 10 == 0:
            self.policy_incentive_active = not self.policy_incentive_active
            for agent in self.schedule.agents:
                agent.cognition["policy_incentives"] += 0.05 if self.policy_incentive_active else -0.05
                agent.cognition["policy_incentives"] = min(max(agent.cognition["policy_incentives"], 0.0), 1.0)
        self.schedule.step()
        self.current_step += 1
        ALPHA = 0.05
        num_PV_adopters = sum(1 for agent in self.schedule.agents if agent.decision == 1)
        self.remaining_fund = max(0, self.remaining_fund - ALPHA * num_PV_adopters)
        effective_subsidy = self.base_subsidy * (self.remaining_fund / self.initial_fund)
        print(f"Remaining Fund: {self.remaining_fund:.3f} / {self.initial_fund}, Effective Subsidy: {effective_subsidy:.3f}")
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
        self.pecs_params = pecs_params
        self.scenario_params = scenario_params or {}
        self.climate_losses = climate_losses or {}
        self.model = LandUseModel(
            self.env_data,
            width=self.width,
            height=self.height,
            pecs_params=self.pecs_params,
            scenario_params=self.scenario_params,
            climate_losses=self.climate_losses
        )
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
MAIN_DIR = "/app"
INPUT_JSON_PATH = os.path.join(MAIN_DIR, "input_mongo_full2_past.json")
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
    Create aggregated environmental data for a past scenario.
    For past data, we expect the scenario to be "PAST" and read from the past data directories.
    """
    scenario_name = scenario  # For past data, scenario is "PAST"
    INPUT_RL_DIR = os.path.join(MAIN_DIR, "INPUT_RL")
    # Assume past data are stored in directories ending with _PAST.
    WHEAT_PAST = os.path.join(INPUT_RL_DIR, "WHEAT_PAST")
    MAIZE_PAST = os.path.join(INPUT_RL_DIR, "MAIZE_PAST")
    PV_PAST    = os.path.join(INPUT_RL_DIR, "PV_PAST")
    
    # Define file paths for past data.
    pv_suitability_file = os.path.join(PV_PAST, "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    pv_yield_file       = os.path.join(PV_PAST, "PAST_PV_YIELD.csv")
    cs_wheat_file       = os.path.join(WHEAT_PAST, "WHEAT_PAST_LUSA_PREDICTIONS.csv")
    cs_maize_file       = os.path.join(MAIZE_PAST, "MAIZE_PAST_LUSA_PREDICTIONS.csv")
    cy_wheat_file       = os.path.join(WHEAT_PAST, "AquaCrop_Results_PAST.csv")
    cy_maize_file       = os.path.join(MAIZE_PAST, "AquaCrop_Results_PAST.csv")
    
    # Read and filter the PV suitability data.
    try:
        PV_ = pd.read_csv(pv_suitability_file)
        PV_ = PV_.rename(columns={"score": "pv_suitability"})
        PV_ = PV_[(PV_["year"] >= 2021) & (PV_["year"] <= 2030)]
    except FileNotFoundError:
        print(f"Error: File '{pv_suitability_file}' not found.")
        return None

    # Read crop suitability data.
    try:
        CS_Wheat_ = pd.read_csv(cs_wheat_file)
        CS_Maize_ = pd.read_csv(cs_maize_file)
        crop_suitability = pd.concat([CS_Wheat_, CS_Maize_], ignore_index=True)
        crop_suitability = crop_suitability[(crop_suitability["year"] >= 2021) & (crop_suitability["year"] <= 2030)]
        crop_suitability = crop_suitability.rename(columns={"score": "crop_suitability"})
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # Read PV yield data.
    try:
        PV_Yield_ = pd.read_csv(pv_yield_file)
        PV_Yield_ = PV_Yield_.rename(columns={'annual_electricity_savings_€': 'pv_profit'})
    except FileNotFoundError:
        print(f"Error: File '{pv_yield_file}' not found.")
        return None

    # Process crop yield for Wheat.
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

    # Process crop yield for Maize.
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

    crop_profit = pd.concat([CY_Wheat, CY_Maize], ignore_index=True)
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
    def __init__(self, verbose=0):
        super(PVAdoptionCallback, self).__init__(verbose)
    def _on_step(self) -> bool:
        pv_adopters = sum([agent.decision for agent in self.training_env.envs[0].model.schedule.agents])
        total_agents = len(self.training_env.envs[0].model.schedule.agents)
        pv_adoption_rate = pv_adopters / total_agents
        logging.info(f"Step {self.num_timesteps}: PV Adoption Rate = {pv_adoption_rate:.2f}")
        return True

def get_processing_stats():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    gpu_status = "Using Apple's MPS GPU" if torch.backends.mps.is_available() else "GPU not available"
    return cpu_usage, memory_usage, gpu_status

# ---------------------------
# Past training and simulation function for Full2 RL Past
# ---------------------------
def train_and_simulate_past(config):
    """
    Extract parameters from the configuration and run both training and simulation for past data.
    """
    model_params = config.get("model_parameters", {})
    period = model_params.get("period", "past")
    if period.lower() != "past":
        logging.error("This script is configured for past data only.")
        return

    # For past data, we fix scenario as "PAST"
    scenario_name = "PAST"
    pecs_params = {
        "physis": model_params.get("physis", DEFAULT_PECS_PARAMS["physis"]),
        "emotion": model_params.get("emotion", DEFAULT_PECS_PARAMS["emotion"]),
        "cognition": model_params.get("cognition", DEFAULT_PECS_PARAMS["cognition"]),
        "social": model_params.get("social", DEFAULT_PECS_PARAMS["social"])
    }
    scenario_params = model_params.get("scenario_params", DEFAULT_SCENARIO_PARAMS)
    initial_subsidy_fund = model_params.get("initial_subsidy_fund", 1.0)
    climate_losses = model_params.get("climate_losses", {})

    for cat, params in pecs_params.items():
        for key, value in params.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"PECS parameter '{cat}.{key}' must be between 0 and 1. Received: {value}")

    task_id = config.get("task_id", "default_task")
    user_id = config.get("user_id", "default_user")
    results_task_dir = os.path.join(RESULTS_DIR, f"{task_id}_{user_id}")
    os.makedirs(results_task_dir, exist_ok=True)

    scenario_output_dir = os.path.join(results_task_dir, scenario_name)
    os.makedirs(scenario_output_dir, exist_ok=True)

    aggregated_data = create_aggregated_data(scenario_name, MAIN_DIR, scenario_output_dir)
    if aggregated_data is None:
        print("Data aggregation failed. Exiting.")
        return

    width = 10
    height = 10
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
        return

    learning_rate = 0.0009809274356741915
    batch_size = 64
    n_epochs = 6
    total_timesteps = 5000

    ppo_model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )

    logging.info(f"Starting training for {scenario_name} with PECS parameters: {pecs_params}")
    start_time = time.time()
    env.reset()
    ppo_model.learn(total_timesteps=total_timesteps, callback=PVAdoptionCallback())
    env.close()
    end_time = time.time()
    training_time = end_time - start_time
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    gpu_status = "Using Apple's MPS GPU" if torch.backends.mps.is_available() else "GPU not available"
    logging.info(f"Training Time: {training_time:.2f} seconds, CPU: {cpu_usage}%, Memory: {memory_usage}%, GPU: {gpu_status}")
    logging.info("Training completed successfully.")

    trained_models_dir = os.path.join(scenario_output_dir, "trained_models")
    os.makedirs(trained_models_dir, exist_ok=True)
    model_path = os.path.join(trained_models_dir, "PAST_ppo_mesa_model.zip")
    ppo_model.save(model_path)

    logging.info("Loading trained model from disk for simulation.")
    ppo_model = PPO.load(model_path, env=env)

    num_episodes = 5
    all_final_decisions = []
    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step = 0
        while not done and not truncated:
            action, _states = ppo_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
        print(f"Episode {episode + 1} completed with total reward: {total_reward}")
        for agent in env.model.schedule.agents:
            try:
                all_final_decisions.append({
                    'episode': episode + 1,
                    'agent_id': agent.unique_id,
                    'lat': agent.lat,
                    'lon': agent.lon,
                    'decision': getattr(agent, 'decision', None)
                })
            except AttributeError as e:
                logging.error(f"Agent {agent.unique_id} missing 'decision': {e}")
    if all_final_decisions:
        decisions_df = pd.DataFrame(all_final_decisions)
        decisions_output_file = os.path.join(scenario_output_dir, "agent_data_over_time.csv")
        decisions_df.to_csv(decisions_output_file, index=False)
        print(f"Final agent decisions saved to '{decisions_output_file}'.")
    else:
        print("Warning: No agent decisions were collected.")

# ---------------------------
# Main function for past data
# ---------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s"
    )
    config = load_config(INPUT_JSON_PATH)
    train_and_simulate_past(config)

if __name__ == "__main__":
    main()
