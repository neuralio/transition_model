"""
Unified RL Pipeline Script (No Plotting, Fixed JSON Input, Custom Save/Load Directory)

This script integrates configuration, data import, Mesa agent/model definitions,
the custom Gym environment, and training/evaluation routines. Model parameters (such as
"scenario", "period", "task_id", and "user_id") are read from a fixed JSON file.
The PPO model and agent data are saved in a directory with the format task_id_user_id
inside /home/ESA-AGENTS/BACKEND_API/Demo_With_Geoserver/RESULTS_RL/BASE_RL.
"""

import os
import time
import json
import argparse
import psutil
import torch
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import SingleGrid

#################################
# Configurations
#################################

# MAIN_DIR is for input data.
# MAIN_DIR = "/home/ESA-AGENTS/BACKEND_API/Demo_With_Geoserver/INPUT_RL"
MAIN_DIR = "/app/INPUT_RL"
# INPUT_JSON_PATH = os.path.join(MAIN_DIR, "input_mongo_base_future.json")
INPUT_JSON_PATH = "/app/input_mongo_base_future.json"

# RESULTS_DIR should be at: /home/ESA-AGENTS/BACKEND_API/Demo_With_Geoserver/RESULTS_RL/BASE_RL
BASE_DIR = os.path.dirname(MAIN_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "RESULTS_RL", "BASE_RL")
os.makedirs(RESULTS_DIR, exist_ok=True)

WHEAT_PAST = os.path.join(MAIN_DIR, "WHEAT_PAST")
WHEAT_FUTURE = os.path.join(MAIN_DIR, "WHEAT_FUTURE")
MAIZE_PAST = os.path.join(MAIN_DIR, "MAIZE_PAST")
MAIZE_FUTURE = os.path.join(MAIN_DIR, "MAIZE_FUTURE")
PV_PAST = os.path.join(MAIN_DIR, "PV_PAST")
PV_FUTURE = os.path.join(MAIN_DIR, "PV_FUTURE")

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
            "pv_yield_file": os.path.join(PV_FUTURE, f"RCP{scenario}_PV_YIELD.csv"),
            "cs_maize_file": os.path.join(MAIZE_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv"),
            "cs_wheat_file": os.path.join(WHEAT_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv"),
            "cy_maize_file": os.path.join(MAIZE_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv"),
            "cy_wheat_file": os.path.join(MAIZE_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv"),
        }

#################################
# Data Import Functions
#################################

def load_pv_data(pv_suitability_file, pv_yield_file):
    PV_ = pd.read_csv(pv_suitability_file, sep=',')
    PV_ = PV_.rename(columns={"score": "pv_suitability"})
    PV_ = PV_[(PV_["year"] >= 2021) & (PV_["year"] <= 2030)]
    print("Loaded PV Suitability")

    PV_Yield_ = pd.read_csv(pv_yield_file, sep=',')
    PV_Yield_ = PV_Yield_.rename(columns={'annual_electricity_savings_€': 'pv_profit'})
    print("Loaded PV Yield")

    return PV_, PV_Yield_

def load_crop_data(cs_maize_file, cs_wheat_file, cy_maize_file, cy_wheat_file):
    CS_Maize_ = pd.read_csv(cs_maize_file, sep=',')
    CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
    crop_suitability = pd.concat([CS_Maize_, CS_Wheat_], ignore_index=True)
    
    if "year" in crop_suitability.columns:
        crop_suitability = crop_suitability[(crop_suitability["year"] >= 2021) & (crop_suitability["year"] <= 2030)]
    crop_suitability = crop_suitability.rename(columns={"score": "crop_suitability"})
    print("Loaded Crop Suitability (Maize & Wheat)")

    CY_Maize = pd.read_csv(cy_maize_file, sep=',')[['Dates', 'Profits']].copy()
    CY_Maize['Dates'] = pd.to_datetime(CY_Maize['Dates'])
    CY_Maize['year'] = CY_Maize['Dates'].dt.year
    CY_Maize = CY_Maize.groupby('year').last().reset_index()
    CY_Maize = CY_Maize.rename(columns={'Profits': 'crop_profit'})
    print("Processed Crop Yield for Maize")

    CY_Wheat = pd.read_csv(cy_wheat_file, sep=',')[['Dates', 'Profits']].copy()
    CY_Wheat['Dates'] = pd.to_datetime(CY_Wheat['Dates'])
    CY_Wheat['year'] = CY_Wheat['Dates'].dt.year
    CY_Wheat = CY_Wheat.groupby('year').last().reset_index()
    CY_Wheat = CY_Wheat.rename(columns={'Profits': 'crop_profit'})
    print("Processed Crop Yield for Wheat")

    crop_profit = pd.concat([CY_Maize, CY_Wheat], ignore_index=True)
    return crop_suitability, crop_profit

def merge_data(crop_suitability, PV_, PV_Yield_, crop_profit):
    crop_profit = crop_suitability.merge(crop_profit, on="year", how="inner")[["year", "lat", "lon", "crop_profit"]]
    for df in [PV_, crop_suitability, PV_Yield_, crop_profit]:
        df["lat"] = df["lat"].astype(float)
        df["lon"] = df["lon"].astype(float)
    env_data = crop_suitability.merge(PV_, on=["year", "lat", "lon"], how="inner")
    env_data = env_data.merge(PV_Yield_, on=["year", "lat", "lon"], how="inner")
    env_data = env_data.merge(crop_profit, on=["year", "lat", "lon"], how="inner")
    print("Merged environmental dataset")
    return env_data

def aggregate_data(env_data):
    aggregated_data = env_data.groupby(['lat', 'lon'], as_index=False).agg({
        'crop_suitability': 'mean',
        'pv_suitability': 'mean',
        'crop_profit': 'sum',
        'pv_profit': 'sum'
    })
    unique_lat_lon_count = aggregated_data[["lat", "lon"]].drop_duplicates().shape[0]
    print(f"Number of unique lat/lon pairs: {unique_lat_lon_count}")
    return aggregated_data

#################################
# Agent and Model Definitions
#################################

class LandUseAgent(Agent):
    def __init__(self, unique_id, model, crop_suitability, pv_suitability, lat, lon):
        super().__init__(unique_id, model)
        self.crop_suitability = crop_suitability
        self.pv_suitability = pv_suitability
        self.lat = lat
        self.lon = lon
        self.decision = 0  # 0: Keep crops, 1: Convert to PV

    def step(self):
        if self.crop_suitability >= 85:
            self.decision = 0
        elif self.pv_suitability >= 85:
            self.decision = 1
        elif self.crop_suitability <= 60:
            self.decision = 1
        elif self.pv_suitability <= 60:
            self.decision = 0
        else:
            self.decision = 0

class LandUseModel(Model):
    def __init__(self, env_data):
        super().__init__()
        self.env_data = env_data
        self.schedule = BaseScheduler(self)
        self.current_step = 0

        for agent_id, row in self.env_data.iterrows():
            crop_suitability = row["crop_suitability"]
            pv_suitability = row["pv_suitability"]
            lat = row["lat"]
            lon = row["lon"]
            agent = LandUseAgent(agent_id, self, crop_suitability, pv_suitability, lat, lon)
            self.schedule.add(agent)

    def step(self):
        self.schedule.step()
        self.current_step += 1

    def get_agent_data(self):
        data = []
        for agent in self.schedule.agents:
            data.append({
                "lat": agent.lat,
                "lon": agent.lon,
                "crop_suitability": agent.crop_suitability,
                "pv_suitability": agent.pv_suitability,
                "decision": agent.decision
            })
        return pd.DataFrame(data)

#################################
# Gym Environment Definition
#################################

class MesaEnv(gym.Env):
    def __init__(self, env_data, max_steps):
        super(MesaEnv, self).__init__()
        self.env_data = env_data.reset_index(drop=True)
        self.max_steps = max_steps
        self.model = LandUseModel(self.env_data)

        suitability_vars = ["crop_suitability", "pv_suitability"]
        min_values = self.env_data[suitability_vars].min().values
        max_values = self.env_data[suitability_vars].max().values
        self.observation_space = spaces.Box(low=min_values, high=max_values, dtype=np.float32)

        num_agents = len(self.env_data)
        self.action_space = spaces.MultiDiscrete([2] * num_agents)

        self.current_step = 0
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.model = LandUseModel(self.env_data)
        self.state = self._get_current_state()
        return self.state, {}

    def step(self, action):
        for idx, agent in enumerate(self.model.schedule.agents):
            if action is not None:
                agent.decision = action[idx]

        self.model.step()
        profits = self.env_data.iloc[self.current_step][["pv_profit", "crop_profit"]]
        reward = self._calculate_reward(profits)
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        self.state = self._get_current_state() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def _get_current_state(self):
        suitability_vars = ["crop_suitability", "pv_suitability"]
        return self.env_data[suitability_vars].mean().values.astype(np.float32)

    def _calculate_reward(self, profits):
        total_reward = 0.0
        for agent in self.model.schedule.agents:
            if agent.decision == 0:
                total_reward += profits["crop_profit"]
            else:
                total_reward += profits["pv_profit"]
        return total_reward / len(self.model.schedule.agents)

    def render(self, mode="human"):
        agent_data = self.model.get_agent_data()
        print(agent_data)

    def close(self):
        pass

#################################
# RL Training and Evaluation Functions
#################################

def get_processing_stats():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    if torch.cuda.is_available():
        gpu_status = "Using CUDA"
    else:
        gpu_status = "CPU only"
    return cpu_usage, memory_usage, gpu_status

def train_rl_model(env, scenario, results_subdir, learning_rate=0.0009809274356741915, batch_size=64, n_epochs=6):
    print("Initializing PPO model...")
    # Instantiate PPO without specifying a device.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs
    )

    print("Starting training...")
    start_time = time.time()
    model.learn(total_timesteps=5000)
    end_time = time.time()

    # Save the model in the provided results subdirectory.
    save_path = os.path.join(results_subdir, f"RCP{scenario}_ppo_mesa_model")
    print(f"Saving model to {save_path}")
    model.save(save_path)

    training_time = end_time - start_time
    cpu_usage, memory_usage, gpu_status = get_processing_stats()

    print(f"Training Time: {training_time:.2f} seconds")
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_usage}%")
    print(f"GPU Status: {gpu_status}")


def test_check_env(scenario):
    file_paths = get_file_paths(scenario)
    PV_, PV_Yield_ = load_pv_data(file_paths["pv_suitability_file"], file_paths["pv_yield_file"])
    crop_suitability, crop_profit = load_crop_data(
        file_paths["cs_maize_file"],
        file_paths["cs_wheat_file"],
        file_paths["cy_maize_file"],
        file_paths["cy_wheat_file"]
    )
    env_data = merge_data(crop_suitability, PV_, PV_Yield_, crop_profit)
    aggregated_data = aggregate_data(env_data)
    env = MesaEnv(env_data=aggregated_data, max_steps=10)
    print("Checking environment compatibility with Gym API...")
    check_env(env, warn=True)
    print("Environment passed the check!")

def evaluate_model(scenario, env, results_subdir):
    # Use the provided results_subdir directly as the output folder.
    output_folder = results_subdir
    print(f"Using results subdirectory: {output_folder}")

    # Load the model from the results_subdir.
    load_path = os.path.join(results_subdir, f"RCP{scenario}_ppo_mesa_model")
    model = PPO.load(load_path, env=env)
    print(f"Loaded model from {load_path}")

    decisions_over_time = []
    rewards_over_time = []
    cumulative_rewards = []
    agent_data_over_time = []
    cumulative_reward = 0

    obs, _ = env.reset()
    for step in range(env.max_steps):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        agent_data = env.model.get_agent_data()
        agent_data["Timestep"] = step
        agent_data_over_time.append(agent_data)
        decisions_over_time.append(action)
        rewards_over_time.append(reward)
        cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)
        if terminated or truncated:
            break

    env.close()
    agent_data_df = pd.concat(agent_data_over_time, ignore_index=True)
    agent_data_csv = os.path.join(output_folder, "agent_data_over_time.csv")
    agent_data_df.to_csv(agent_data_csv, index=False)
    print(f"Saved agent data to '{agent_data_csv}'")

    np.save(os.path.join(output_folder, "decisions_over_time.npy"), decisions_over_time)
    np.save(os.path.join(output_folder, "rewards_over_time.npy"), rewards_over_time)
    np.save(os.path.join(output_folder, "cumulative_rewards.npy"), cumulative_rewards)

    print("Evaluation complete.")


#################################
# Main Function (CLI entry point)
#################################

def main():
    parser = argparse.ArgumentParser(
        description="Run the RL model using parameters from a fixed JSON input file."
    )
    parser.add_argument("--check-env", action="store_true", help="Check environment compatibility with Gym.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model and save agent data over time.")
    args = parser.parse_args()

    # Read model parameters from the fixed JSON file
    with open(INPUT_JSON_PATH, "r") as f:
        input_data = json.load(f)
    model_params = input_data.get("model_parameters", {})
    period = model_params.get("period", "future")
    scenario = model_params.get("scenario", None)
    if scenario is None:
        raise ValueError("The input JSON must contain a 'scenario' in model_parameters.")
    past_flag = (period.lower() == "past")
    print(f"Running with {period} data. Scenario: {scenario}")

    # Read additional identifiers from the JSON
    task_id = input_data.get("task_id")
    user_id = input_data.get("user_id")
    if not task_id or not user_id:
        raise ValueError("The input JSON must contain both 'task_id' and 'user_id'.")
    
    # Create a subdirectory with format task_id_user_id inside RESULTS_DIR
    results_subdir = os.path.join(RESULTS_DIR, f"{task_id}_{user_id}")
    os.makedirs(results_subdir, exist_ok=True)
    print(f"Results will be saved in: {results_subdir}")

    # Get file paths based on period and scenario
    if past_flag:
        file_paths = get_file_paths(scenario=None, past=True)
    else:
        file_paths = get_file_paths(scenario, past=False)

    PV_, PV_Yield_ = load_pv_data(file_paths["pv_suitability_file"], file_paths["pv_yield_file"])
    crop_suitability, crop_profit = load_crop_data(
        file_paths["cs_maize_file"],
        file_paths["cs_wheat_file"],
        file_paths["cy_maize_file"],
        file_paths["cy_wheat_file"]
    )
    env_data = merge_data(crop_suitability, PV_, PV_Yield_, crop_profit)
    aggregated_data = aggregate_data(env_data)
    env = MesaEnv(env_data=aggregated_data, max_steps=10)

    if args.check_env:
        print("Checking environment...")
        check_env(env, warn=True)

    if args.train:
        print("Training RL model...")
        train_rl_model(env, scenario if not past_flag else "past", results_subdir)

    if args.evaluate:
        print("Evaluating model...")
        evaluate_model(scenario if not past_flag else "past", env, results_subdir)

if __name__ == "__main__":
    main()
