"""
pecs_RL_future.py

This unified script combines the RL training and simulation execution for the PECS-based
Mesa environment. It is designed to be called by an API endpoint.

Directory structure:
  - MAIN_DIR = "/app/INPUT_RL"
  - INPUT_JSON_PATH = "/app/input_mongo_pecs_future.json"
  - BASE_DIR = os.path.dirname(MAIN_DIR)
  - RESULTS_DIR = os.path.join(BASE_DIR, "RESULTS_RL", "PECS_RL")

For future runs, the "scenario" and modified PECS parameters are read from the input JSON.
If any PECS parameter group is missing in the JSON, default values (same as used in training)
will be used.
"""

import os
import json
import time
import psutil
import torch
import random
import logging
import shutil
import argparse
import pandas as pd
import numpy as np

# Gym and Mesa imports
import gymnasium as gym
from gymnasium import spaces
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import MultiGrid
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# -----------------------------
# Directory definitions
# -----------------------------
MAIN_DIR = "/app/INPUT_RL"
INPUT_JSON_PATH = "/app/input_mongo_pecs_past.json"
BASE_DIR = os.path.dirname(MAIN_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "RESULTS_RL", "PECS_RL")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Default PECS Parameters (used for training)
# -----------------------------
DEFAULT_PEC_PARAMS = {
    "physis": {"health_status": 0.8, "labor_availability": 0.6},
    "emotion": {"stress_level": 0.3, "satisfaction": 0.7},
    "cognition": {"policy_incentives": 0.5, "information_access": 0.6},
    "social": {"social_influence": 0.4, "community_participation": 0.5}
}

# -----------------------------
# Mesa Environment Classes
# -----------------------------
class LandUseAgent(Agent):
    def __init__(self, unique_id, model, crop_suitability, pv_suitability, crop_profit, pv_profit, lat, lon, pecs_params):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.crop_suitability = crop_suitability
        self.pv_suitability = pv_suitability
        self.crop_profit = crop_profit
        self.pv_profit = pv_profit
        self.lat = lat
        self.lon = lon
        self.decision = 0  # 0: Crop, 1: PV

        # Initialize PECS attributes (make a copy so each agent is independent)
        self.physis = pecs_params["physis"].copy()
        self.emotion = pecs_params["emotion"].copy()
        self.cognition = pecs_params["cognition"].copy()
        self.social = pecs_params["social"].copy()

    def step(self):
        self.update_physis()
        self.update_emotion()
        self.update_cognition()
        self.update_social()
        self.make_decision()

    def update_physis(self):
        if self.decision == 0:  # Farming
            self.physis["health_status"] -= 0.01
            self.physis["labor_availability"] -= 0.01
        elif self.decision == 1:  # PV installation
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
        if neighbors:
            avg_neighbor_decision = sum([neighbor.decision for neighbor in neighbors]) / len(neighbors)
        else:
            avg_neighbor_decision = 0
        if self.decision == 0:
            social_influence = 1 - avg_neighbor_decision
        elif self.decision == 1:
            social_influence = avg_neighbor_decision
        else:
            social_influence = 0
        social_influence_weight = self.cognition["policy_incentives"]
        self.social["social_influence"] = social_influence * social_influence_weight

    def make_decision(self):
        if self.crop_profit > self.pv_profit and self.physis["health_status"] > 0.5:
            self.decision = 0
        elif self.pv_profit > self.crop_profit and self.cognition["policy_incentives"] > 0.6:
            self.decision = 1
        elif self.emotion["stress_level"] > 0.7:
            self.decision = 0
        elif self.social["social_influence"] > 0.5:
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
        }

class LandUseModel(Model):
    def __init__(self, env_data, width, height, pecs_params, policy_incentive_active=False, seed=None):
        super().__init__()
        self.env_data = env_data
        self.schedule = BaseScheduler(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.current_step = 0
        self.policy_incentive_active = policy_incentive_active

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
                pecs_params=pecs_params
            )
            self.schedule.add(agent)
            # Normalize lat/lon to grid coordinates
            x = int((lat - self.env_data["lat"].min()) / (self.env_data["lat"].max() - self.env_data["lat"].min()) * (width - 1))
            y = int((lon - self.env_data["lon"].min()) / (self.env_data["lon"].max() - self.env_data["lon"].min()) * (height - 1))
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            self.grid.place_agent(agent, (x, y))

    def step(self):
        if self.current_step % 10 == 0:
            self.policy_incentive_active = not self.policy_incentive_active
            for agent in self.schedule.agents:
                delta = 0.05 if self.policy_incentive_active else -0.05
                agent.cognition["policy_incentives"] = min(max(agent.cognition["policy_incentives"] + delta, 0.0), 1.0)
        self.schedule.step()
        self.current_step += 1

    def collect_agent_data(self):
        agent_data = []
        for agent in self.schedule.agents:
            agent_data.append({
                "Agent ID": agent.unique_id,
                "lat": agent.lat,
                "lon": agent.lon,
                "crop_suitability": agent.crop_suitability,
                "pv_suitability": agent.pv_suitability,
                "crop_profit": agent.crop_profit,
                "pv_profit": agent.pv_profit,
                "decision": agent.decision,
                "health_status": agent.physis["health_status"],
                "stress_level": agent.emotion["stress_level"],
                "policy_incentives": agent.cognition["policy_incentives"],
                "social_influence": agent.social["social_influence"]
            })
        return agent_data

    def save_agent_data(self, filename="agent_data.csv"):
        df = pd.DataFrame(self.collect_agent_data())
        df.to_csv(filename, index=False)

class MesaEnv(gym.Env):
    def __init__(self, env_data, max_steps, pecs_params, width=5, height=5):
        super(MesaEnv, self).__init__()
        self.env_data = env_data.reset_index(drop=True)
        self.max_steps = max_steps
        self.width = width
        self.height = height
        self.pecs_params = pecs_params

        self.model = LandUseModel(
            self.env_data,
            width=self.width,
            height=self.height,
            pecs_params=self.pecs_params
        )

        suitability_vars = ["crop_suitability", "pv_suitability"]
        pecs_vars = ["health_status", "stress_level", "policy_incentives", "social_influence"]
        min_values = np.concatenate([
            self.env_data[suitability_vars].min().values,
            np.array([0.0] * len(pecs_vars))
        ])
        max_values = np.concatenate([
            self.env_data[suitability_vars].max().values,
            np.array([1.0] * len(pecs_vars))
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
            seed=seed
        )
        self.state = self._get_current_state()
        if self.state.shape != self.observation_space.shape:
            raise ValueError(f"Expected {self.observation_space.shape}, got {self.state.shape}")
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
        return self.state, reward, terminated, False, {}

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
                    "decision": agent.decision
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
        }
        return np.array(list(aggregated_state.values()), dtype=np.float32)

    def _calculate_reward(self):
        total_reward = 0.0
        for agent in self.model.schedule.agents:
            neighbors = self.model.grid.get_neighbors(agent.pos, moore=True, include_center=False)
            if neighbors:
                avg_neighbor_decision = sum([neighbor.decision for neighbor in neighbors]) / len(neighbors)
            else:
                avg_neighbor_decision = 0
            if agent.decision == 0:
                social_influence = 1 - avg_neighbor_decision
                total_reward += (
                    agent.crop_profit * 0.4 +
                    agent.emotion["satisfaction"] * 0.2 -
                    agent.emotion["stress_level"] * 0.2 +
                    social_influence * agent.cognition["policy_incentives"]
                )
            elif agent.decision == 1:
                social_influence = avg_neighbor_decision
                total_reward += (
                    agent.pv_profit * 0.4 +
                    agent.cognition["policy_incentives"] * 0.2 -
                    agent.physis["labor_availability"] * 0.2 +
                    social_influence * agent.cognition["policy_incentives"]
                )
        return total_reward / len(self.model.schedule.agents)

    def close(self):
        pass

# -----------------------------
# Utility Functions
# -----------------------------
def load_input_json(json_path):
    """Load the input JSON parameters."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def get_pecs_params(input_params):
    """
    Extract the PECS parameters from the input JSON.
    If any of the groups is missing, use the defaults.
    """
    model_params = input_params.get("model_parameters", {})
    pecs = {}
    for key in ["physis", "emotion", "cognition", "social"]:
        pecs[key] = model_params.get(key, DEFAULT_PEC_PARAMS[key])
    return pecs

def load_and_process_data(scenario=None):
    """
    Load and aggregate environmental data based on period (past or future).
    """
    if scenario is None or scenario == "past":
        # Handle past data scenario
        pv_suitability_file = os.path.join(MAIN_DIR, "PV_PAST", "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
        pv_yield_file = os.path.join(MAIN_DIR, "PV_PAST", "PAST_PV_YIELD.csv")
        cs_wheat_file = os.path.join(MAIN_DIR, "WHEAT_PAST", "PAST_LUSA_PREDICTIONS.csv")
        cy_wheat_file = os.path.join(MAIN_DIR, "WHEAT_PAST", "AquaCrop_Results_PAST.csv")
        scenario_str = "PAST"
        logging.info("Loading past data files.")
    else:
        # Future scenario handling remains unchanged
        scenario_str = f"RCP{scenario}"
        pv_suitability_file = os.path.join(MAIN_DIR, "PV_FUTURE", f"{scenario_str}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
        pv_yield_file = os.path.join(MAIN_DIR, "PV_FUTURE", f"{scenario_str}_PV_YIELD.csv")
        cs_wheat_file = os.path.join(MAIN_DIR, "WHEAT_FUTURE", f"{scenario_str}_LUSA_PREDICTIONS.csv")
        cy_wheat_file = os.path.join(MAIN_DIR, "WHEAT_FUTURE", f"AquaCrop_Results_{scenario_str}.csv")
        logging.info(f"Loading future data files for scenario {scenario_str}.")

    PV_ = pd.read_csv(pv_suitability_file, sep=',')
    PV_.rename(columns={"score": "pv_suitability"}, inplace=True)
    logging.info(f"Loaded PV Suitability for {scenario_str}")

    CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
    CS_Wheat_.rename(columns={"score": "crop_suitability"}, inplace=True)
    logging.info(f"Loaded Crop Suitability for Wheat for {scenario_str}")

    PV_Yield_ = pd.read_csv(pv_yield_file, sep=',')
    PV_Yield_.rename(columns={'annual_electricity_savings_€': 'pv_profit'}, inplace=True)
    logging.info(f"Loaded PV Yield for {scenario_str}")

    CY_Wheat_FULL = pd.read_csv(cy_wheat_file, sep=',')
    CY_Wheat = CY_Wheat_FULL[['Dates', 'Profits']].copy()
    CY_Wheat['Dates'] = pd.to_datetime(CY_Wheat['Dates'])
    CY_Wheat['year'] = CY_Wheat['Dates'].dt.year
    CY_Wheat = CY_Wheat.groupby('year').last().reset_index()
    CY_Wheat.rename(columns={'Profits': 'crop_profit'}, inplace=True)
    logging.info(f"Processed Crop Yield for Wheat for {scenario_str}")

    crop_profit = CS_Wheat_.merge(CY_Wheat, on="year", how="inner")[["year", "lat", "lon", "crop_profit"]]
    for df in [PV_, CS_Wheat_, PV_Yield_, crop_profit]:
        df["lat"] = df["lat"].astype(float).round(5)
        df["lon"] = df["lon"].astype(float).round(5)

    env_data = CS_Wheat_.merge(PV_, on=["year", "lat", "lon"], how="inner")
    env_data = env_data.merge(PV_Yield_, on=["year", "lat", "lon"], how="inner")
    env_data = env_data.merge(crop_profit, on=["year", "lat", "lon"], how="inner")

    aggregated_data = env_data.groupby(['lat', 'lon'], as_index=False).agg({
        'crop_suitability': 'mean',
        'pv_suitability': 'mean',
        'crop_profit': 'sum',
        'pv_profit': 'sum'
    })

    logging.info(f"Aggregated data contains {aggregated_data.shape[0]} unique lat/lon pairs.")
    return aggregated_data


def get_processing_stats():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    if torch.cuda.is_available():
        gpu_status = "Using CUDA"
    else:
        gpu_status = "CPU only"
    return cpu_usage, memory_usage, gpu_status

def train_rl_model(env, scenario_dir, scenario_str, learning_rate=0.00098, batch_size=64, n_epochs=6, total_timesteps=5000):
    """Train a PPO model on the provided environment and save the model."""
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    end_time = time.time()
    model_path = os.path.join(scenario_dir, "past_ppo_mesa_model")
    model.save(model_path)
    training_time = end_time - start_time
    cpu_usage, memory_usage, gpu_status = get_processing_stats()
    logging.info(f"Training Time: {training_time:.2f} seconds")
    logging.info(f"CPU Usage: {cpu_usage}% | Memory Usage: {memory_usage}% | GPU Status: {gpu_status}")
    print(f"Training completed in {training_time:.2f} seconds")
    return model

# def run_simulation(env, model, scenario_dir, num_episodes=5):
#     """Run simulation episodes using the trained model and save final agent decisions."""
#     all_final_decisions = []
#     for episode in range(num_episodes):
#         obs, _ = env.reset()
#         done = False
#         total_reward, step = 0.0, 0
#         while not done:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, done, _, _ = env.step(action)
#             total_reward += reward
#             step += 1
#         episode_decisions = [
#             {'episode': episode + 1, 'agent_id': agent.unique_id, 'decision': agent.decision}
#             for agent in env.model.schedule.agents
#         ]
#         all_final_decisions.extend(episode_decisions)
#         logging.info(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Steps = {step}")
#         print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Steps = {step}")
    
#     decisions_df = pd.DataFrame(all_final_decisions)
#     decisions_file = os.path.join(scenario_dir, "agent_data_over_time.csv")
#     decisions_df.to_csv(decisions_file, index=False)
#     logging.info(f"Final agent decisions saved to {decisions_file}")
def run_simulation(env, model, scenario_dir, num_episodes=5):
    all_agent_data = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward, step = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1

            # Collect data at each timestep
            timestep_data = env.model.collect_agent_data()
            for agent_entry in timestep_data:
                agent_entry["Step"] = step
                agent_entry["Episode"] = episode + 1
                all_agent_data.append(agent_entry)
        
        logging.info(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Steps = {step}")
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Steps = {step}")

    # Save all collected data
    decisions_df = pd.DataFrame(all_agent_data)
    decisions_file = os.path.join(scenario_dir, "agent_data_over_time.csv")
    decisions_df.to_csv(decisions_file, index=False)
    logging.info(f"Saved agent data over time to {decisions_file}")
# -----------------------------
# Main Execution
# -----------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    input_params = load_input_json(INPUT_JSON_PATH)
    model_params = input_params.get("model_parameters", {})

    if model_params.get("period", "").lower() != "past":
        logging.error("This script is intended for past scenario runs only.")
        return

    scenario_str = "past"

    
    # Create a subdirectory with the format task_id_user_id
    task_id = input_params.get("task_id")
    user_id = input_params.get("user_id")
    if not task_id or not user_id:
        logging.error("Task ID or User ID not found in the input JSON.")
        return
    result_subdir = os.path.join(RESULTS_DIR, f"{task_id}_{user_id}")
    os.makedirs(result_subdir, exist_ok=True)
    
    # Save all results in the subdirectory (no separate scenario folder)
    scenario_dir = result_subdir
    
    pecs_params = get_pecs_params(input_params)
    for group, params in pecs_params.items():
        for key, value in params.items():
            if not (0.0 <= value <= 1.0):
                logging.error(f"PECS parameter '{group}.{key}' must be between 0 and 1. Got {value}")
                return
    
    logging.info(f"Running simulation for scenario {scenario_str} with PECS parameters: {pecs_params}")
    
    aggregated_data = load_and_process_data()
    if aggregated_data is None or aggregated_data.empty:
        logging.error("Environmental data could not be loaded or is empty.")
        return
    
    width, height, max_steps = 10, 10, 10
    env = MesaEnv(
        env_data=aggregated_data,
        max_steps=max_steps,
        pecs_params=pecs_params,
        width=width,
        height=height
    )
    
    try:
        check_env(env)
        logging.info("Environment passed the Gym API check.")
    except Exception as e:
        logging.error(f"Environment check failed: {e}")
        return
    
    train_model = True
    if train_model:
        model = train_rl_model(env, scenario_dir, scenario_str)
    else:
        model_path = os.path.join(scenario_dir, "past_ppo_mesa_model.zip")
        try:
            model = PPO.load(model_path, env=env)
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            return

    run_simulation(env, model, scenario_dir, num_episodes=5)
    logging.info("Simulation run completed.")

if __name__ == "__main__":
    main()
