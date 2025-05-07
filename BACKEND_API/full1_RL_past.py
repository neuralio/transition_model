#!/usr/bin/env python
"""
full1_RL_past.py

Description:
    This script combines training and simulation runs for a reinforcement learning (RL) pipeline 
    using a Mesa-based environment for past data. It reads configuration (with period "past" and 
    other PECS parameters) from an input JSON file, aggregates environmental data from a fixed directory 
    structure for past data, and then either trains a PPO model (if --train_model is set) or runs simulation 
    episodes using a pre-trained model.
    
Directory structure:
    MAIN_DIR = "/app/INPUT_RL"
    INPUT_JSON_PATH = "/app/input_mongo_full1_past.json"
    BASE_DIR = os.path.dirname(MAIN_DIR)
    RESULTS_DIR = os.path.join(BASE_DIR, "RESULTS_RL", "FULL1_RL")
    
Usage:
    python full1_RL_past.py [--train_model] [--num_episodes NUM]
    
    --train_model   Flag to run training; if not set, the script loads a pre-trained model.
    --num_episodes  Number of simulation episodes to run (default: 5).

Note: This script is meant to be later called via an API endpoint.
"""

import os
import json
import argparse
import logging
import shutil
import time
import random
import psutil

import pandas as pd
import numpy as np
import torch

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement


from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import MultiGrid

# --------------------------
# Directory constants
# --------------------------
MAIN_DIR = "/app/INPUT_RL"
INPUT_JSON_PATH = "/app/input_mongo_full1_past.json"
BASE_DIR = os.path.dirname(MAIN_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "RESULTS_RL", "FULL1_RL")
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

# --------------------------
# Mesa Environment Classes
# --------------------------
class LandUseAgent(Agent):
    def __init__(self, unique_id, model, crop_suitability, pv_suitability, crop_profit, pv_profit, lat, lon, pecs_params, scenario_params):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.crop_suitability = crop_suitability
        self.pv_suitability = pv_suitability
        self.crop_profit = crop_profit
        self.pv_profit = pv_profit
        self.lat = lat
        self.lon = lon
        self.decision = 0  # 0: keep crops, 1: convert to PV

        # Initialize PECS attributes (each agent gets its own instance)
        self.physis = pecs_params["physis"].copy()
        self.emotion = pecs_params["emotion"].copy()
        self.cognition = pecs_params["cognition"].copy()
        self.social = pecs_params["social"].copy()

        # Scenario-specific parameters
        self.subsidy = scenario_params["subsidy"]
        self.loan_interest_rate = scenario_params["loan_interest_rate"]
        self.tax_incentive_rate = scenario_params["tax_incentive_rate"]
        self.social_prestige_weight = scenario_params["social_prestige_weight"]

    def step(self):
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
        self.cognition["policy_incentives"] = min(max(self.cognition["policy_incentives"], 0.0), 1.0)

    def update_social(self):
        if not hasattr(self, 'pos') or self.pos is None:
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
        crop_profit = self.crop_profit
        pv_profit = self.pv_profit
        pv_profit_with_subsidy = pv_profit + self.subsidy - (pv_profit * self.loan_interest_rate)
        pv_profit_with_tax = pv_profit_with_subsidy + (pv_profit * self.tax_incentive_rate)
        if crop_profit > pv_profit_with_tax and self.physis["health_status"] > 0.5:
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
            "lat": self.lat,
            "lon": self.lon,
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
        }

class LandUseModel(Model):
    def __init__(self, env_data, width, height, pecs_params, scenario_params, seed=None):
        super().__init__()
        self.env_data = env_data
        self.schedule = BaseScheduler(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.current_step = 0
        self.policy_incentive_active = False

        self.subsidy = scenario_params["subsidy"]
        self.loan_interest_rate = scenario_params["loan_interest_rate"]
        self.tax_incentive_rate = scenario_params["tax_incentive_rate"]
        self.social_prestige_weight = scenario_params["social_prestige_weight"]

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
                scenario_params=scenario_params
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
                agent.cognition["policy_incentives"] += 0.05 if self.policy_incentive_active else -0.05
                agent.cognition["policy_incentives"] = min(max(agent.cognition["policy_incentives"], 0.0), 1.0)
        self.schedule.step()
        self.current_step += 1

    def collect_agent_data(self):
        agent_data = []
        for agent in self.schedule.agents:
            agent_data.append(agent.get_state())
        return agent_data

    def save_agent_data(self, filename="agent_data_over_time.csv"):
        agent_data = self.collect_agent_data()
        df = pd.DataFrame(agent_data)
        df.to_csv(filename, index=False)

class MesaEnv(gym.Env):
    def __init__(self, env_data, max_steps, pecs_params, scenario_params, width=10, height=10):
        super(MesaEnv, self).__init__()
        self.env_data = env_data.reset_index(drop=True)
        self.max_steps = max_steps
        self.width = width
        self.height = height
        self.pecs_params = pecs_params
        self.scenario_params = scenario_params

        self.model = LandUseModel(
            self.env_data,
            width=self.width,
            height=self.height,
            pecs_params=self.pecs_params,
            scenario_params=self.scenario_params
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
        self.state = self._get_current_state()

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.model = LandUseModel(
            self.env_data,
            width=self.width,
            height=self.height,
            pecs_params=self.pecs_params,
            scenario_params=self.scenario_params,
            seed=seed
        )
        self.state = self._get_current_state()
        if self.state.shape != self.observation_space.shape:
            raise ValueError(f"Expected state shape {self.observation_space.shape}, got {self.state.shape}")
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

    def _get_current_state(self):
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
                total_reward += (agent.crop_profit * 0.4 +
                                 agent.emotion["satisfaction"] * 0.2 -
                                 agent.emotion["stress_level"] * 0.2 +
                                 social_influence * agent.cognition["policy_incentives"])
            elif agent.decision == 1:
                social_influence = avg_neighbor_decision
                total_reward += (agent.pv_profit * 0.4 +
                                 agent.cognition["policy_incentives"] * 0.2 -
                                 agent.physis["labor_availability"] * 0.2 +
                                 social_influence * agent.cognition["policy_incentives"])
        return total_reward / len(self.model.schedule.agents)

    def render(self, mode="human"):
        if mode == "human":
            print(self.model.collect_agent_data())

    def close(self):
        pass

# --------------------------
# Utility Functions
# --------------------------
def cleanup_files(log_file_path, output_dir_path):
    if os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
            logging.info(f"Deleted old log file: {log_file_path}")
        except Exception as e:
            logging.error(f"Error deleting log file '{log_file_path}': {e}")
    if os.path.exists(output_dir_path):
        try:
            shutil.rmtree(output_dir_path)
            logging.info(f"Deleted old output directory: {output_dir_path}")
            os.makedirs(output_dir_path, exist_ok=True)
            logging.info(f"Recreated output directory: {output_dir_path}")
        except Exception as e:
            logging.error(f"Error deleting output directory '{output_dir_path}': {e}")
    else:
        os.makedirs(output_dir_path, exist_ok=True)
        logging.info(f"Created output directory: {output_dir_path}")

def create_aggregated_data(scenario, main_dir, output_dir):
    # For past data, use the PAST directories.
    logging.info("Loading past data files.")
    pv_suitability_file = os.path.join(MAIN_DIR, "PV_PAST", "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    pv_yield_file       = os.path.join(MAIN_DIR, "PV_PAST", "PAST_PV_YIELD.csv")
    cs_wheat_file       = os.path.join(MAIN_DIR, "WHEAT_PAST", "PAST_LUSA_PREDICTIONS.csv")
    cy_wheat_file       = os.path.join(MAIN_DIR, "WHEAT_PAST", "AquaCrop_Results_PAST.csv")

    try:
        PV_ = pd.read_csv(pv_suitability_file, sep=',')
        PV_ = PV_.rename(columns={"score": "pv_suitability"})
        PV_ = PV_[(PV_["year"] >= 2021) & (PV_["year"] <= 2030)]
        logging.info("Loaded PV Suitability for PAST data.")
    except Exception as e:
        logging.error(f"Error loading PV suitability: {e}")
        return None

    try:
        CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
        CS_Wheat_ = CS_Wheat_[(CS_Wheat_["year"] >= 2021) & (CS_Wheat_["year"] <= 2030)]
        CS_Wheat_ = CS_Wheat_.rename(columns={"score": "crop_suitability"})
        logging.info("Loaded Crop Suitability for Wheat (PAST).")
    except Exception as e:
        logging.error(f"Error loading crop suitability: {e}")
        return None

    try:
        PV_Yield_ = pd.read_csv(pv_yield_file, sep=',')
        PV_Yield_ = PV_Yield_.rename(columns={'annual_electricity_savings_€': 'pv_profit'})
        logging.info("Loaded PV Yield for PAST data.")
    except Exception as e:
        logging.error(f"Error loading PV yield: {e}")
        return None

    try:
        CY_Wheat_FULL = pd.read_csv(cy_wheat_file, sep=',')
        CY_Wheat = CY_Wheat_FULL[['Dates', 'Profits']].copy()
        CY_Wheat['Dates'] = pd.to_datetime(CY_Wheat['Dates'])
        CY_Wheat['year'] = CY_Wheat['Dates'].dt.year
        CY_Wheat = CY_Wheat.groupby('year').last().reset_index()
        CY_Wheat = CY_Wheat.rename(columns={'Profits': 'crop_profit'})
        logging.info("Processed Crop Yield for Wheat (PAST).")
    except Exception as e:
        logging.error(f"Error processing crop yield: {e}")
        return None

    try:
        crop_profit = CS_Wheat_.merge(CY_Wheat, on="year", how="inner")[["year", "lat", "lon", "crop_profit"]]
    except Exception as e:
        logging.error(f"Error merging crop profit: {e}")
        return None

    for df in [PV_, CS_Wheat_, PV_Yield_, crop_profit]:
        df["lat"] = df["lat"].astype(float).round(5)
        df["lon"] = df["lon"].astype(float).round(5)

    try:
        env_data = CS_Wheat_.merge(PV_, on=["year", "lat", "lon"], how="inner")
        env_data = env_data.merge(PV_Yield_, on=["year", "lat", "lon"], how="inner")
        env_data = env_data.merge(crop_profit, on=["year", "lat", "lon"], how="inner")
        logging.info("Merged environmental dataset for PAST data successfully.")
    except Exception as e:
        logging.error(f"Error merging datasets: {e}")
        return None

    aggregated_data = env_data.groupby(['lat', 'lon'], as_index=False).agg({
        'crop_suitability': 'mean',
        'pv_suitability': 'mean',
        'crop_profit': 'sum',
        'pv_profit': 'sum'
    })

    logging.info(f"Number of unique lat/lon pairs: {aggregated_data.shape[0]}")
    output_file = os.path.join(output_dir, f"aggregated_data_PAST.csv")
    aggregated_data.to_csv(output_file, index=False)
    logging.info(f"Aggregated data saved to '{output_file}'.")
    return aggregated_data

def load_input_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error reading input JSON: {e}")
        return None

# --------------------------
# Training and Simulation Functions
# --------------------------
# def train_rl_phase(env, result_subdir, scenario_str, pecs_params, scenario_params,
#                    total_timesteps=5000, learning_rate=0.0009809274356741915, batch_size=64, n_epochs=6):
#     logging.info(f"Starting training for {scenario_str} data with PECS: {pecs_params} and scenario params: {scenario_params}")
#     ppo_model = PPO(
#         'MlpPolicy',
#         env,
#         verbose=1,
#         learning_rate=learning_rate,
#         batch_size=batch_size,
#         n_epochs=n_epochs,
#         device="mps" if torch.backends.mps.is_available() else "cpu"
#     )
#     start_time = time.time()
#     ppo_model.learn(total_timesteps=total_timesteps)
#     end_time = time.time()
#     model_path = os.path.join(result_subdir, "past_ppo_mesa_model.zip")
#     ppo_model.save(model_path)
#     training_time = end_time - start_time
#     cpu_usage = psutil.cpu_percent(interval=1)
#     memory_usage = psutil.virtual_memory().percent
#     gpu_status = "Using Apple's MPS GPU" if torch.backends.mps.is_available() else "GPU not available"
#     logging.info(f"Training Time: {training_time:.2f} seconds, CPU: {cpu_usage}%, Memory: {memory_usage}%, GPU: {gpu_status}")
#     logging.info("Training completed successfully.")
#     return ppo_model, model_path
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv

def train_rl_phase(env, result_subdir, scenario_str, pecs_params, scenario_params,
                   total_timesteps=5000, learning_rate=0.0009809274356741915,
                   batch_size=64, n_epochs=6,
                   eval_freq=1000, patience=5, min_delta=1e-2):
    logging.info(f"Starting training for {scenario_str} data with PECS: {pecs_params} and scenario params: {scenario_params}")
    # Wrap the env for evaluation
    eval_env = DummyVecEnv([lambda: env])
    # Early stopping: stop after `patience` evaluations with < min_delta improvement
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=patience,
        min_delta=min_delta,
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_eval=stop_callback,
        eval_freq=eval_freq,
        best_model_save_path=result_subdir,
        verbose=1
    )

    ppo_model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )
    start_time = time.time()
    # Pass our EvalCallback so training will stop early if no improvement
    ppo_model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )
    end_time = time.time()

    # Save final (or best) model
    model_path = os.path.join(result_subdir, "past_ppo_mesa_model.zip")
    ppo_model.save(model_path)

    # Log resource usage
    training_time = end_time - start_time
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    gpu_status = "Using Apple's MPS GPU" if torch.backends.mps.is_available() else "GPU not available"
    logging.info(f"Training Time: {training_time:.2f} seconds, CPU: {cpu_usage}%, Memory: {memory_usage}%, GPU: {gpu_status}")
    logging.info("Training completed successfully.")
    return ppo_model, model_path


def simulation_rl_phase(ppo_model, env, result_subdir, num_episodes=5):
    all_final_decisions = []
    for episode in range(num_episodes):
        logging.info(f"Starting episode {episode + 1}.")
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step = 0
        while not done and not truncated:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
        logging.info(f"Episode {episode + 1} completed: Total Reward = {total_reward:.2f}, Steps = {step}.")
        episode_decisions = []
        for agent in env.model.schedule.agents:
            episode_decisions.append({
                'episode': episode + 1,
                'agent_id': agent.unique_id,
                'lat': agent.lat,
                'lon': agent.lon,
                'decision': agent.decision
            })
        all_final_decisions.extend(episode_decisions)
    if all_final_decisions:
        decisions_df = pd.DataFrame(all_final_decisions)
        agent_data_file = os.path.join(result_subdir, "agent_data_over_time.csv")
        decisions_df.to_csv(agent_data_file, index=False)
        logging.info(f"Agent data over time saved to '{agent_data_file}'.")
    else:
        logging.warning("No agent decisions were collected during simulation.")
        agent_data_file = None
    return agent_data_file

# --------------------------
# Main Execution Function
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified RL training and simulation for full1_rl past.")
    parser.add_argument("--train_model", action="store_true", help="Train the RL model if set; otherwise use pre-trained model.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of simulation episodes to run (default: 5).")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file = os.path.join(RESULTS_DIR, "run.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")
    logging.info("Starting full1_RL_past run.")

    cleanup_files(log_file, RESULTS_DIR)

    input_json = load_input_json(INPUT_JSON_PATH)
    if input_json is None:
        logging.error("Failed to load input JSON. Exiting.")
        return

    # Retrieve task_id and user_id for naming the results subdirectory.
    task_id = input_json.get("task_id", "default_task")
    user_id = input_json.get("user_id", "default_user")
    subdir_name = f"{task_id}_{user_id}"
    result_subdir = os.path.join(RESULTS_DIR, subdir_name)
    os.makedirs(result_subdir, exist_ok=True)

    model_params = input_json.get("model_parameters", {})
    period = model_params.get("period", "past")
    if period.lower() != "past":
        logging.error("This script is configured for past scenarios only.")
        return

    # For past, there is no scenario field; we use "PAST" as scenario_str.
    scenario_str = "PAST"
    pecs_params = {
        "physis": model_params.get("physis", DEFAULT_PECS_PARAMS["physis"]),
        "emotion": model_params.get("emotion", DEFAULT_PECS_PARAMS["emotion"]),
        "cognition": model_params.get("cognition", DEFAULT_PECS_PARAMS["cognition"]),
        "social": model_params.get("social", DEFAULT_PECS_PARAMS["social"])
    }
    scenario_params = model_params.get("scenario_params", DEFAULT_SCENARIO_PARAMS)

    aggregated_data = create_aggregated_data(scenario_str, MAIN_DIR, result_subdir)
    if aggregated_data is None:
        logging.error("Data aggregation failed. Exiting.")
        return

    width, height, max_steps = 10, 10, 10
    env = MesaEnv(
        env_data=aggregated_data,
        max_steps=max_steps,
        pecs_params=pecs_params,
        scenario_params=scenario_params,
        width=width,
        height=height
    )
    try:
        check_env(env)
        logging.info("Environment passed Gym API check.")
    except Exception as e:
        logging.error(f"Environment check failed: {e}")
        return

    # Set model filename and path (for past, we use "past_ppo_mesa_model.zip")
    model_path = os.path.join(result_subdir, "past_ppo_mesa_model.zip")

    if args.train_model:
        ppo_model, model_path = train_rl_phase(env, result_subdir, scenario_str, pecs_params, scenario_params)
    else:
        try:
            ppo_model = PPO.load(model_path, env=env)
            logging.info(f"Loaded pre-trained model from {model_path}.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return

    agent_data_file = simulation_rl_phase(ppo_model, env, result_subdir, num_episodes=args.num_episodes)

    logging.info("full1_RL_past run completed.")

if __name__ == "__main__":
    main()
