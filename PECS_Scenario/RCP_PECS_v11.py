import os
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import MultiGrid
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import time
import psutil
import torch
import json
import argparse
import logging
import matplotlib.pyplot as plt
import random  # Import random for seeding

# ================================
# Define LandUseAgent Class
# ================================
class LandUseAgent(Agent):
    def __init__(self, unique_id, model, crop_suitability, pv_suitability, crop_profit, pv_profit, lat, lon, pecs_params):
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

    def step(self):
        """
        Update the agent's state and make decisions based on PECS attributes.
        """
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
        if self.pos is None:
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
        crop_profit = self.crop_profit  # Profit from crop farming
        pv_profit = self.pv_profit      # Profit from PV installation
      
        # Decision-making logic
        if crop_profit > pv_profit and self.physis["health_status"] > 0.5:
            self.decision = 0  # Prioritize crops
        elif pv_profit > crop_profit and self.cognition["policy_incentives"] > 0.6:
            self.decision = 1  # Prioritize PV
        elif self.emotion["stress_level"] > 0.7:
            self.decision = 0  # Avoid risk under high stress
        elif self.social["social_influence"] > 0.5:
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
        }

# ================================
# Define LandUseModel Class
# ================================
class LandUseModel(Model):
    def __init__(self, env_data, width, height, pecs_params, policy_incentive_active=False, seed=None):
        super().__init__()
        self.env_data = env_data
        self.schedule = BaseScheduler(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.current_step = 0
        self.policy_incentive_active = policy_incentive_active

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            # Add any other necessary seeding for reproducibility

        # Create agents and place them on the grid
        for agent_id, row in self.env_data.iterrows():
            crop_suitability = row["crop_suitability"]
            pv_suitability = row["pv_suitability"]
            pv_profit = row["pv_profit"]
            crop_profit = row["crop_profit"]
            lat = row["lat"]
            lon = row["lon"]

            # Create agent with PECS parameters
            agent = LandUseAgent(
                unique_id=agent_id,
                model=self,
                crop_suitability=crop_suitability,
                pv_suitability=pv_suitability,
                crop_profit=crop_profit,
                pv_profit=pv_profit,
                lat=lat,
                lon=lon,
                pecs_params=pecs_params  # Pass pecs_params here
            )
            self.schedule.add(agent)

            # Normalize lat/lon to grid coordinates
            x = int((lat - self.env_data["lat"].min()) / (self.env_data["lat"].max() - self.env_data["lat"].min()) * (width - 1))
            y = int((lon - self.env_data["lon"].min()) / (self.env_data["lon"].max() - self.env_data["lon"].min()) * (height - 1))

            # Ensure coordinates are within bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            # Place agent on the grid
            self.grid.place_agent(agent, (x, y))

    def step(self):
        """
        Advance the model by one step.
        """
        # Example: Dynamically update policy incentives (toggle every 10 steps)
        if self.current_step % 10 == 0:
            self.policy_incentive_active = not self.policy_incentive_active
            # Optionally, adjust PECS parameters here
            for agent in self.schedule.agents:
                agent.cognition["policy_incentives"] += 0.05 if self.policy_incentive_active else -0.05
                # Ensure the values stay within bounds
                agent.cognition["policy_incentives"] = min(max(agent.cognition["policy_incentives"], 0.0), 1.0)

        # Step all agents
        self.schedule.step()
        self.current_step += 1

    def collect_agent_data(self):
        """
        Collect data from all agents for analysis.
        """
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
                "Decision": agent.decision,  # 0: Crop, 1: PV
                "Health Status": agent.physis["health_status"],
                "Stress Level": agent.emotion["stress_level"],
                "Policy Incentives": agent.cognition["policy_incentives"],
                "Social Influence": agent.social["social_influence"]
            })
        return agent_data

    def save_agent_data(self, filename="agent_data.csv"):
        """
        Save agent data to a CSV file.
        """
        agent_data = self.collect_agent_data()
        df = pd.DataFrame(agent_data)
        df.to_csv(filename, index=False)
        #print(f"Agent data saved to {filename}")

# ================================
# Define MesaEnv Class
# ================================
class MesaEnv(gym.Env):
    def __init__(self, env_data, max_steps, pecs_params, width=5, height=5):
        super(MesaEnv, self).__init__()
        self.env_data = env_data.reset_index(drop=True)
        self.max_steps = max_steps

        # Grid dimensions
        self.width = width
        self.height = height

        # Store pecs_params as an instance variable
        self.pecs_params = pecs_params

        # Initialize the model with PECS-enabled agents and grid dimensions
        self.model = LandUseModel(
            self.env_data,
            width=self.width,
            height=self.height,
            pecs_params=self.pecs_params
        )

        # Observation space: Include PECS variables and suitability scores
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

        # Action space: Decision for each agent (0: Crop, 1: PV)
        num_agents = len(self.env_data)
        self.action_space = spaces.MultiDiscrete([2] * num_agents)

        self.current_step = 0
        self.state = None

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        # Remove or comment out the following line if it causes issues
        # super().reset(seed=seed)

        self.current_step = 0
        # Reinitialize the model with pecs_params and seed
        self.model = LandUseModel(
            self.env_data,
            width=self.width,
            height=self.height,
            pecs_params=self.pecs_params,
            seed=seed  # Pass seed here if you handle it in LandUseModel
        )
        self.state = self._get_current_state()
        expected_shape = self.observation_space.shape
        if self.state.shape != expected_shape:
            raise ValueError(f"Mismatch: Expected {expected_shape}, but got {self.state.shape}")
        return self.state, {}

    def step(self, action):
        """
        Apply actions, update the model, and calculate rewards.
        """
        # Override RL model decisions with agent actions
        for idx, agent in enumerate(self.model.schedule.agents):
            if action is not None:
                agent.decision = action[idx]

        # Update the model
        self.model.step()

        # Calculate rewards and update state
        reward = self._calculate_reward()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        self.state = self._get_current_state() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def _get_current_state(self, detailed=False):
        """
        Get aggregated or detailed state across all agents, including profit.
        """
        if detailed:
            # Detailed state: Return per-agent attributes
            return np.array([
                {
                    "crop_suitability": agent.crop_suitability,
                    "pv_suitability": agent.pv_suitability,
                    "health_status": agent.physis["health_status"],
                    "stress_level": agent.emotion["stress_level"],
                    "policy_incentives": agent.cognition["policy_incentives"],
                    "social_influence": agent.social["social_influence"],
                    "social_influence_weight": agent.cognition["policy_incentives"],  # Dynamic weight for social influence
                    "decision": agent.decision
                } for agent in self.model.schedule.agents
            ], dtype=object)

        # Aggregated state: Weighted averages for the entire model
        total_agents = len(self.model.schedule.agents)

        # Calculate weighted social influence
        weighted_social_influence = np.sum([
            agent.social["social_influence"] * agent.cognition["policy_incentives"]
            for agent in self.model.schedule.agents
        ]) / total_agents

        # Aggregate profits
        mean_crop_profit = np.mean([agent.crop_profit for agent in self.model.schedule.agents])
        mean_pv_profit = np.mean([agent.pv_profit for agent in self.model.schedule.agents])

        # Default: Aggregate state
        aggregated_state = {
            "crop_suitability": np.mean([agent.crop_suitability for agent in self.model.schedule.agents]),
            "pv_suitability": np.mean([agent.pv_suitability for agent in self.model.schedule.agents]),
            "health_status": np.mean([agent.physis["health_status"] for agent in self.model.schedule.agents]),
            "stress_level": np.mean([agent.emotion["stress_level"] for agent in self.model.schedule.agents]),
            "policy_incentives": np.mean([agent.cognition["policy_incentives"] for agent in self.model.schedule.agents]),
            "social_influence": weighted_social_influence,  # Use weighted social influence
        }
        return np.array(list(aggregated_state.values()), dtype=np.float32)

    def _calculate_reward(self):
        """
        Calculate rewards based on agent decisions, crop/PV profit, PECS metrics, and social influence.
        """
        total_reward = 0.0

        for agent in self.model.schedule.agents:
            # Calculate social influence as the average decision of neighbors
            neighbors = self.model.grid.get_neighbors(agent.pos, moore=True, include_center=False)
            if neighbors:
                avg_neighbor_decision = sum([neighbor.decision for neighbor in neighbors]) / len(neighbors)
            else:
                avg_neighbor_decision = 0  # No neighbors

            # Adjust social influence based on the agent's current decision
            if agent.decision == 0:  # Crop farming
                social_influence = 1 - avg_neighbor_decision  # More influence from PV neighbors
            elif agent.decision == 1:  # PV installation
                social_influence = avg_neighbor_decision  # More influence from PV neighbors
            else:
                social_influence = 0  # Default for unexpected cases

            # Dynamic weight for social influence
            social_influence_weight = agent.cognition["policy_incentives"]

            # Reward calculation
            if agent.decision == 0:  # Crop farming
                total_reward += (
                    agent.crop_profit * 0.4 +  # Weight crop profit
                    agent.emotion["satisfaction"] * 0.2 -  # Positive impact of satisfaction
                    agent.emotion["stress_level"] * 0.2 +  # Negative impact of stress
                    social_influence * social_influence_weight  # Positive impact of social conformity
                )
            elif agent.decision == 1:  # PV installation
                total_reward += (
                    agent.pv_profit * 0.4 +  # Weight PV profit
                    agent.cognition["policy_incentives"] * 0.2 -  # Positive impact of policy incentives
                    agent.physis["labor_availability"] * 0.2 +  # Negative impact of labor availability
                    social_influence * social_influence_weight  # Positive impact of social conformity
                )

        # Normalize reward by the number of agents
        return total_reward / len(self.model.schedule.agents)

    def render(self, mode="human"):
        """
        Display or visualize the current state of all agents.
        """
        if mode == "human":
            # Collect agent data
            agent_data = self.model.collect_agent_data()
            # Optionally, print or log agent data
            pass
        elif mode == "plot":
            # Collect agent decisions and positions
            agent_data = self.model.collect_agent_data()
            lats = [data["Lat"] for data in agent_data]
            lons = [data["Lon"] for data in agent_data]
            decisions = [data["Decision"] for data in agent_data]
            # Scatter plot of agent decisions
            plt.figure(figsize=(10, 6))
            plt.scatter(lons, lats, c=decisions, cmap="coolwarm", edgecolor="k")
            plt.colorbar(label="Decision (0: Crop, 1: PV)")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Agent Decisions")
            plt.grid()
            # Save the plot
            plt.savefig("agent_decisions_plot.png", dpi=300)
            plt.close()

    def close(self):
        pass

# ================================
# Main Execution Block
# ================================
# ================================
# Main Execution Block
# ================================
if __name__ == "__main__":
    # ================================
    # Argument Parsing
    # ================================
    # Define default PECS parameters
    DEFAULT_PEC_PARAMS = {
        "physis": {"health_status": 0.8, "labor_availability": 0.6},
        "emotion": {"stress_level": 0.3, "satisfaction": 0.7},
        "cognition": {"policy_incentives": 0.5, "information_access": 0.6},
        "social": {"social_influence": 0.4, "community_participation": 0.5}
    }

    # Argument parsing with PECS parameters
    parser = argparse.ArgumentParser(description="Run the land-use simulation with a specific RCP scenario or past data.")
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
    # Add PECS parameters as arguments
    parser.add_argument("--physis_health_status", type=float, default=0.8, help="Initial health status of agents (default: 0.8)")
    parser.add_argument("--physis_labor_availability", type=float, default=0.6, help="Initial labor availability of agents (default: 0.6)")
    parser.add_argument("--emotion_stress_level", type=float, default=0.3, help="Initial stress level of agents (default: 0.3)")
    parser.add_argument("--emotion_satisfaction", type=float, default=0.7, help="Initial satisfaction of agents (default: 0.7)")
    parser.add_argument("--cognition_policy_incentives", type=float, default=0.5, help="Initial policy incentives (default: 0.5)")
    parser.add_argument("--cognition_information_access", type=float, default=0.6, help="Initial information access (default: 0.6)")
    parser.add_argument("--social_social_influence", type=float, default=0.4, help="Initial social influence (default: 0.4)")
    parser.add_argument("--social_community_participation", type=float, default=0.5, help="Initial community participation (default: 0.5)")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file for PECS parameters.")

    args = parser.parse_args()

    # Ensure that either --scenario or --past is specified
    if not args.past and not args.scenario:
        raise ValueError("You must specify either --scenario or --past.")

    # Load PECS parameters from config file if provided
    if args.config:
        with open(args.config, "r") as f:
            pecs_params = json.load(f)
    else:
        # Construct PECS parameters from parsed arguments
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

    # Validate PECS parameters
    def validate_pecs_params(pecs_params):
        for category, params in pecs_params.items():
            for param, value in params.items():
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"PECS parameter '{category}.{param}' must be between 0 and 1. Received: {value}")

    validate_pecs_params(pecs_params)

    MAIN_DIR = "/Users/mbanti/Documents/Projects/esa_agents_local/PECS/data"
    WHEAT_PAST   = os.path.join(MAIN_DIR, "WHEAT_PAST")
    WHEAT_FUTURE = os.path.join(MAIN_DIR, "WHEAT_FUTURE")
    MAIZE_PAST   = os.path.join(MAIN_DIR, "MAIZE_PAST")
    MAIZE_FUTURE = os.path.join(MAIN_DIR, "MAIZE_FUTURE")
    PV_PAST      = os.path.join(MAIN_DIR, "PV_PAST")
    PV_FUTURE    = os.path.join(MAIN_DIR, "PV_FUTURE")

    # Determine whether to run with past or future scenario data
    if args.past:
        scenario_name = "PAST"
        print("Running simulation with past data...")
        pv_suitability_file = os.path.join(PV_PAST, "PAST_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
        pv_yield_file       = os.path.join(PV_PAST, "PAST_PV_YIELD.csv")
        cs_wheat_file       = os.path.join(WHEAT_PAST, "PAST_LUSA_PREDICTIONS.csv")
        cy_wheat_file       = os.path.join(WHEAT_PAST, "AquaCrop_Results_PAST.csv")
    else:
        scenario_name = f"RCP{args.scenario}"
        print(f"Running simulation for scenario {scenario_name}...")
        pv_suitability_file = os.path.join(PV_FUTURE, f"RCP{args.scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
        pv_yield_file       = os.path.join(PV_FUTURE, f"RCP{args.scenario}_PV_SUITABILITY_ANNUAL_ELECTRICITY_SAVINGS_2021_2030.csv")
        cs_wheat_file       = os.path.join(WHEAT_FUTURE, f"RCP{args.scenario}_LUSA_PREDICTIONS.csv")
        cy_wheat_file       = os.path.join(WHEAT_FUTURE, f"AquaCrop_Results_RCP{args.scenario}.csv")

    print(f"Using dataset: {scenario_name}")

    # Flag to control whether to train the model
    train_model = True  # Set to False to skip training

# if __name__ == "__main__":
#     # ================================
#     # Argument Parsing
#     # ================================
#     # Define default PECS parameters
#     DEFAULT_PEC_PARAMS = {
#         "physis": {"health_status": 0.8, "labor_availability": 0.6},
#         "emotion": {"stress_level": 0.3, "satisfaction": 0.7},
#         "cognition": {"policy_incentives": 0.5, "information_access": 0.6},
#         "social": {"social_influence": 0.4, "community_participation": 0.5}
#     }

#     # Argument parsing with PECS parameters
#     parser = argparse.ArgumentParser(description="Run the land-use simulation with a specific RCP scenario.")
#     parser.add_argument(
#         "--scenario",
#         type=str,
#         choices=["26", "45", "85"],
#         required=True,
#         help="Specify the RCP scenario to run the simulation (options: '26', '45', '85')."
#     )
#     # Add PECS parameters as arguments
#     parser.add_argument("--physis_health_status", type=float, default=0.8, help="Initial health status of agents (default: 0.8)")
#     parser.add_argument("--physis_labor_availability", type=float, default=0.6, help="Initial labor availability of agents (default: 0.6)")
#     parser.add_argument("--emotion_stress_level", type=float, default=0.3, help="Initial stress level of agents (default: 0.3)")
#     parser.add_argument("--emotion_satisfaction", type=float, default=0.7, help="Initial satisfaction of agents (default: 0.7)")
#     parser.add_argument("--cognition_policy_incentives", type=float, default=0.5, help="Initial policy incentives (default: 0.5)")
#     parser.add_argument("--cognition_information_access", type=float, default=0.6, help="Initial information access (default: 0.6)")
#     parser.add_argument("--social_social_influence", type=float, default=0.4, help="Initial social influence (default: 0.4)")
#     parser.add_argument("--social_community_participation", type=float, default=0.5, help="Initial community participation (default: 0.5)")
#     parser.add_argument("--config", type=str, default=None, help="Path to JSON config file for PECS parameters.")

#     args = parser.parse_args()
#     scenario = args.scenario

#     # Load PECS parameters from config file if provided
#     if args.config:
#         with open(args.config, "r") as f:
#             pecs_params = json.load(f)
#     else:
#         # Construct PECS parameters from parsed arguments
#         pecs_params = {
#             "physis": {
#                 "health_status": args.physis_health_status,
#                 "labor_availability": args.physis_labor_availability
#             },
#             "emotion": {
#                 "stress_level": args.emotion_stress_level,
#                 "satisfaction": args.emotion_satisfaction
#             },
#             "cognition": {
#                 "policy_incentives": args.cognition_policy_incentives,
#                 "information_access": args.cognition_information_access
#             },
#             "social": {
#                 "social_influence": args.social_social_influence,
#                 "community_participation": args.social_community_participation
#             }
#         }

#     # Validate PECS parameters
#     def validate_pecs_params(pecs_params):
#         for category, params in pecs_params.items():
#             for param, value in params.items():
#                 if not (0.0 <= value <= 1.0):
#                     raise ValueError(f"PECS parameter '{category}.{param}' must be between 0 and 1. Received: {value}")

#     validate_pecs_params(pecs_params)

#     print(f"Running simulation for RCP scenario: {scenario} with PECS parameters: {pecs_params}")

#     # Flag to control whether to train the model
#     train_model = True  # Set to False to skip training

    # ########################################################################
    # # PATH DEFINITION
    # ########################################################################
    # MAIN_DIR = "/Users/mbanti/Documents/Projects/esa_agents_local/PECS/data"
    # WHEAT_PAST   = os.path.join(MAIN_DIR, "WHEAT_PAST")
    # WHEAT_FUTURE = os.path.join(MAIN_DIR, "WHEAT_FUTURE")
    # MAIZE_PAST   = os.path.join(MAIN_DIR, "MAIZE_PAST")
    # MAIZE_FUTURE = os.path.join(MAIN_DIR, "MAIZE_FUTURE")
    # PV_PAST      = os.path.join(MAIN_DIR, "PV_PAST")
    # PV_FUTURE    = os.path.join(MAIN_DIR, "PV_FUTURE")

    ########################################################################
    # LOAD AND PROCESS DATA
    ########################################################################

    # # Define file paths based on scenario
    # pv_suitability_file = os.path.join(PV_FUTURE, f"RCP{scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    # pv_yield_file       = os.path.join(PV_FUTURE, f"RCP{scenario}_PV_SUITABILITY_ANNUAL_ELECTRICITY_SAVINGS_2021_2030.csv")
    # # cs_maize_file = os.path.join(MAIZE_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv")
    # cs_wheat_file = os.path.join(WHEAT_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv")
    # # cy_maize_file = os.path.join(MAIZE_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv")
    # cy_wheat_file = os.path.join(WHEAT_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv")

    # # --- PV Suitability ---
    # PV_ = pd.read_csv(pv_suitability_file, sep=',')
    # PV_ = PV_.rename(columns={"score": "pv_suitability"})
    # PV_ = PV_[(PV_["year"] >= 2021) & (PV_["year"] <= 2030)]
    # print(f"Loaded PV Suitability for RCP {scenario}")

    # # --- Crop Suitability for Maize and Wheat ---
    # CS_Maize_ = pd.read_csv(cs_maize_file, sep=',')
    # CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
    # crop_suitability = pd.concat([CS_Maize_, CS_Wheat_], ignore_index=True)
    # crop_suitability = crop_suitability[(crop_suitability["year"] >= 2021) & (crop_suitability["year"] <= 2030)]
    # crop_suitability = crop_suitability.rename(columns={"score": "crop_suitability"})
    # print(f"Loaded Crop Suitability (Maize & Wheat) for RCP {scenario}")

    # # --- PV Yield (Profit) ---
    # PV_Yield_ = pd.read_csv(pv_yield_file, sep=',')
    # PV_Yield_ = PV_Yield_.rename(columns={'annual_electricity_savings_€': 'pv_profit'})
    # print(f"Loaded PV Yield for RCP {scenario}")

    # # --- Crop Yield (Profit) for Maize ---
    # CY_Maize_26_FULL = pd.read_csv(cy_maize_file, sep=',')
    # CY_Maize_26 = CY_Maize_26_FULL[['Dates', 'Profits']].copy()
    # CY_Maize_26['Dates'] = pd.to_datetime(CY_Maize_26['Dates'])
    # CY_Maize_26['year'] = CY_Maize_26['Dates'].dt.year
    # CY_Maize_26 = CY_Maize_26.groupby('year').last().reset_index()
    # CY_Maize_26 = CY_Maize_26.rename(columns={'Profits': 'crop_profit'})
    # print(f"Processed Crop Yield for Maize for RCP {scenario}")

    # # --- Crop Yield (Profit) for Wheat ---
    # CY_Wheat_26_FULL = pd.read_csv(cy_wheat_file, sep=',')
    # CY_Wheat_26 = CY_Wheat_26_FULL[['Dates', 'Profits']].copy()
    # CY_Wheat_26['Dates'] = pd.to_datetime(CY_Wheat_26['Dates'])
    # CY_Wheat_26['year'] = CY_Wheat_26['Dates'].dt.year
    # CY_Wheat_26 = CY_Wheat_26.groupby('year').last().reset_index()
    # CY_Wheat_26 = CY_Wheat_26.rename(columns={'Profits': 'crop_profit'})
    # print(f"Processed Crop Yield for Wheat for RCP {scenario}")

    # # --- Combine Crop Profit for Maize and Wheat ---
    # crop_profit = pd.concat([CY_Maize_26, CY_Wheat_26], ignore_index=True)
    # # Merge with crop_suitability based on "year", "lat", and "lon"
    # crop_profit = crop_suitability.merge(crop_profit, on="year", how="inner")[["year", "lat", "lon", "crop_profit"]]

    # # Convert latitude and longitude columns to float for compatibility
    # for df in [PV_, crop_suitability, PV_Yield_, crop_profit]:
    #     df["lat"] = df["lat"].astype(float).round(5)
    #     df["lon"] = df["lon"].astype(float).round(5)

    # # --- Merge Datasets ---
    # env_data = crop_suitability.merge(PV_, on=["year", "lat", "lon"], how="inner")
    # env_data = env_data.merge(PV_Yield_, on=["year", "lat", "lon"], how="inner")
    # env_data = env_data.merge(crop_profit, on=["year", "lat", "lon"], how="inner")

    # print(env_data.head())
    # print("Merged environmental dataset:")
    # Define file paths based on scenario
    # pv_suitability_file = os.path.join(PV_FUTURE, f"RCP{scenario}_PV_SUITABILITY_PREDICTIONS_UPDATED.csv")
    # pv_yield_file       = os.path.join(PV_FUTURE, f"RCP{scenario}_PV_SUITABILITY_ANNUAL_ELECTRICITY_SAVINGS_2021_2030.csv")
    # cs_wheat_file       = os.path.join(WHEAT_FUTURE, f"RCP{scenario}_LUSA_PREDICTIONS.csv")
    # cy_wheat_file       = os.path.join(WHEAT_FUTURE, f"AquaCrop_Results_RCP{scenario}.csv")

    # --- PV Suitability ---
    PV_ = pd.read_csv(pv_suitability_file, sep=',')
    PV_ = PV_.rename(columns={"score": "pv_suitability"})
    PV_ = PV_[(PV_["year"] >= 2021) & (PV_["year"] <= 2030)]
    print(f"Loaded PV Suitability for {scenario_name}")

    # --- Crop Suitability for Wheat ---
    CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
    CS_Wheat_ = CS_Wheat_[(CS_Wheat_["year"] >= 2021) & (CS_Wheat_["year"] <= 2030)]
    CS_Wheat_ = CS_Wheat_.rename(columns={"score": "crop_suitability"})
    print(f"Loaded Crop Suitability for Wheat for {scenario_name}")

    # --- PV Yield (Profit) ---
    PV_Yield_ = pd.read_csv(pv_yield_file, sep=',')
    PV_Yield_ = PV_Yield_.rename(columns={'annual_electricity_savings_€': 'pv_profit'})
    print(f"Loaded PV Yield for {scenario_name}")


    # --- Crop Yield (Profit) for Wheat ---
    CY_Wheat_FULL = pd.read_csv(cy_wheat_file, sep=',')
    CY_Wheat = CY_Wheat_FULL[['Dates', 'Profits']].copy()
    CY_Wheat['Dates'] = pd.to_datetime(CY_Wheat['Dates'])
    CY_Wheat['year'] = CY_Wheat['Dates'].dt.year
    CY_Wheat = CY_Wheat.groupby('year').last().reset_index()
    CY_Wheat = CY_Wheat.rename(columns={'Profits': 'crop_profit'})
    print(f"Processed Crop Yield for Wheat for {scenario_name}")

    # --- Merge Crop Profit with Crop Suitability ---
    crop_profit = CS_Wheat_.merge(CY_Wheat, on="year", how="inner")[["year", "lat", "lon", "crop_profit"]]

    # Convert latitude and longitude columns to float for compatibility
    for df in [PV_, CS_Wheat_, PV_Yield_, crop_profit]:
        df["lat"] = df["lat"].astype(float).round(5)
        df["lon"] = df["lon"].astype(float).round(5)

    # --- Merge Datasets ---
    env_data = CS_Wheat_.merge(PV_, on=["year", "lat", "lon"], how="inner")
    env_data = env_data.merge(PV_Yield_, on=["year", "lat", "lon"], how="inner")
    env_data = env_data.merge(crop_profit, on=["year", "lat", "lon"], how="inner")

    print(env_data.head())
    print("Merged environmental dataset:")    



    # Aggregate env_data by unique lat/lon pairs
    aggregated_data = env_data.groupby(['lat', 'lon'], as_index=False).agg({
        'crop_suitability': 'mean',  # Average crop suitability over years
        'pv_suitability': 'mean',   # Average PV suitability over years
        'crop_profit': 'sum',       # Total crop profit over years
        'pv_profit': 'sum'          # Total PV profit over years
    })

    # Count the number of unique lat/lon pairs
    unique_lat_lon_count = aggregated_data[["lat", "lon"]].drop_duplicates().shape[0]

    # Display the result
    print(f"Number of unique lat/lon pairs: {unique_lat_lon_count}")

    ###########################
    #  Define Mesa Classes
    ###########################
    # (Already defined above)

    ###########################
    #  Instantiate and Check Env
    ###########################
    print(f"Number of columns: {aggregated_data.shape[1]}")
    # Define the grid dimensions BEFORE instantiating the environment
    width = 10
    height = 10

    # Instantiate the environment with PECs parameters
    env = MesaEnv(
        env_data=aggregated_data,
        max_steps=10,
        pecs_params=pecs_params,
        width=width,
        height=height
    )

    # Check if the environment follows the Gym API
    try:
        check_env(env)
        print("Environment passed the Gym API check.")
    except Exception as e:
        print(f"Environment check failed: {e}")
        exit(1)

    ##############################################
    # TRAIN RL AGENTS 
    ##############################################
    learning_rate = 0.0009809274356741915
    batch_size = 64
    n_epochs = 6

    if train_model:
        # Create a directory for saving models if it doesn't exist
        SAVE_DIR = "trained_models"
        os.makedirs(SAVE_DIR, exist_ok=True)

        # Create a subdirectory for each RCP scenario
        scenario_name = "PAST" if args.past else f"RCP{args.scenario}"
        scenario_dir = os.path.join(SAVE_DIR, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            filename=os.path.join(scenario_dir, "training.log"),
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s"
        )

        # Log PECS parameters
        logging.info(f"Starting training for {scenario_name} with PECS parameters: {pecs_params}")

        # Create the PPO RL model with GPU acceleration using the MPS backend.
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            device="mps"  # Use MPS on Apple Silicon
        )

        # Function to monitor CPU and memory usage (GPU stats from GPUtil are skipped for MPS)
        def get_processing_stats():
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            # Memory usage
            memory_usage = psutil.virtual_memory().percent
            # Indicate that MPS is used
            gpu_status = "Using Apple's MPS GPU" if torch.backends.mps.is_available() else "GPU not available"
            return cpu_usage, memory_usage, gpu_status

        # Record the start time
        start_time = time.time()

        # Train the RL model
        model.learn(total_timesteps=5000)

        # Record the end time
        end_time = time.time()

        # Save the trained PPO model
        model.save(os.path.join(scenario_dir, "ppo_model"))

        # Save PECS parameters
        with open(os.path.join(scenario_dir, "pecs_params.json"), "w") as f:
            json.dump(pecs_params, f, indent=4)

        # Calculate training time
        training_time = end_time - start_time

        # Get processing stats
        cpu_usage, memory_usage, gpu_status = get_processing_stats()

        # Log the stats
        logging.info(f"Training Time: {training_time:.2f} seconds")
        logging.info(f"CPU Usage: {cpu_usage}%")
        logging.info(f"Memory Usage: {memory_usage}%")
        logging.info(f"GPU Status: {gpu_status}")

        # Print the stats
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"CPU Usage: {cpu_usage}%")
        print(f"Memory Usage: {memory_usage}%")
        print(f"GPU Status: {gpu_status}")

