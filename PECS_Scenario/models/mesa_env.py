# models/mesa_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .land_use_model import LandUseModel
import matplotlib.pyplot as plt
import pandas as pd
import random
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

