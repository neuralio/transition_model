import os
import pandas as pd
import xarray as xr
import pandas as pd
import numpy as np 
# Import necessary libraries
import numpy as np 
import gymnasium as gym        # Provides tools for creating environments compatible with RL algorithms.
from gymnasium import spaces   # Contains classes to define action and observation spaces in Gym.
from mesa import Agent, Model  # Base classes for creating agents and models.
from mesa.time import BaseScheduler # A scheduler in Mesa that activates agents in the order they were added.
from mesa.space import SingleGrid   # A grid where each cell contains at most one agent.
from stable_baselines3 import PPO   # Proximal Policy Optimization algorithm from Stable Baselines 3.
from stable_baselines3.common.env_checker import check_env # A utility function to verify that the custom environment adheres to the Gym API.
from stable_baselines3 import PPO
import time
import psutil
import GPUtil
import os  
import argparse
from model import LandUseModel
##############################
# Create The Gym Environment 
###############################
# 
# In this step, we'll create a custom Gym environment called MesaEnv that wraps our Mesa model. 
# This environment will allow us to interface the Mesa simulation with the reinforcement
# learning algorithms provided by Stable Baselines 3.

class MesaEnv(gym.Env):
    def __init__(self, env_data, max_steps):
        super(MesaEnv, self).__init__()
        self.env_data = env_data.reset_index(drop=True)
        self.max_steps = max_steps
        self.model = LandUseModel(self.env_data)  # Use the LandUseModel with env_data

        # Observation space: Suitability scores
        suitability_vars = ["crop_suitability", "pv_suitability"]
        min_values = self.env_data[suitability_vars].min().values
        max_values = self.env_data[suitability_vars].max().values
        self.observation_space = spaces.Box(low=min_values, high=max_values, dtype=np.float32)

        # Action space: Decision for each agent
        num_agents = len(self.env_data)
        self.action_space = spaces.MultiDiscrete([2] * num_agents)

        self.current_step = 0
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.model = LandUseModel(self.env_data)  # Reinitialize the model
        self.state = self._get_current_state()
        return self.state, {}


    def step(self, action):
        # Override RL model decisions with agent decisions
        for idx, agent in enumerate(self.model.schedule.agents):
            if action is not None:
                agent.decision = action[idx]

        # Update the model
        self.model.step()


        # Calculate the reward AUTO PROSTHESA
        profits = self.env_data.iloc[self.current_step][["pv_profit", "crop_profit"]]

        # Collect rewards and update state
        # reward = self._calculate_reward()
        reward = self._calculate_reward(profits) ## kai auto
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

    # def _calculate_reward(self):
    #     total_reward = 0.0
    #     for agent in self.model.schedule.agents:
    #         if agent.decision == 0:
    #             total_reward += agent.crop_suitability  # Example reward based on suitability
    #         else:
    #             total_reward += agent.pv_suitability
    #     return total_reward / len(self.model.schedule.agents)

    def render(self, mode="human"):
        agent_data = self.model.get_agent_data()
        print(agent_data)

    def close(self):
        pass

