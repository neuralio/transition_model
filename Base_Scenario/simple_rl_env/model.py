from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import MultiGrid
from agents import LandUseAgent
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


###########################
#  Define mesa Model
###########################
#
# In this step, we'll define the LandUseModel, which represents the overall simulation. 
# The model will create a grid of agents and manage the simulation steps.

class LandUseModel(Model):
    def __init__(self, env_data):
        super().__init__()
        self.env_data = env_data  # Use aggregated data aggregated_data
        self.schedule = BaseScheduler(self)
        self.current_step = 0

        # Create agents for each unique lat-lon pair
        for agent_id, row in self.env_data.iterrows():
            crop_suitability = row["crop_suitability"]
            pv_suitability = row["pv_suitability"]
            lat = row["lat"]
            lon = row["lon"]

            # Create agent with unique lat-lon pair and suitability scores
            agent = LandUseAgent(agent_id, self, crop_suitability, pv_suitability, lat, lon)
            self.schedule.add(agent)

    def step(self):
        # Advance the model by one step
        self.schedule.step()
        self.current_step += 1

    def get_agent_data(self):
        # Collect data from all agents
        data = []
        for agent in self.schedule.agents:
            data.append({
                "lat": agent.lat,
                "lon": agent.lon,
                "crop_suitability": agent.crop_suitability,
                "pv_suitability": agent.pv_suitability,
                "decision": agent.decision  # 0: Crop, 1: PV
            })
        return pd.DataFrame(data)