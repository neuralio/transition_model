#!/usr/bin/env python
# models/mesa_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .land_use_model import LandUseModel
import matplotlib.pyplot as plt
import pandas as pd
import random

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

        # Initialize the model without any EU target.
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

