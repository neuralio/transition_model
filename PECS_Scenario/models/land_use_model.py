# models/land_use_model.py
from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import MultiGrid
import numpy as np
import random
from .land_use_agent import LandUseAgent

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

