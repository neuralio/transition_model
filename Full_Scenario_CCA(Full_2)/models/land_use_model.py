from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import MultiGrid
import numpy as np
import random
from .land_use_agent import LandUseAgent

class LandUseModel(Model):
    def __init__(self, env_data, width, height, pecs_params, scenario_params=None, climate_losses=None, seed=None):
        super().__init__()
        self.env_data = env_data
        self.schedule = BaseScheduler(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.current_step = 0
        self.policy_incentive_active = False

        # Store scenario parameters and climate losses
        self.scenario_params = scenario_params or {}
        self.climate_losses = climate_losses or {}

        # Extract scenario parameters
        self.base_subsidy = self.scenario_params.get("subsidy", 0.0)  # Base subsidy for PV installations
        self.loan_interest_rate = self.scenario_params.get("loan_interest_rate", 0.05)
        self.tax_incentive_rate = self.scenario_params.get("tax_incentive_rate", 0.0)
        self.social_prestige_weight = self.scenario_params.get("social_prestige_weight", 0.0)

        # Initialize the saturation fund dedicated to PV installations.
        self.initial_fund = self.scenario_params.get("initial_subsidy_fund", 1.0)
        self.remaining_fund = self.initial_fund

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Create agents and place them on the grid.
        for agent_id, row in self.env_data.iterrows():
            crop_suitability = row["crop_suitability"]
            pv_suitability = row["pv_suitability"]
            pv_profit = row["pv_profit"]
            crop_profit = row["crop_profit"]
            lat = row["lat"]
            lon = row["lon"]

            # Create agent without passing any EU target.
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

            # Normalize lat/lon to grid coordinates.
            x = int((lat - self.env_data["lat"].min()) / (self.env_data["lat"].max() - self.env_data["lat"].min()) * (width - 1))
            y = int((lon - self.env_data["lon"].min()) / (self.env_data["lon"].max() - self.env_data["lon"].min()) * (height - 1))
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            self.grid.place_agent(agent, (x, y))

    def step(self):
        current_year = self.current_step + 2021
        # Apply climate losses dynamically
        if current_year in self.climate_losses:
            loss_factor = self.climate_losses[current_year]
            for agent in self.schedule.agents:
                agent.crop_profit *= (1 - loss_factor)
                print(f"Agent {agent.unique_id}: Climate loss applied for {current_year}, new crop profit: {agent.crop_profit:.2f}")

        # Update policy incentives (toggle every 10 steps)
        if self.current_step % 10 == 0:
            self.policy_incentive_active = not self.policy_incentive_active
            for agent in self.schedule.agents:
                agent.cognition["policy_incentives"] += 0.05 if self.policy_incentive_active else -0.05
                agent.cognition["policy_incentives"] = min(max(agent.cognition["policy_incentives"], 0.0), 1.0)

        # Step all agents
        self.schedule.step()
        self.current_step += 1

        # --- Saturation Fund Mechanism ---
        ALPHA = 0.05  # Fixed deduction per PV adoption per step
        num_PV_adopters = sum(1 for agent in self.schedule.agents if agent.decision == 1)
        self.remaining_fund = max(0, self.remaining_fund - ALPHA * num_PV_adopters)
        effective_subsidy = self.base_subsidy * (self.remaining_fund / self.initial_fund)
        print(f"Remaining Fund: {self.remaining_fund:.3f} / {self.initial_fund}, Effective Subsidy: {effective_subsidy:.3f}")
        # Update each agent’s subsidy value.
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
