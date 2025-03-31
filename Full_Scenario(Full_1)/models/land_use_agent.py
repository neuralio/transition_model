# models/land_use_agent.py
from mesa import Agent
import random 


class LandUseAgent(Agent):
    def __init__(self, unique_id, model, crop_suitability, pv_suitability, crop_profit, pv_profit, lat, lon, pecs_params, scenario_params=None):
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


        # Scenario-specific parameters
        self.subsidy = scenario_params.get("subsidy", 0.0) if scenario_params else 0.0
        self.loan_interest_rate = scenario_params.get("loan_interest_rate", 0.05) if scenario_params else 0.05
        self.tax_incentive_rate = scenario_params.get("tax_incentive_rate", 0.0) if scenario_params else 0.0
        self.social_prestige_weight = scenario_params.get("social_prestige_weight", 0.0) if scenario_params else 0.0


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

        # Adjust PV profit with scenario-specific parameters
        pv_profit_with_subsidy = pv_profit + self.subsidy - (pv_profit * self.loan_interest_rate)
        pv_profit_with_tax = pv_profit_with_subsidy + (pv_profit * self.tax_incentive_rate)

      
        # Decision-making logic
        if crop_profit > pv_profit_with_tax and self.physis["health_status"] > 0.5:
            self.decision = 0  # Prioritize crops
        elif pv_profit_with_tax > crop_profit and self.cognition["policy_incentives"] > 0.6:
            self.decision = 1  # Prioritize PV
        elif self.emotion["stress_level"] > 0.7:
            self.decision = 0  # Avoid risk under high stress
        elif self.social["social_influence"] > 0.5 or self.social["social_influence"] * self.social_prestige_weight > 0.6:
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
            "Subsidy": self.subsidy,  # Added subsidy parameter
            "Loan Interest Rate": self.loan_interest_rate,  # Added loan interest rate
            "Tax Incentive Rate": self.tax_incentive_rate,  # Added tax incentive rate
            "Social Prestige Weight": self.social_prestige_weight,  # Added social prestige weight
        }