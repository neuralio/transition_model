from mesa import Agent

class LandUseAgent(Agent):
    def __init__(self, unique_id, model, crop_suitability, pv_suitability, lat, lon):
        super().__init__(unique_id, model)
        self.crop_suitability = crop_suitability
        self.pv_suitability = pv_suitability
        self.lat = lat
        self.lon = lon
        self.decision = 0  # 0: Keep crops, 1: Convert to PV

    def step(self):
        # Apply decision rules
        if self.crop_suitability >= 85:
            self.decision = 0  # Suitable for crops
        elif self.pv_suitability >= 85:
            self.decision = 1  # Suitable for PV
        elif self.crop_suitability <= 60:
            self.decision = 1  # Not suitable for crops, convert to PV
        elif self.pv_suitability <= 60:
            self.decision = 0  # Not suitable for PV, keep crops
        else:
            # Let the RL model handle intermediate cases
            self.decision = 0  # Default action