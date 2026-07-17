"""
Hybrid Agriculture Environment for DQN
- Real data calibrated parameters (Fertilizer kg/ha distributions)
- Daily physical plant growth model
- Numerical fertilizer amounts (discrete levels in kg/ha)
"""
import numpy as np
import pandas as pd
import os

class AgricultureEnv:
    def __init__(self, use_real_data=True, data_path=None, max_days=30, seed=42):
        self.max_days = max_days
        self.seed = seed
        np.random.seed(seed)

        # Load calibration data
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'indian_crop_yield_synthetic.csv')
        
        if use_real_data and os.path.exists(data_path):
            self.calib = pd.read_csv(data_path)
            self.fert_mean = float(self.calib['Fertilizer_per_ha'].median())
            self.fert_std = float(self.calib['Fertilizer_per_ha'].std())
            self.yield_mean = float(self.calib['Yield'].median())
            self.rain_mean = float(self.calib['Annual_Rainfall'].mean())
        else:
            self.fert_mean = 140.0
            self.fert_std = 90.0
            self.yield_mean = 3800.0
            self.rain_mean = 1050.0

        # Actions aligned to real Fertilizer_per_ha percentiles
        # p10~43, p25~74, p50~137, p75~259 kg/ha (season totals)
        # Daily doses sized so 2-4 applications reach realistic season totals
        self.actions = {
            0: {'name': 'Wait',             'water': 0,  'fert_kg': 0,   'cost': 0.0},
            1: {'name': 'Light Irrig',      'water': 14, 'fert_kg': 0,   'cost': 0.6},
            2: {'name': 'Heavy Irrig',      'water': 28, 'fert_kg': 0,   'cost': 1.2},
            3: {'name': 'Fert 40kg',        'water': 0,  'fert_kg': 40,  'cost': 1.0},
            4: {'name': 'Fert 70kg',        'water': 0,  'fert_kg': 70,  'cost': 1.6},
            5: {'name': 'Fert 140kg',       'water': 0,  'fert_kg': 140, 'cost': 2.8},
            6: {'name': 'Irrig+Fert 50kg',  'water': 16, 'fert_kg': 50,  'cost': 2.0},
            7: {'name': 'Irrig+Fert 90kg',  'water': 18, 'fert_kg': 90,  'cost': 2.6},
        }
        self.n_actions = len(self.actions)

        self.observation_space = type('obj', (object,), {'shape': (9,)})()
        self.action_space = type('obj', (object,), {'n': self.n_actions})()

        self.reset()

    def reset(self):
        self.day = 0
        self.cumulative_growth = 1.0
        self.total_fert_used = 0.0
        self.total_water_used = 0.0

        # Realistic initial state (calibrated ranges)
        self.state = np.array([
            0.0,                                      # day norm
            np.random.uniform(38, 62),                # moisture %
            np.random.uniform(40, 90),                # nutrient (relative)
            np.random.uniform(6.0, 7.2),              # pH
            np.random.uniform(18, 32),                # temperature
            np.random.uniform(45, 85),                # humidity
            np.random.uniform(350, 650),              # light
            np.random.uniform(70, 92),                # health
            np.random.uniform(1.5, 8.0),              # toxicity
        ], dtype=np.float32)

        self.base_yield = np.random.normal(self.yield_mean, 600)
        self.base_yield = np.clip(self.base_yield, 1800, 9000)
        self.current_yield = self.base_yield * 0.85

        return self.state.copy(), {}

    def step(self, action):
        if action not in self.actions:
            raise ValueError(f"Invalid action {action}")

        act = self.actions[action]
        moisture, nutrient, pH, temp, humidity, light, health, toxicity = self.state[1:]

        # Apply action effects
        new_moisture = np.clip(moisture + act['water'] + np.random.normal(0, 2.5) - 3.5, 5, 100)  # daily loss
        # Stronger nutrient response so fertilizing matters
        new_nutrient = np.clip(nutrient + act['fert_kg'] * 0.90 + np.random.normal(0, 1.2) - 3.8, 5, 180)
        new_toxicity = np.clip(toxicity + act['fert_kg'] * 0.012 + np.random.normal(0, 0.2), 0, 30)
        new_pH = np.clip(pH + np.random.normal(0, 0.03), 5.2, 8.3)
        new_temp = np.clip(temp + np.random.normal(0, 1.1), 12, 42)
        new_humidity = np.clip(humidity + np.random.normal(0, 3.5), 20, 98)
        new_light = np.clip(light + np.random.normal(0, 25), 200, 900)

        # Stress calculation
        water_stress = 0.0
        if new_moisture < 40:
            water_stress = (40 - new_moisture) / 35
        elif new_moisture > 78:
            water_stress = (new_moisture - 78) / 25

        nutrient_stress = 0.0
        if new_nutrient < 50:
            nutrient_stress = (50 - new_nutrient) / 38   # stronger low-nutrient penalty
        elif new_nutrient > 140:
            nutrient_stress = (new_nutrient - 140) / 55

        tox_stress = min(new_toxicity / 28.0, 1.0)  # softer toxicity

        # Health update
        health_delta = 2.5 - 4.0 * water_stress - 4.5 * nutrient_stress - 4.0 * tox_stress
        health_delta += np.random.normal(0, 1.0)
        new_health = np.clip(health + health_delta, 0, 100)

        # Daily growth — nutrient stress weighs more so fertilizing pays off
        daily_potential = 0.036
        growth = daily_potential * (1 - 0.65 * water_stress) * (1 - 0.72 * nutrient_stress) * (1 - 0.70 * tox_stress)
        self.cumulative_growth += max(growth, 0.001)
        self.cumulative_growth = np.clip(self.cumulative_growth, 0.55, 3.8)

        # Yield
        tox_penalty = min(new_toxicity / 25.0, 0.55)
        self.current_yield = self.base_yield * self.cumulative_growth * (1 - tox_penalty)

        self.total_fert_used += act['fert_kg']
        self.total_water_used += act['water']

        # New state
        new_state = np.array([
            (self.day + 1) / self.max_days,
            new_moisture, new_nutrient, new_pH, new_temp,
            new_humidity, new_light, new_health, new_toxicity
        ], dtype=np.float32)

        reward = self._compute_reward(action, new_state, act)

        self.state = new_state
        self.day += 1
        terminated = self.day >= self.max_days

        info = {
            'day': self.day,
            'current_yield': float(self.current_yield),
            'action_name': act['name'],
            'fert_kg': act['fert_kg'],
            'water': act['water'],
            'cumulative_growth': float(self.cumulative_growth),
            'water_stress': float(water_stress),
            'nutrient_stress': float(nutrient_stress),
            'total_fert_used': self.total_fert_used,
            'health': float(new_health),
            'toxicity': float(new_toxicity),
            'moisture': float(new_moisture),
        }

        return self.state.copy(), float(reward), terminated, False, info

    def _compute_reward(self, action, state, act):
        health = state[7]
        toxicity = state[8]
        moisture = state[1]
        nutrient = state[2]

        r = 0.0
        r += (health - 50) * 1.0

        # Toxicity only hurts when clearly high
        if toxicity > 14:
            r -= (toxicity - 14) * 2.5
        elif toxicity > 9:
            r -= (toxicity - 9) * 0.9

        # Moisture & nutrient bands
        if 38 <= moisture <= 78:
            r += 6.0
        if 55 <= nutrient <= 130:
            r += 16.0   # nutrient in good range is highly valuable
        if nutrient < 48:
            r -= (48 - nutrient) * 0.55  # push fertilize when low

        r -= act['cost'] * 1.2

        if health < 15:
            r -= 50.0

        if self.day >= self.max_days - 1:
            # Yield bonus
            y_norm = (self.current_yield - 2000) / 4500.0
            r += max(0.0, y_norm) * 200.0
            # Season total fert aligned to real-data median (~137 kg/ha)
            # Reward being in the realistic band [70, 260]
            tf = self.total_fert_used
            if 70 <= tf <= 260:
                r += 45.0
            elif 40 <= tf < 70 or 260 < tf <= 350:
                r += 15.0
            elif tf == 0:
                r -= 20.0  # no fertilizer at all is bad vs real data

        return r

    def render(self):
        print(f"Day {self.day:2d} | Yield: {self.current_yield:7.0f} | Growth: {self.cumulative_growth:.2f}x | "
              f"Health: {self.state[7]:5.1f} | Tox: {self.state[8]:4.1f} | Fert used: {self.total_fert_used:.0f}kg")
