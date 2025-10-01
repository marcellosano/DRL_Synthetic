import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .base_environment import BaseEnvironment

@dataclass
class FOZZone:
    """Represents a Flood Operational Zone"""
    id: int
    name: str
    centroid: Tuple[float, float]
    area: float
    population: int
    elevation: float
    vulnerability: float
    current_water_level: float = 0.0
    is_evacuated: bool = False
    has_protection: bool = False

class FOZEnvironment(BaseEnvironment):
    """Flood Operational Zone environment for real-world locations"""

    def __init__(self, config_path: str):
        """Initialize FOZ environment from configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        super().__init__(config)

        self.location = config['location']['name']
        self.zones = {}
        self.zone_connectivity = None
        self.current_hazard = None
        self.time_step = 0
        self.resources = 100

        # Load zone definitions
        self._load_zones(config)

        # Load action definitions
        self.actions = config['actions']['available']
        self.action_map = {a['id']: a for a in self.actions}

        # Metrics tracking
        self.cumulative_damage = 0
        self.cumulative_lives_lost = 0
        self.actions_taken = []

    def _load_zones(self, config):
        """Load zone definitions from configuration"""
        # This would load from actual CSV files in production
        # For now, create synthetic zones
        num_zones = config['zones']['count']

        for i in range(num_zones):
            zone = FOZZone(
                id=i,
                name=f"Zone_{i}",
                centroid=(np.random.uniform(0, 100), np.random.uniform(0, 100)),
                area=np.random.uniform(0.5, 5.0),  # km²
                population=np.random.randint(100, 5000),
                elevation=np.random.uniform(0, 10),  # meters
                vulnerability=np.random.uniform(0.3, 0.9)
            )
            self.zones[i] = zone

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        super().reset()

        # Reset all zones
        for zone in self.zones.values():
            zone.current_water_level = 0.0
            zone.is_evacuated = False
            zone.has_protection = False

        # Reset metrics
        self.cumulative_damage = 0
        self.cumulative_lives_lost = 0
        self.actions_taken = []
        self.resources = 100

        # Generate new hazard scenario
        self._generate_hazard()

        return self._get_state()

    def _generate_hazard(self):
        """Generate a hazard scenario (storm surge, flooding, etc.)"""
        # Simple hazard model - would use historical data in production
        hazard_intensity = np.random.uniform(0.5, 2.0)  # meters
        affected_zones = np.random.choice(
            list(self.zones.keys()),
            size=min(10, len(self.zones) // 3),
            replace=False
        )

        self.current_hazard = {
            'intensity': hazard_intensity,
            'affected_zones': affected_zones,
            'arrival_time': np.random.randint(3, 10)  # time steps
        }

    def _get_state(self) -> np.ndarray:
        """Get current state as numpy array"""
        state_features = []

        # Time features
        state_features.append(self.time_step / self.max_steps)
        state_features.append(self.resources / 100)

        # Hazard features
        if self.current_hazard:
            state_features.append(self.current_hazard['intensity'] / 3.0)
            state_features.append(self.current_hazard['arrival_time'] / 10.0)
        else:
            state_features.extend([0, 0])

        # Zone features (sorted by vulnerability)
        sorted_zones = sorted(self.zones.values(),
                            key=lambda z: z.vulnerability * (1 + z.current_water_level),
                            reverse=True)

        # Add features for top N zones
        max_zones = 20  # Fixed size for neural network
        for i in range(max_zones):
            if i < len(sorted_zones):
                zone = sorted_zones[i]
                state_features.extend([
                    zone.centroid[0] / 100,  # Normalized x
                    zone.centroid[1] / 100,  # Normalized y
                    zone.population / 5000,  # Normalized population
                    zone.elevation / 10,  # Normalized elevation
                    zone.vulnerability,
                    zone.current_water_level / 3.0,  # Normalized water level
                    float(zone.is_evacuated),
                    float(zone.has_protection)
                ])
            else:
                state_features.extend([0] * 8)  # Padding

        self.state = np.array(state_features, dtype=np.float32)
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return results"""
        self.step_count += 1
        self.time_step += 1

        # Apply action
        action_info = self._apply_action(action)

        # Update hazard progression
        self._update_hazard()

        # Calculate impacts
        damage, lives_lost = self._calculate_impacts()

        # Update cumulative metrics
        self.cumulative_damage += damage
        self.cumulative_lives_lost += lives_lost

        # Calculate reward
        reward = self._calculate_reward(action, damage, lives_lost)

        # Check if done
        self.done = (self.time_step >= self.max_steps) or (self.resources <= 0)

        # Get new state
        new_state = self._get_state()

        # Compile info
        info = {
            'action_taken': action,
            'action_cost': action_info['cost'],
            'damage': damage,
            'lives_lost': lives_lost,
            'cumulative_damage': self.cumulative_damage,
            'cumulative_lives_lost': self.cumulative_lives_lost,
            'resources_remaining': self.resources
        }

        return new_state, reward, self.done, info

    def _apply_action(self, action: int) -> Dict:
        """Apply the selected action"""
        if action not in self.action_map:
            return {'cost': 0, 'success': False}

        action_def = self.action_map[action]
        cost = action_def['cost']

        # Check if we have enough resources
        if cost > self.resources:
            return {'cost': 0, 'success': False}

        # Deduct cost
        self.resources -= cost
        self.actions_taken.append(action)

        # Apply action effects
        if action_def['name'] == 'evacuate_zone':
            # Evacuate highest risk zones
            high_risk_zones = sorted(
                self.zones.values(),
                key=lambda z: z.vulnerability * (1 + z.current_water_level),
                reverse=True
            )[:3]
            for zone in high_risk_zones:
                zone.is_evacuated = True

        elif action_def['name'] in ['sandbag_deployment', 'activate_flood_gates']:
            # Protect vulnerable zones
            vulnerable_zones = sorted(
                self.zones.values(),
                key=lambda z: z.vulnerability,
                reverse=True
            )[:5]
            for zone in vulnerable_zones:
                zone.has_protection = True

        return {'cost': cost, 'success': True}

    def _update_hazard(self):
        """Update hazard progression"""
        if not self.current_hazard:
            return

        # Decrease arrival time
        self.current_hazard['arrival_time'] -= 1

        # If hazard has arrived, update water levels
        if self.current_hazard['arrival_time'] <= 0:
            for zone_id in self.current_hazard['affected_zones']:
                zone = self.zones[zone_id]

                # Calculate water level increase
                base_increase = self.current_hazard['intensity']

                # Modify based on elevation and protection
                elevation_factor = max(0, 1 - zone.elevation / 5)
                protection_factor = 0.5 if zone.has_protection else 1.0

                zone.current_water_level += base_increase * elevation_factor * protection_factor
                zone.current_water_level = min(zone.current_water_level, 3.0)  # Cap at 3m

    def _calculate_impacts(self) -> Tuple[float, int]:
        """Calculate damage and lives lost"""
        total_damage = 0
        total_lives_lost = 0

        for zone in self.zones.values():
            if zone.current_water_level > 0:
                # Damage calculation
                damage_factor = min(zone.current_water_level / 2.0, 1.0)
                zone_damage = damage_factor * zone.vulnerability * zone.area * 1000  # Economic units
                total_damage += zone_damage

                # Lives lost calculation
                if not zone.is_evacuated and zone.current_water_level > 0.5:
                    risk_factor = min((zone.current_water_level - 0.5) / 2.0, 1.0)
                    lives_at_risk = int(zone.population * risk_factor * zone.vulnerability * 0.01)
                    total_lives_lost += lives_at_risk

        return total_damage, total_lives_lost

    def _calculate_reward(self, action: int, damage: float, lives_lost: int) -> float:
        """Calculate reward for the action taken"""
        # Base reward components
        life_penalty = -lives_lost * 100  # Heavy penalty for lives lost
        damage_penalty = -damage * 0.01  # Economic damage penalty

        # Action cost penalty (efficiency)
        action_def = self.action_map.get(action, {'cost': 0})
        cost_penalty = -action_def['cost'] * 0.5

        # Bonus for successful protection
        protection_bonus = 0
        if lives_lost == 0 and damage < 100:
            protection_bonus = 50

        # Time bonus for early action
        time_bonus = max(0, (self.max_steps - self.time_step) * 2)

        reward = life_penalty + damage_penalty + cost_penalty + protection_bonus + time_bonus

        return reward

    def get_valid_actions(self) -> np.ndarray:
        """Return mask of valid actions based on resources"""
        mask = np.ones(len(self.actions), dtype=bool)

        for i, action in enumerate(self.actions):
            if action['cost'] > self.resources:
                mask[i] = False

        # Always allow do nothing
        mask[0] = True

        return mask

    def render(self, mode: str = 'human'):
        """Render the environment (visualization)"""
        if mode == 'human':
            print(f"\n--- Time Step: {self.time_step}/{self.max_steps} ---")
            print(f"Location: {self.location}")
            print(f"Resources: {self.resources}")
            print(f"Cumulative Lives Lost: {self.cumulative_lives_lost}")
            print(f"Cumulative Damage: ${self.cumulative_damage:.0f}")

            if self.current_hazard:
                print(f"Hazard Intensity: {self.current_hazard['intensity']:.1f}m")
                print(f"Arrival Time: {max(0, self.current_hazard['arrival_time'])} steps")

            # Show top 5 at-risk zones
            at_risk = sorted(
                self.zones.values(),
                key=lambda z: z.current_water_level * z.population,
                reverse=True
            )[:5]

            if at_risk[0].current_water_level > 0:
                print("\nTop At-Risk Zones:")
                for zone in at_risk:
                    if zone.current_water_level > 0:
                        status = "🚁 Evacuated" if zone.is_evacuated else "⚠️  At Risk"
                        print(f"  {zone.name}: {zone.current_water_level:.1f}m water, "
                              f"{zone.population} people - {status}")

    @property
    def state_dim(self) -> int:
        """Return state dimension"""
        # 4 global features + 20 zones * 8 features each
        return 4 + 20 * 8

    @property
    def action_dim(self) -> int:
        """Return action dimension"""
        return len(self.actions)