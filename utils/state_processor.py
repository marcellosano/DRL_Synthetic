import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class StateProcessor:
    """Process and transform environment states for neural networks"""

    def __init__(self, grid_size=20, device='cpu'):
        self.grid_size = grid_size
        self.device = device
        self.scaler = StandardScaler()
        self.clusterer = None

    def process_state(self, state):
        """Convert environment state to neural network input"""
        # Handle different state formats
        if isinstance(state, dict):
            return self._process_dict_state(state)
        elif isinstance(state, np.ndarray):
            return torch.FloatTensor(state).to(self.device)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

    def _process_dict_state(self, state):
        """Process dictionary state from environment"""
        features = []

        # Grid features
        if 'grid' in state:
            grid_features = state['grid'].flatten()
            features.extend(grid_features)

        # Storm features
        if 'storm' in state:
            storm = state['storm']
            features.extend([
                storm.get('x', 0) / self.grid_size,
                storm.get('y', 0) / self.grid_size,
                storm.get('radius', 0),
                storm.get('intensity', 0),
                storm.get('dx', 0),
                storm.get('dy', 0)
            ])

        # House features
        if 'houses' in state:
            house_features = self._process_houses(state['houses'])
            features.extend(house_features)

        # Time features
        if 'time_step' in state:
            features.append(state['time_step'] / 24)  # Normalize to [0,1]

        # Resources
        if 'resources' in state:
            features.append(state['resources'] / 100)  # Normalize

        return torch.FloatTensor(features).to(self.device)

    def _process_houses(self, houses, max_houses=50):
        """Process house information into fixed-size feature vector"""
        features = []

        # Sort houses by risk level
        sorted_houses = sorted(houses.values(),
                              key=lambda h: h.get('flood_risk', 0),
                              reverse=True)[:max_houses]

        for i in range(max_houses):
            if i < len(sorted_houses):
                house = sorted_houses[i]
                features.extend([
                    house.get('x', 0) / self.grid_size,
                    house.get('y', 0) / self.grid_size,
                    house.get('flood_risk', 0),
                    house.get('evacuated', 0),
                    house.get('population', 0) / 100
                ])
            else:
                features.extend([0, 0, 0, 0, 0])  # Padding

        return features

    def extract_cluster_features(self, state):
        """Extract hazard cluster features using DBSCAN"""
        if not isinstance(state, dict) or 'hazards' not in state:
            return torch.zeros(10).to(self.device)  # Return empty features

        hazards = state['hazards']
        if not hazards:
            return torch.zeros(10).to(self.device)

        # Extract hazard positions
        positions = np.array([[h['x'], h['y']] for h in hazards])

        if len(positions) < 2:
            # Single hazard - return basic features
            h = hazards[0]
            features = [
                1,  # Number of clusters
                h['x'] / self.grid_size,
                h['y'] / self.grid_size,
                h.get('intensity', 0.5),
                0,  # No spread for single hazard
                0, 0, 0, 0, 0  # Padding
            ]
            return torch.FloatTensor(features).to(self.device)

        # Cluster hazards
        clustering = DBSCAN(eps=3.0, min_samples=2).fit(positions)
        labels = clustering.labels_

        # Calculate cluster statistics
        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        n_noise = list(labels).count(-1)

        cluster_features = []
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            cluster_mask = labels == label
            cluster_positions = positions[cluster_mask]
            cluster_hazards = [h for i, h in enumerate(hazards) if cluster_mask[i]]

            # Cluster center
            center = np.mean(cluster_positions, axis=0)

            # Cluster spread
            spread = np.std(cluster_positions)

            # Average intensity
            avg_intensity = np.mean([h.get('intensity', 0.5) for h in cluster_hazards])

            cluster_features.extend([
                center[0] / self.grid_size,
                center[1] / self.grid_size,
                avg_intensity,
                spread / self.grid_size
            ])

        # Pad or truncate to fixed size
        features = [n_clusters] + cluster_features[:9]
        while len(features) < 10:
            features.append(0)

        return torch.FloatTensor(features[:10]).to(self.device)

    def get_action_mask(self, state):
        """Generate action mask based on current state"""
        # Default: all actions are valid
        mask = np.ones(12, dtype=bool)

        if isinstance(state, dict):
            # Check resource constraints
            resources = state.get('resources', 100)

            # Action costs (example)
            action_costs = {
                0: 0,   # Do nothing
                1: 10,  # Evacuate high risk
                2: 20,  # Evacuate medium risk
                3: 30,  # Evacuate all
                4: 15,  # Deploy barriers
                5: 25,  # Emergency services
                6: 20,  # Shelter setup
                7: 50,  # Full emergency
                8: 10,  # Issue warning
                9: 5,   # Monitor
                10: 15, # Reinforce infrastructure
                11: 40  # Request aid
            }

            # Disable actions that cost more than available resources
            for action, cost in action_costs.items():
                if cost > resources:
                    mask[action] = False

            # Always allow do nothing and monitor
            mask[0] = True
            mask[9] = True

        return mask