import numpy as np

class CurriculumScheduler:
    """Manages curriculum learning difficulty progression with fixed thresholds"""

    def __init__(self, levels=10, fixed_thresholds=True):
        self.levels = levels
        self.current_level = 0
        self.episode_count = 0
        self.fixed_thresholds = fixed_thresholds
        self.level_history = []

        # Fixed performance thresholds for level progression
        self.performance_thresholds = {
            0: 0.3,   # Level 0 -> 1: 30% success
            1: 0.4,   # Level 1 -> 2: 40% success
            2: 0.5,   # Level 2 -> 3: 50% success
            3: 0.55,  # Level 3 -> 4: 55% success
            4: 0.6,   # Level 4 -> 5: 60% success
            5: 0.65,  # Level 5 -> 6: 65% success
            6: 0.7,   # Level 6 -> 7: 70% success
            7: 0.75,  # Level 7 -> 8: 75% success
            8: 0.8,   # Level 8 -> 9: 80% success
            9: 0.85,  # Level 9 -> 10: 85% success
        }

        # Window for averaging performance
        self.performance_window = []
        self.window_size = 100  # Average over last 100 episodes

    def update(self, reward, info=None):
        """Update curriculum based on episode performance"""
        self.episode_count += 1

        # Add to performance window (normalize reward to [0, 1])
        normalized_reward = (reward + 1000) / 2000  # Assuming rewards in [-1000, 1000]
        self.performance_window.append(normalized_reward)

        # Keep window size limited
        if len(self.performance_window) > self.window_size:
            self.performance_window.pop(0)

        # Check for level progression every 50 episodes
        if self.episode_count % 50 == 0 and len(self.performance_window) >= 50:
            avg_performance = np.mean(self.performance_window[-50:])

            # Check if should progress to next level
            if self.current_level < self.levels - 1:
                threshold = self.performance_thresholds.get(self.current_level, 0.8)
                if avg_performance >= threshold:
                    self.current_level += 1
                    print(f"📈 Curriculum advanced to level {self.current_level} (performance: {avg_performance:.2%})")
                    self.level_history.append({
                        'episode': self.episode_count,
                        'level': self.current_level,
                        'performance': avg_performance
                    })

    def get_difficulty_params(self):
        """Get current difficulty parameters based on level"""
        # Map level to difficulty parameters
        difficulty_scale = self.current_level / (self.levels - 1)

        params = {
            'storm_intensity': 0.3 + 0.7 * difficulty_scale,  # 0.3 to 1.0
            'storm_radius': 0.2 + 0.3 * difficulty_scale,     # 0.2 to 0.5
            'storm_speed': 0.5 + 0.5 * difficulty_scale,      # 0.5 to 1.0
            'num_initial_hazards': 1 + int(2 * difficulty_scale),  # 1 to 3
            'population_density': 0.5 + 0.5 * difficulty_scale,    # 0.5 to 1.0
            'infrastructure_vulnerability': 0.3 + 0.4 * difficulty_scale  # 0.3 to 0.7
        }

        return params

    def reset(self):
        """Reset scheduler to initial state"""
        self.current_level = 0
        self.episode_count = 0
        self.performance_window = []
        self.level_history = []

    def get_stats(self):
        """Get curriculum statistics"""
        return {
            'current_level': self.current_level,
            'total_episodes': self.episode_count,
            'recent_performance': np.mean(self.performance_window[-10:]) if self.performance_window else 0,
            'level_history': self.level_history
        }