"""
Parameter optimization utilities for hyperparameter tuning and reward function optimization.
Supports grid search, random search, and Bayesian optimization.
"""

import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple
import itertools
import random
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ParameterRange:
    """Define a parameter range for optimization"""
    name: str
    values: List[Any] = None
    min_val: float = None
    max_val: float = None
    param_type: str = "discrete"  # discrete, continuous, integer

    def sample(self, n: int = 1) -> List[Any]:
        """Sample n values from this parameter range"""
        if self.param_type == "discrete":
            return random.choices(self.values, k=n)
        elif self.param_type == "continuous":
            return [random.uniform(self.min_val, self.max_val) for _ in range(n)]
        elif self.param_type == "integer":
            return [random.randint(int(self.min_val), int(self.max_val)) for _ in range(n)]


class ParameterOptimizer:
    """Optimize hyperparameters and reward function parameters"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.optimization_history = []

    def grid_search(self, base_config: str, parameter_ranges: Dict[str, ParameterRange],
                   max_combinations: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for grid search

        Args:
            base_config: Base configuration file path
            parameter_ranges: Dictionary of parameter ranges
            max_combinations: Limit number of combinations (randomly sampled if exceeded)

        Returns:
            List of parameter override dictionaries
        """
        # Generate all possible combinations
        param_names = list(parameter_ranges.keys())
        param_values = []

        for param_name in param_names:
            param_range = parameter_ranges[param_name]
            if param_range.param_type == "discrete":
                param_values.append(param_range.values)
            else:
                # For continuous/integer, use a reasonable number of samples
                samples = np.linspace(param_range.min_val, param_range.max_val, 5)
                if param_range.param_type == "integer":
                    samples = [int(x) for x in samples]
                param_values.append(samples.tolist())

        # Generate all combinations
        all_combinations = list(itertools.product(*param_values))

        # Limit combinations if needed
        if max_combinations and len(all_combinations) > max_combinations:
            all_combinations = random.sample(all_combinations, max_combinations)

        # Convert to override dictionaries
        parameter_sets = []
        for combination in all_combinations:
            overrides = dict(zip(param_names, combination))
            parameter_sets.append(overrides)

        return parameter_sets

    def random_search(self, base_config: str, parameter_ranges: Dict[str, ParameterRange],
                     num_samples: int = 50) -> List[Dict[str, Any]]:
        """
        Generate random parameter combinations

        Args:
            base_config: Base configuration file path
            parameter_ranges: Dictionary of parameter ranges
            num_samples: Number of random samples to generate

        Returns:
            List of parameter override dictionaries
        """
        parameter_sets = []

        for _ in range(num_samples):
            overrides = {}
            for param_name, param_range in parameter_ranges.items():
                sampled_value = param_range.sample(1)[0]
                overrides[param_name] = sampled_value
            parameter_sets.append(overrides)

        return parameter_sets

    def suggest_reward_parameters(self) -> Dict[str, ParameterRange]:
        """Suggest parameter ranges for reward function optimization"""
        return {
            'reward.lives_saved_weight': ParameterRange(
                name='lives_saved_weight',
                min_val=5.0, max_val=25.0,
                param_type='continuous'
            ),
            'reward.economic_cost_weight': ParameterRange(
                name='economic_cost_weight',
                min_val=-1.0, max_val=-0.01,
                param_type='continuous'
            ),
            'reward.early_warning_bonus': ParameterRange(
                name='early_warning_bonus',
                min_val=1.0, max_val=15.0,
                param_type='continuous'
            ),
            'reward.false_alarm_penalty': ParameterRange(
                name='false_alarm_penalty',
                min_val=-10.0, max_val=-0.5,
                param_type='continuous'
            ),
            'reward.time_penalty_factor': ParameterRange(
                name='time_penalty_factor',
                min_val=-0.5, max_val=-0.01,
                param_type='continuous'
            ),
            'reward.objective_weights.safety': ParameterRange(
                name='safety_weight',
                values=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                param_type='discrete'
            )
        }

    def suggest_training_parameters(self) -> Dict[str, ParameterRange]:
        """Suggest parameter ranges for training hyperparameters"""
        return {
            'training.learning_rate': ParameterRange(
                name='learning_rate',
                values=[0.0001, 0.0003, 0.001, 0.003, 0.01],
                param_type='discrete'
            ),
            'training.batch_size': ParameterRange(
                name='batch_size',
                values=[16, 32, 64, 128, 256],
                param_type='discrete'
            ),
            'training.clip_epsilon': ParameterRange(
                name='clip_epsilon',
                min_val=0.05, max_val=0.4,
                param_type='continuous'
            ),
            'training.entropy_coef': ParameterRange(
                name='entropy_coef',
                min_val=0.001, max_val=0.2,
                param_type='continuous'
            ),
            'training.gamma': ParameterRange(
                name='gamma',
                values=[0.9, 0.95, 0.99, 0.995, 0.999],
                param_type='discrete'
            ),
            'network.attention_heads': ParameterRange(
                name='attention_heads',
                values=[2, 4, 8, 12, 16],
                param_type='discrete'
            )
        }

    def create_optimization_campaign(self, campaign_name: str, base_config: str,
                                   parameter_ranges: Dict[str, ParameterRange],
                                   method: str = "grid", max_runs: int = 50) -> List[str]:
        """
        Create a complete optimization campaign

        Args:
            campaign_name: Name for this optimization campaign
            base_config: Base configuration file
            parameter_ranges: Parameters to optimize
            method: Optimization method (grid, random)
            max_runs: Maximum number of runs

        Returns:
            List of experiment run names
        """
        # Generate parameter sets
        if method == "grid":
            parameter_sets = self.grid_search(base_config, parameter_ranges, max_runs)
        elif method == "random":
            parameter_sets = self.random_search(base_config, parameter_ranges, max_runs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Create experiment runs
        run_names = []
        for i, overrides in enumerate(parameter_sets):
            # Add campaign identifier to experiment name
            overrides['experiment.name'] = f"{campaign_name}_run_{i:03d}"

            # Create experiment run
            experiment_run = self.config_manager.create_experiment_run(base_config, overrides)
            run_names.append(experiment_run.name)

        # Save campaign info
        campaign_info = {
            'name': campaign_name,
            'base_config': base_config,
            'method': method,
            'parameter_ranges': {name: {
                'values': range_obj.values,
                'min_val': range_obj.min_val,
                'max_val': range_obj.max_val,
                'param_type': range_obj.param_type
            } for name, range_obj in parameter_ranges.items()},
            'runs': run_names,
            'total_runs': len(run_names)
        }

        campaign_dir = Path("campaigns")
        campaign_dir.mkdir(exist_ok=True)
        with open(campaign_dir / f"{campaign_name}.json", 'w') as f:
            json.dump(campaign_info, f, indent=2)

        return run_names

    def analyze_campaign_results(self, campaign_name: str,
                                metric: str = "episode_reward") -> Dict[str, Any]:
        """Analyze results from an optimization campaign"""
        # Load campaign info
        campaign_file = Path("campaigns") / f"{campaign_name}.json"
        if not campaign_file.exists():
            raise FileNotFoundError(f"Campaign not found: {campaign_name}")

        with open(campaign_file, 'r') as f:
            campaign_info = json.load(f)

        # Collect results from all runs
        results = []
        for run_name in campaign_info['runs']:
            run = next((r for r in self.config_manager.experiment_runs
                       if r.name == run_name), None)
            if run and metric in run.metrics and run.metrics[metric]:
                # Get final performance
                final_score = run.metrics[metric][-1]
                # Get average performance over last 10% of training
                recent_scores = run.metrics[metric][-max(1, len(run.metrics[metric])//10):]
                avg_score = sum(recent_scores) / len(recent_scores)

                # Extract parameter values from config
                param_values = {}
                for param_name in campaign_info['parameter_ranges'].keys():
                    keys = param_name.split('.')
                    value = run.config
                    for key in keys:
                        value = value[key]
                    param_values[param_name] = value

                results.append({
                    'run_name': run_name,
                    'final_score': final_score,
                    'avg_score': avg_score,
                    'parameters': param_values
                })

        if not results:
            return {'error': 'No completed runs found'}

        # Find best configurations
        best_final = max(results, key=lambda x: x['final_score'])
        best_avg = max(results, key=lambda x: x['avg_score'])

        # Parameter impact analysis
        param_impact = {}
        for param_name in campaign_info['parameter_ranges'].keys():
            # Group results by parameter value
            param_groups = {}
            for result in results:
                param_val = result['parameters'][param_name]
                if param_val not in param_groups:
                    param_groups[param_val] = []
                param_groups[param_val].append(result['avg_score'])

            # Calculate average performance for each parameter value
            param_averages = {val: sum(scores)/len(scores)
                            for val, scores in param_groups.items()}

            param_impact[param_name] = {
                'values': param_averages,
                'best_value': max(param_averages.items(), key=lambda x: x[1]),
                'worst_value': min(param_averages.items(), key=lambda x: x[1]),
                'range': max(param_averages.values()) - min(param_averages.values())
            }

        return {
            'campaign': campaign_name,
            'total_runs': len(results),
            'metric': metric,
            'best_final_run': best_final,
            'best_average_run': best_avg,
            'parameter_impact': param_impact,
            'all_results': results
        }


# Example usage functions
def create_reward_optimization_campaign(config_manager, campaign_name: str = "reward_opt"):
    """Create a campaign focused on reward function optimization"""
    optimizer = ParameterOptimizer(config_manager)
    reward_ranges = optimizer.suggest_reward_parameters()

    return optimizer.create_optimization_campaign(
        campaign_name=campaign_name,
        base_config="base.yaml",
        parameter_ranges=reward_ranges,
        method="random",
        max_runs=30
    )


def create_training_optimization_campaign(config_manager, campaign_name: str = "training_opt"):
    """Create a campaign focused on training hyperparameters"""
    optimizer = ParameterOptimizer(config_manager)
    training_ranges = optimizer.suggest_training_parameters()

    return optimizer.create_optimization_campaign(
        campaign_name=campaign_name,
        base_config="base.yaml",
        parameter_ranges=training_ranges,
        method="grid",
        max_runs=50
    )


if __name__ == "__main__":
    from config_manager import ConfigManager

    # Example usage
    config_manager = ConfigManager()
    optimizer = ParameterOptimizer(config_manager)

    # Create reward optimization campaign
    print("Creating reward optimization campaign...")
    reward_ranges = optimizer.suggest_reward_parameters()
    runs = optimizer.create_optimization_campaign(
        "test_reward_opt",
        "base.yaml",
        reward_ranges,
        method="random",
        max_runs=10
    )

    print(f"Created {len(runs)} experiment runs")
    print("Run names:", runs[:3], "...")  # Show first 3