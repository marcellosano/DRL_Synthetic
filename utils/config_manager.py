"""
Configuration management system for DRL coastal environment.
Supports YAML configuration files with inheritance and parameter overrides.
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import copy
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class ExperimentRun:
    """Track individual experiment runs"""
    name: str
    config: Dict[str, Any]
    start_time: datetime
    metrics: Dict[str, List[float]]
    status: str = "running"  # running, completed, failed


class ConfigManager:
    """Manages configuration loading, inheritance, and parameter overrides"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.experiments_dir = self.config_dir / "experiments"
        self.runs_dir = Path("runs")
        self.runs_dir.mkdir(exist_ok=True)

        # Track experiment runs
        self.experiment_runs: List[ExperimentRun] = []

    def load_config(self, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration with inheritance support and parameter overrides.

        Args:
            config_path: Path to the config file (relative to config_dir)
            overrides: Dictionary of parameter overrides

        Returns:
            Complete configuration dictionary
        """
        # Load the main config file
        full_path = self.config_dir / config_path
        if not full_path.exists():
            raise FileNotFoundError(f"Config file not found: {full_path}")

        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)

        # Handle inheritance
        if 'extends' in config:
            parent_path = config['extends']
            # Resolve relative paths
            if not os.path.isabs(parent_path):
                parent_path = (full_path.parent / parent_path).resolve()
                # Make relative to config_dir, handling both absolute and relative cases
                try:
                    parent_path = parent_path.relative_to(self.config_dir.resolve())
                except ValueError:
                    # If already relative or outside config_dir, use as-is
                    parent_path = Path(parent_path).relative_to(Path.cwd() / self.config_dir)

            parent_config = self.load_config(str(parent_path))
            config = self._merge_configs(parent_config, config)
            # Remove the 'extends' key from final config
            config.pop('extends', None)

        # Apply parameter overrides
        if overrides:
            config = self._apply_overrides(config, overrides)

        # Validate configuration
        self._validate_config(config)

        return config

    def _merge_configs(self, parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge child config into parent config"""
        merged = copy.deepcopy(parent)

        for key, value in child.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def _apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameter overrides using dot notation (e.g., 'training.learning_rate': 0.001)"""
        config = copy.deepcopy(config)

        for key, value in overrides.items():
            self._set_nested_value(config, key, value)

        return config

    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set a nested dictionary value using dot notation"""
        keys = key.split('.')
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration values"""
        # Check required sections
        required_sections = ['experiment', 'training', 'environment', 'reward']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate value ranges
        training = config.get('training', {})
        if 'learning_rate' in training and not (0 < training['learning_rate'] <= 1):
            raise ValueError("Learning rate must be between 0 and 1")

        if 'batch_size' in training and training['batch_size'] <= 0:
            raise ValueError("Batch size must be positive")

        # Validate reward weights sum to reasonable total
        reward = config.get('reward', {})
        if 'objective_weights' in reward:
            weights = reward['objective_weights']
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                raise ValueError(f"Objective weights must sum to 1.0, got {total}")

    def create_experiment_run(self, config_path: str, overrides: Optional[Dict[str, Any]] = None) -> ExperimentRun:
        """Create and track a new experiment run"""
        config = self.load_config(config_path, overrides)

        # Generate unique run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = config['experiment']['name']
        run_name = f"{base_name}_{timestamp}"

        # Create experiment run
        experiment_run = ExperimentRun(
            name=run_name,
            config=config,
            start_time=datetime.now(),
            metrics={}
        )

        self.experiment_runs.append(experiment_run)

        # Save run configuration
        run_dir = self.runs_dir / run_name
        run_dir.mkdir(exist_ok=True)

        with open(run_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        return experiment_run

    def update_run_metrics(self, run_name: str, metrics: Dict[str, float]):
        """Update metrics for a running experiment"""
        for run in self.experiment_runs:
            if run.name == run_name:
                for metric_name, value in metrics.items():
                    if metric_name not in run.metrics:
                        run.metrics[metric_name] = []
                    run.metrics[metric_name].append(value)
                break

    def complete_run(self, run_name: str, final_metrics: Optional[Dict[str, float]] = None):
        """Mark an experiment run as completed"""
        for run in self.experiment_runs:
            if run.name == run_name:
                run.status = "completed"
                if final_metrics:
                    self.update_run_metrics(run_name, final_metrics)

                # Save final metrics
                run_dir = self.runs_dir / run_name
                with open(run_dir / "metrics.json", 'w') as f:
                    json.dump(run.metrics, f, indent=2)
                break

    def get_parameter_suggestions(self, base_config_path: str,
                                 parameter_name: str,
                                 num_suggestions: int = 5) -> List[Any]:
        """Get parameter suggestions based on typical ranges"""
        suggestions_map = {
            'training.learning_rate': [0.0001, 0.0003, 0.001, 0.003, 0.01],
            'training.batch_size': [16, 32, 64, 128, 256],
            'training.clip_epsilon': [0.1, 0.15, 0.2, 0.25, 0.3],
            'training.entropy_coef': [0.001, 0.01, 0.05, 0.1, 0.2],
            'network.attention_heads': [2, 4, 8, 12, 16],
            'reward.lives_saved_weight': [5.0, 10.0, 15.0, 20.0, 25.0],
            'reward.early_warning_bonus': [1.0, 3.0, 5.0, 7.0, 10.0],
            'environment.max_storms': [1, 2, 3, 4, 5]
        }

        return suggestions_map.get(parameter_name, [])[:num_suggestions]

    def compare_runs(self, run_names: List[str], metric: str = "episode_reward") -> Dict[str, Any]:
        """Compare multiple experiment runs on a specific metric"""
        comparison = {
            'runs': {},
            'metric': metric,
            'summary': {}
        }

        for run_name in run_names:
            run = next((r for r in self.experiment_runs if r.name == run_name), None)
            if run and metric in run.metrics:
                values = run.metrics[metric]
                comparison['runs'][run_name] = {
                    'values': values,
                    'mean': sum(values) / len(values),
                    'max': max(values),
                    'min': min(values),
                    'final': values[-1] if values else None,
                    'config': run.config
                }

        # Generate summary
        if comparison['runs']:
            best_run = max(comparison['runs'].items(),
                          key=lambda x: x[1]['mean'])
            comparison['summary']['best_run'] = best_run[0]
            comparison['summary']['best_score'] = best_run[1]['mean']

        return comparison


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to load configuration"""
    manager = ConfigManager()
    return manager.load_config(config_path, overrides)


if __name__ == "__main__":
    # Example usage
    manager = ConfigManager()

    # Load base configuration
    print("Loading base configuration...")
    base_config = manager.load_config("base.yaml")
    print(f"Base experiment: {base_config['experiment']['name']}")

    # Load with overrides
    print("\nLoading with overrides...")
    overrides = {
        'training.learning_rate': 0.001,
        'reward.lives_saved_weight': 15.0
    }
    config = manager.load_config("base.yaml", overrides)
    print(f"LR: {config['training']['learning_rate']}")
    print(f"Lives saved weight: {config['reward']['lives_saved_weight']}")

    # Test inheritance
    print("\nTesting inheritance...")
    quick_config = manager.load_config("experiments/quick_test.yaml")
    print(f"Quick test episodes: {quick_config['training']['episodes']}")
    print(f"Grid size: {quick_config['environment']['grid_size']}")