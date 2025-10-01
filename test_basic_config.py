"""
Simple test of configuration system without numpy dependency.
"""

import sys
import os
sys.path.append('utils')

from config_manager import ConfigManager


def test_config_system():
    """Test the basic configuration loading and management"""
    print("Testing DRL Configuration System")
    print("=" * 50)

    manager = ConfigManager()

    # Test 1: Load base configuration
    print("\n1. Loading base configuration...")
    try:
        base_config = manager.load_config("base.yaml")
        print(f"✓ Loaded base config: {base_config['experiment']['name']}")
        print(f"✓ Learning rate: {base_config['training']['learning_rate']}")
        print(f"✓ Grid size: {base_config['environment']['grid_size']}")
        print(f"✓ Reward weight: {base_config['reward']['lives_saved_weight']}")
    except Exception as e:
        print(f"✗ Failed to load base config: {e}")
        return

    # Test 2: Test inheritance
    print("\n2. Testing configuration inheritance...")
    try:
        quick_config = manager.load_config("experiments/quick_test.yaml")
        print(f"✓ Inherited experiment: {quick_config['experiment']['name']}")
        print(f"✓ Base LR (inherited): {quick_config['training']['learning_rate']}")
        print(f"✓ Override episodes: {quick_config['training']['episodes']}")
        print(f"✓ Override grid size: {quick_config['environment']['grid_size']}")
    except Exception as e:
        print(f"✗ Failed inheritance test: {e}")

    # Test 3: Parameter overrides
    print("\n3. Testing parameter overrides...")
    try:
        overrides = {
            'training.learning_rate': 0.001,
            'reward.lives_saved_weight': 20.0,
            'experiment.name': 'override_test'
        }
        override_config = manager.load_config("base.yaml", overrides)
        print(f"✓ Override LR: {override_config['training']['learning_rate']}")
        print(f"✓ Override reward: {override_config['reward']['lives_saved_weight']}")
        print(f"✓ Override name: {override_config['experiment']['name']}")
    except Exception as e:
        print(f"✗ Failed override test: {e}")

    # Test 4: Experiment tracking
    print("\n4. Testing experiment tracking...")
    try:
        run1 = manager.create_experiment_run("base.yaml")
        print(f"✓ Created experiment: {run1.name}")
        print(f"✓ Run status: {run1.status}")
        print(f"✓ Config saved to: runs/{run1.name}/config.yaml")

        # Test metrics updating
        fake_metrics = {
            'episode_reward': 150.0,
            'lives_saved': 25
        }
        manager.update_run_metrics(run1.name, fake_metrics)
        print(f"✓ Updated metrics: {len(run1.metrics)} metric types")

        manager.complete_run(run1.name)
        print(f"✓ Completed run: {run1.status}")

    except Exception as e:
        print(f"✗ Failed tracking test: {e}")

    # Test 5: Configuration suggestions
    print("\n5. Testing parameter suggestions...")
    try:
        suggestions = manager.get_parameter_suggestions("base.yaml", "training.learning_rate")
        print(f"✓ LR suggestions: {suggestions}")

        batch_suggestions = manager.get_parameter_suggestions("base.yaml", "training.batch_size")
        print(f"✓ Batch size suggestions: {batch_suggestions}")
    except Exception as e:
        print(f"✗ Failed suggestions test: {e}")

    print("\n" + "=" * 50)
    print("Configuration system test completed!")

    print("\nHow to use the system:")
    print("1. Edit config/base.yaml for default settings")
    print("2. Create experiment configs in config/experiments/")
    print("3. Override parameters: config.load_config('base.yaml', {'training.learning_rate': 0.001})")
    print("4. Track experiments: manager.create_experiment_run('config.yaml')")


if __name__ == "__main__":
    test_config_system()