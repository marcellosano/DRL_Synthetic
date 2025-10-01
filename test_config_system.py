"""
Test script for the configuration system.
Demonstrates how to use the config manager and parameter optimizer.
"""

import sys
import os
sys.path.append('utils')

from config_manager import ConfigManager
from parameter_optimizer import ParameterOptimizer, ParameterRange


def test_basic_config_loading():
    """Test basic configuration loading and inheritance"""
    print("=== Testing Basic Configuration Loading ===")

    manager = ConfigManager()

    # Test base config
    print("\n1. Loading base configuration...")
    base_config = manager.load_config("base.yaml")
    print(f"✓ Base experiment: {base_config['experiment']['name']}")
    print(f"✓ Learning rate: {base_config['training']['learning_rate']}")
    print(f"✓ Grid size: {base_config['environment']['grid_size']}")

    # Test inheritance
    print("\n2. Testing inheritance with quick_test.yaml...")
    quick_config = manager.load_config("experiments/quick_test.yaml")
    print(f"✓ Inherited LR: {quick_config['training']['learning_rate']}")  # From base
    print(f"✓ Overridden episodes: {quick_config['training']['episodes']}")  # Overridden
    print(f"✓ Overridden grid size: {quick_config['environment']['grid_size']}")

    # Test parameter overrides
    print("\n3. Testing parameter overrides...")
    overrides = {
        'training.learning_rate': 0.001,
        'reward.lives_saved_weight': 20.0,
        'experiment.name': 'test_override'
    }
    override_config = manager.load_config("base.yaml", overrides)
    print(f"✓ Overridden LR: {override_config['training']['learning_rate']}")
    print(f"✓ Overridden reward weight: {override_config['reward']['lives_saved_weight']}")
    print(f"✓ Overridden name: {override_config['experiment']['name']}")


def test_experiment_tracking():
    """Test experiment run creation and tracking"""
    print("\n=== Testing Experiment Tracking ===")

    manager = ConfigManager()

    # Create experiment runs
    print("\n1. Creating experiment runs...")
    run1 = manager.create_experiment_run("base.yaml")
    print(f"✓ Created run: {run1.name}")

    overrides = {'training.learning_rate': 0.001}
    run2 = manager.create_experiment_run("experiments/quick_test.yaml", overrides)
    print(f"✓ Created run: {run2.name}")

    # Update metrics
    print("\n2. Updating metrics...")
    fake_metrics = {
        'episode_reward': 150.0,
        'lives_saved': 25,
        'training_loss': 0.05
    }
    manager.update_run_metrics(run1.name, fake_metrics)
    print(f"✓ Updated metrics for {run1.name}")

    # Add more metrics over time
    for i in range(5):
        metrics = {
            'episode_reward': 150.0 + i * 10,
            'lives_saved': 25 + i * 2
        }
        manager.update_run_metrics(run1.name, metrics)

    print(f"✓ Run has {len(run1.metrics['episode_reward'])} metric points")

    # Complete run
    manager.complete_run(run1.name, {'final_score': 200.0})
    print(f"✓ Completed run: {run1.status}")


def test_parameter_optimization():
    """Test parameter optimization capabilities"""
    print("\n=== Testing Parameter Optimization ===")

    manager = ConfigManager()
    optimizer = ParameterOptimizer(manager)

    # Test parameter range creation
    print("\n1. Creating parameter ranges...")
    reward_ranges = optimizer.suggest_reward_parameters()
    print(f"✓ Created {len(reward_ranges)} reward parameter ranges")
    for name, param_range in list(reward_ranges.items())[:3]:
        print(f"  - {name}: {param_range.param_type}")

    training_ranges = optimizer.suggest_training_parameters()
    print(f"✓ Created {len(training_ranges)} training parameter ranges")

    # Test parameter sampling
    print("\n2. Testing parameter sampling...")
    lr_range = training_ranges['training.learning_rate']
    samples = lr_range.sample(3)
    print(f"✓ LR samples: {samples}")

    # Test grid search generation
    print("\n3. Testing grid search...")
    small_ranges = {
        'training.learning_rate': ParameterRange(
            name='lr',
            values=[0.001, 0.003],
            param_type='discrete'
        ),
        'training.batch_size': ParameterRange(
            name='batch',
            values=[32, 64],
            param_type='discrete'
        )
    }

    grid_combinations = optimizer.grid_search("base.yaml", small_ranges)
    print(f"✓ Generated {len(grid_combinations)} combinations")
    for i, combo in enumerate(grid_combinations):
        print(f"  Combo {i+1}: {combo}")

    # Test random search
    print("\n4. Testing random search...")
    random_combinations = optimizer.random_search("base.yaml", small_ranges, num_samples=3)
    print(f"✓ Generated {len(random_combinations)} random combinations")
    for i, combo in enumerate(random_combinations):
        print(f"  Random {i+1}: {combo}")


def test_campaign_creation():
    """Test optimization campaign creation"""
    print("\n=== Testing Campaign Creation ===")

    manager = ConfigManager()
    optimizer = ParameterOptimizer(manager)

    # Create a small test campaign
    print("\n1. Creating test campaign...")
    small_ranges = {
        'training.learning_rate': ParameterRange(
            name='lr',
            values=[0.001, 0.003],
            param_type='discrete'
        ),
        'reward.lives_saved_weight': ParameterRange(
            name='reward_weight',
            values=[10.0, 15.0],
            param_type='discrete'
        )
    }

    run_names = optimizer.create_optimization_campaign(
        campaign_name="test_campaign",
        base_config="base.yaml",
        parameter_ranges=small_ranges,
        method="grid",
        max_runs=10
    )

    print(f"✓ Created campaign with {len(run_names)} runs")
    print(f"✓ First few runs: {run_names[:3]}")

    # Simulate some results
    print("\n2. Simulating campaign results...")
    for i, run_name in enumerate(run_names):
        # Simulate training metrics
        for episode in range(10):
            fake_reward = 100 + i * 20 + episode * 5 + (i * episode) % 30
            manager.update_run_metrics(run_name, {'episode_reward': fake_reward})

        # Complete the run
        manager.complete_run(run_name)

    print("✓ Simulated training for all runs")

    # Analyze results
    print("\n3. Analyzing campaign results...")
    analysis = optimizer.analyze_campaign_results("test_campaign")

    if 'error' not in analysis:
        print(f"✓ Analyzed {analysis['total_runs']} runs")
        print(f"✓ Best run: {analysis['best_average_run']['run_name']}")
        print(f"✓ Best score: {analysis['best_average_run']['avg_score']:.2f}")

        # Show parameter impact
        print("\n   Parameter Impact Analysis:")
        for param, impact in analysis['parameter_impact'].items():
            best_val, best_score = impact['best_value']
            print(f"   - {param}: {best_val} (score: {best_score:.2f})")
    else:
        print(f"✗ Analysis error: {analysis['error']}")


def demonstrate_usage():
    """Demonstrate typical usage patterns"""
    print("\n=== Usage Demonstration ===")

    print("\n1. Quick parameter override for testing:")
    print("python train.py --config base.yaml --override training.learning_rate=0.001")

    print("\n2. Use pre-configured experiment:")
    print("python train.py --config experiments/quick_test.yaml")

    print("\n3. Reward function tuning:")
    print("python train.py --config experiments/reward_tuning.yaml")

    print("\n4. Hyperparameter sweep:")
    print("python optimize.py --campaign reward_optimization --method random --runs 50")

    print("\n5. Compare experiment results:")
    print("python analyze.py --compare run1,run2,run3 --metric episode_reward")


if __name__ == "__main__":
    print("Testing DRL Configuration System")
    print("=" * 50)

    try:
        test_basic_config_loading()
        test_experiment_tracking()
        test_parameter_optimization()
        test_campaign_creation()
        demonstrate_usage()

        print("\n" + "=" * 50)
        print("✅ All tests passed! Configuration system is working.")
        print("\nYou can now:")
        print("1. Modify config/base.yaml to change default settings")
        print("2. Create new experiment configs in config/experiments/")
        print("3. Use the parameter optimizer for systematic tuning")
        print("4. Track experiments with the config manager")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()