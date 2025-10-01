#!/usr/bin/env python3
"""
Local testing script to validate modular structure
Run this before pushing to GitHub
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all module imports"""
    print("🧪 Testing imports...")

    try:
        from environments import BaseEnvironment, FOZEnvironment
        print("  ✅ Environment imports successful")
    except Exception as e:
        print(f"  ❌ Environment import failed: {e}")
        return False

    try:
        from models import MultiHeadAttention, SpatialAttentionLayer
        print("  ✅ Model imports successful")
    except Exception as e:
        print(f"  ❌ Model import failed: {e}")
        return False

    try:
        from utils import CurriculumScheduler, StateProcessor
        print("  ✅ Utils imports successful")
    except Exception as e:
        print(f"  ❌ Utils import failed: {e}")
        return False

    return True

def test_environment_creation():
    """Test FOZ environment creation"""
    print("\n🧪 Testing environment creation...")

    try:
        from environments import FOZEnvironment

        # Test with Venice config
        venice_config = "config/venice.yaml"
        if Path(venice_config).exists():
            env = FOZEnvironment(venice_config)
            print(f"  ✅ Venice environment created")
            print(f"     State dimension: {env.state_dim}")
            print(f"     Action dimension: {env.action_dim}")
        else:
            print(f"  ⚠️  Venice config not found, skipping")

        return True
    except Exception as e:
        print(f"  ❌ Environment creation failed: {e}")
        traceback.print_exc()
        return False

def test_basic_interaction():
    """Test basic environment interaction"""
    print("\n🧪 Testing basic interaction...")

    try:
        from environments import FOZEnvironment

        # Create environment
        env = FOZEnvironment("config/venice.yaml")

        # Reset
        state = env.reset()
        print(f"  ✅ Environment reset successful")
        print(f"     State shape: {state.shape}")

        # Get valid actions
        mask = env.get_valid_actions()
        print(f"  ✅ Valid actions mask: {mask.sum()} actions available")

        # Take a step
        action = 0  # Do nothing
        next_state, reward, done, info = env.step(action)
        print(f"  ✅ Step execution successful")
        print(f"     Reward: {reward:.2f}")
        print(f"     Resources remaining: {info['resources_remaining']}")

        return True
    except Exception as e:
        print(f"  ❌ Interaction test failed: {e}")
        traceback.print_exc()
        return False

def test_state_processor():
    """Test state processor"""
    print("\n🧪 Testing state processor...")

    try:
        from utils import StateProcessor
        import numpy as np

        processor = StateProcessor()
        print("  ✅ StateProcessor created")

        # Test with numpy array
        test_state = np.random.randn(100)
        processed = processor.process_state(test_state)
        print(f"  ✅ Array processing successful: {processed.shape}")

        # Test with dict state
        dict_state = {
            'grid': np.random.randn(20, 20),
            'time_step': 5,
            'resources': 80
        }
        processed = processor.process_state(dict_state)
        print(f"  ✅ Dict processing successful: {processed.shape}")

        return True
    except Exception as e:
        print(f"  ❌ State processor test failed: {e}")
        traceback.print_exc()
        return False

def test_curriculum():
    """Test curriculum scheduler"""
    print("\n🧪 Testing curriculum scheduler...")

    try:
        from utils import CurriculumScheduler

        scheduler = CurriculumScheduler(levels=10)
        print("  ✅ CurriculumScheduler created")

        # Test update
        for i in range(100):
            reward = np.random.randn() * 100
            scheduler.update(reward)

        stats = scheduler.get_stats()
        print(f"  ✅ Curriculum updated successfully")
        print(f"     Current level: {stats['current_level']}")
        print(f"     Total episodes: {stats['total_episodes']}")

        return True
    except Exception as e:
        print(f"  ❌ Curriculum test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 DRL FOZ Local Testing Suite")
    print("=" * 60)

    tests = [
        test_imports,
        test_environment_creation,
        test_basic_interaction,
        test_state_processor,
        test_curriculum
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)

    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! Ready to push to GitHub.")
    else:
        print("\n⚠️  Some tests failed. Please fix before pushing.")
        sys.exit(1)

if __name__ == "__main__":
    main()