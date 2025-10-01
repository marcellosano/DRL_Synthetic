"""
Quick verification script to check dashboard setup
"""

import sys
from pathlib import Path

def check_installation():
    """Check if all required packages are installed"""
    print("🔍 Checking installations...")

    required_packages = [
        'streamlit',
        'plotly',
        'torch',
        'numpy',
        'pandas',
        'yaml'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package if package != 'yaml' else 'yaml')
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)

    return missing

def check_files():
    """Check if all dashboard files exist"""
    print("\n📁 Checking dashboard files...")

    required_files = [
        'app.py',
        'dashboard/pages/1_Configuration.py',
        'dashboard/pages/2_Training.py',
        'dashboard/pages/3_Evaluation.py',
        'dashboard/pages/4_Deployment.py',
        'dashboard/utils/trainer.py',
        'models/ppo_agent.py',
        'environments/foz_environment.py',
        'utils/config_manager.py',
        'config/base.yaml'
    ]

    missing = []
    for filepath in required_files:
        path = Path(filepath)
        if path.exists():
            print(f"  ✅ {filepath}")
        else:
            print(f"  ❌ {filepath}")
            missing.append(filepath)

    return missing

def check_directories():
    """Check if required directories exist"""
    print("\n📂 Checking directories...")

    required_dirs = [
        'config',
        'config/experiments',
        'dashboard',
        'dashboard/pages',
        'dashboard/utils',
        'environments',
        'models',
        'utils'
    ]

    missing = []
    for dirpath in required_dirs:
        path = Path(dirpath)
        if path.exists() and path.is_dir():
            print(f"  ✅ {dirpath}/")
        else:
            print(f"  ❌ {dirpath}/")
            missing.append(dirpath)

    return missing

def main():
    print("=" * 60)
    print("DRL Coastal EWS Dashboard - Verification")
    print("=" * 60)
    print()

    # Check installations
    missing_packages = check_installation()

    # Check files
    missing_files = check_files()

    # Check directories
    missing_dirs = check_directories()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if not missing_packages and not missing_files and not missing_dirs:
        print("✅ All checks passed! Dashboard is ready to launch.")
        print("\nTo start the dashboard, run:")
        print("  ./run_dashboard.sh")
        print("\nOr:")
        print("  streamlit run app.py")
        return 0
    else:
        print("❌ Some issues found:")

        if missing_packages:
            print(f"\n  Missing packages ({len(missing_packages)}):")
            for pkg in missing_packages:
                print(f"    - {pkg}")
            print("\n  Install with:")
            print("    pip3 install -r requirements.txt --break-system-packages")

        if missing_files:
            print(f"\n  Missing files ({len(missing_files)}):")
            for f in missing_files:
                print(f"    - {f}")

        if missing_dirs:
            print(f"\n  Missing directories ({len(missing_dirs)}):")
            for d in missing_dirs:
                print(f"    - {d}")

        return 1

if __name__ == "__main__":
    sys.exit(main())