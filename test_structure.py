#!/usr/bin/env python3
"""
Simple structure test that doesn't require external packages
"""

import os
from pathlib import Path

def check_structure():
    """Check if all files and directories exist"""
    print("🔍 Checking project structure...")

    required_dirs = [
        "config",
        "environments",
        "models",
        "utils",
        "data/venice",
        "data/goldcoast"
    ]

    required_files = [
        "requirements.txt",
        ".gitignore",
        "config/venice.yaml",
        "config/goldcoast.yaml",
        "environments/__init__.py",
        "environments/base_environment.py",
        "environments/foz_environment.py",
        "models/__init__.py",
        "models/attention.py",
        "utils/__init__.py",
        "utils/curriculum.py",
        "utils/state_processor.py"
    ]

    all_good = True

    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ missing")
            all_good = False

    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} missing")
            all_good = False

    return all_good

def main():
    print("=" * 60)
    print("📁 DRL FOZ Project Structure Check")
    print("=" * 60)

    if check_structure():
        print("\n✅ All files and directories are in place!")
        print("Ready to push to GitHub.")
    else:
        print("\n⚠️  Some files/directories are missing!")

if __name__ == "__main__":
    main()