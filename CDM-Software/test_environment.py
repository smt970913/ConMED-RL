#!/usr/bin/env python3
"""
Test script to verify that all required packages are installed and working.
Run this before trying to start the Flask application.
"""
import os
import sys
print(f"Python version: {sys.version}")
print("=" * 50)

# Test imports
packages_to_test = [
    ("Flask", "flask"),
    ("PyTorch", "torch"), 
    ("NumPy", "numpy"),
    ("Scikit-learn", "sklearn"),
    ("Pillow", "PIL"),
    ("Interactive Support", "interactive_support")
]

failed_imports = []

for package_name, module_name in packages_to_test:
    try:
        __import__(module_name)
        print(f"✓ {package_name} - OK")
    except ImportError as e:
        print(f"✗ {package_name} - FAILED: {e}")
        failed_imports.append(package_name)

print("=" * 50)

if failed_imports:
    print(f"❌ {len(failed_imports)} package(s) failed to import:")
    for package in failed_imports:
        print(f"   - {package}")
    print("\nPlease install missing packages using:")
    print("pip install flask torch numpy scikit-learn pillow")
else:
    print("✅ All packages imported successfully!")
    
    # Test file paths
    print("\nTesting file paths...")
    
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (project root)
    project_root = os.path.dirname(script_dir)
    
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    
    files_to_check = [
        os.path.join(project_root, "image", "ConMED-RL Logo.png"),
        os.path.join(project_root, "Software_FQE_models", "discharge_decision_making", "ocrl_agent_s1_fqe_obj_20250508_v0.pth"),
        os.path.join(project_root, "Software_FQE_models", "discharge_decision_making", "ocrl_agent_s1_fqe_con_rr_20250508_v0.pth"),
        os.path.join(project_root, "Software_FQE_models", "discharge_decision_making", "ocrl_agent_s1_fqe_con_los_20250508_v0.pth")
    ]
    
    missing_files = []
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {os.path.basename(file_path)} - Found")
        else:
            print(f"✗ {os.path.basename(file_path)} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ {len(missing_files)} file(s) missing:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    else:
        print("\n✅ All required files found!")
        print("\n🚀 Ready to run the Flask application!")

print("\n" + "=" * 50)
print("Test completed. Press Enter to exit...")
input() 