#!/usr/bin/env python
"""
Build and package ConCare-RL for distribution
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description if description else cmd}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    else:
        print(f"Success: {result.stdout}")
    
    return result

def clean_build():
    """Clean previous build artifacts"""
    print("Cleaning previous build artifacts...")
    
    dirs_to_clean = ['build', 'dist', '*.egg-info', '__pycache__']
    
    for pattern in dirs_to_clean:
        if '*' in pattern:
            # Handle glob patterns
            import glob
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        else:
            if os.path.exists(pattern):
                shutil.rmtree(pattern)
    
    print("✓ Build artifacts cleaned")

def build_package():
    """Build the package"""
    print("Building ConCare-RL package...")
    
    # Build using both methods for compatibility
    run_command("python setup.py sdist bdist_wheel", "Building with setup.py")
    run_command("python -m build", "Building with pyproject.toml")
    
    print("✓ Package built successfully")

def check_package():
    """Check the built package"""
    print("Checking package...")
    
    run_command("python -m twine check dist/*", "Checking package with twine")
    
    print("✓ Package check completed")

def test_install():
    """Test installation in a virtual environment"""
    print("Testing package installation...")
    
    # Create test virtual environment
    run_command("python -m venv test_env", "Creating test environment")
    
    # Install the package
    if os.name == 'nt':  # Windows
        pip_cmd = "test_env\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        pip_cmd = "test_env/bin/pip"
    
    # Find the built wheel file
    wheel_files = list(Path("dist").glob("*.whl"))
    if not wheel_files:
        print("Error: No wheel file found in dist/")
        sys.exit(1)
    
    wheel_file = wheel_files[0]
    run_command(f"{pip_cmd} install {wheel_file}", f"Installing {wheel_file}")
    
    # Test import
    if os.name == 'nt':  # Windows
        python_cmd = "test_env\\Scripts\\python"
    else:  # Unix/Linux/Mac
        python_cmd = "test_env/bin/python"
    
    run_command(f"{python_cmd} -c \"import ConMedRL; print('ConMedRL imported successfully')\"", 
                "Testing import")
    
    # Cleanup test environment
    shutil.rmtree("test_env")
    
    print("✓ Package installation test completed")

def publish_package(test=True):
    """Publish package to PyPI"""
    repository = "testpypi" if test else "pypi"
    
    print(f"Publishing to {'Test ' if test else ''}PyPI...")
    
    if test:
        run_command("python -m twine upload --repository testpypi dist/*", 
                   "Uploading to Test PyPI")
        print("\n✓ Package uploaded to Test PyPI")
        print("Install with: pip install --index-url https://test.pypi.org/simple/ concarerl")
    else:
        run_command("python -m twine upload dist/*", "Uploading to PyPI")
        print("\n✓ Package uploaded to PyPI")
        print("Install with: pip install concarerl")

def main():
    """Main function"""
    print("ConCare-RL Package Builder")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
    else:
        print("Available actions:")
        print("  clean    - Clean build artifacts")
        print("  build    - Build package")
        print("  check    - Check package")
        print("  test     - Test package installation")
        print("  testpypi - Upload to Test PyPI") 
        print("  pypi     - Upload to PyPI")
        print("  all      - Run clean, build, check, and test")
        action = input("\nSelect action: ").lower()
    
    try:
        if action == "clean":
            clean_build()
        elif action == "build":
            build_package()
        elif action == "check":
            check_package()
        elif action == "test":
            test_install()
        elif action == "testpypi":
            publish_package(test=True)
        elif action == "pypi":
            publish_package(test=False)
        elif action == "all":
            clean_build()
            build_package()
            check_package()
            test_install()
        else:
            print(f"Unknown action: {action}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 