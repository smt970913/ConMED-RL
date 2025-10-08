#!/usr/bin/env python
"""
Setup script for ConMED-RL: An OCRL-Based Toolkit for Medical Decision Support
"""

from setuptools import setup, find_packages
import os

# Read the PyPI-specific README file
try:
    with open("PYPI_README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    # Fallback to regular README if PYPI_README doesn't exist
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

# Read the requirements file
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except UnicodeDecodeError:
    with open("requirements.txt", "r", encoding="utf-8-sig") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version from ConMED_RL/__init__.py
def get_version():
    """Extract version from __init__.py"""
    version_file = os.path.join("ConMedRL", "__init__.py")
    with open(version_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="ConMedRL",
    version=get_version(),
    author="Maotong Sun, Jingui Xie",
    author_email="maotong.sun@tum.de, jingui.xie@tum.de",
    description="An Offline Constrained Reinforcement Learning Toolkit for Critical Care Decision Making",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smt970913/ConMED-RL",
    project_urls={
        "Bug Tracker": "https://github.com/smt970913/ConMED-RL/issues",
        "Documentation": "https://github.com/smt970913/ConMED-RL#readme",
        "Source Code": "https://github.com/smt970913/ConMED-RL",
    },
    packages=find_packages(include=["ConMedRL", "ConMedRL.*", "Data", "Data.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research", 
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "jupyter>=1.0",
            "notebook>=6.0",
        ],
        "models": [
            "torch>=1.8.0",
            "torchvision>=0.9.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ]
    },
    package_data={
        "ConMedRL": [
            "*.py",
        ],
        "Data": [
            "**/*.py",
        ]
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "reinforcement learning",
        "constrained reinforcement learning", 
        "offline reinforcement learning",
        "clinical decision support",
        "healthcare",
        "ICU",
        "critical care",
        "machine learning",
        "artificial intelligence"
    ],
)
