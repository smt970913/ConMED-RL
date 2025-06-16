"""
ConCare-RL: An Offline Constrained Reinforcement Learning Toolkit for Critical Care Decision Making

This package provides the core OCRL framework for critical care decision support,
including policy evaluation, policy optimization, and data handling utilities.

Authors: Maotong Sun (maotong.sun@tum.de), Jingui Xie (jingui.xie@tum.de)
"""

__version__ = "1.0.0"
__author__ = "Maotong Sun, Jingui Xie"
__email__ = "maotong.sun@tum.de, jingui.xie@tum.de"

# Core OCRL components
from .concarerl import (
    FCN_fqe,
    FCN_fqi, 
    ReplayBuffer,
    FQE,
    FQI,
    RLConfig_custom,
    RLConfigurator,
    RLTraining
)

# Data loading utilities
from .data_loader import (
    TrainDataLoader,
    ValTestDataLoader
)

# Done condition functions
from .done_condition_function_examples import *

__all__ = [
    # Core OCRL classes
    'FCN_fqe',
    'FCN_fqi',
    'ReplayBuffer', 
    'FQE',
    'FQI',
    'RLConfig_custom',
    'RLConfigurator',
    'RLTraining',
    
    # Data loading classes
    'TrainDataLoader',
    'ValTestDataLoader',
    
    # Package metadata
    '__version__',
    '__author__',
    '__email__'
] 