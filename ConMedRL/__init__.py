"""
ConMedRL: An Offline Constrained Reinforcement Learning Toolkit for Critical Care Decision Making

This package provides the core OCRL framework for critical care decision support,
including policy evaluation, policy optimization, and data handling utilities.

Authors: Maotong Sun (maotong.sun@tum.de), Jingui Xie (jingui.xie@tum.de)
"""

__version__ = "1.1.0"
__author__ = "Maotong Sun, Jingui Xie"
__email__ = "maotong.sun@tum.de, jingui.xie@tum.de"

# Core OCRL components
from .conmedrl import (
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

from .model_artifacts import (
    EXACT_RETRAIN_METHOD,
    ModelArtifactManifest,
    ModelCompatibilityError,
    assert_model_compatible,
    create_model_artifact_manifest,
    dataset_content_hash,
    exact_retrain,
    invalidate_model_artifact,
    load_model_artifact_manifest,
    set_deterministic_seed,
    write_model_artifact_manifest,
)

# Unified preprocessing subpackage. Importing it is lightweight; database
# adapters and optional dependencies are loaded only when build_dataset runs.
from . import data
from .data import (
    UNLEARNING_API_VERSION,
    build_dataset,
    compute_dataset_content_hash,
    invalidate_dataset_manifest,
    load_dataset,
    purge_observation_cache,
    rebuild_dataset_after_withdrawal,
    withdrawal_request_digest,
)

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

    # Model lineage and exact machine unlearning
    'EXACT_RETRAIN_METHOD',
    'ModelArtifactManifest',
    'ModelCompatibilityError',
    'assert_model_compatible',
    'create_model_artifact_manifest',
    'dataset_content_hash',
    'exact_retrain',
    'invalidate_model_artifact',
    'load_model_artifact_manifest',
    'set_deterministic_seed',
    'write_model_artifact_manifest',

    # Data preprocessing
    'data',
    'build_dataset',
    'compute_dataset_content_hash',
    'invalidate_dataset_manifest',
    'load_dataset',
    'UNLEARNING_API_VERSION',
    'purge_observation_cache',
    'rebuild_dataset_after_withdrawal',
    'withdrawal_request_digest',
    
    # Package metadata
    '__version__',
    '__author__',
    '__email__'
] 