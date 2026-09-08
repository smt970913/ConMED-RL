"""
Data preprocessing modules for ConMedRL

This package contains data preprocessing scripts for different datasets:
- mimic_iv_icu_discharge: MIMIC-IV ICU discharge decision data preprocessing
- mimic_iv_icu_extubation: MIMIC-IV ICU extubation decision data preprocessing  
- SICdb_discharge: SICdb discharge decision data preprocessing
- SICdb_extubation: SICdb extubation decision data preprocessing
"""

__version__ = "1.1.0"  # compatibility package; canonical API is ConMedRL.data

# Import data preprocessing modules
from . import mimic_iv_icu_discharge
from . import mimic_iv_icu_extubation
from . import SICdb_discharge
from . import SICdb_extubation
from ConMedRL.data import (
    PreprocessConfig,
    VariableSearch,
    build_dataset,
    load_dataset,
)

__all__ = [
    "mimic_iv_icu_discharge",
    "mimic_iv_icu_extubation", 
    "SICdb_discharge",
    "SICdb_extubation",
    "PreprocessConfig",
    "VariableSearch",
    "build_dataset",
    "load_dataset",
] 