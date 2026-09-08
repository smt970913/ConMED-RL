"""Data preprocessing for offline constrained RL on ICU databases.

Turns a local copy of MIMIC-IV or SICdb into the tables
:class:`ConMedRL.TrainDataLoader` and :class:`ConMedRL.ValTestDataLoader`
expect, optionally also as Parquet or a ``d3rlpy`` ``MDPDataset``.

The usual entry point is a single call::

    from ConMedRL.data import build_dataset

    bundle = build_dataset(
        database="mimic-iv",
        task="discharge",
        data_dir="/path/to/mimic-iv",
        output_dir="./processed",
        llm_api_key=None,          # optional, only used for variable search
    )

Everything the run decided is on the returned bundle: the resolved state
space (:attr:`~ConMedRL.data.RLDatasetBundle.schema`), the per-split tables,
and the reports explaining which variables were dropped and why.

Because the two databases do not carry the same measurements, the state space
is resolved per run rather than fixed in advance -- SICdb, for example, records
no Glasgow Coma Scale and no haematocrit at all. Use
:meth:`~ConMedRL.data.RLDatasetBundle.describe` to see what survived.
"""

from __future__ import annotations

from .config import (
    CohortConfig,
    Database,
    FHIRConfig,
    ImputationConfig,
    LLMConfig,
    OutlierConfig,
    OutputFormat,
    PreprocessConfig,
    SplitConfig,
    Task,
)
from .dataset import (
    MDPDatasetBundle,
    ROW_INDEX_COLUMN,
    RLDatasetBundle,
    SplitTables,
    compute_dataset_content_hash,
    validate_rl_contract,
)
from .export import available_formats, write_bundle
from .llm import (
    AnthropicBackend,
    LLMBackend,
    NullLLMBackend,
    OpenAIBackend,
    get_llm_backend,
)
from .schema import (
    CANONICAL_VARIABLES,
    StateSpaceSchema,
    TASK_STATE_SPACE,
    VariableSpec,
    resolve_state_space,
    task_state_space,
)
from .sources import (
    MIMIC_IV_TABLES,
    MissingTableError,
    SICDB_TABLES,
    SourceRegistry,
    TableSpec,
)
from .variables import (
    LOINC_HINTS,
    MIMIC_KNOWN_ABSENT,
    MIMIC_SEED_ITEMIDS,
    MappedVariable,
    SICDB_KNOWN_ABSENT,
    SICDB_SEED_IDS,
    VariableMapping,
    VariableSearch,
    resolve_variable_mapping,
)
from .specs import (
    SPEC_VERSION,
    ActionSpec,
    ApprovalRequiredError,
    CostRule,
    DatasetSpec,
    DictionarySpec,
    EventRule,
    SafeRuleEvaluator,
    SpecValidationError,
    TableRoleSpec,
    TaskSpec,
    UnitRule,
)
from .profiler import DatasetProfile, DatasetProfiler, profile_dataset
from .planner import (
    GenericPlanner,
    PlannerValidationError,
    approve_plan,
    recommend_dataset_plan,
    recommend_task_plan,
    require_approved_plan,
    validate_plan,
)
from .profiles import load_nwicu_profile
from .tasks import evaluate_rule, register_declarative_task
from .unlearning import (
    UNLEARNING_API_VERSION,
    canonical_withdrawal_ids,
    invalidate_dataset_manifest,
    purge_observation_cache,
    rebuild_dataset_after_withdrawal,
    withdrawal_request_digest,
)
from .fhir import (
    FHIRExportResult,
    FHIRR4Exporter,
    export_fhir_r4,
    run_hl7_validator,
    validate_resources,
)

__all__ = [
    # entry point
    "build_dataset",
    "load_dataset",
    # configuration
    "PreprocessConfig",
    "Database",
    "Task",
    "OutputFormat",
    "LLMConfig",
    "SplitConfig",
    "ImputationConfig",
    "OutlierConfig",
    "CohortConfig",
    "FHIRConfig",
    # results
    "RLDatasetBundle",
    "SplitTables",
    "MDPDatasetBundle",
    "ROW_INDEX_COLUMN",
    "compute_dataset_content_hash",
    "validate_rl_contract",
    # exact patient withdrawal
    "UNLEARNING_API_VERSION",
    "canonical_withdrawal_ids",
    "withdrawal_request_digest",
    "invalidate_dataset_manifest",
    "purge_observation_cache",
    "rebuild_dataset_after_withdrawal",
    # declarative specifications and planning
    "SPEC_VERSION",
    "DatasetSpec",
    "TableRoleSpec",
    "DictionarySpec",
    "UnitRule",
    "EventRule",
    "TaskSpec",
    "ActionSpec",
    "CostRule",
    "SafeRuleEvaluator",
    "SpecValidationError",
    "ApprovalRequiredError",
    "DatasetProfiler",
    "DatasetProfile",
    "profile_dataset",
    "GenericPlanner",
    "PlannerValidationError",
    "recommend_dataset_plan",
    "recommend_task_plan",
    "validate_plan",
    "approve_plan",
    "require_approved_plan",
    "register_declarative_task",
    "evaluate_rule",
    "load_nwicu_profile",
    # FHIR R4 exchange
    "FHIRR4Exporter",
    "FHIRExportResult",
    "export_fhir_r4",
    "validate_resources",
    "run_hl7_validator",
    # state space
    "VariableSpec",
    "StateSpaceSchema",
    "CANONICAL_VARIABLES",
    "TASK_STATE_SPACE",
    "task_state_space",
    "resolve_state_space",
    # sources
    "SourceRegistry",
    "TableSpec",
    "MIMIC_IV_TABLES",
    "SICDB_TABLES",
    "MissingTableError",
    # variable resolution
    "VariableSearch",
    "VariableMapping",
    "MappedVariable",
    "resolve_variable_mapping",
    "SICDB_SEED_IDS",
    "SICDB_KNOWN_ABSENT",
    "MIMIC_SEED_ITEMIDS",
    "MIMIC_KNOWN_ABSENT",
    "LOINC_HINTS",
    # llm
    "LLMBackend",
    "NullLLMBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "get_llm_backend",
    # export
    "write_bundle",
    "available_formats",
]


def __getattr__(name: str):
    """Defer the pipeline import so the light modules stay cheap.

    ``build_dataset`` pulls in the adapters, which pull in Dask; importing
    :mod:`ConMedRL.data` just to read a config should not pay for that.
    """
    if name in ("build_dataset", "load_dataset"):
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
