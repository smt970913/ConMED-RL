"""Configuration objects for the ConMedRL data-preprocessing module.

Everything a user needs to describe *what* to build is expressed here as plain
dataclasses, so a whole preprocessing run can be serialised to / from JSON and
reproduced later.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .unlearning import canonical_withdrawal_ids, withdrawal_request_digest

__all__ = [
    "Database",
    "Task",
    "OutputFormat",
    "LLMConfig",
    "SplitConfig",
    "ImputationConfig",
    "OutlierConfig",
    "CohortConfig",
    "FHIRConfig",
    "PreprocessConfig",
]

PathLike = Union[str, "os.PathLike[str]"]


class Database(object):
    """Supported source databases."""

    MIMIC_IV = "mimic-iv"
    SICDB = "sicdb"
    GENERIC = "generic"

    ALL = (MIMIC_IV, SICDB, GENERIC)

    #: Spellings we accept from users and map onto the canonical value.
    _ALIASES = {
        "mimic": MIMIC_IV,
        "mimic4": MIMIC_IV,
        "mimic-4": MIMIC_IV,
        "mimic_iv": MIMIC_IV,
        "mimic-iv": MIMIC_IV,
        "mimiciv": MIMIC_IV,
        "sicdb": SICDB,
        "sic": SICDB,
        "salzburg": SICDB,
        "salzburgicu": SICDB,
        "salzburg-icu": SICDB,
        "generic": GENERIC,
        "mimic-like": GENERIC,
        "mimiclike": GENERIC,
        "custom": GENERIC,
    }

    @classmethod
    def normalize(cls, value: str) -> str:
        key = str(value).strip().lower().replace(" ", "").replace("_", "_")
        canonical = cls._ALIASES.get(key) or cls._ALIASES.get(key.replace("_", "-"))
        if canonical is None:
            raise ValueError(
                "Unknown database {0!r}. Supported: {1}".format(value, ", ".join(cls.ALL))
            )
        return canonical


class Task(object):
    """Supported offline-RL decision-making tasks."""

    DISCHARGE = "discharge"
    EXTUBATION = "extubation"

    ALL = (DISCHARGE, EXTUBATION)

    _ALIASES = {
        "discharge": DISCHARGE,
        "icudischarge": DISCHARGE,
        "discharge-decision-making": DISCHARGE,
        "dischargedecisionmaking": DISCHARGE,
        "extubation": EXTUBATION,
        "extubate": EXTUBATION,
        "weaning": EXTUBATION,
        "extubation-decision-making": EXTUBATION,
        "extubationdecisionmaking": EXTUBATION,
    }

    @classmethod
    def normalize(cls, value: str, allow_custom: bool = False) -> str:
        key = str(value).strip().lower().replace(" ", "").replace("_", "")
        canonical = cls._ALIASES.get(key)
        if canonical is None:
            canonical = cls._ALIASES.get(str(value).strip().lower())
        if canonical is None:
            if allow_custom:
                custom = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
                if custom:
                    return custom
            raise ValueError(
                "Unknown task {0!r}. Supported: {1}".format(value, ", ".join(cls.ALL))
            )
        return canonical


class OutputFormat(object):
    """Serialisation targets for the finished dataset.

    ``csv`` is always produced because the ConMedRL data loaders read the
    outcome / state tables from CSV; requesting other formats adds to it rather
    than replacing it.
    """

    CSV = "csv"
    PARQUET = "parquet"
    D3RLPY = "d3rlpy"
    FHIR = "fhir"

    ALL = (CSV, PARQUET, D3RLPY, FHIR)

    _ALIASES = {
        "csv": CSV,
        "parquet": PARQUET,
        "pq": PARQUET,
        "d3rlpy": D3RLPY,
        "mdpdataset": D3RLPY,
        "mdp": D3RLPY,
        "fhir": FHIR,
        "fhir-r4": FHIR,
    }

    @classmethod
    def normalize(cls, value: str) -> str:
        key = str(value).strip().lower().replace(".", "")
        canonical = cls._ALIASES.get(key)
        if canonical is None:
            raise ValueError(
                "Unknown output format {0!r}. Supported: {1}".format(
                    value, ", ".join(cls.ALL)
                )
            )
        return canonical


@dataclass
class LLMConfig:
    """How (and whether) to call a large language model.

    The LLM is only ever used to *translate clinical term names* and to help
    rank candidate variable names during variable search. No patient-level data
    is sent to the provider.
    """

    provider: str = "none"
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_retries: int = 3
    timeout: float = 60.0
    #: Cache translations on disk so repeated runs don't re-pay for tokens.
    cache_path: Optional[PathLike] = None

    _PROVIDER_ALIASES = {
        "none": "none",
        "off": "none",
        "disabled": "none",
        "openai": "openai",
        "gpt": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
    }

    _DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-20241022",
    }

    #: Environment variables consulted when ``api_key`` is not given.
    _ENV_KEYS = {
        "openai": ("CONMEDRL_LLM_API_KEY", "OPENAI_API_KEY"),
        "anthropic": ("CONMEDRL_LLM_API_KEY", "ANTHROPIC_API_KEY"),
    }

    def __post_init__(self) -> None:
        key = str(self.provider).strip().lower()
        canonical = self._PROVIDER_ALIASES.get(key)
        if canonical is None:
            raise ValueError(
                "Unknown LLM provider {0!r}. Supported: none, openai, anthropic".format(
                    self.provider
                )
            )
        self.provider = canonical

        if self.model is None and self.provider != "none":
            self.model = self._DEFAULT_MODELS[self.provider]

        if self.cache_path is not None:
            self.cache_path = Path(self.cache_path)

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    def resolve_api_key(self) -> Optional[str]:
        """Return the API key, falling back to environment variables.

        Keeping the fallback here means notebooks can stay free of literal
        secrets while still working with an explicit key when one is passed.
        """
        if not self.enabled:
            return None
        if self.api_key:
            return self.api_key
        for env_name in self._ENV_KEYS[self.provider]:
            value = os.environ.get(env_name)
            if value:
                return value
        return None


@dataclass
class SplitConfig:
    """Grouped train / validation / test partitioning.

    When ``group_key`` is left as ``None``, the pipeline follows the reference
    task design: discharge is grouped by ``subject_id``; extubation is grouped
    by the complete mechanical-ventilation treatment episode (the canonical
    ``stay_id`` in the current built-in cohort) and stratified by extubation
    failure.  Custom tasks default to patient-level grouping.  Explicit values
    override these task defaults.
    """

    #: Fraction of groups held out of training (split further into val + test).
    test_prop: float = 0.2
    #: Fraction of the held-out groups that become the *test* set; the rest
    #: become validation.
    val_prop: float = 0.5
    random_seed: int = 42
    #: Grouping column. ``None`` selects the task-specific default.
    group_key: Optional[str] = None
    #: Optional group-level outcome used for stratified sampling.
    stratify_column: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.test_prop < 1.0:
            raise ValueError("test_prop must lie in (0, 1), got {0}".format(self.test_prop))
        if not 0.0 < self.val_prop < 1.0:
            raise ValueError("val_prop must lie in (0, 1), got {0}".format(self.val_prop))


@dataclass
class ImputationConfig:
    """Missing-value handling for the physiological variables."""

    #: Columns missing more than this fraction are candidates for dropping.
    missing_threshold_drop: float = 0.75
    #: Columns missing less than this fraction go straight to KNN imputation.
    missing_threshold_knn: float = 0.10
    forward_fill: bool = True
    # Interpolation uses a future measurement to fill an earlier state. That
    # is useful for retrospective description but leaks information to a
    # decision policy, so it is opt-in.
    linear_interpolate: bool = False
    knn_impute: bool = True
    knn_neighbors: int = 5
    knn_chunk_size: int = 10000
    #: Cap reference rows to avoid quadratic distance work on full ICU tables.
    knn_fit_max_rows: int = 10000
    n_jobs: int = -1
    #: Drop the columns classified as "too sparse" instead of imputing them.
    drop_sparse_columns: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.missing_threshold_knn <= self.missing_threshold_drop <= 1.0:
            raise ValueError(
                "Expected 0 <= missing_threshold_knn <= missing_threshold_drop <= 1, "
                "got {0} and {1}".format(
                    self.missing_threshold_knn, self.missing_threshold_drop
                )
            )
        if self.knn_fit_max_rows < self.knn_neighbors:
            raise ValueError("knn_fit_max_rows must be >= knn_neighbors")


@dataclass
class OutlierConfig:
    """Physiologically implausible value removal."""

    #: One of ``"iqr"``, ``"zscore"``, ``"range"`` or ``"none"``.
    method: str = "iqr"
    iqr_factor: float = 3.0
    z_threshold: float = 3.0
    #: Restrict filtering to these variables; ``None`` means every numeric
    #: physiological variable in the resolved state space.
    variables: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        allowed = ("iqr", "zscore", "range", "none")
        method = str(self.method).strip().lower()
        if method not in allowed:
            raise ValueError(
                "Unknown outlier method {0!r}. Supported: {1}".format(
                    self.method, ", ".join(allowed)
                )
            )
        self.method = method


@dataclass
class CohortConfig:
    """Inclusion / exclusion criteria and decision-epoch construction."""

    #: Window used to label an ICU readmission as such.
    readmission_observation_days: int = 7
    #: Window used to attribute an out-of-ICU death to the discharge decision.
    death_observation_days: int = 7
    #: Window used to call a later intubation an extubation failure.
    reintubation_observation_days: int = 7
    #: Stays with more readmissions than this are dropped as outliers.
    readmission_count_threshold: int = 6
    #: Stays longer than this (days) are dropped as outliers.
    los_threshold: float = 15.0
    #: Spacing of decision epochs, in hours.
    decision_epoch_hours: float = 12.0
    #: Minimum age (years) for inclusion.
    min_age: float = 18.0
    #: ICU unit names to keep; ``None`` keeps every unit.
    icu_units: Optional[Sequence[str]] = None
    #: For extubation: minimum invasive-ventilation duration (hours) to include.
    min_ventilation_hours: float = 12.0

    def __post_init__(self) -> None:
        for name in (
            "readmission_observation_days",
            "death_observation_days",
            "reintubation_observation_days",
        ):
            if getattr(self, name) < 0:
                raise ValueError("{0} must be non-negative".format(name))
        if self.readmission_count_threshold is not None and self.readmission_count_threshold < 1:
            raise ValueError("readmission_count_threshold must be at least 1")
        if self.los_threshold is not None and self.los_threshold <= 0:
            raise ValueError("los_threshold must be positive")
        if self.decision_epoch_hours <= 0:
            raise ValueError("decision_epoch_hours must be positive")


@dataclass
class FHIRConfig:
    """FHIR R4 exchange and conformance options."""

    version: str = "4.0.1"
    validator_path: Optional[PathLike] = None
    terminology_server: Optional[str] = None
    validate_profiles: bool = True
    #: Optional environment variable containing a pseudonymisation salt.
    id_salt_env: str = "CONMEDRL_FHIR_ID_SALT"

    def __post_init__(self) -> None:
        if not str(self.version).startswith("4"):
            raise ValueError("ConMedRL currently exports FHIR R4 only.")
        if self.validator_path is not None:
            self.validator_path = Path(self.validator_path)


@dataclass
class PreprocessConfig:
    """Top-level description of one preprocessing run.

    Example
    -------
    >>> cfg = PreprocessConfig(
    ...     database="mimic-iv",
    ...     task="discharge",
    ...     data_dir="/data/mimic-iv/3.1",
    ...     output_dir="./conmedrl_out",
    ...     output_formats=("csv", "parquet", "d3rlpy"),
    ... )
    """

    database: str
    task: str
    data_dir: PathLike
    output_dir: PathLike = "./conmedrl_data"

    #: ``csv`` is implied; listing it explicitly is harmless.
    output_formats: Sequence[str] = ("csv",)

    #: Prefix for every written file, e.g. ``discharge_sample``.
    output_prefix: Optional[str] = None

    #: Compression of the *source* files ("gzip", "zip", None, ...). ``"infer"``
    #: lets pandas decide from each file's extension.
    source_compression: Optional[str] = "infer"

    llm: LLMConfig = field(default_factory=LLMConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    cohort: CohortConfig = field(default_factory=CohortConfig)
    fhir: FHIRConfig = field(default_factory=FHIRConfig)

    #: Approved declarative specifications for generic datasets/tasks.
    dataset_spec: Optional[Mapping[str, Any]] = None
    task_spec: Optional[Mapping[str, Any]] = None
    approved_plan_hash: Optional[str] = None
    #: Configuring a provider does not implicitly authorize LLM mappings.
    llm_allow_mapping: bool = False

    # --- dynamic state space -------------------------------------------------
    #: Keep a canonical variable only if this fraction of rows is observed for
    #: it in the source database. Variables below the bar are dropped from the
    #: state space, which is how the same task adapts to SICdb's narrower
    #: coverage without editing code.
    min_variable_coverage: float = 0.0
    #: Force these canonical variables into the state space even if coverage is
    #: low (they will be imputed).
    require_variables: Sequence[str] = ()
    #: Force these canonical variables out of the state space.
    exclude_variables: Sequence[str] = ()

    # --- execution -----------------------------------------------------------
    #: Rows per chunk when streaming very large event tables.
    chunk_size: int = 1_000_000
    #: Cache expensive intermediate artefacts under ``output_dir/_cache``.
    use_cache: bool = True
    #: Exact patient IDs to omit. Kept in memory and never serialised.
    withdrawn_subject_ids: Sequence[Any] = field(default_factory=tuple, repr=False)
    #: Redacted withdrawal metadata retained in manifests/config round-trips.
    withdrawal_count: int = 0
    withdrawal_digest: Optional[str] = None
    #: ``0`` quiet, ``1`` progress, ``2`` debug.
    verbosity: int = 1
    random_seed: int = 42

    #: Explicit overrides for individual source tables, keyed by logical name
    #: (e.g. ``{"chartevents": "/mnt/big/chartevents.csv.gz"}``). Anything not
    #: listed is resolved by searching ``data_dir``.
    source_paths: Mapping[str, PathLike] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.database = Database.normalize(self.database)
        if self.database == Database.GENERIC and self.dataset_spec is not None:
            raw_spec = (
                self.dataset_spec.to_dict()
                if hasattr(self.dataset_spec, "to_dict")
                else dict(self.dataset_spec)
            )
            if self.task_spec is None:
                wanted = str(self.task).strip().lower().replace("_", "-")
                for candidate in raw_spec.get("tasks", ()):
                    candidate_dict = (
                        candidate.to_dict()
                        if hasattr(candidate, "to_dict")
                        else dict(candidate)
                    )
                    name = str(candidate_dict.get("name", "")).strip().lower().replace(
                        "_", "-"
                    )
                    if name == wanted:
                        self.task_spec = candidate_dict
                        break
            if self.approved_plan_hash is None:
                self.approved_plan_hash = raw_spec.get("approval_hash")
        self.task = Task.normalize(
            self.task,
            allow_custom=self.database == Database.GENERIC or self.task_spec is not None,
        )

        self.data_dir = Path(self.data_dir).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()

        formats = [OutputFormat.normalize(f) for f in self.output_formats]
        if OutputFormat.CSV not in formats:
            formats.insert(0, OutputFormat.CSV)
        # De-duplicate while keeping the caller's ordering.
        self.output_formats = tuple(dict.fromkeys(formats))

        if self.output_prefix is None:
            self.output_prefix = "{0}_{1}".format(
                self.database.replace("-", "_"), self.task
            )

        if isinstance(self.llm, Mapping):
            self.llm = LLMConfig(**dict(self.llm))
        if isinstance(self.split, Mapping):
            self.split = SplitConfig(**dict(self.split))
        if isinstance(self.imputation, Mapping):
            self.imputation = ImputationConfig(**dict(self.imputation))
        if isinstance(self.outliers, Mapping):
            self.outliers = OutlierConfig(**dict(self.outliers))
        if isinstance(self.cohort, Mapping):
            self.cohort = CohortConfig(**dict(self.cohort))
        if isinstance(self.fhir, Mapping):
            self.fhir = FHIRConfig(**dict(self.fhir))

        if self.database == Database.GENERIC:
            if self.dataset_spec is None or self.task_spec is None:
                raise ValueError(
                    "Generic datasets require approved dataset_spec and task_spec mappings."
                )
            if not self.approved_plan_hash:
                raise ValueError(
                    "Generic dataset execution requires an approved_plan_hash."
                )

        if not 0.0 <= self.min_variable_coverage <= 1.0:
            raise ValueError(
                "min_variable_coverage must lie in [0, 1], got {0}".format(
                    self.min_variable_coverage
                )
            )

        overlap = set(self.require_variables) & set(self.exclude_variables)
        if overlap:
            raise ValueError(
                "Variables listed as both required and excluded: {0}".format(
                    ", ".join(sorted(overlap))
                )
            )

        self.source_paths = {
            str(k): Path(v).expanduser() for k, v in dict(self.source_paths).items()
        }
        self.withdrawn_subject_ids = tuple(self.withdrawn_subject_ids or ())
        if self.withdrawn_subject_ids:
            canonical = canonical_withdrawal_ids(self.withdrawn_subject_ids)
            digest = withdrawal_request_digest(self.withdrawn_subject_ids)
            if self.withdrawal_count not in (0, len(canonical)):
                raise ValueError(
                    "withdrawal_count does not match the supplied unique subject IDs."
                )
            if self.withdrawal_digest not in (None, digest):
                raise ValueError(
                    "withdrawal_digest does not match the supplied subject IDs."
                )
            self.withdrawal_count = len(canonical)
            self.withdrawal_digest = digest
        else:
            if self.withdrawal_count < 0:
                raise ValueError("withdrawal_count cannot be negative.")
            if bool(self.withdrawal_count) != bool(self.withdrawal_digest):
                raise ValueError(
                    "Redacted withdrawal metadata requires both count and digest."
                )

    # -- convenience ----------------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        return Path(self.output_dir) / "_cache"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain JSON-compatible types, redacting the API key."""
        payload = asdict(self)
        # Raw withdrawal IDs are memory-only. The count and stable digest are
        # sufficient to audit a manifest, but not to reconstruct the request.
        payload.pop("withdrawn_subject_ids", None)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["output_formats"] = list(self.output_formats)
        payload["require_variables"] = list(self.require_variables)
        payload["exclude_variables"] = list(self.exclude_variables)
        payload["source_paths"] = {k: str(v) for k, v in self.source_paths.items()}
        if payload["llm"].get("api_key"):
            payload["llm"]["api_key"] = "***redacted***"
        if payload["llm"].get("cache_path") is not None:
            payload["llm"]["cache_path"] = str(payload["llm"]["cache_path"])
        if payload["outliers"].get("variables") is not None:
            payload["outliers"]["variables"] = list(payload["outliers"]["variables"])
        if payload["cohort"].get("icu_units") is not None:
            payload["cohort"]["icu_units"] = list(payload["cohort"]["icu_units"])
        if payload["fhir"].get("validator_path") is not None:
            payload["fhir"]["validator_path"] = str(payload["fhir"]["validator_path"])
        return payload

    def save_json(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreprocessConfig":
        # Accept either a standalone config or the redacted config embedded in
        # a public dataset manifest.
        if "config" in payload and "database" not in payload:
            embedded = payload["config"]
            if not isinstance(embedded, Mapping):
                raise ValueError("Manifest `config` must be a mapping.")
            payload = embedded
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                "Unknown configuration keys: {0}".format(", ".join(sorted(unknown)))
            )
        kwargs = dict(payload)
        if kwargs.get("llm", {}).get("api_key") == "***redacted***":
            kwargs["llm"] = dict(kwargs["llm"], api_key=None)
        return cls(**kwargs)

    @classmethod
    def load_json(cls, path: PathLike) -> "PreprocessConfig":
        with open(Path(path), "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
