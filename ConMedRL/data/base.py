"""The adapter contract, and the shared assembly that follows it.

A database adapter is responsible for exactly three things:

1. :meth:`BaseAdapter.build_cohort` -- decide which episodes are in the study
   and label their outcomes. One row per episode.
2. :meth:`BaseAdapter.extract_observations` -- pull the physiology as a long
   table of ``(episode_id, time_offset_hours, variable, value)``.
3. :meth:`BaseAdapter.action_events` -- say when the clinical action of
   interest occurred.

Everything after that -- pivoting to a state matrix, epoch alignment, cost
assignment, splitting, imputation, scaling, export -- is database-independent
and lives in :mod:`ConMedRL.data.pipeline`. That boundary is what the original
four scripts lacked: each re-implemented the whole chain, so a fix to one never
reached the others.

Time axis
---------
Adapters report time as ``time_offset_hours``: floating-point hours since **ICU
admission**. MIMIC-IV stores wall-clock timestamps and SICdb stores integer
seconds since *PDMS* admission (which may include a preceding surgery, hence
``cases.ICUOffset``); normalising both to hours-since-ICU-admission is what
lets one epoch definition serve both. Adapters may additionally carry a
``time`` datetime column for readability.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PreprocessConfig
from .llm import LLMBackend, NullLLMBackend
from .schema import VariableSpec, task_state_space
from .sources import SourceRegistry
from .unlearning import canonical_subject_id
from .variables import (
    DictionarySchema,
    VariableMapping,
    dictionary_schema_for,
    resolve_variable_mapping,
)

__all__ = [
    "EPISODE_KEY",
    "TIME_OFFSET_COLUMN",
    "CohortFrame",
    "ObservationFrame",
    "AdapterResult",
    "BaseAdapter",
]

logger = logging.getLogger(__name__)

#: Episode identifier used throughout the pipeline. Adapters map their native
#: key onto it (MIMIC-IV ``stay_id``, SICdb ``CaseID``) so splits, grouping and
#: next-state lookups have one column name to rely on.
EPISODE_KEY = "stay_id"

#: Canonical time axis: hours since ICU admission.
TIME_OFFSET_COLUMN = "time_offset_hours"


# Columns every adapter must place on its cohort frame.
_REQUIRED_COHORT_COLUMNS: Tuple[str, ...] = (
    EPISODE_KEY,
    "subject_id",
    "los",
)

# Columns every adapter must place on its observation frame.
_REQUIRED_OBSERVATION_COLUMNS: Tuple[str, ...] = (
    EPISODE_KEY,
    TIME_OFFSET_COLUMN,
    "variable",
    "value",
)


CohortFrame = pd.DataFrame
ObservationFrame = pd.DataFrame


@dataclass
class AdapterResult:
    """What an adapter hands back to the shared pipeline.

    Attributes
    ----------
    cohort:
        One row per episode. Must carry ``stay_id``, ``subject_id`` and ``los``
        (days), plus whatever outcome flags the task's cost function reads
        (``death_in_ICU``, ``readmission``, ``reintubation``, ...). Static
        per-episode state variables (``age``, ``M``, ``weight``) belong here
        too; the assembler broadcasts them across the episode's rows.
    observations:
        Long table: ``stay_id``, ``time_offset_hours``, ``variable`` (a
        canonical name), ``value`` (float).
    actions:
        ``stay_id`` -> ``time_offset_hours`` at which the terminal clinical
        action was taken, or ``NaN`` if it never was (censored episode).
    mapping:
        How canonical variables were resolved to source ids, for the report.
    report:
        Free-form provenance: row counts at each filter step, warnings.
    """

    cohort: CohortFrame
    observations: ObservationFrame
    actions: pd.DataFrame
    mapping: Optional[VariableMapping] = None
    report: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "AdapterResult":
        """Fail loudly on a malformed adapter output.

        Catching this here, rather than three transforms later, is the
        difference between "the SICdb adapter forgot ``los``" and a
        ``KeyError`` inside a groupby.
        """
        missing_cohort = [c for c in _REQUIRED_COHORT_COLUMNS if c not in self.cohort.columns]
        if missing_cohort:
            raise ValueError(
                "Adapter cohort frame is missing required column(s): {0}. "
                "Present: {1}".format(
                    ", ".join(missing_cohort), ", ".join(map(str, self.cohort.columns))
                )
            )

        missing_obs = [
            c for c in _REQUIRED_OBSERVATION_COLUMNS if c not in self.observations.columns
        ]
        if missing_obs:
            raise ValueError(
                "Adapter observation frame is missing required column(s): {0}. "
                "Present: {1}".format(
                    ", ".join(missing_obs), ", ".join(map(str, self.observations.columns))
                )
            )

        if self.cohort[EPISODE_KEY].duplicated().any():
            duplicates = int(self.cohort[EPISODE_KEY].duplicated().sum())
            raise ValueError(
                "Adapter cohort frame has {0} duplicated {1} value(s); it must "
                "be one row per episode.".format(duplicates, EPISODE_KEY)
            )

        if self.cohort.empty:
            raise ValueError(
                "The cohort is empty after filtering. Loosen the CohortConfig "
                "criteria, or check that the source tables cover the expected "
                "patients."
            )

        orphans = set(self.observations[EPISODE_KEY]) - set(self.cohort[EPISODE_KEY])
        if orphans:
            logger.warning(
                "Dropping observations for %d episode(s) absent from the "
                "cohort.", len(orphans),
            )
            self.observations = self.observations[
                self.observations[EPISODE_KEY].isin(set(self.cohort[EPISODE_KEY]))
            ].copy()

        negative = int((self.observations[TIME_OFFSET_COLUMN] < 0).sum())
        if negative:
            logger.info(
                "%d observation(s) precede ICU admission and will be dropped "
                "during epoch alignment.", negative,
            )

        return self


class BaseAdapter(abc.ABC):
    """Base class for database adapters.

    Subclasses implement the three abstract methods; the constructor and
    variable resolution are shared.
    """

    #: Canonical database name this adapter serves.
    database: str = ""

    #: Dictionary groups worth searching when resolving variables, e.g.
    #: ``("chartevents",)`` for MIMIC-IV.
    dictionary_groups: Tuple[str, ...] = ()

    #: Languages to translate canonical names into before fuzzy matching.
    target_languages: Tuple[str, ...] = ()

    def __init__(
        self,
        config: PreprocessConfig,
        sources: Optional[SourceRegistry] = None,
        llm: Optional[LLMBackend] = None,
    ) -> None:
        self.config = config
        self.llm = llm or NullLLMBackend()
        self.sources = sources or SourceRegistry(
            data_dir=config.data_dir,
            database=config.database,
            overrides=config.source_paths,
            compression=config.source_compression,
        )
        self.report: Dict[str, Any] = {}
        self._mapping: Optional[VariableMapping] = None

    # -- abstract surface -----------------------------------------------------

    @abc.abstractmethod
    def required_tables(self) -> List[str]:
        """Logical table names this adapter needs for the configured task."""

    @abc.abstractmethod
    def load_dictionary(self) -> pd.DataFrame:
        """Return the variable dictionary (``d_items`` / ``d_references``)."""

    @abc.abstractmethod
    def build_cohort(self) -> CohortFrame:
        """Select the study episodes and label their outcomes."""

    @abc.abstractmethod
    def extract_observations(
        self, cohort: CohortFrame, mapping: VariableMapping
    ) -> ObservationFrame:
        """Pull physiology for the cohort as a long canonical-name table."""

    @abc.abstractmethod
    def action_events(self, cohort: CohortFrame) -> pd.DataFrame:
        """When the terminal clinical action happened, per episode."""

    # -- shared ---------------------------------------------------------------

    @property
    def dictionary_schema(self) -> DictionarySchema:
        return dictionary_schema_for(self.database)

    def wanted_variables(self) -> List[VariableSpec]:
        """The task's canonical state space, minus anything excluded."""
        excluded = set(self.config.exclude_variables)
        return [spec for spec in task_state_space(self.config.task) if spec.name not in excluded]

    def resolve_variables(
        self, overrides: Optional[Mapping[str, Sequence[Any]]] = None
    ) -> VariableMapping:
        """Resolve the task's state space against this database's dictionary."""
        if self._mapping is not None:
            return self._mapping

        dictionary = self.load_dictionary()
        self._mapping = resolve_variable_mapping(
            dictionary=dictionary,
            database=self.database,
            wanted=self.wanted_variables(),
            schema=self.dictionary_schema,
            llm=self.llm,
            overrides=overrides,
            target_languages=self.target_languages,
            use_llm=self.llm.is_llm and self.config.llm_allow_mapping,
            allowed_groups=self.dictionary_groups or None,
        )
        return self._mapping

    def run(
        self, variable_overrides: Optional[Mapping[str, Sequence[Any]]] = None
    ) -> AdapterResult:
        """Execute the adapter and return a validated result."""
        self.sources.require(self.required_tables())

        mapping = self.resolve_variables(variable_overrides)

        cohort = self.build_cohort()
        cohort = self._apply_patient_withdrawals(cohort)
        logger.info("Cohort: %d episode(s)", len(cohort))

        cache_path = self._observation_cache_path(mapping)
        observations: Optional[pd.DataFrame] = None
        if cache_path is not None and cache_path.exists():
            try:
                observations = pd.read_parquet(cache_path)
                self.report["observation_cache"] = "hit"
                logger.info("Loaded %d cached observations from %s", len(observations), cache_path)
            except Exception as exc:  # cache failure must never break a run
                logger.warning("Ignoring unreadable observation cache %s: %s", cache_path, exc)

        if observations is None:
            observations = self.extract_observations(cohort, mapping)
            self.report["observation_cache"] = "miss"
            if cache_path is not None:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    observations.to_parquet(cache_path, index=False)
                except Exception as exc:
                    logger.warning("Could not write observation cache %s: %s", cache_path, exc)
        logger.info(
            "Extracted %d observation(s) across %d variable(s)",
            len(observations),
            observations["variable"].nunique() if len(observations) else 0,
        )

        actions = self.action_events(cohort)

        result = AdapterResult(
            cohort=cohort,
            observations=observations,
            actions=actions,
            mapping=mapping,
            report=dict(self.report),
        )
        return result.validate()

    def _apply_patient_withdrawals(self, cohort: CohortFrame) -> CohortFrame:
        """Remove exact subjects before observation extraction or cache use."""
        requested = tuple(self.config.withdrawn_subject_ids)
        if self.config.withdrawal_count and not requested:
            raise ValueError(
                "This config contains only redacted withdrawal metadata. Exact "
                "rebuild requires withdrawn_subject_ids to be supplied again."
            )
        if not requested:
            return cohort

        tokens = {canonical_subject_id(value) for value in requested}
        subject_tokens = cohort["subject_id"].map(canonical_subject_id)
        removed_mask = subject_tokens.isin(tokens)
        removed = cohort.loc[removed_mask]
        kept = cohort.loc[~removed_mask].copy()
        affected_subjects = int(removed["subject_id"].nunique(dropna=True))
        affected_episodes = int(len(removed))
        self.report["withdrawal"] = {
            "requested_subject_count": int(self.config.withdrawal_count),
            "request_digest": self.config.withdrawal_digest,
            "affected_subject_count": affected_subjects,
            "affected_episode_count": affected_episodes,
            "remaining_episode_count": int(len(kept)),
        }
        self._log_filter("patient withdrawal", len(cohort), len(kept))
        if kept.empty:
            raise ValueError("The cohort is empty after exact patient withdrawal.")
        return kept

    def _observation_cache_path(
        self, mapping: VariableMapping
    ) -> Optional[Path]:
        """Return a content-addressed cache path for expensive extraction.

        Source file size/mtime, cohort settings and resolved source IDs all
        participate in the key. Changing the database download, task filters,
        or a variable override therefore creates a new cache instead of
        silently reusing stale observations.
        """
        if not self.config.use_cache:
            return None
        sources: Dict[str, Dict[str, Any]] = {}
        for name in self.required_tables():
            path = self.sources.resolve(name, required=True)
            stat = path.stat()
            sources[name] = {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        payload = {
            "database": self.config.database,
            "task": self.config.task,
            "cohort": vars(self.config.cohort),
            "exclude_variables": list(self.config.exclude_variables),
            "withdrawal_digest": self.config.withdrawal_digest,
            "mapping": {
                name: list(entry.source_ids)
                for name, entry in sorted(mapping.entries.items())
                if entry.resolved
            },
            "sources": sources,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        filename = "{0}_{1}_observations_{2}.parquet".format(
            self.database.replace("-", "_"), self.config.task, digest
        )
        return self.config.cache_dir / filename

    # -- helpers for subclasses ----------------------------------------------

    def _log_filter(self, step: str, before: int, after: int) -> None:
        """Record a cohort filter's effect, so attrition is auditable."""
        self.report.setdefault("cohort_filters", []).append(
            {"step": step, "before": before, "after": after, "removed": before - after}
        )
        logger.info(
            "%-42s %7d -> %7d (removed %d)", step, before, after, before - after
        )

    @staticmethod
    def _to_long(
        wide: pd.DataFrame,
        id_columns: Sequence[str],
        value_columns: Sequence[str],
    ) -> pd.DataFrame:
        """Melt a wide observation block into the canonical long form."""
        present = [c for c in value_columns if c in wide.columns]
        if not present:
            return pd.DataFrame(columns=list(_REQUIRED_OBSERVATION_COLUMNS))
        long = wide.melt(
            id_vars=list(id_columns),
            value_vars=present,
            var_name="variable",
            value_name="value",
        )
        return long.dropna(subset=["value"])

    @staticmethod
    def _relabel_by_mapping(
        events: pd.DataFrame,
        id_column: str,
        mapping: VariableMapping,
    ) -> pd.DataFrame:
        """Replace source ids with canonical variable names, dropping the rest.

        Rows whose id is not claimed by any canonical variable are discarded
        here rather than carried along, which is what keeps a 300-million-row
        event table from being pivoted into hundreds of unused columns.
        """
        reverse = mapping.id_to_canonical()
        if not reverse:
            return events.iloc[0:0].assign(variable=pd.Series(dtype=object))
        out = events[events[id_column].isin(reverse.keys())].copy()
        out["variable"] = out[id_column].map(reverse)
        return out
