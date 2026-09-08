"""Unified end-to-end preprocessing entry point."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .adapters import GenericMimicAdapter, MimicIVAdapter, SICdbAdapter
from .base import EPISODE_KEY, TIME_OFFSET_COLUMN, AdapterResult, BaseAdapter
from .config import Database, LLMConfig, PreprocessConfig, SplitConfig, Task
from .dataset import (
    ROW_INDEX_COLUMN,
    RLDatasetBundle,
    SplitTables,
    compute_dataset_content_hash,
    validate_rl_contract,
)
from .export import write_bundle
from .llm import get_llm_backend
from .mdp import assign_costs, select_decision_points, split_by_group
from .schema import (
    CANONICAL_VARIABLES,
    StateSpaceSchema,
    VariableSpec,
    resolve_state_space,
    task_state_space,
)
from .tasks import register_declarative_task
from .specs import SafeRuleEvaluator
from .transforms import (
    GroupedFiller,
    OutlierFilter,
    StateImputer,
    StateScaler,
    clip_to_plausible_range,
    compute_coverage,
)

__all__ = ["build_dataset", "load_dataset"]

logger = logging.getLogger(__name__)


def _adapter_for(config: PreprocessConfig, llm: Any) -> BaseAdapter:
    if config.database == Database.MIMIC_IV:
        return MimicIVAdapter(config, llm=llm)
    if config.database == Database.SICDB:
        return SICdbAdapter(config, llm=llm)
    if config.database == Database.GENERIC:
        return GenericMimicAdapter(config, llm=llm)
    raise ValueError("Unsupported database: {0}".format(config.database))


def _coerce_config(
    config: Optional[Union[PreprocessConfig, Mapping[str, Any]]],
    *,
    database: Optional[str],
    task: Optional[str],
    data_dir: Optional[Union[str, Path]],
    output_dir: Optional[Union[str, Path]],
    output_formats: Sequence[str],
    llm_api_key: Optional[str],
    llm_provider: str,
    llm_model: Optional[str],
    overrides: Mapping[str, Any],
) -> PreprocessConfig:
    if config is not None:
        if any(value is not None for value in (database, task, data_dir, output_dir)):
            raise ValueError(
                "Pass either `config` or the database/task/data_dir arguments, not both."
            )
        if overrides or llm_api_key is not None or llm_provider != "none":
            raise ValueError(
                "When `config` is supplied, put LLM and preprocessing options on it."
            )
        if isinstance(config, PreprocessConfig):
            return config
        return PreprocessConfig.from_dict(config)

    missing = [
        name
        for name, value in (
            ("database", database),
            ("task", task),
            ("data_dir", data_dir),
        )
        if value is None
    ]
    if missing:
        raise TypeError("Missing required argument(s): {0}".format(", ".join(missing)))

    kwargs = dict(overrides)
    if llm_api_key is not None and llm_provider == "none":
        llm_provider = "openai"
    kwargs.update(
        {
            "database": database,
            "task": task,
            "data_dir": data_dir,
            "output_formats": output_formats,
            "llm": LLMConfig(
                provider=llm_provider, api_key=llm_api_key, model=llm_model
            ),
        }
    )
    if output_dir is not None:
        kwargs["output_dir"] = output_dir
    return PreprocessConfig(**kwargs)


def _assemble_epochs(result: AdapterResult, config: PreprocessConfig) -> pd.DataFrame:
    """Bin irregular observations onto one row per clinical decision epoch."""
    if config.task not in (Task.DISCHARGE, Task.EXTUBATION):
        return _assemble_custom_epochs(result, config)
    cohort = result.cohort.copy()
    observations = result.observations.copy()
    actions = result.actions.rename(
        columns={TIME_OFFSET_COLUMN: "_action_hours"}
    ).copy()
    cohort = cohort.merge(actions, on=EPISODE_KEY, how="inner", validate="one_to_one")

    if config.task == Task.EXTUBATION:
        if "intubation_hours" not in cohort:
            raise KeyError("The extubation cohort has no `intubation_hours` column.")
        cohort["_decision_start"] = cohort["intubation_hours"].astype(float)
    else:
        cohort["_decision_start"] = 0.0

    epoch_hours = float(config.cohort.decision_epoch_hours)
    duration = cohort["_action_hours"] - cohort["_decision_start"]
    cohort["_n_epochs"] = np.maximum(
        1, np.ceil(duration.clip(lower=0) / epoch_hours).astype(int)
    )

    # At most los_threshold/epoch_hours rows per stay under the default cohort
    # criteria, so constructing this compact grid is much cheaper than pivoting
    # the raw event table directly.
    repeated = cohort.loc[cohort.index.repeat(cohort["_n_epochs"])].copy()
    repeated["epoch"] = repeated.groupby(EPISODE_KEY, sort=False).cumcount() + 1
    repeated[TIME_OFFSET_COLUMN] = np.minimum(
        repeated["_decision_start"] + repeated["epoch"] * epoch_hours,
        repeated["_action_hours"],
    )

    timing = cohort[
        [EPISODE_KEY, "_decision_start", "_action_hours", "_n_epochs"]
    ]
    observations = observations.merge(timing, on=EPISODE_KEY, how="inner")
    observations = observations[
        (observations[TIME_OFFSET_COLUMN] >= observations["_decision_start"])
        & (observations[TIME_OFFSET_COLUMN] <= observations["_action_hours"])
    ].copy()
    observations["epoch"] = (
        np.floor(
            (observations[TIME_OFFSET_COLUMN] - observations["_decision_start"])
            / epoch_hours
        ).astype(int)
        + 1
    )
    observations["epoch"] = np.minimum(
        observations["epoch"], observations["_n_epochs"]
    )

    if observations.empty:
        aggregated = pd.DataFrame(columns=[EPISODE_KEY, "epoch"])
    else:
        dataset_payload = (
            config.dataset_spec.to_dict()
            if hasattr(config.dataset_spec, "to_dict")
            else dict(config.dataset_spec or {})
        )
        aggregation_by_variable = {
            str(rule.get("name")): str(rule.get("aggregation", "median")).lower()
            for rule in (
                dict(value) for value in dataset_payload.get("event_rules", ())
            )
        }
        pieces: List[pd.Series] = []
        for variable, block in observations.groupby(
            "variable", sort=False, observed=True
        ):
            method = aggregation_by_variable.get(str(variable), "median")
            if method == "none":
                method = "median"
            if method not in ("first", "last", "min", "max", "mean", "sum", "count", "median"):
                raise ValueError(
                    "Unsupported observation aggregation {0!r}.".format(method)
                )
            values = block.groupby([EPISODE_KEY, "epoch"], sort=False)["value"].agg(
                method
            )
            values.name = variable
            pieces.append(values)
        aggregated = (
            pd.concat(pieces, axis=1).reset_index()
            if pieces
            else pd.DataFrame(columns=[EPISODE_KEY, "epoch"])
        )
    observed_names = [
        c for c in aggregated.columns if c not in (EPISODE_KEY, "epoch")
    ]
    aggregated = aggregated.rename(
        columns={name: "__observed__{0}".format(name) for name in observed_names}
    )
    frame = repeated.merge(
        aggregated, on=[EPISODE_KEY, "epoch"], how="left", validate="one_to_one"
    )

    # Measurement values override static cohort values of the same name, while
    # demographics fall back to the per-stay value.
    for spec in task_state_space(config.task):
        measured = "__observed__{0}".format(spec.name)
        if measured in frame and spec.name in frame:
            frame[spec.name] = frame[measured].combine_first(frame[spec.name])
        elif measured in frame:
            frame[spec.name] = frame[measured]

    if config.task == Task.DISCHARGE:
        action_name = "discharge_action"
    else:
        action_name = "extubation_action"
    frame[action_name] = (
        frame["epoch"].to_numpy() == frame["_n_epochs"].to_numpy()
    ).astype(float)

    # Derived variables are computed only where their inputs exist. The dynamic
    # state-space resolver will drop a derived column with zero training
    # coverage rather than inventing it.
    if "Arterial O2 pressure" in frame and "Inspired O2 Fraction" in frame:
        fio2 = pd.to_numeric(frame["Inspired O2 Fraction"], errors="coerce")
        fio2_fraction = np.where(fio2 > 1.5, fio2 / 100.0, fio2)
        calculated_pf = pd.to_numeric(
            frame["Arterial O2 pressure"], errors="coerce"
        ) / fio2_fraction
        if "PaO2/FiO2 Ratio" in frame:
            frame["PaO2/FiO2 Ratio"] = frame[
                "PaO2/FiO2 Ratio"
            ].combine_first(pd.Series(calculated_pf, index=frame.index))
        else:
            frame["PaO2/FiO2 Ratio"] = calculated_pf
    if "Respiratory Rate" in frame and "Tidal Volume" in frame:
        tidal_litres = pd.to_numeric(frame["Tidal Volume"], errors="coerce") / 1000.0
        frame["Rapid Shallow Breathing Index"] = pd.to_numeric(
            frame["Respiratory Rate"], errors="coerce"
        ) / tidal_litres.replace(0, np.nan)
    if config.task == Task.EXTUBATION:
        frame["Mechanical Ventilation Duration"] = (
            frame[TIME_OFFSET_COLUMN] - frame["_decision_start"]
        ).clip(lower=0)
        frame["extubation_count"] = 0.0

    # A portable, human-readable time axis. MIMIC retains real timestamps;
    # SICdb has no public wall-clock date, so its numeric offset is the honest
    # representation rather than a fabricated date.
    if "intime" in frame:
        frame["time"] = frame["intime"] + pd.to_timedelta(
            frame[TIME_OFFSET_COLUMN], unit="h"
        )
    else:
        frame["time"] = frame[TIME_OFFSET_COLUMN]

    temporary = [
        c
        for c in frame.columns
        if c.startswith("__observed__")
        or c in ("_decision_start", "_action_hours", "_n_epochs")
    ]
    return frame.drop(columns=temporary).sort_values(
        [EPISODE_KEY, "epoch"], kind="mergesort"
    ).reset_index(drop=True)


def _assemble_custom_epochs(
    result: AdapterResult, config: PreprocessConfig
) -> pd.DataFrame:
    """Assemble an approved custom task, including non-terminal actions."""
    task_spec = dict(config.task_spec or {})
    timeline = dict(task_spec.get("timeline") or {})
    action_spec = dict(task_spec.get("action") or {})
    cohort = result.cohort.copy()
    observations = result.observations.copy()
    epoch_hours = float(
        timeline.get(
            "decision_epoch_hours",
            task_spec.get(
                "decision_epoch_hours", config.cohort.decision_epoch_hours
            ),
        )
    )

    start_column = timeline.get("start_column")
    if start_column and start_column in cohort:
        start = cohort[start_column]
        if pd.api.types.is_datetime64_any_dtype(start):
            cohort["_decision_start"] = (
                start - pd.to_datetime(cohort["intime"])
            ).dt.total_seconds() / 3600.0
        else:
            cohort["_decision_start"] = pd.to_numeric(start, errors="coerce")
    else:
        cohort["_decision_start"] = float(timeline.get("start_hours", 0.0))

    end_column = timeline.get("end_column")
    if end_column and end_column in cohort:
        end = cohort[end_column]
        if pd.api.types.is_datetime64_any_dtype(end):
            cohort["_decision_end"] = (
                end - pd.to_datetime(cohort["intime"])
            ).dt.total_seconds() / 3600.0
        else:
            cohort["_decision_end"] = pd.to_numeric(end, errors="coerce")
            if timeline.get("end_unit") == "days":
                cohort["_decision_end"] *= 24.0
    elif "outtime" in cohort and "intime" in cohort:
        cohort["_decision_end"] = (
            pd.to_datetime(cohort["outtime"]) - pd.to_datetime(cohort["intime"])
        ).dt.total_seconds() / 3600.0
    else:
        cohort["_decision_end"] = pd.to_numeric(
            cohort["los"], errors="coerce"
        ) * 24.0

    duration = (cohort["_decision_end"] - cohort["_decision_start"]).clip(lower=0)
    cohort["_n_epochs"] = np.maximum(
        1, np.ceil(duration / epoch_hours).astype(int)
    )
    repeated = cohort.loc[cohort.index.repeat(cohort["_n_epochs"])].copy()
    repeated["epoch"] = repeated.groupby(EPISODE_KEY, sort=False).cumcount() + 1
    repeated[TIME_OFFSET_COLUMN] = np.minimum(
        repeated["_decision_start"] + repeated["epoch"] * epoch_hours,
        repeated["_decision_end"],
    )

    timing = cohort[
        [EPISODE_KEY, "_decision_start", "_decision_end", "_n_epochs"]
    ]
    observations = observations.merge(timing, on=EPISODE_KEY, how="inner")
    observations = observations[
        (observations[TIME_OFFSET_COLUMN] >= observations["_decision_start"])
        & (observations[TIME_OFFSET_COLUMN] <= observations["_decision_end"])
    ].copy()
    observations["epoch"] = (
        np.floor(
            (observations[TIME_OFFSET_COLUMN] - observations["_decision_start"])
            / epoch_hours
        ).astype(int)
        + 1
    )
    observations["epoch"] = np.minimum(
        observations["epoch"], observations["_n_epochs"]
    )
    if observations.empty:
        aggregated = pd.DataFrame(columns=[EPISODE_KEY, "epoch"])
    else:
        dataset_payload = (
            config.dataset_spec.to_dict()
            if hasattr(config.dataset_spec, "to_dict")
            else dict(config.dataset_spec or {})
        )
        aggregation_by_variable = {
            str(rule.get("name")): str(rule.get("aggregation", "median")).lower()
            for rule in (
                dict(value) for value in dataset_payload.get("event_rules", ())
            )
        }
        pieces: List[pd.Series] = []
        for variable, block in observations.groupby(
            "variable", sort=False, observed=True
        ):
            method = aggregation_by_variable.get(str(variable), "median")
            if method == "none":
                method = "median"
            if method not in (
                "first", "last", "min", "max", "mean", "sum", "count", "median"
            ):
                raise ValueError(
                    "Unsupported observation aggregation {0!r}.".format(method)
                )
            values = block.groupby(
                [EPISODE_KEY, "epoch"], sort=False
            )["value"].agg(method)
            values.name = variable
            pieces.append(values)
        aggregated = (
            pd.concat(pieces, axis=1).reset_index()
            if pieces
            else pd.DataFrame(columns=[EPISODE_KEY, "epoch"])
        )
    observed_names = [
        column for column in aggregated if column not in (EPISODE_KEY, "epoch")
    ]
    aggregated = aggregated.rename(
        columns={
            name: "__observed__{0}".format(name) for name in observed_names
        }
    )
    frame = repeated.merge(
        aggregated, on=[EPISODE_KEY, "epoch"], how="left", validate="one_to_one"
    )
    for variable in task_state_space(config.task):
        measured = "__observed__{0}".format(variable.name)
        if measured in frame and variable.name in frame:
            frame[variable.name] = frame[measured].combine_first(frame[variable.name])
        elif measured in frame:
            frame[variable.name] = frame[measured]

    action_columns = list(
        action_spec.get("columns") or (action_spec.get("name") or "action",)
    )
    actions = result.actions.copy()
    if len(actions):
        actions = actions.merge(timing, on=EPISODE_KEY, how="inner")
        actions = actions[
            (actions[TIME_OFFSET_COLUMN] >= actions["_decision_start"])
            & (actions[TIME_OFFSET_COLUMN] <= actions["_decision_end"])
        ].copy()
        actions["epoch"] = (
            np.floor(
                (actions[TIME_OFFSET_COLUMN] - actions["_decision_start"])
                / epoch_hours
            ).astype(int)
            + 1
        )
        actions["epoch"] = np.minimum(actions["epoch"], actions["_n_epochs"])
        present = [column for column in action_columns if column in actions]
        if present:
            aggregation = str(action_spec.get("aggregation", "last")).lower()
            if aggregation not in ("last", "mean", "sum", "max", "min"):
                raise ValueError(
                    "Unsupported custom action aggregation {0!r}.".format(aggregation)
                )
            action_table = (
                actions.groupby([EPISODE_KEY, "epoch"], sort=False)[present]
                .agg(aggregation)
                .reset_index()
            )
            frame = frame.merge(
                action_table, on=[EPISODE_KEY, "epoch"], how="left"
            )
    expression = action_spec.get("expression")
    if expression is not None:
        context = {name: frame[name] for name in frame.columns}
        computed = SafeRuleEvaluator().evaluate(expression, context)
        values = np.asarray(computed)
        if len(action_columns) == 1:
            frame[action_columns[0]] = (
                values.item() if values.ndim == 0 else values
            )
        elif values.ndim == 2 and values.shape[1] == len(action_columns):
            for index, column in enumerate(action_columns):
                frame[column] = values[:, index]
        elif all(column in frame for column in action_columns):
            pass
        else:
            raise ValueError(
                "Multi-column action expression must produce an N x action_dim array."
            )

    context = {name: frame[name] for name in frame.columns}
    if task_spec.get("terminal_rule") is not None:
        terminal_values = np.asarray(
            SafeRuleEvaluator().evaluate(task_spec["terminal_rule"], context), dtype=float
        )
        frame["__terminal_rule__"] = (
            terminal_values.item() if terminal_values.ndim == 0 else terminal_values
        )
    if task_spec.get("censoring_rule") is not None:
        censor_values = np.asarray(
            SafeRuleEvaluator().evaluate(task_spec["censoring_rule"], context), dtype=float
        )
        frame["__censoring_rule__"] = (
            censor_values.item() if censor_values.ndim == 0 else censor_values
        )
    default = action_spec.get("default", 0.0)
    carry_forward = bool(action_spec.get("carry_forward", False))
    for column in action_columns:
        if column not in frame:
            frame[column] = default
        elif carry_forward:
            frame[column] = frame.groupby(EPISODE_KEY, sort=False)[column].ffill()
        frame[column] = frame[column].fillna(default)

    if "intime" in frame:
        frame["time"] = pd.to_datetime(frame["intime"]) + pd.to_timedelta(
            frame[TIME_OFFSET_COLUMN], unit="h"
        )
    else:
        frame["time"] = frame[TIME_OFFSET_COLUMN]
    return frame.drop(
        columns=[
            column
            for column in frame
            if column.startswith("__observed__")
            or column in ("_decision_start", "_decision_end", "_n_epochs")
        ],
        errors="ignore",
    ).sort_values([EPISODE_KEY, "epoch"], kind="mergesort").reset_index(drop=True)


def _resolve_schema(
    splits: Dict[str, pd.DataFrame],
    result: AdapterResult,
    config: PreprocessConfig,
) -> StateSpaceSchema:
    candidates = [spec.name for spec in task_state_space(config.task)]
    coverage = compute_coverage(splits["train"], candidates)
    available = [name for name, value in coverage.items() if value > 0.0]

    min_coverage = config.min_variable_coverage
    if config.imputation.drop_sparse_columns:
        min_coverage = max(
            min_coverage, 1.0 - config.imputation.missing_threshold_drop
        )
    schema = resolve_state_space(
        task=config.task,
        database=config.database,
        available=available,
        coverage=coverage,
        min_coverage=min_coverage,
        require=config.require_variables,
        exclude=config.exclude_variables,
    )

    # Replace a generic "not available" message with the database-specific,
    # audited reason from the mapping (for example SICdb's Quick percentage is
    # not interchangeable with a PT measured in seconds).
    if result.mapping is not None:
        for name, reason in list(schema.dropped.items()):
            entry = result.mapping.entries.get(name)
            if entry is not None and entry.notes and "not available" in reason:
                schema.dropped[name] = entry.notes
    return schema


def _transform_splits(
    splits: Dict[str, pd.DataFrame],
    schema: StateSpaceSchema,
    config: PreprocessConfig,
) -> Tuple[Dict[str, pd.DataFrame], StateScaler, Dict[str, Any]]:
    names = schema.names
    report: Dict[str, Any] = {}

    outlier = OutlierFilter(config.outliers)
    outlier.fit(splits["train"], names)
    for split_name, frame in splits.items():
        splits[split_name], cleared = outlier.transform(frame)
        if cleared:
            report["statistical_outliers_{0}".format(split_name)] = cleared

    filler = GroupedFiller(
        group_key=EPISODE_KEY,
        forward_fill=config.imputation.forward_fill,
        # Backward fill would copy a future measurement into an earlier policy
        # state. It is deliberately disabled even when interpolation is opted in.
        backward_fill=False,
        interpolate=config.imputation.linear_interpolate,
    )
    for split_name, frame in splits.items():
        splits[split_name] = filler.transform(frame, names)

    imputer = StateImputer(config.imputation)
    imputer.fit(splits["train"], names)
    for split_name, frame in splits.items():
        splits[split_name] = imputer.transform(frame)
    report["imputation_buckets"] = dict(imputer.buckets_)

    scaler = StateScaler()
    scaler.fit(splits["train"], names)
    return splits, scaler, report


def _make_split_tables(
    split_name: str,
    frame: pd.DataFrame,
    schema: StateSpaceSchema,
    scaler: StateScaler,
    cost_spec: Any,
) -> Tuple[SplitTables, pd.DataFrame]:
    frame = frame.reset_index(drop=True)
    frame[ROW_INDEX_COLUMN] = np.arange(len(frame), dtype=np.int64)
    raw_state = frame[schema.names].copy()
    state = scaler.transform(frame).reset_index(drop=True)

    all_state_names = set(CANONICAL_VARIABLES)
    outcome_columns = [c for c in frame.columns if c not in all_state_names]
    outcome = frame[outcome_columns].reset_index(drop=True)

    if split_name == "train":
        return SplitTables(split_name, outcome, state), raw_state

    selected = select_decision_points(
        outcome, cost_spec, group_key=EPISODE_KEY, strategy="first"
    )
    parent_positions = selected[ROW_INDEX_COLUMN].to_numpy(dtype=int)
    state_select = state.iloc[parent_positions].reset_index(drop=True)
    outcome_select = selected.reset_index(drop=True)
    return (
        SplitTables(
            split_name,
            outcome,
            state,
            outcome_select=outcome_select,
            state_select=state_select,
        ),
        raw_state,
    )


def _resolve_split_config(config: PreprocessConfig) -> SplitConfig:
    """Resolve task-specific grouping without overriding an explicit choice."""
    if config.split.group_key is not None:
        return config.split
    if config.task == Task.EXTUBATION:
        return replace(
            config.split,
            group_key=EPISODE_KEY,
            stratify_column=config.split.stratify_column or "extubation_fail",
        )
    return replace(config.split, group_key="subject_id")


def build_dataset(
    config: Optional[Union[PreprocessConfig, Mapping[str, Any]]] = None,
    *,
    database: Optional[str] = None,
    task: Optional[str] = None,
    data_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    output_formats: Sequence[str] = ("csv",),
    llm_api_key: Optional[str] = None,
    llm_provider: str = "none",
    llm_model: Optional[str] = None,
    variable_overrides: Optional[Mapping[str, Sequence[Any]]] = None,
    lineage: Optional[Mapping[str, Any]] = None,
    write: bool = True,
    **config_options: Any,
) -> RLDatasetBundle:
    """Build ConMedRL-ready data from MIMIC-IV or SICdb.

    Either pass a :class:`PreprocessConfig`, or use the concise keyword API::

        bundle = build_dataset(
            database="sicdb",
            task="extubation",
            data_dir="/data/sicdb",
            output_dir="./processed",
            output_formats=("csv", "parquet", "d3rlpy"),
            llm_provider="openai",
            llm_api_key="...",  # optional; patient data is never sent
        )
    """
    cfg = _coerce_config(
        config,
        database=database,
        task=task,
        data_dir=data_dir,
        output_dir=output_dir,
        output_formats=output_formats,
        llm_api_key=llm_api_key,
        llm_provider=llm_provider,
        llm_model=llm_model,
        overrides=config_options,
    )

    if cfg.task not in (Task.DISCHARGE, Task.EXTUBATION):
        register_declarative_task(cfg.task_spec, replace=True)

    logging.getLogger("ConMedRL.data").setLevel(
        logging.DEBUG if cfg.verbosity >= 2 else logging.INFO
    )
    llm = get_llm_backend(cfg.llm)
    adapter = _adapter_for(cfg, llm)
    result = adapter.run(variable_overrides)
    frame = _assemble_epochs(result, cfg)

    # Physiological ranges are clinical constants, not learned statistics, so
    # applying them before splitting does not leak validation information.
    candidate_ranges = {
        spec.name: spec.plausible_range
        for spec in task_state_space(cfg.task)
        if spec.plausible_range is not None
    }
    frame, implausible = clip_to_plausible_range(frame, candidate_ranges)
    resolved_split = _resolve_split_config(cfg)
    cfg.split = resolved_split
    split_frames = split_by_group(frame, resolved_split)
    schema = _resolve_schema(split_frames, result, cfg)
    split_frames, scaler, transform_report = _transform_splits(
        split_frames, schema, cfg
    )

    cost_spec = None
    for name in ("train", "val", "test"):
        split_frames[name], cost_spec = assign_costs(
            split_frames[name], cfg.task, cfg.cohort, group_key=EPISODE_KEY
        )

    tables: Dict[str, SplitTables] = {}
    raw_states: Dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        tables[name], raw_states[name] = _make_split_tables(
            name, split_frames[name], schema, scaler, cost_spec
        )

    report = dict(result.report)
    report.update(transform_report)
    report["implausible_values_cleared"] = implausible
    report["split"] = {
        "group_key": resolved_split.group_key,
        "stratify_column": resolved_split.stratify_column,
        "random_seed": resolved_split.random_seed,
    }
    report["mapping"] = (
        result.mapping.to_frame().to_dict(orient="records")
        if result.mapping is not None
        else []
    )
    if lineage is not None:
        lineage_report = dict(lineage)
        withdrawal = dict(result.report.get("withdrawal") or {})
        remaining_transitions = int(len(frame))
        parent_transitions = int(
            lineage_report.pop("parent_transition_count", remaining_transitions)
        )
        lineage_report["affected_counts"] = {
            "subjects": int(withdrawal.get("affected_subject_count", 0)),
            "episodes": int(withdrawal.get("affected_episode_count", 0)),
            "transitions": max(parent_transitions - remaining_transitions, 0),
            "remaining_episodes": int(
                withdrawal.get("remaining_episode_count", len(result.cohort))
            ),
            "remaining_transitions": remaining_transitions,
        }
        report["lineage"] = lineage_report

    bundle = RLDatasetBundle(
        config=cfg,
        schema=schema,
        train=tables["train"],
        val=tables["val"],
        test=tables["test"],
        terminal_state=np.zeros(schema.state_dim, dtype=np.float32),
        action_name=cost_spec.action_column,
        action_type=cost_spec.action_type,
        action_columns=cost_spec.action_names,
        action_bounds=(
            {
                column: tuple(bounds)
                for column, bounds in zip(
                    cost_spec.action_names, cost_spec.action_bounds
                )
            }
            or None
        ),
        action_categories=(
            cost_spec.action_categories
            or (
                (0, 1)
                if cost_spec.action_type == "discrete"
                and cost_spec.task in (Task.DISCHARGE, Task.EXTUBATION)
                else None
            )
        ),
        num_constraints=cost_spec.num_constraints,
        scaler=scaler,
        raw_state_tables=raw_states,
        report=report,
    )
    bundle.report["contract_validation"] = validate_rl_contract(bundle)
    bundle.content_hash = compute_dataset_content_hash(bundle)
    if write:
        write_bundle(bundle)
    return bundle


def _portable_path(manifest_dir: Path, recorded: str) -> Path:
    path = Path(recorded)
    if path.exists():
        return path
    relocated = manifest_dir / path.name
    if relocated.exists():
        return relocated
    raise FileNotFoundError("Dataset artefact is missing: {0}".format(recorded))


def load_dataset(
    manifest_path: Union[str, Path],
    *,
    allow_invalidated: bool = False,
) -> RLDatasetBundle:
    """Reload a dataset written by :func:`build_dataset`.

    Superseded datasets are rejected by default after a patient-withdrawal
    rebuild. ``allow_invalidated=True`` exists only for controlled audit work.
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("status", "active") == "invalidated" and not allow_invalidated:
        raise ValueError(
            "Dataset manifest is invalidated by a patient-withdrawal request; "
            "load its recorded successor instead."
        )
    directory = manifest_path.parent
    files = manifest["written_files"]
    cfg = PreprocessConfig.from_dict(manifest["config"])

    variables = []
    for item in manifest["schema"]["variables"]:
        variables.append(
            CANONICAL_VARIABLES.get(
                item["name"],
                VariableSpec(item["name"], item.get("kind", "unknown"), item.get("unit")),
            )
        )
    schema = StateSpaceSchema(
        task=manifest["schema"]["task"],
        database=manifest["schema"]["database"],
        variables=variables,
        dropped=dict(manifest["schema"].get("dropped", {})),
        coverage=dict(manifest["schema"].get("coverage", {})),
    )

    tables: Dict[str, SplitTables] = {}
    raw_states: Dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        outcome = pd.read_csv(
            _portable_path(directory, files["outcome_table_{0}".format(name)])
        )
        state = pd.read_csv(
            _portable_path(directory, files["state_var_table_{0}".format(name)])
        )
        outcome_select = state_select = None
        outcome_key = "outcome_table_{0}_select".format(name)
        state_key = "state_var_table_{0}_select".format(name)
        if outcome_key in files and state_key in files:
            outcome_select = pd.read_csv(
                _portable_path(directory, files[outcome_key])
            )
            state_select = pd.read_csv(_portable_path(directory, files[state_key]))
        tables[name] = SplitTables(
            name, outcome, state, outcome_select, state_select
        )
        raw_key = "state_var_table_{0}_unscaled".format(name)
        if raw_key in files:
            raw_states[name] = pd.read_csv(
                _portable_path(directory, files[raw_key])
            )

    terminal = pd.read_csv(
        _portable_path(directory, files["terminal_state"])
    ).iloc[0].to_numpy(dtype=np.float32)
    scaler = None
    if "scaler" in files:
        import joblib

        scaler = joblib.load(_portable_path(directory, files["scaler"]))

    bundle = RLDatasetBundle(
        config=cfg,
        schema=schema,
        train=tables["train"],
        val=tables["val"],
        test=tables["test"],
        terminal_state=terminal,
        action_name=manifest["action_name"],
        action_type=manifest.get("action_type", "discrete"),
        action_columns=tuple(
            manifest.get("action_columns", (manifest["action_name"],))
        ),
        action_bounds=manifest.get("action_bounds") or None,
        action_categories=manifest.get("action_categories"),
        num_constraints=int(manifest["num_constraints"]),
        scaler=scaler,
        raw_state_tables=raw_states,
        report=dict(manifest.get("report", {})),
        written_files=dict(files),
        content_hash=manifest.get("content_hash"),
    )
    expected_hash = manifest.get("content_hash")
    actual_hash = compute_dataset_content_hash(bundle)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(
            "Dataset content hash mismatch: manifest records {0}, loaded "
            "artefacts produce {1}.".format(expected_hash, actual_hash)
        )
    bundle.content_hash = actual_hash
    bundle.report["content_hash_validation"] = {
        "present_in_manifest": expected_hash is not None,
        "valid": True,
        "content_hash": actual_hash,
    }
    bundle.report["contract_validation"] = validate_rl_contract(bundle)
    return bundle
