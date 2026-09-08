"""Turning a per-timestep clinical table into an MDP: epochs, costs, splits.

This is the database-agnostic half of the pipeline. An adapter's job ends when
it has produced one tidy frame -- one row per (stay, decision epoch), carrying
the resolved state variables plus the outcome bookkeeping columns. Everything
from there to ``obj_cost`` / ``con_cost_i`` / ``done`` is shared, which is what
keeps the discharge and extubation tasks from drifting apart across databases.

Sign convention: these are **costs**, and the OCRL agents minimise them. A cost
of 0 is the good outcome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CohortConfig, PreprocessConfig, SplitConfig, Task
from .dataset import ROW_INDEX_COLUMN

__all__ = [
    "CostSpec",
    "TASK_COSTS",
    "task_cost_spec",
    "register_cost_spec",
    "denote_decision_epochs",
    "assign_costs",
    "split_by_group",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostSpec:
    """What the objective and each constraint mean for one task.

    ``builder`` receives the assembled frame plus the cohort config and returns
    a frame of named cost columns. Keeping it a function rather than a formula
    string lets a task express costs that depend on several outcome columns.
    """

    task: str
    action_column: str
    objective: str
    constraints: Tuple[str, ...]
    builder: Callable[[pd.DataFrame, CohortConfig], pd.DataFrame]
    description: str = ""
    action_type: str = "discrete"
    action_columns: Tuple[str, ...] = ()
    action_categories: Tuple[Any, ...] = ()
    action_bounds: Tuple[Tuple[float, float], ...] = ()
    terminal_column: Optional[str] = None
    censoring_column: Optional[str] = None

    @property
    def num_constraints(self) -> int:
        return len(self.constraints)

    @property
    def action_names(self) -> Tuple[str, ...]:
        return self.action_columns or (self.action_column,)


def _discharge_costs(frame: pd.DataFrame, cohort: CohortConfig) -> pd.DataFrame:
    """Discharge task: minimise death, subject to readmission and LOS limits.

    * objective -- ``mortality_costs``: 1 if the patient died in the ICU or
      within ``death_observation_days`` of leaving it. Charged once, on the
      transition where the discharge happened.
    * constraint 0 -- ``readmission_costs``: 1 if the patient bounced back to
      the ICU within ``readmission_observation_days``.
    * constraint 1 -- ``los_costs_scaled``: one unit per decision epoch the
      patient remains in the ICU, so the dual variable prices delay.
    """
    action = frame["discharge_action"].to_numpy(dtype=float)
    discharged = action > 0.0

    died = np.zeros(len(frame), dtype=float)
    for column in ("death_in_ICU", "death_out_ICU"):
        if column in frame.columns:
            died = np.maximum(died, frame[column].fillna(0).to_numpy(dtype=float))
    # Mortality is attributed to the discharge decision, not to waiting.
    mortality_costs = np.where(discharged, died, 0.0)

    readmitted = (
        frame["readmission"].fillna(0).to_numpy(dtype=float)
        if "readmission" in frame.columns
        else np.zeros(len(frame), dtype=float)
    )
    readmission_costs = np.where(discharged, readmitted, 0.0)

    # Staying costs one epoch of ICU time; discharging costs none.
    epoch_hours = float(cohort.decision_epoch_hours)
    los_costs = np.where(discharged, 0.0, epoch_hours)
    los_costs_scaled = np.where(discharged, 0.0, 1.0)

    return pd.DataFrame(
        {
            "mortality_costs": mortality_costs,
            "readmission_costs": readmission_costs,
            "los_costs": los_costs,
            "los_costs_scaled": los_costs_scaled,
        },
        index=frame.index,
    )


def _extubation_costs(frame: pd.DataFrame, cohort: CohortConfig) -> pd.DataFrame:
    """Extubation task: minimise extubation failure, subject to ICU LOS.

    * objective -- ``extubation_failure_costs``: 1 when an extubation action is
      followed by extubation failure/reintubation in the observation window.
    * constraint 0 -- ``icu_los_costs`` (hours): if extubated, the observed
      remaining ICU length of stay; if not, one decision epoch. This preserves
      the original extubation experiment's RLOS-vs-waiting cost without
      changing its clinical unit.
    """
    action = frame["extubation_action"].to_numpy(dtype=float)
    extubated = action > 0.0

    failed = np.zeros(len(frame), dtype=float)
    for column in ("reintubation", "extubation_fail"):
        if column in frame.columns:
            failed = np.maximum(failed, frame[column].fillna(0).to_numpy(dtype=float))
    extubation_failure_costs = np.where(extubated, failed, 0.0)

    epoch_hours = float(cohort.decision_epoch_hours)
    if "time_offset_hours" in frame.columns:
        elapsed_hours = frame["time_offset_hours"].fillna(0).to_numpy(dtype=float)
    elif "epoch" in frame.columns:
        elapsed_hours = frame["epoch"].fillna(1).to_numpy(dtype=float) * epoch_hours
    else:
        elapsed_hours = np.zeros(len(frame), dtype=float)
    total_los_hours = (
        frame["los"].fillna(0).to_numpy(dtype=float) * 24.0
        if "los" in frame.columns
        else np.zeros(len(frame), dtype=float)
    )
    remaining_icu_los = np.maximum(total_los_hours - elapsed_hours, 0.0)
    icu_los_costs = np.where(extubated, remaining_icu_los, epoch_hours)

    return pd.DataFrame(
        {
            "extubation_failure_costs": extubation_failure_costs,
            "icu_los_costs": icu_los_costs,
        },
        index=frame.index,
    )


#: Cost structure per task.
TASK_COSTS: Dict[str, CostSpec] = {
    Task.DISCHARGE: CostSpec(
        task=Task.DISCHARGE,
        action_column="discharge_action",
        objective="mortality_costs",
        constraints=("readmission_costs", "los_costs_scaled"),
        builder=_discharge_costs,
        description=(
            "Minimise post-discharge mortality subject to a readmission-rate "
            "limit and an ICU length-of-stay limit."
        ),
    ),
    Task.EXTUBATION: CostSpec(
        task=Task.EXTUBATION,
        action_column="extubation_action",
        objective="extubation_failure_costs",
        constraints=("icu_los_costs",),
        builder=_extubation_costs,
        description=(
            "Minimise extubation failure subject to one remaining-ICU-"
            "length-of-stay constraint."
        ),
    ),
}


def task_cost_spec(task: str) -> CostSpec:
    key = str(task).strip().lower().replace("_", "-").replace(" ", "-")
    if key in TASK_COSTS:
        return TASK_COSTS[key]
    return TASK_COSTS[Task.normalize(task)]


def register_cost_spec(spec: CostSpec, replace: bool = False) -> str:
    """Register an approved custom task cost builder."""
    key = str(spec.task).strip().lower().replace("_", "-").replace(" ", "-")
    if key in TASK_COSTS and not replace:
        raise ValueError("Cost specification {0!r} is already registered.".format(key))
    if not spec.action_column:
        raise ValueError("CostSpec.action_column cannot be empty.")
    if not spec.objective:
        raise ValueError("CostSpec.objective cannot be empty.")
    TASK_COSTS[key] = spec
    return key


# ---------------------------------------------------------------------------
# Decision epochs
# ---------------------------------------------------------------------------


def denote_decision_epochs(
    frame: pd.DataFrame,
    group_key: str = "stay_id",
    time_column: str = "time_offset_hours",
    start_column: str = "icu_starttime",
    epoch_hours: float = 12.0,
) -> pd.DataFrame:
    """Add an ``epoch`` column counting decision points within each stay.

    Epochs are anchored to each stay's own admission time, so epoch 1 always
    means "the first review after admission" regardless of wall-clock time.
    Rows are returned sorted by ``(group_key, time_column)`` with a fresh
    ``RangeIndex``, because both the data loaders and the within-stay fillers
    assume consecutive positional order.
    """
    for column in (group_key, time_column):
        if column not in frame.columns:
            raise KeyError("denote_decision_epochs needs a {0!r} column.".format(column))
    if epoch_hours <= 0:
        raise ValueError("epoch_hours must be positive, got {0}".format(epoch_hours))

    result = frame.copy()
    numeric_time = pd.to_numeric(result[time_column], errors="coerce")
    if numeric_time.notna().any():
        if start_column in result.columns:
            numeric_start = pd.to_numeric(result[start_column], errors="coerce")
            elapsed_hours = numeric_time - numeric_start.fillna(0.0)
        else:
            elapsed_hours = numeric_time
    else:
        # Backward-compatible path for wall-clock timestamp inputs.
        result[time_column] = pd.to_datetime(result[time_column], errors="coerce")
        if start_column in result.columns:
            anchor = pd.to_datetime(result[start_column], errors="coerce")
        else:
            anchor = result.groupby(group_key, sort=False)[time_column].transform("min")
        elapsed_hours = (result[time_column] - anchor).dt.total_seconds() / 3600.0
    # Epoch 1 covers the first `epoch_hours` after admission.
    result["epoch"] = np.floor(elapsed_hours / float(epoch_hours)).astype("Int64") + 1
    result.loc[result["epoch"] < 1, "epoch"] = 1

    result = result.sort_values([group_key, time_column], kind="mergesort")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cost assignment
# ---------------------------------------------------------------------------


def assign_costs(
    frame: pd.DataFrame,
    task: str,
    cohort: CohortConfig,
    group_key: str = "stay_id",
    los_scaler: Optional[Any] = None,
) -> Tuple[pd.DataFrame, CostSpec]:
    """Attach ``done``, ``obj_cost`` and ``con_cost_{i}`` to ``frame``.

    ``done`` marks the transition after which the episode has no successor:
    either the clinician took the terminal action, or the stay's record ends.
    Both cases matter -- a truncated stay whose ``done`` stays 0 would have its
    next-state read from the following *patient*.
    """
    spec = task_cost_spec(task)

    missing_actions = [name for name in spec.action_names if name not in frame.columns]
    if missing_actions:
        raise KeyError(
            "Task {0!r} expects action column(s) {1}; the adapter did not "
            "produce them.".format(spec.task, missing_actions)
        )

    result = frame.copy()
    costs = spec.builder(result, cohort)
    for column in costs.columns:
        result[column] = costs[column].to_numpy()

    result["obj_cost"] = result[spec.objective].astype(float)
    for i, constraint in enumerate(spec.constraints):
        result["con_cost_{0}".format(i)] = result[constraint].astype(float)

    if group_key in result.columns:
        stay = result[group_key].to_numpy()
        last_of_stay = np.zeros(len(result), dtype=bool)
        if len(result):
            last_of_stay[:-1] = stay[1:] != stay[:-1]
            last_of_stay[-1] = True
    else:
        last_of_stay = np.zeros(len(result), dtype=bool)
        if len(result):
            last_of_stay[-1] = True

    if spec.terminal_column and spec.terminal_column in result.columns:
        terminal_action = (
            result[spec.terminal_column].fillna(0).to_numpy(dtype=float) > 0
        )
    elif spec.task in (Task.DISCHARGE, Task.EXTUBATION):
        action = result[spec.action_column].fillna(0).to_numpy(dtype=float)
        terminal_action = action > 0.0
    else:
        # Generic actions (especially doses) are not terminal merely because
        # they are non-zero. A task must declare a terminal indicator.
        terminal_action = np.zeros(len(result), dtype=bool)

    if spec.censoring_column and spec.censoring_column in result.columns:
        explicitly_censored = (
            result[spec.censoring_column].fillna(0).to_numpy(dtype=float) > 0
        )
    else:
        explicitly_censored = np.zeros(len(result), dtype=bool)
    result["done"] = np.where(
        terminal_action | last_of_stay | explicitly_censored, 1.0, 0.0
    )
    # Distinguishing the two lets a caller build a d3rlpy timeout mask, or
    # exclude truncated episodes entirely.
    result["is_terminal_action"] = terminal_action.astype(float)
    result["is_truncated"] = (
        (last_of_stay | explicitly_censored) & ~terminal_action
    ).astype(float)

    if spec.task in (Task.DISCHARGE, Task.EXTUBATION):
        # An action of 0 is the conservative choice for built-in stop tasks.
        result["safe_action"] = np.where(terminal_action, 0.0, 1.0)

    return result, spec


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_by_group(
    frame: pd.DataFrame,
    split: SplitConfig,
) -> Dict[str, pd.DataFrame]:
    """Partition ``frame`` into train / val / test over whole groups.

    ``group_key=None`` falls back to ``subject_id`` when this low-level helper
    is called directly.  The end-to-end pipeline resolves its task-specific
    grouping and optional stratification before calling this function.
    Splitting rows rather than whole groups could put early transitions in
    training and their terminal outcome in test.  Returns frames with fresh
    ``RangeIndex`` values, since next-state lookups are positional.
    """
    group_key = split.group_key or "subject_id"
    if group_key not in frame.columns:
        raise KeyError(
            "split_by_group needs the grouping column {0!r}.".format(group_key)
        )

    groups = pd.unique(frame[group_key])
    if len(groups) < 3:
        raise ValueError(
            "Need at least 3 distinct {0} values to build three splits, got "
            "{1}.".format(group_key, len(groups))
        )

    stratify_column = split.stratify_column
    if stratify_column is not None:
        if stratify_column not in frame.columns:
            raise KeyError(
                "split_by_group needs stratification column {0!r}.".format(
                    stratify_column
                )
            )
        label_counts = frame.groupby(group_key, dropna=False)[
            stratify_column
        ].nunique(dropna=False)
        inconsistent = label_counts[label_counts != 1]
        if not inconsistent.empty:
            raise ValueError(
                "{0!r} must be constant within every {1!r} group; "
                "inconsistent groups include {2}.".format(
                    stratify_column,
                    group_key,
                    list(inconsistent.index[:5]),
                )
            )
        group_rows = frame[[group_key, stratify_column]].drop_duplicates(group_key)
        if group_rows[stratify_column].isna().any():
            raise ValueError(
                "Cannot stratify by {0!r} because some groups have missing labels.".format(
                    stratify_column
                )
            )
        group_values = group_rows[group_key].to_numpy(dtype=object)
        group_labels = group_rows[stratify_column].to_numpy()
        try:
            train_values, holdout_values, _, holdout_labels = train_test_split(
                group_values,
                group_labels,
                test_size=split.test_prop,
                random_state=split.random_seed,
                stratify=group_labels,
            )
            val_values, test_values = train_test_split(
                holdout_values,
                test_size=split.val_prop,
                random_state=split.random_seed,
                stratify=holdout_labels,
            )
        except ValueError as exc:
            raise ValueError(
                "Could not stratify {0} groups by {1!r}: {2}".format(
                    group_key, stratify_column, exc
                )
            ) from exc
        train_groups = set(train_values)
        val_groups = set(val_values)
        test_groups = set(test_values)
    else:
        rng = np.random.default_rng(split.random_seed)
        shuffled = np.array(groups, dtype=object)
        rng.shuffle(shuffled)

        n_holdout = int(round(len(shuffled) * split.test_prop))
        n_holdout = max(2, min(n_holdout, len(shuffled) - 1))
        n_test = int(round(n_holdout * split.val_prop))
        n_test = max(1, min(n_test, n_holdout - 1))

        test_groups = set(shuffled[:n_test])
        val_groups = set(shuffled[n_test:n_holdout])
        train_groups = set(shuffled[n_holdout:])

    out: Dict[str, pd.DataFrame] = {}
    for name, wanted in (
        ("train", train_groups),
        ("val", val_groups),
        ("test", test_groups),
    ):
        subset = frame[frame[group_key].isin(wanted)].copy()
        subset = subset.reset_index(drop=True)
        subset[ROW_INDEX_COLUMN] = np.arange(len(subset), dtype=np.int64)
        out[name] = subset

    logger.info(
        "Split %d %s(s) into train=%d, val=%d, test=%d%s",
        len(groups),
        group_key,
        len(train_groups),
        len(val_groups),
        len(test_groups),
        (
            " stratified by {0}".format(stratify_column)
            if stratify_column is not None
            else ""
        ),
    )
    return out


def select_decision_points(
    frame: pd.DataFrame,
    spec: CostSpec,
    epoch_column: str = "epoch",
    group_key: str = "stay_id",
    strategy: str = "first",
) -> pd.DataFrame:
    """Pick the rows at which policy evaluation should be scored.

    Validation and test estimates in the OCRL loop average Q-values over a set
    of *states a clinician actually had to decide from*. Averaging over every
    logged timestep instead over-weights long stays, which is why the published
    example carries a separate ``*_select`` table.

    ``strategy`` is ``"first"`` (one state per stay, its first decision epoch),
    ``"terminal"`` (the row where the terminal action was taken) or ``"all"``.
    Rows keep their ``row_index``, so next-state lookups still resolve against
    the full split table.
    """
    if strategy == "all":
        return frame.copy()

    if group_key not in frame.columns:
        raise KeyError("select_decision_points needs a {0!r} column.".format(group_key))

    if strategy == "first":
        sort_columns = [c for c in (group_key, epoch_column) if c in frame.columns]
        ordered = frame.sort_values(sort_columns, kind="mergesort")
        return ordered.groupby(group_key, sort=False, as_index=False).head(1).copy()

    if strategy == "terminal":
        mask = frame.get("is_terminal_action")
        if mask is None:
            mask = frame[spec.action_column] > 0
        selected = frame[mask.astype(bool)].copy()
        if selected.empty:
            logger.warning(
                "No terminal actions found; falling back to the first decision "
                "point per %s.", group_key,
            )
            return select_decision_points(
                frame, spec, epoch_column, group_key, strategy="first"
            )
        return selected

    raise ValueError(
        "Unknown selection strategy {0!r}; expected 'first', 'terminal' or "
        "'all'.".format(strategy)
    )
