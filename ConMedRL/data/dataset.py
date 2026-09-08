"""Containers for a finished offline-RL dataset, plus d3rlpy interop.

The pipeline's product is a :class:`RLDatasetBundle`: three
:class:`SplitTables` (train / validation / test), the terminal state vector,
the fitted scaler and the resolved state-space schema. Those are exactly the
arguments ``ConMedRL.TrainDataLoader`` and ``ConMedRL.ValTestDataLoader``
expect, so a bundle can go straight into training:

>>> bundle = build_dataset(cfg)
>>> loader = TrainDataLoader(
...     cfg=rl_cfg,
...     outcome_table=bundle.train.outcome,
...     state_var_table=bundle.train.state,
...     terminal_state=bundle.terminal_state,
... )

Column contract
---------------
``outcome`` carries the bookkeeping columns the loaders read by name:

==================  ========================================================
``<action_name>``   The clinical action taken (0/1); its name is task-specific
                    (``discharge_action`` / ``extubation_action``).
``done``            1.0 when the transition ends the episode.
``obj_cost``        Objective cost being minimised.
``con_cost_{i}``    Cost of constraint ``i``, one column per constraint.
``stay_id``         Episode identifier; splits never straddle it.
``row_index``       Position of this row inside its split's ``state`` table.
                    ``*_select`` tables keep it so next-state lookups stay
                    correct after a CSV round-trip.
==================  ========================================================

``state`` holds only the resolved state variables, one column per entry of
``schema.names``, already scaled to ``[0, 1]``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PreprocessConfig
from .schema import StateSpaceSchema

__all__ = [
    "SplitTables",
    "RLDatasetBundle",
    "MDPDatasetBundle",
    "ROW_INDEX_COLUMN",
    "compute_dataset_content_hash",
    "validate_rl_contract",
]

#: Name of the column that records a row's position inside its split's state
#: table. Written to every outcome table so the decision-point subsets survive
#: serialisation.
ROW_INDEX_COLUMN = "row_index"


@dataclass
class SplitTables:
    """One partition of the dataset.

    Attributes
    ----------
    outcome, state:
        The *full* trajectory tables: every recorded time step of every stay in
        this partition, index-aligned with each other and ordered by
        ``(stay_id, time)``. Next-state lookups walk this table.
    outcome_select, state_select:
        The decision-point subset -- the rows at which a clinician actually
        faced the decision. Validation and testing evaluate here, while
        next-states still come from the full tables. ``None`` for the training
        split, which trains on every transition.
    """

    name: str
    outcome: pd.DataFrame
    state: pd.DataFrame
    outcome_select: Optional[pd.DataFrame] = None
    state_select: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        if len(self.outcome) != len(self.state):
            raise ValueError(
                "{0}: outcome table has {1} rows but state table has {2}".format(
                    self.name, len(self.outcome), len(self.state)
                )
            )
        if not self.outcome.index.equals(self.state.index):
            raise ValueError(
                "{0}: outcome and state tables must share an index; the data "
                "loaders pair a state with its successor by index.".format(self.name)
            )
        if self.outcome_select is not None and self.state_select is not None:
            if len(self.outcome_select) != len(self.state_select):
                raise ValueError(
                    "{0}: select tables disagree in length ({1} vs {2})".format(
                        self.name, len(self.outcome_select), len(self.state_select)
                    )
                )

    @property
    def n_rows(self) -> int:
        return len(self.outcome)

    @property
    def n_episodes(self) -> int:
        if "stay_id" not in self.outcome.columns:
            return 0
        return int(self.outcome["stay_id"].nunique())

    @property
    def n_decision_points(self) -> int:
        return 0 if self.outcome_select is None else len(self.outcome_select)

    def describe(self) -> Dict[str, Any]:
        return {
            "split": self.name,
            "rows": self.n_rows,
            "episodes": self.n_episodes,
            "decision_points": self.n_decision_points,
            "state_dim": self.state.shape[1],
        }


@dataclass
class MDPDatasetBundle:
    """d3rlpy datasets built from a split, one per cost signal.

    d3rlpy models a single scalar reward, but a constrained problem has an
    objective plus one cost per constraint. Rather than collapse them, this
    exposes ``objective`` for the primary problem and ``constraints[i]`` for
    each constraint, all sharing the same transitions. ``arrays`` keeps the raw
    NumPy views so other libraries can be wired up without re-deriving them.
    """

    objective: Any
    constraints: List[Any] = field(default_factory=list)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    #: ``True`` if rewards were negated from costs (the default).
    rewards_negated: bool = True

    def __len__(self) -> int:
        return int(self.arrays["observations"].shape[0]) if self.arrays else 0


@dataclass
class RLDatasetBundle:
    """Everything one preprocessing run produced."""

    config: PreprocessConfig
    schema: StateSpaceSchema
    train: SplitTables
    val: SplitTables
    test: SplitTables

    #: State vector substituted for the successor of a terminal transition.
    terminal_state: np.ndarray

    #: Name of the action column inside the outcome tables.
    action_name: str = "action"
    #: ``"discrete"`` or ``"continuous"``.
    action_type: str = "discrete"
    #: Ordered action columns. Empty preserves the legacy single-column API.
    action_columns: Tuple[str, ...] = ()
    #: Optional per-column lower/upper limits for continuous actions.
    action_bounds: Optional[Mapping[str, Tuple[float, float]]] = None
    #: Optional discrete labels. Integer values remain the training encoding.
    action_categories: Optional[Sequence[Any]] = None
    #: Number of ``con_cost_{i}`` columns present.
    num_constraints: int = 0

    #: Scaler fitted on the *training* rows only, reused for val/test and for
    #: scaling live patient input at inference time.
    scaler: Optional[Any] = None
    #: Unscaled copies of the state tables, when retained.
    raw_state_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Provenance: cohort counts, coverage, dropped variables, timings.
    report: Dict[str, Any] = field(default_factory=dict)
    #: Paths of everything written to disk, keyed by logical artefact name.
    written_files: Dict[str, str] = field(default_factory=dict)
    #: Stable SHA-256 over all RL tables and the terminal state.
    content_hash: Optional[str] = None

    # -- convenience ---------------------------------------------------------

    def __post_init__(self) -> None:
        self.action_type = str(self.action_type).strip().lower()
        if self.action_type not in ("discrete", "continuous"):
            raise ValueError("action_type must be 'discrete' or 'continuous'")
        if not self.action_columns:
            self.action_columns = (self.action_name,)
        else:
            self.action_columns = tuple(self.action_columns)
            self.action_name = self.action_columns[0]
        if self.action_type == "discrete" and len(self.action_columns) != 1:
            raise ValueError("Discrete bundles require exactly one encoded action column.")
        if self.action_type == "continuous" and self.action_bounds:
            missing = set(self.action_columns) - set(self.action_bounds)
            if missing:
                raise ValueError(
                    "Continuous action bounds missing for {0}.".format(sorted(missing))
                )
            for column in self.action_columns:
                low, high = self.action_bounds[column]
                if not np.isfinite([low, high]).all() or low >= high:
                    raise ValueError(
                        "Invalid action bounds for {0!r}: {1}".format(
                            column, self.action_bounds[column]
                        )
                    )

    @property
    def state_dim(self) -> int:
        return self.schema.state_dim

    @property
    def action_dim(self) -> int:
        if self.action_type == "discrete" and self.action_categories is not None:
            return len(self.action_categories)
        return len(self.action_columns)

    @property
    def loader_action(self) -> Any:
        """Argument to pass as ``action_name`` to the legacy data loaders."""
        return (
            self.action_name
            if self.action_type == "discrete"
            else list(self.action_columns)
        )

    @property
    def ordered_action_bounds(self) -> Optional[List[Tuple[float, float]]]:
        if not self.action_bounds:
            return None
        return [tuple(self.action_bounds[column]) for column in self.action_columns]

    @property
    def splits(self) -> Dict[str, SplitTables]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def summary(self) -> str:
        lines = [
            "ConMedRL dataset: database={0}, task={1}".format(
                self.config.database, self.config.task
            ),
            "  state_dim      : {0}".format(self.state_dim),
            "  action         : {0} {1}".format(
                self.action_type, ", ".join(self.action_columns)
            ),
            "  constraints    : {0}".format(self.num_constraints),
        ]
        for split in (self.train, self.val, self.test):
            info = split.describe()
            lines.append(
                "  {0:<6}: {1:>8} rows | {2:>6} episodes | {3:>6} decision points".format(
                    info["split"], info["rows"], info["episodes"], info["decision_points"]
                )
            )
        if self.schema.dropped:
            lines.append(
                "  dropped {0} canonical variable(s); see schema.dropped_frame()".format(
                    len(self.schema.dropped)
                )
            )
        return "\n".join(lines)

    def loader_kwargs(self, split: str = "train") -> Dict[str, Any]:
        """Keyword arguments for the matching ConMedRL data loader.

        >>> TrainDataLoader(cfg=rl_cfg, **bundle.loader_kwargs("train"))
        >>> ValTestDataLoader(cfg=rl_cfg, **bundle.loader_kwargs("val"))
        """
        if split == "train":
            return {
                "outcome_table": self.train.outcome,
                "state_var_table": self.train.state,
                "terminal_state": self.terminal_state,
            }
        if split not in ("val", "test"):
            raise ValueError("split must be 'train', 'val' or 'test', got {0!r}".format(split))

        tables = self.splits[split]
        if tables.outcome_select is None or tables.state_select is None:
            raise ValueError(
                "The {0} split has no decision-point subset; ValTestDataLoader "
                "needs one.".format(split)
            )
        return {
            "outcome_table_select": tables.outcome_select,
            "state_var_table_select": tables.state_select,
            "outcome_table": tables.outcome,
            "state_var_table": tables.state,
            "terminal_state": self.terminal_state,
        }

    def to_dict(self) -> Dict[str, Any]:
        self.content_hash = compute_dataset_content_hash(self)
        return {
            "status": "active",
            "config": self.config.to_dict(),
            "schema": self.schema.to_dict(),
            "action_name": self.action_name,
            "action_type": self.action_type,
            "action_columns": list(self.action_columns),
            "action_dim": self.action_dim,
            "action_bounds": dict(self.action_bounds or {}),
            "action_categories": (
                list(self.action_categories)
                if self.action_categories is not None
                else None
            ),
            "num_constraints": self.num_constraints,
            "splits": [s.describe() for s in (self.train, self.val, self.test)],
            "report": self.report,
            "written_files": dict(self.written_files),
            "content_hash": self.content_hash,
        }

    # -- d3rlpy ---------------------------------------------------------------

    def to_mdp_dataset(
        self,
        split: str = "train",
        negate_costs: bool = True,
        include_constraints: bool = True,
        use_select: bool = False,
    ) -> MDPDatasetBundle:
        """Build ``d3rlpy`` ``MDPDataset`` objects from one split.

        Parameters
        ----------
        split:
            ``"train"``, ``"val"`` or ``"test"``.
        negate_costs:
            ConMedRL *minimises* cost; d3rlpy *maximises* reward. Leaving this
            at ``True`` sets ``reward = -cost`` so d3rlpy's agents optimise the
            same thing. Set to ``False`` only if you intend to flip the sign
            yourself.
        include_constraints:
            Also build one dataset per ``con_cost_{i}`` column.
        use_select:
            Build from the decision-point subset instead of the full
            trajectories. Off by default: d3rlpy needs contiguous transitions,
            and the subset is not contiguous.
        """
        if split not in self.splits:
            raise ValueError("split must be 'train', 'val' or 'test', got {0!r}".format(split))
        tables = self.splits[split]

        outcome = tables.outcome_select if use_select else tables.outcome
        state = tables.state_select if use_select else tables.state
        if outcome is None or state is None:
            raise ValueError("The {0} split has no select subset.".format(split))

        return build_mdp_dataset(
            outcome=outcome,
            state=state,
            action_name=self.action_name,
            action_columns=self.action_columns,
            action_type=self.action_type,
            num_constraints=self.num_constraints if include_constraints else 0,
            negate_costs=negate_costs,
        )


# ---------------------------------------------------------------------------
# d3rlpy construction
# ---------------------------------------------------------------------------


def _canonical_cell(value: Any) -> Optional[str]:
    """Normalise a scalar independently of pandas/CSV inferred dtypes."""
    if value is None or bool(pd.isna(value)):
        return None
    text = str(value).strip()
    if text.lower() in ("true", "false"):
        return text.lower()
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite():
        return text
    if number == 0:
        return "0"
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    # Pandas' CSV writer may shorten a binary float by one or two terminal
    # digits while preserving its value. Fifteen significant decimal digits
    # give a stable semantic representation across that round trip.
    with localcontext() as context:
        context.prec = 15
        number = +number
    return format(number.normalize(), "f")


def _update_table_hash(digest: Any, label: str, table: Optional[pd.DataFrame]) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    if table is None:
        digest.update(b"<absent>\0")
        return
    header = json.dumps(
        [str(column) for column in table.columns],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest.update(header.encode("utf-8"))
    digest.update(b"\n")
    for row in table.itertuples(index=False, name=None):
        encoded = json.dumps(
            [_canonical_cell(value) for value in row],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(encoded.encode("utf-8"))
        digest.update(b"\n")
    digest.update(b"\0")


def compute_dataset_content_hash(bundle: RLDatasetBundle) -> str:
    """Return a canonical SHA-256 over every split table and terminal state."""
    digest = hashlib.sha256()
    digest.update(b"ConMedRL-dataset-content-v1\0")
    for name in ("train", "val", "test"):
        split = bundle.splits[name]
        _update_table_hash(digest, name + "/outcome", split.outcome)
        _update_table_hash(digest, name + "/state", split.state)
        _update_table_hash(digest, name + "/outcome_select", split.outcome_select)
        _update_table_hash(digest, name + "/state_select", split.state_select)
    terminal = pd.DataFrame(
        [np.asarray(bundle.terminal_state).reshape(-1)],
        columns=bundle.schema.names,
    )
    _update_table_hash(digest, "terminal_state", terminal)
    metadata = {
        "schema": bundle.schema.to_dict(),
        "action_name": bundle.action_name,
        "action_type": bundle.action_type,
        "action_columns": list(bundle.action_columns),
        "action_bounds": dict(bundle.action_bounds or {}),
        "action_categories": (
            list(bundle.action_categories)
            if bundle.action_categories is not None
            else None
        ),
        "num_constraints": int(bundle.num_constraints),
    }
    digest.update(b"bundle_metadata\0")
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _episode_boundaries(outcome: pd.DataFrame) -> np.ndarray:
    """Boolean mask marking the last row of each episode.

    Uses ``stay_id`` when available; otherwise the whole table is one episode.
    """
    n = len(outcome)
    last = np.zeros(n, dtype=bool)
    if n == 0:
        return last
    if "stay_id" in outcome.columns:
        stay = outcome["stay_id"].to_numpy()
        last[:-1] = stay[1:] != stay[:-1]
    last[-1] = True
    return last


def _make_mdp_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    action_type: str = "discrete",
) -> Any:
    """Construct an ``MDPDataset`` across d3rlpy's two incompatible APIs.

    v2 takes ``timeouts`` alongside ``terminals``; v1 instead takes
    ``episode_terminals`` meaning "episode ended for any reason".
    """
    try:
        from d3rlpy.dataset import MDPDataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Building an MDPDataset needs d3rlpy: pip install d3rlpy"
        ) from exc

    try:  # d3rlpy >= 2.0
        return MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts,
        )
    except TypeError:
        pass

    # d3rlpy 1.x: collapse terminals+timeouts into episode_terminals.
    episode_terminals = np.clip(terminals + timeouts, 0.0, 1.0)
    return MDPDataset(
        observations=observations,
        actions=(
            actions.reshape(-1)
            if action_type == "discrete"
            else actions.astype(np.float32)
        ),
        rewards=rewards.reshape(-1),
        terminals=terminals,
        episode_terminals=episode_terminals,
        discrete_action=action_type == "discrete",
    )


def build_mdp_dataset(
    outcome: pd.DataFrame,
    state: pd.DataFrame,
    action_name: str,
    action_columns: Optional[Sequence[str]] = None,
    action_type: str = "discrete",
    num_constraints: int = 0,
    negate_costs: bool = True,
) -> MDPDatasetBundle:
    """Convert outcome/state tables into d3rlpy ``MDPDataset`` objects.

    Terminal vs. timeout is distinguished deliberately: a stay whose final row
    has ``done == 1`` truly ended (the patient was discharged or extubated), so
    its value is bootstrapped from nothing. A stay that merely stops being
    recorded is a *timeout*, and treating it as terminal would teach the agent
    that running out of data is an absorbing state worth zero cost.
    """
    action_type = str(action_type).strip().lower()
    columns = tuple(action_columns or (action_name,))
    missing_actions = [column for column in columns if column not in outcome.columns]
    if missing_actions:
        raise KeyError(
            "Action column(s) {0} are missing from the outcome table; found: "
            "{1}".format(missing_actions, ", ".join(map(str, outcome.columns[:20])))
        )
    if action_type not in ("discrete", "continuous"):
        raise ValueError("action_type must be 'discrete' or 'continuous'")
    if action_type == "discrete" and len(columns) != 1:
        raise ValueError("Discrete MDPDataset export expects one encoded action column.")
    if "obj_cost" not in outcome.columns:
        raise KeyError("The outcome table has no 'obj_cost' column.")

    observations = np.ascontiguousarray(state.to_numpy(dtype=np.float32))
    if action_type == "discrete":
        actions = outcome[columns[0]].to_numpy(dtype=np.int64).reshape(-1, 1)
    else:
        actions = outcome[list(columns)].to_numpy(dtype=np.float32)

    sign = -1.0 if negate_costs else 1.0
    obj_cost = outcome["obj_cost"].to_numpy(dtype=np.float32)
    rewards = (sign * obj_cost).reshape(-1, 1).astype(np.float32)

    is_last = _episode_boundaries(outcome)
    if "is_terminal_action" in outcome.columns:
        terminal_action = outcome["is_terminal_action"].to_numpy(dtype=np.float32) > 0
    else:
        done = outcome.get("done", pd.Series(np.zeros(len(outcome)), index=outcome.index))
        truncated = outcome.get(
            "is_truncated", pd.Series(np.zeros(len(outcome)), index=outcome.index)
        )
        terminal_action = (done.to_numpy(dtype=np.float32) > 0) & ~(
            truncated.to_numpy(dtype=np.float32) > 0
        )
    terminals = np.where(terminal_action, 1.0, 0.0).astype(np.float32)
    if "is_truncated" in outcome.columns:
        truncated = outcome["is_truncated"].to_numpy(dtype=np.float32) > 0
        timeouts = np.where(truncated, 1.0, 0.0).astype(np.float32)
    else:
        timeouts = np.where(is_last & ~terminal_action, 1.0, 0.0).astype(np.float32)

    objective = _make_mdp_dataset(
        observations, actions, rewards, terminals, timeouts, action_type
    )

    constraint_datasets: List[Any] = []
    arrays: Dict[str, np.ndarray] = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "obj_cost": obj_cost,
    }

    for i in range(num_constraints):
        column = "con_cost_{0}".format(i)
        if column not in outcome.columns:
            continue
        con_cost = outcome[column].to_numpy(dtype=np.float32)
        arrays[column] = con_cost
        constraint_datasets.append(
            _make_mdp_dataset(
                observations,
                actions,
                (sign * con_cost).reshape(-1, 1).astype(np.float32),
                terminals,
                timeouts,
                action_type,
            )
        )

    return MDPDatasetBundle(
        objective=objective,
        constraints=constraint_datasets,
        arrays=arrays,
        rewards_negated=negate_costs,
    )


def validate_rl_contract(bundle: RLDatasetBundle) -> Dict[str, Any]:
    """Validate the complete loader/export contract before training.

    The check is intentionally stricter than pandas construction: silent row
    misalignment or a configured trajectory group occurring in two splits
    invalidates an offline evaluation even though every individual table
    remains readable.
    """
    errors: List[str] = []
    warnings: List[str] = []
    required = {"stay_id", "done", "obj_cost"}
    required.update("con_cost_{0}".format(i) for i in range(bundle.num_constraints))
    required.update(bundle.action_columns)

    split_report = bundle.report.get("split", {})
    group_key = (
        split_report.get("group_key")
        or bundle.config.split.group_key
        or "subject_id"
    )
    groups: Dict[str, set] = {}
    for name, tables in bundle.splits.items():
        outcome, state = tables.outcome, tables.state
        missing = required - set(outcome.columns)
        if missing:
            errors.append("{0}: missing outcome columns {1}".format(name, sorted(missing)))
        if list(state.columns) != bundle.schema.names:
            errors.append("{0}: state columns do not match the ordered schema".format(name))
        if len(outcome) != len(state) or not outcome.index.equals(state.index):
            errors.append("{0}: outcome/state rows are not index-aligned".format(name))
        if "stay_id" in outcome:
            repeated = outcome["stay_id"].ne(outcome["stay_id"].shift()).groupby(
                outcome["stay_id"]
            ).sum()
            if (repeated > 1).any():
                errors.append("{0}: episode rows are not contiguous".format(name))
            last = outcome["stay_id"].ne(outcome["stay_id"].shift(-1))
            if "done" in outcome and not (outcome.loc[last, "done"] > 0).all():
                errors.append("{0}: at least one episode does not end with done=1".format(name))
        if group_key in outcome:
            groups[name] = set(outcome[group_key].dropna())
        else:
            errors.append(
                "{0}: split grouping column {1!r} is missing".format(
                    name, group_key
                )
            )

        actions = outcome[[c for c in bundle.action_columns if c in outcome]]
        if not actions.empty:
            numeric = actions.apply(pd.to_numeric, errors="coerce")
            if numeric.isna().any().any():
                errors.append("{0}: actions contain missing/non-numeric values".format(name))
            if bundle.action_type == "discrete":
                values = numeric.iloc[:, 0].dropna().to_numpy()
                if not np.equal(values, np.floor(values)).all():
                    errors.append("{0}: discrete actions are not integer encoded".format(name))
            for column, bounds in (bundle.action_bounds or {}).items():
                if column in numeric and bounds is not None:
                    low, high = bounds
                    if ((numeric[column] < low) | (numeric[column] > high)).any():
                        errors.append(
                            "{0}: action {1!r} lies outside [{2}, {3}]".format(
                                name, column, low, high
                            )
                        )

    split_names = list(groups)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                errors.append(
                    "{0} split leakage between {1}/{2}: {3} group(s)".format(
                        group_key, left, right, len(overlap)
                    )
                )

    if errors:
        raise ValueError("Invalid RL dataset contract:\n- " + "\n- ".join(errors))
    return {"valid": True, "errors": [], "warnings": warnings}
