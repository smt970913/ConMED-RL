"""Versioned declarative specifications for generic clinical datasets.

The objects in this module are deliberately data-only.  They can be reviewed,
hashed and persisted as JSON, and they never contain executable Python.  Rule
expressions use a small, explicitly whitelisted expression tree interpreted by
:class:`SafeRuleEvaluator`; neither ``eval`` nor generated source is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar

__all__ = [
    "SPEC_VERSION",
    "SpecValidationError",
    "ApprovalRequiredError",
    "TableRoleSpec",
    "DictionarySpec",
    "UnitRule",
    "EventRule",
    "ActionSpec",
    "CostRule",
    "TaskSpec",
    "DatasetSpec",
    "SafeRuleEvaluator",
    "validate_expression",
    "evaluate_expression",
    "evaluate_rule",
]

SPEC_VERSION = "1.0"
_MAX_EXPRESSION_DEPTH = 32
_MAX_EXPRESSION_NODES = 1000


class SpecValidationError(ValueError):
    """A declarative specification is malformed or unsafe."""


class ApprovalRequiredError(RuntimeError):
    """Execution was attempted with an unapproved or stale specification."""


def _fail(path: str, message: str) -> None:
    raise SpecValidationError("{0}: {1}".format(path, message))


def _string(value: Any, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    value = value.strip()
    if not value and not allow_empty:
        _fail(path, "must not be empty")
    return value


def _optional_string(value: Any, path: str) -> Optional[str]:
    if value is None:
        return None
    return _string(value, path)


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "expected a boolean")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected a finite number")
    value = float(value)
    if not math.isfinite(value):
        _fail(path, "expected a finite number")
    return value


def _json_value(value: Any, path: str = "value") -> Any:
    """Validate and copy a value that must survive a strict JSON round-trip."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "NaN and infinity are not JSON-compatible")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, "{0}[{1}]".format(path, i)) for i, item in enumerate(value)]
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "object keys must be strings")
            out[key] = _json_value(item, "{0}.{1}".format(path, key))
        return out
    _fail(path, "value of type {0} is not JSON-compatible".format(type(value).__name__))


def _mapping(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected an object")
    return dict(_json_value(value, path))


def _strings(value: Any, path: str, unique: bool = True) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        _fail(path, "expected an array of strings")
    result = tuple(_string(item, "{0}[{1}]".format(path, i)) for i, item in enumerate(value))
    if unique and len(set(result)) != len(result):
        _fail(path, "entries must be unique")
    return result


def _strict_kwargs(cls: Type[Any], payload: Mapping[str, Any], path: str) -> Dict[str, Any]:
    raw = _mapping(payload, path)
    known = {item.name for item in fields(cls) if item.init}
    unknown = set(raw) - known
    if unknown:
        _fail(path, "unknown field(s): {0}".format(", ".join(sorted(unknown))))
    return raw


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _expression(value: Any, path: str) -> Dict[str, Any]:
    expression = _mapping(value, path)
    validate_expression(expression, path)
    return expression


@dataclass(frozen=True)
class TableRoleSpec:
    """How one local source file participates in canonical ingestion."""

    name: str
    file: str
    role: str
    columns: Tuple[str, ...]
    episode_id_column: str
    subject_id_column: Optional[str] = None
    time_column: Optional[str] = None
    start_time_column: Optional[str] = None
    end_time_column: Optional[str] = None
    item_id_column: Optional[str] = None
    value_column: Optional[str] = None
    unit_column: Optional[str] = None
    large: bool = False
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, "TableRoleSpec.name"))
        object.__setattr__(self, "file", _string(self.file, "TableRoleSpec.file"))
        object.__setattr__(self, "role", _string(self.role, "TableRoleSpec.role").lower())
        object.__setattr__(self, "columns", _strings(self.columns, "TableRoleSpec.columns"))
        object.__setattr__(
            self, "episode_id_column", _string(self.episode_id_column, "TableRoleSpec.episode_id_column")
        )
        for name in (
            "subject_id_column",
            "time_column",
            "start_time_column",
            "end_time_column",
            "item_id_column",
            "value_column",
            "unit_column",
        ):
            object.__setattr__(
                self, name, _optional_string(getattr(self, name), "TableRoleSpec.{0}".format(name))
            )
        object.__setattr__(self, "large", _bool(self.large, "TableRoleSpec.large"))
        object.__setattr__(self, "required", _bool(self.required, "TableRoleSpec.required"))
        selected = set(self.columns)
        referenced = [
            self.episode_id_column,
            self.subject_id_column,
            self.time_column,
            self.start_time_column,
            self.end_time_column,
            self.item_id_column,
            self.value_column,
            self.unit_column,
        ]
        missing = [column for column in referenced if column and column not in selected]
        if missing:
            _fail("TableRoleSpec.columns", "must include referenced column(s): {0}".format(", ".join(missing)))
        if self.time_column and (self.start_time_column or self.end_time_column):
            _fail("TableRoleSpec", "use either time_column or interval start/end columns")

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TableRoleSpec":
        data = _strict_kwargs(cls, payload, "TableRoleSpec")
        if "columns" in data:
            data["columns"] = _strings(data["columns"], "TableRoleSpec.columns")
        return cls(**data)


@dataclass(frozen=True)
class DictionarySpec:
    """A non-patient lookup table used to resolve clinical item identifiers."""

    name: str
    file: str
    id_column: str
    name_column: str
    unit_column: Optional[str] = None
    group_column: Optional[str] = None
    code_column: Optional[str] = None
    description_column: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("name", "file", "id_column", "name_column"):
            object.__setattr__(self, name, _string(getattr(self, name), "DictionarySpec.{0}".format(name)))
        for name in ("unit_column", "group_column", "code_column", "description_column"):
            object.__setattr__(
                self, name, _optional_string(getattr(self, name), "DictionarySpec.{0}".format(name))
            )

    @property
    def columns(self) -> Tuple[str, ...]:
        values = (
            self.id_column,
            self.name_column,
            self.unit_column,
            self.group_column,
            self.code_column,
            self.description_column,
        )
        return tuple(value for value in values if value is not None)

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DictionarySpec":
        return cls(**_strict_kwargs(cls, payload, "DictionarySpec"))


@dataclass(frozen=True)
class UnitRule:
    """Affine conversion selected by an optional declarative predicate."""

    name: str
    source_table: str
    source_unit: str
    target_unit: str
    scale: float = 1.0
    offset: float = 0.0
    when: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        for name in ("name", "source_table", "source_unit", "target_unit"):
            object.__setattr__(self, name, _string(getattr(self, name), "UnitRule.{0}".format(name)))
        object.__setattr__(self, "scale", _number(self.scale, "UnitRule.scale"))
        object.__setattr__(self, "offset", _number(self.offset, "UnitRule.offset"))
        if self.scale == 0:
            _fail("UnitRule.scale", "must not be zero")
        if self.when is not None:
            object.__setattr__(self, "when", _expression(self.when, "UnitRule.when"))

    def convert(self, value: Any) -> Any:
        return value * self.scale + self.offset

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnitRule":
        return cls(**_strict_kwargs(cls, payload, "UnitRule"))


@dataclass(frozen=True)
class EventRule:
    """Derive a canonical event from a source table without executable code."""

    name: str
    source_table: str
    predicate: Dict[str, Any]
    time_expression: Dict[str, Any]
    value_expression: Optional[Dict[str, Any]] = None
    operation: str = "point"
    aggregation: str = "none"
    pair_with: Optional[str] = None
    window_hours: Optional[float] = None

    _OPERATIONS = ("point", "interval_start", "interval_end", "pair")
    _AGGREGATIONS = ("none", "first", "last", "min", "max", "mean", "sum", "count")

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, "EventRule.name"))
        object.__setattr__(self, "source_table", _string(self.source_table, "EventRule.source_table"))
        object.__setattr__(self, "predicate", _expression(self.predicate, "EventRule.predicate"))
        object.__setattr__(
            self, "time_expression", _expression(self.time_expression, "EventRule.time_expression")
        )
        if self.value_expression is not None:
            object.__setattr__(
                self, "value_expression", _expression(self.value_expression, "EventRule.value_expression")
            )
        operation = _string(self.operation, "EventRule.operation").lower()
        aggregation = _string(self.aggregation, "EventRule.aggregation").lower()
        if operation not in self._OPERATIONS:
            _fail("EventRule.operation", "supported values: {0}".format(", ".join(self._OPERATIONS)))
        if aggregation not in self._AGGREGATIONS:
            _fail("EventRule.aggregation", "supported values: {0}".format(", ".join(self._AGGREGATIONS)))
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "aggregation", aggregation)
        object.__setattr__(self, "pair_with", _optional_string(self.pair_with, "EventRule.pair_with"))
        if operation == "pair" and not self.pair_with:
            _fail("EventRule.pair_with", "is required for pair operations")
        if self.window_hours is not None:
            window = _number(self.window_hours, "EventRule.window_hours")
            if window <= 0:
                _fail("EventRule.window_hours", "must be positive")
            object.__setattr__(self, "window_hours", window)

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EventRule":
        return cls(**_strict_kwargs(cls, payload, "EventRule"))


@dataclass(frozen=True)
class ActionSpec:
    """Discrete categorical or bounded continuous action definition."""

    name: str
    kind: str
    columns: Tuple[str, ...]
    expression: Dict[str, Any]
    categories: Tuple[Any, ...] = ()
    bounds: Tuple[Tuple[float, float], ...] = ()
    terminal: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, "ActionSpec.name"))
        kind = _string(self.kind, "ActionSpec.kind").lower()
        if kind not in ("discrete", "continuous"):
            _fail("ActionSpec.kind", "must be 'discrete' or 'continuous'")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "columns", _strings(self.columns, "ActionSpec.columns"))
        if not self.columns:
            _fail("ActionSpec.columns", "must contain at least one action column")
        object.__setattr__(self, "expression", _expression(self.expression, "ActionSpec.expression"))
        categories = tuple(_json_value(self.categories, "ActionSpec.categories"))
        bounds: List[Tuple[float, float]] = []
        if not isinstance(self.bounds, (list, tuple)):
            _fail("ActionSpec.bounds", "expected an array")
        for i, pair in enumerate(self.bounds):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                _fail("ActionSpec.bounds[{0}]".format(i), "expected [low, high]")
            low = _number(pair[0], "ActionSpec.bounds[{0}][0]".format(i))
            high = _number(pair[1], "ActionSpec.bounds[{0}][1]".format(i))
            if low >= high:
                _fail("ActionSpec.bounds[{0}]".format(i), "low must be less than high")
            bounds.append((low, high))
        if kind == "discrete":
            if len(self.columns) != 1:
                _fail("ActionSpec.columns", "discrete actions require exactly one column")
            if len(categories) < 2:
                _fail("ActionSpec.categories", "discrete actions require at least two categories")
            if bounds:
                _fail("ActionSpec.bounds", "discrete actions do not use bounds")
            canonical = [_canonical_json(item) for item in categories]
            if len(set(canonical)) != len(canonical):
                _fail("ActionSpec.categories", "categories must be unique")
        else:
            if categories:
                _fail("ActionSpec.categories", "continuous actions do not use categories")
            if len(bounds) != len(self.columns):
                _fail("ActionSpec.bounds", "continuous actions require one bound per column")
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "bounds", tuple(bounds))
        object.__setattr__(self, "terminal", _bool(self.terminal, "ActionSpec.terminal"))

    @property
    def action_dim(self) -> int:
        return len(self.columns)

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionSpec":
        data = _strict_kwargs(cls, payload, "ActionSpec")
        if "columns" in data:
            data["columns"] = _strings(data["columns"], "ActionSpec.columns")
        if "categories" in data:
            if not isinstance(data["categories"], (list, tuple)):
                _fail("ActionSpec.categories", "expected an array")
            data["categories"] = tuple(data["categories"])
        if "bounds" in data:
            if not isinstance(data["bounds"], (list, tuple)):
                _fail("ActionSpec.bounds", "expected an array")
            data["bounds"] = tuple(tuple(pair) if isinstance(pair, (list, tuple)) else pair for pair in data["bounds"])
        return cls(**data)


@dataclass(frozen=True)
class CostRule:
    """One objective or constraint cost derived from a safe expression."""

    name: str
    kind: str
    expression: Dict[str, Any]
    weight: float = 1.0
    threshold: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, "CostRule.name"))
        kind = _string(self.kind, "CostRule.kind").lower()
        if kind not in ("objective", "constraint"):
            _fail("CostRule.kind", "must be 'objective' or 'constraint'")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "expression", _expression(self.expression, "CostRule.expression"))
        object.__setattr__(self, "weight", _number(self.weight, "CostRule.weight"))
        if self.threshold is not None:
            object.__setattr__(self, "threshold", _number(self.threshold, "CostRule.threshold"))
        if kind == "objective" and self.threshold is not None:
            _fail("CostRule.threshold", "is only valid for constraints")

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CostRule":
        return cls(**_strict_kwargs(cls, payload, "CostRule"))


@dataclass(frozen=True)
class TaskSpec:
    """A complete, deterministic offline-RL clinical task declaration."""

    name: str
    episode_table: str
    episode_id_column: str
    subject_id_column: str
    timeline_anchor: Dict[str, Any]
    state_columns: Tuple[str, ...]
    action: ActionSpec
    objective: CostRule
    constraints: Tuple[CostRule, ...] = ()
    cohort_filters: Tuple[Dict[str, Any], ...] = ()
    terminal_rule: Optional[Dict[str, Any]] = None
    censoring_rule: Optional[Dict[str, Any]] = None
    decision_epoch_hours: float = 12.0

    def __post_init__(self) -> None:
        for name in ("name", "episode_table", "episode_id_column", "subject_id_column"):
            object.__setattr__(self, name, _string(getattr(self, name), "TaskSpec.{0}".format(name)))
        object.__setattr__(self, "timeline_anchor", _expression(self.timeline_anchor, "TaskSpec.timeline_anchor"))
        object.__setattr__(self, "state_columns", _strings(self.state_columns, "TaskSpec.state_columns"))
        if not self.state_columns:
            _fail("TaskSpec.state_columns", "must not be empty")
        if not isinstance(self.action, ActionSpec):
            _fail("TaskSpec.action", "expected ActionSpec")
        if not isinstance(self.objective, CostRule) or self.objective.kind != "objective":
            _fail("TaskSpec.objective", "expected an objective CostRule")
        if not isinstance(self.constraints, (list, tuple)):
            _fail("TaskSpec.constraints", "expected an array")
        constraints = tuple(self.constraints)
        if any(not isinstance(item, CostRule) or item.kind != "constraint" for item in constraints):
            _fail("TaskSpec.constraints", "all entries must be constraint CostRule objects")
        names = [self.objective.name] + [item.name for item in constraints]
        if len(set(names)) != len(names):
            _fail("TaskSpec", "cost names must be unique")
        object.__setattr__(self, "constraints", constraints)
        if not isinstance(self.cohort_filters, (list, tuple)):
            _fail("TaskSpec.cohort_filters", "expected an array")
        object.__setattr__(
            self,
            "cohort_filters",
            tuple(_expression(item, "TaskSpec.cohort_filters[{0}]".format(i)) for i, item in enumerate(self.cohort_filters)),
        )
        for name in ("terminal_rule", "censoring_rule"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _expression(value, "TaskSpec.{0}".format(name)))
        epoch = _number(self.decision_epoch_hours, "TaskSpec.decision_epoch_hours")
        if epoch <= 0:
            _fail("TaskSpec.decision_epoch_hours", "must be positive")
        object.__setattr__(self, "decision_epoch_hours", epoch)

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskSpec":
        data = _strict_kwargs(cls, payload, "TaskSpec")
        if "action" in data:
            data["action"] = ActionSpec.from_dict(data["action"])
        if "objective" in data:
            data["objective"] = CostRule.from_dict(data["objective"])
        if "constraints" in data:
            if not isinstance(data["constraints"], (list, tuple)):
                _fail("TaskSpec.constraints", "expected an array")
            data["constraints"] = tuple(CostRule.from_dict(item) for item in data["constraints"])
        if "state_columns" in data:
            data["state_columns"] = _strings(data["state_columns"], "TaskSpec.state_columns")
        if "cohort_filters" in data:
            if not isinstance(data["cohort_filters"], (list, tuple)):
                _fail("TaskSpec.cohort_filters", "expected an array")
            data["cohort_filters"] = tuple(data["cohort_filters"])
        return cls(**data)


@dataclass(frozen=True)
class DatasetSpec:
    """Versioned dataset/task plan with cryptographic review provenance."""

    name: str
    tables: Tuple[TableRoleSpec, ...]
    tasks: Tuple[TaskSpec, ...]
    version: str = SPEC_VERSION
    dictionaries: Tuple[DictionarySpec, ...] = ()
    unit_rules: Tuple[UnitRule, ...] = ()
    event_rules: Tuple[EventRule, ...] = ()
    source_fingerprints: Dict[str, str] = field(default_factory=dict)
    model: Optional[str] = None
    prompt_hash: Optional[str] = None
    response_hash: Optional[str] = None
    confidence: float = 0.0
    warnings: Tuple[str, ...] = ()
    unresolved_decisions: Tuple[str, ...] = ()
    approval_hash: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, "DatasetSpec.name"))
        version = _string(self.version, "DatasetSpec.version")
        if version != SPEC_VERSION:
            _fail("DatasetSpec.version", "unsupported version {0!r}; expected {1!r}".format(version, SPEC_VERSION))
        object.__setattr__(self, "version", version)
        for name, expected in (
            ("tables", TableRoleSpec),
            ("tasks", TaskSpec),
            ("dictionaries", DictionarySpec),
            ("unit_rules", UnitRule),
            ("event_rules", EventRule),
        ):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or any(not isinstance(item, expected) for item in value):
                _fail("DatasetSpec.{0}".format(name), "expected an array of {0}".format(expected.__name__))
            object.__setattr__(self, name, tuple(value))
        if not self.tables:
            _fail("DatasetSpec.tables", "must not be empty")
        if not self.tasks:
            _fail("DatasetSpec.tasks", "must not be empty")
        fingerprints = _mapping(self.source_fingerprints, "DatasetSpec.source_fingerprints")
        if any(not isinstance(value, str) or not value for value in fingerprints.values()):
            _fail("DatasetSpec.source_fingerprints", "fingerprints must be non-empty strings")
        object.__setattr__(self, "source_fingerprints", dict(sorted(fingerprints.items())))
        object.__setattr__(self, "model", _optional_string(self.model, "DatasetSpec.model"))
        for name in ("prompt_hash", "response_hash", "approval_hash"):
            value = _optional_string(getattr(self, name), "DatasetSpec.{0}".format(name))
            if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower())):
                _fail("DatasetSpec.{0}".format(name), "expected a 64-character SHA-256 hex digest")
            object.__setattr__(self, name, value.lower() if value else None)
        confidence = _number(self.confidence, "DatasetSpec.confidence")
        if not 0.0 <= confidence <= 1.0:
            _fail("DatasetSpec.confidence", "must lie in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "warnings", _strings(self.warnings, "DatasetSpec.warnings", unique=False))
        object.__setattr__(
            self,
            "unresolved_decisions",
            _strings(self.unresolved_decisions, "DatasetSpec.unresolved_decisions", unique=False),
        )
        self._validate_references()

    def _validate_references(self) -> None:
        table_names = [table.name for table in self.tables]
        if len(set(table_names)) != len(table_names):
            _fail("DatasetSpec.tables", "table names must be unique")
        files = [table.file for table in self.tables]
        if len(set(files)) != len(files):
            _fail("DatasetSpec.tables", "each source file may be selected only once")
        dictionary_names = [item.name for item in self.dictionaries]
        if len(set(dictionary_names)) != len(dictionary_names):
            _fail("DatasetSpec.dictionaries", "dictionary names must be unique")
        event_names = [item.name for item in self.event_rules]
        if len(set(event_names)) != len(event_names):
            _fail("DatasetSpec.event_rules", "event names must be unique")
        task_names = [item.name for item in self.tasks]
        if len(set(task_names)) != len(task_names):
            _fail("DatasetSpec.tasks", "task names must be unique")
        known_tables = set(table_names)
        missing_refs: List[str] = []
        for rule in self.unit_rules:
            if rule.source_table not in known_tables:
                missing_refs.append("unit rule {0!r} -> {1!r}".format(rule.name, rule.source_table))
        for rule in self.event_rules:
            if rule.source_table not in known_tables:
                missing_refs.append("event rule {0!r} -> {1!r}".format(rule.name, rule.source_table))
            if rule.pair_with and rule.pair_with not in set(event_names):
                missing_refs.append("event rule {0!r} pair -> {1!r}".format(rule.name, rule.pair_with))
        for task in self.tasks:
            if task.episode_table not in known_tables:
                missing_refs.append("task {0!r} -> {1!r}".format(task.name, task.episode_table))
        if missing_refs:
            _fail("DatasetSpec", "unknown references: {0}".format("; ".join(missing_refs)))

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetSpec":
        data = _strict_kwargs(cls, payload, "DatasetSpec")
        nested = (
            ("tables", TableRoleSpec),
            ("tasks", TaskSpec),
            ("dictionaries", DictionarySpec),
            ("unit_rules", UnitRule),
            ("event_rules", EventRule),
        )
        for name, nested_cls in nested:
            if name not in data:
                continue
            value = data[name]
            if not isinstance(value, (list, tuple)):
                _fail("DatasetSpec.{0}".format(name), "expected an array")
            data[name] = tuple(nested_cls.from_dict(item) for item in value)
        for name in ("warnings", "unresolved_decisions"):
            if name in data:
                data[name] = _strings(data[name], "DatasetSpec.{0}".format(name), unique=False)
        return cls(**data)

    def _approval_payload(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload["approval_hash"] = None
        return payload

    def compute_approval_hash(self) -> str:
        """Hash every executable choice and its source fingerprints."""
        return hashlib.sha256(_canonical_json(self._approval_payload()).encode("utf-8")).hexdigest()

    @property
    def is_approved(self) -> bool:
        return bool(self.approval_hash) and self.approval_hash == self.compute_approval_hash()

    def validate(
        self,
        require_approved: bool = False,
        source_fingerprints: Optional[Mapping[str, str]] = None,
    ) -> "DatasetSpec":
        """Revalidate source identity and, optionally, explicit approval."""
        if source_fingerprints is not None:
            current = dict(source_fingerprints)
            if current != self.source_fingerprints:
                raise ApprovalRequiredError("Source fingerprints changed; the plan must be reviewed again.")
        if require_approved:
            self.assert_approved(source_fingerprints)
        return self

    def approve(
        self, source_fingerprints: Optional[Mapping[str, str]] = None
    ) -> "DatasetSpec":
        """Return an immutable approved copy after all blocking checks pass."""
        if self.unresolved_decisions:
            raise ApprovalRequiredError(
                "Cannot approve a plan with unresolved decisions: {0}".format(
                    "; ".join(self.unresolved_decisions)
                )
            )
        if not self.source_fingerprints:
            raise ApprovalRequiredError("Cannot approve a plan without source fingerprints.")
        if self.confidence < 0.75:
            raise ApprovalRequiredError(
                "Cannot approve a low-confidence plan ({0:.2f} < 0.75).".format(
                    self.confidence
                )
            )
        blocking_warning_terms = (
            "temporal leakage",
            "possible leakage",
            "incompatible unit",
            "unsupported outcome window",
            "low-confidence clinical",
        )
        blocking_warnings = [
            warning
            for warning in self.warnings
            if any(term in warning.lower() for term in blocking_warning_terms)
        ]
        if blocking_warnings:
            raise ApprovalRequiredError(
                "Cannot approve plan with blocking warning(s): {0}".format(
                    "; ".join(blocking_warnings)
                )
            )
        if source_fingerprints is not None and dict(source_fingerprints) != self.source_fingerprints:
            raise ApprovalRequiredError("Source fingerprints changed; regenerate or revalidate the plan.")
        unsigned = replace(self, approval_hash=None)
        return replace(unsigned, approval_hash=unsigned.compute_approval_hash())

    def assert_approved(
        self, source_fingerprints: Optional[Mapping[str, str]] = None
    ) -> "DatasetSpec":
        if not self.is_approved:
            raise ApprovalRequiredError("This dataset plan has not been explicitly approved.")
        if source_fingerprints is not None and dict(source_fingerprints) != self.source_fingerprints:
            raise ApprovalRequiredError("Approved source fingerprints no longer match the local files.")
        return self


def _dataclass_dict(value: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in fields(value):
        current = getattr(value, item.name)
        if hasattr(current, "to_dict"):
            out[item.name] = current.to_dict()
        elif isinstance(current, tuple):
            out[item.name] = [
                entry.to_dict() if hasattr(entry, "to_dict") else _json_value(entry)
                for entry in current
            ]
        elif isinstance(current, Mapping):
            out[item.name] = _json_value(current)
        else:
            out[item.name] = _json_value(current)
    return out


_BINARY_OPERATORS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "mod": operator.mod,
}
_NARY_OPERATORS = ("and", "or", "coalesce", "minimum", "maximum")
_UNARY_OPERATORS = ("not", "neg", "abs", "is_null", "not_null", "lower", "upper")
_SPECIAL_OPERATORS = ("column", "literal", "in", "not_in", "between", "contains", "starts_with", "ends_with")
_ALLOWED_OPERATORS = set(_BINARY_OPERATORS) | set(_NARY_OPERATORS) | set(_UNARY_OPERATORS) | set(_SPECIAL_OPERATORS)


def validate_expression(expression: Mapping[str, Any], path: str = "expression") -> None:
    """Validate the shape and complexity of a safe expression tree."""
    counter = [0]

    def visit(node: Any, node_path: str, depth: int) -> None:
        counter[0] += 1
        if counter[0] > _MAX_EXPRESSION_NODES:
            _fail(path, "expression exceeds {0} nodes".format(_MAX_EXPRESSION_NODES))
        if depth > _MAX_EXPRESSION_DEPTH:
            _fail(path, "expression exceeds maximum depth {0}".format(_MAX_EXPRESSION_DEPTH))
        if not isinstance(node, Mapping):
            _fail(node_path, "expression nodes must be objects")
        unknown = set(node) - {"op", "name", "value", "args"}
        if unknown:
            _fail(node_path, "unknown field(s): {0}".format(", ".join(sorted(unknown))))
        op = node.get("op")
        if not isinstance(op, str) or op not in _ALLOWED_OPERATORS:
            _fail(node_path + ".op", "unsupported operator {0!r}".format(op))
        if op == "column":
            if set(node) != {"op", "name"}:
                _fail(node_path, "column requires exactly 'op' and 'name'")
            _string(node["name"], node_path + ".name")
            return
        if op == "literal":
            if set(node) != {"op", "value"}:
                _fail(node_path, "literal requires exactly 'op' and 'value'")
            _json_value(node["value"], node_path + ".value")
            return
        if set(node) != {"op", "args"}:
            _fail(node_path, "{0} requires exactly 'op' and 'args'".format(op))
        args = node["args"]
        if not isinstance(args, (list, tuple)):
            _fail(node_path + ".args", "expected an array")
        if op in _BINARY_OPERATORS or op in ("in", "not_in", "contains", "starts_with", "ends_with"):
            expected = 2
            if len(args) != expected:
                _fail(node_path + ".args", "{0} expects {1} operands".format(op, expected))
        elif op == "between":
            if len(args) != 3:
                _fail(node_path + ".args", "between expects three operands")
        elif op in _UNARY_OPERATORS:
            if len(args) != 1:
                _fail(node_path + ".args", "{0} expects one operand".format(op))
        elif op in ("and", "or", "coalesce", "minimum", "maximum") and len(args) < 1:
            _fail(node_path + ".args", "{0} expects at least one operand".format(op))
        for i, child in enumerate(args):
            visit(child, "{0}.args[{1}]".format(node_path, i), depth + 1)

    visit(expression, path, 0)


class SafeRuleEvaluator:
    """Interpreter for the expression whitelist.

    ``context`` may contain scalars, NumPy arrays or pandas Series.  Operations
    are applied directly to those objects, preserving vectorised behaviour.
    """

    def evaluate(self, expression: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
        validate_expression(expression)
        if not isinstance(context, Mapping):
            raise TypeError("context must be a mapping")
        return self._evaluate(expression, context)

    def evaluate_rule(self, rule: Any, context: Mapping[str, Any]) -> Any:
        """Evaluate one validated rule object against scalar or tabular context.

        Event rules return metadata arrays rather than executing joins or
        mutating a frame.  Pairing and aggregation therefore remain explicit
        downstream operations, while every value-producing expression still
        passes through this interpreter.
        """
        if isinstance(rule, UnitRule):
            applies = True if rule.when is None else self.evaluate(rule.when, context)
            if "value" not in context:
                raise KeyError("UnitRule evaluation requires context['value']")
            converted = rule.convert(context["value"])
            if isinstance(applies, bool):
                return converted if applies else context["value"]
            if hasattr(context["value"], "where"):
                return context["value"].where(operator.invert(applies), converted)
            try:
                import numpy as np

                return np.where(applies, converted, context["value"])
            except ImportError as exc:  # pragma: no cover - package requires numpy
                raise TypeError("vector unit conversion requires NumPy") from exc
        if isinstance(rule, EventRule):
            return {
                "name": rule.name,
                "mask": self.evaluate(rule.predicate, context),
                "time": self.evaluate(rule.time_expression, context),
                "value": (
                    self.evaluate(rule.value_expression, context)
                    if rule.value_expression is not None
                    else None
                ),
                "operation": rule.operation,
                "aggregation": rule.aggregation,
                "pair_with": rule.pair_with,
                "window_hours": rule.window_hours,
            }
        if isinstance(rule, (ActionSpec, CostRule)):
            return self.evaluate(rule.expression, context)
        raise TypeError(
            "rule must be UnitRule, EventRule, ActionSpec or CostRule, got {0}".format(
                type(rule).__name__
            )
        )

    def _evaluate(self, node: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
        op = node["op"]
        if op == "column":
            name = node["name"]
            if name not in context:
                raise KeyError("Expression references unavailable column {0!r}".format(name))
            return context[name]
        if op == "literal":
            return _json_value(node["value"])
        args = [self._evaluate(child, context) for child in node["args"]]
        if op in _BINARY_OPERATORS:
            if op in ("div", "mod") and _is_zero_scalar(args[1]):
                raise ZeroDivisionError("division by zero in declarative expression")
            return _BINARY_OPERATORS[op](args[0], args[1])
        if op == "and":
            result = args[0]
            for value in args[1:]:
                result = operator.and_(result, value)
            return result
        if op == "or":
            result = args[0]
            for value in args[1:]:
                result = operator.or_(result, value)
            return result
        if op == "not":
            return operator.invert(args[0]) if not isinstance(args[0], bool) else not args[0]
        if op == "neg":
            return operator.neg(args[0])
        if op == "abs":
            return abs(args[0])
        if op in ("is_null", "not_null"):
            result = _is_null(args[0])
            return operator.invert(result) if op == "not_null" and not isinstance(result, bool) else (
                not result if op == "not_null" else result
            )
        if op in ("in", "not_in"):
            result = _membership(args[0], args[1])
            return operator.invert(result) if op == "not_in" and not isinstance(result, bool) else (
                not result if op == "not_in" else result
            )
        if op == "between":
            return operator.and_(operator.ge(args[0], args[1]), operator.le(args[0], args[2]))
        if op in ("lower", "upper"):
            method = op
            if hasattr(args[0], "str"):
                return getattr(args[0].str, method)()
            return getattr(str(args[0]), method)()
        if op in ("contains", "starts_with", "ends_with"):
            return _string_operation(op, args[0], args[1])
        if op == "coalesce":
            result = args[0]
            for value in args[1:]:
                if hasattr(result, "fillna"):
                    result = result.fillna(value)
                elif _is_null(result):
                    result = value
            return result
        if op in ("minimum", "maximum"):
            chooser = min if op == "minimum" else max
            if all(_is_scalar(value) for value in args):
                return chooser(args)
            result = args[0]
            comparator = operator.lt if op == "minimum" else operator.gt
            for value in args[1:]:
                condition = comparator(value, result)
                if hasattr(result, "where"):
                    result = result.where(operator.invert(condition), value)
                else:
                    try:
                        import numpy as np

                        result = np.where(condition, value, result)
                    except ImportError as exc:  # pragma: no cover - package requires numpy
                        raise TypeError("vector minimum/maximum requires NumPy") from exc
            return result
        raise SpecValidationError("Unsupported expression operator {0!r}".format(op))


def _is_zero_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int, float))


def _is_null(value: Any) -> Any:
    if value is None:
        return True
    if hasattr(value, "isna"):
        return value.isna()
    try:
        import pandas as pd

        return pd.isna(value)
    except ImportError:  # pragma: no cover - package requires pandas
        return isinstance(value, float) and math.isnan(value)


def _membership(value: Any, choices: Any) -> Any:
    if not isinstance(choices, (list, tuple, set)):
        raise TypeError("The right operand of 'in' must be an array literal")
    if hasattr(value, "isin"):
        return value.isin(list(choices))
    return value in choices


def _string_operation(op: str, value: Any, needle: Any) -> Any:
    needle = str(needle)
    method = {"contains": "contains", "starts_with": "startswith", "ends_with": "endswith"}[op]
    if hasattr(value, "str"):
        return getattr(value.str, method)(needle, na=False)
    return getattr(str(value), method)(needle)


def evaluate_expression(expression: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
    """Convenience wrapper around :class:`SafeRuleEvaluator`."""
    return SafeRuleEvaluator().evaluate(expression, context)


def evaluate_rule(rule: Any, context: Mapping[str, Any]) -> Any:
    """Convenience wrapper for evaluating a validated declarative rule."""
    return SafeRuleEvaluator().evaluate_rule(rule, context)
