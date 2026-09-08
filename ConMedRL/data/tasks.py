"""Safe compiler for approved declarative clinical task specifications."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .mdp import CostSpec, register_cost_spec
from .schema import CANONICAL_VARIABLES, VariableSpec, register_task_state_space
from .specs import SafeRuleEvaluator

__all__ = ["evaluate_rule", "register_declarative_task"]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return value


def _series(value: Any, frame: pd.DataFrame) -> Any:
    value = _plain(value)
    if isinstance(value, str):
        if value not in frame:
            raise KeyError("Task rule references missing column {0!r}.".format(value))
        return frame[value]
    if not isinstance(value, Mapping):
        return value
    if "column" in value:
        column = str(value["column"])
        if column not in frame:
            raise KeyError("Task rule references missing column {0!r}.".format(column))
        return frame[column]
    if "value" in value and "op" not in value:
        return value["value"]
    return evaluate_rule(value, frame)


def evaluate_rule(rule: Any, frame: pd.DataFrame) -> Any:
    """Evaluate a small expression language without Python ``eval``.

    Rules are nested dictionaries such as
    ``{"op": "where", "args": [{"column": "action"}, 1, 0]}``.
    Only the operators below are executable.
    """
    rule = _plain(rule)
    if not isinstance(rule, Mapping):
        return _series(rule, frame)
    op = str(rule.get("op", "")).strip().lower()
    args = [_series(value, frame) for value in rule.get("args", ())]
    if not op:
        return _series(rule, frame)

    binary = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / np.where(np.asarray(b) == 0, np.nan, b),
        "min": lambda a, b: np.minimum(a, b),
        "max": lambda a, b: np.maximum(a, b),
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
        "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b,
        "and": lambda a, b: np.asarray(a, dtype=bool) & np.asarray(b, dtype=bool),
        "or": lambda a, b: np.asarray(a, dtype=bool) | np.asarray(b, dtype=bool),
    }
    if op in binary:
        if len(args) != 2:
            raise ValueError("Operator {0!r} requires two arguments.".format(op))
        return binary[op](args[0], args[1])
    if op == "not":
        return ~np.asarray(args[0], dtype=bool)
    if op == "neg":
        return -args[0]
    if op == "abs":
        return np.abs(args[0])
    if op == "isna":
        return pd.isna(args[0])
    if op == "fillna":
        return pd.Series(args[0], index=frame.index).fillna(args[1])
    if op == "clip":
        return pd.Series(args[0], index=frame.index).clip(args[1], args[2])
    if op == "where":
        if len(args) != 3:
            raise ValueError("Operator 'where' requires condition/true/false.")
        return np.where(np.asarray(args[0], dtype=bool), args[1], args[2])
    raise ValueError("Unsupported task-rule operator {0!r}.".format(op))


def _variable(item: Any) -> VariableSpec:
    if isinstance(item, str):
        return CANONICAL_VARIABLES.get(item, VariableSpec(item, "unknown"))
    value = dict(_plain(item))
    name = str(value.get("name") or value.get("canonical") or "")
    if not name:
        raise ValueError("State variable entries require name/canonical.")
    return CANONICAL_VARIABLES.get(
        name,
        VariableSpec(
            name=name,
            kind=str(value.get("kind", "unknown")),
            unit=value.get("unit"),
            plausible_range=(
                tuple(value["plausible_range"])
                if value.get("plausible_range") is not None
                else None
            ),
            required=bool(value.get("required", False)),
            aliases=tuple(value.get("aliases") or ()),
            description=str(value.get("description", "")),
        ),
    )


def register_declarative_task(task_spec: Any, replace: bool = False) -> CostSpec:
    """Compile and register an approved task specification."""
    spec = dict(_plain(task_spec))
    task = str(spec.get("name") or spec.get("task") or "").strip()
    if not task:
        raise ValueError("Task specification requires a name.")
    states = (
        spec.get("states")
        or spec.get("state_variables")
        or spec.get("state_columns")
        or ()
    )
    register_task_state_space(task, [_variable(item) for item in states], replace=replace)

    action = dict(_plain(spec.get("action") or {}))
    action_type = str(action.get("type") or action.get("kind", "discrete")).lower()
    action_columns = tuple(
        action.get("columns") or (action.get("name") or "action",)
    )
    if action_type not in ("discrete", "continuous"):
        raise ValueError("Action type must be discrete or continuous.")
    if action_type == "discrete" and len(action_columns) != 1:
        raise ValueError("Discrete tasks require one encoded action column.")

    objective = dict(_plain(spec.get("objective") or {}))
    objective_name = str(objective.get("name", "objective_cost"))
    if "expression" not in objective and "rule" not in objective:
        raise ValueError("Task objective requires an expression.")
    constraints: List[Dict[str, Any]] = [
        dict(_plain(item)) for item in (spec.get("constraints") or ())
    ]
    constraint_names = tuple(
        str(item.get("name", "constraint_{0}".format(i)))
        for i, item in enumerate(constraints)
    )

    def builder(frame: pd.DataFrame, cohort: Any) -> pd.DataFrame:
        context = {name: frame[name] for name in frame.columns}

        def compute(expression: Any) -> Any:
            expression = _plain(expression)
            if isinstance(expression, Mapping) and "op" in expression:
                try:
                    return SafeRuleEvaluator().evaluate(expression, context)
                except (TypeError, ValueError):
                    pass
            return evaluate_rule(expression, frame)

        values: Dict[str, Any] = {
            objective_name: compute(
                objective.get("expression", objective.get("rule"))
            )
        }
        for name, item in zip(constraint_names, constraints):
            expression = item.get("expression", item.get("rule"))
            if expression is None:
                raise ValueError("Constraint {0!r} has no expression.".format(name))
            values[name] = compute(expression)
        return pd.DataFrame(values, index=frame.index).astype(float)

    cost_spec = CostSpec(
        task=task,
        action_column=action_columns[0],
        action_type=action_type,
        action_columns=action_columns,
        action_categories=tuple(action.get("categories") or ()),
        action_bounds=tuple(tuple(value) for value in (action.get("bounds") or ())),
        terminal_column=(
            action.get("terminal_column")
            or ("__terminal_rule__" if spec.get("terminal_rule") is not None else None)
        ),
        censoring_column=(
            "__censoring_rule__" if spec.get("censoring_rule") is not None else None
        ),
        objective=objective_name,
        constraints=constraint_names,
        builder=builder,
        description=str(spec.get("description", "")),
    )
    register_cost_spec(cost_spec, replace=replace)
    return cost_spec
