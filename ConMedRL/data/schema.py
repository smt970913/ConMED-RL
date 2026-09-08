"""Canonical state-space definitions and availability-driven resolution.

The same decision problem has to run against databases that simply do not
record the same things: MIMIC-IV carries a venous blood-gas panel that SICdb
mostly lacks, and SICdb records respirator settings MIMIC-IV names differently.
Hard-coding one variable list per (database, task) pair is what made the
original four preprocessing scripts diverge.

Instead, each *task* declares a canonical, ordered state space of
:class:`VariableSpec` entries, and each database adapter reports which of those
it could actually extract. :func:`resolve_state_space` intersects the two and
returns a :class:`StateSpaceSchema` recording both what survived and why
anything was dropped. The resulting state dimension is whatever the data
supports, which is exactly what ``RLTraining(state_dim=...)`` needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import Database, Task

__all__ = [
    "VariableSpec",
    "StateSpaceSchema",
    "CANONICAL_VARIABLES",
    "TASK_STATE_SPACE",
    "task_state_space",
    "register_task_state_space",
    "resolve_state_space",
]


# Variable roles, used for grouping in reports and for choosing sensible
# imputation strategies.
KIND_DEMOGRAPHIC = "demographic"
KIND_VITAL = "vital"
KIND_LAB = "lab"
KIND_VENTILATION = "ventilation"
KIND_NEURO = "neuro"
KIND_DERIVED = "derived"


@dataclass(frozen=True)
class VariableSpec:
    """One canonical state variable, independent of any source database.

    Attributes
    ----------
    name:
        Canonical column name. This is what ends up in the state table, and
        what adapters must map their source labels onto.
    kind:
        Role of the variable (see the ``KIND_*`` constants).
    unit:
        Canonical unit. Adapters are responsible for converting into it.
    plausible_range:
        ``(low, high)`` physiological bounds. Values outside are treated as
        recording errors and set to missing before imputation. ``None`` skips
        range checking.
    required:
        When ``True`` the task cannot be built without this variable, so a
        database that lacks it raises instead of silently shrinking the state.
    aliases:
        Extra source labels that mean the same thing, used by the fuzzy
        column-matching helpers.
    """

    name: str
    kind: str
    unit: Optional[str] = None
    plausible_range: Optional[Tuple[float, float]] = None
    required: bool = False
    aliases: Tuple[str, ...] = ()
    description: str = ""


def _v(
    name: str,
    kind: str,
    unit: Optional[str] = None,
    plausible_range: Optional[Tuple[float, float]] = None,
    required: bool = False,
    aliases: Sequence[str] = (),
    description: str = "",
) -> VariableSpec:
    return VariableSpec(
        name=name,
        kind=kind,
        unit=unit,
        plausible_range=plausible_range,
        required=required,
        aliases=tuple(aliases),
        description=description,
    )


# ---------------------------------------------------------------------------
# Canonical variable dictionary
# ---------------------------------------------------------------------------

#: Every state variable either task can use, keyed by canonical name.
#: Plausible ranges are deliberately wide -- they exist to catch unit mix-ups
#: and charting errors, not to enforce clinical normality.
CANONICAL_VARIABLES: Dict[str, VariableSpec] = {
    spec.name: spec
    for spec in [
        # -- demographics ----------------------------------------------------
        _v("age", KIND_DEMOGRAPHIC, "years", (18.0, 120.0), required=True),
        _v(
            "M",
            KIND_DEMOGRAPHIC,
            "indicator",
            (0.0, 1.0),
            required=True,
            aliases=("gender", "sex", "is_male"),
            description="1 for male, 0 otherwise.",
        ),
        _v("weight", KIND_DEMOGRAPHIC, "kg", (20.0, 350.0), aliases=("patientweight", "Admission Weight (Kg)")),
        _v("height", KIND_DEMOGRAPHIC, "cm", (100.0, 230.0)),
        # -- vital signs -----------------------------------------------------
        _v("Heart Rate", KIND_VITAL, "bpm", (10.0, 300.0), required=True, aliases=("HeartRateECG", "HeartRateSPO2", "HeartRateABP")),
        _v("Blood Pressure Systolic", KIND_VITAL, "mmHg", (30.0, 300.0), aliases=("Arterial Blood Pressure systolic", "Non Invasive Blood Pressure systolic", "ART BP Systolic")),
        _v("Blood Pressure Diastolic", KIND_VITAL, "mmHg", (10.0, 200.0), aliases=("Arterial Blood Pressure diastolic", "Non Invasive Blood Pressure diastolic", "ART BP Diastolic")),
        _v("Blood Pressure Mean", KIND_VITAL, "mmHg", (20.0, 250.0), aliases=("Arterial Blood Pressure mean", "Non Invasive Blood Pressure mean", "ART BP Mean")),
        _v("Temperature C", KIND_VITAL, "degC", (25.0, 45.0), aliases=("Temperature Celsius", "TempCore", "Temperature")),
        _v("SaO2", KIND_VITAL, "%", (30.0, 100.0), aliases=("Arterial O2 Saturation", "O2 saturation pulseoxymetry", "SpO2")),
        _v("Respiratory Rate", KIND_VITAL, "breaths/min", (0.0, 80.0), aliases=("RR", "RespRate", "Respiratory Rate (Total)")),
        # -- neurological ----------------------------------------------------
        _v("GCS Score", KIND_NEURO, "points", (3.0, 15.0), aliases=("GCS score", "Glasgow Coma Scale")),
        _v("RASS", KIND_NEURO, "points", (-5.0, 4.0), aliases=("Richmond Agitation-Sedation Scale",)),
        # -- blood gas / labs ------------------------------------------------
        _v("Arterial O2 pressure", KIND_LAB, "mmHg", (20.0, 700.0), aliases=("PaO2", "pO2 (arterial)")),
        _v("Arterial CO2 Pressure", KIND_LAB, "mmHg", (10.0, 150.0), aliases=("PaCO2", "pCO2 (arterial)")),
        _v("PH (Arterial)", KIND_LAB, "pH", (6.5, 8.0), aliases=("pH arterial",)),
        _v("PH (Venous)", KIND_LAB, "pH", (6.5, 8.0), aliases=("pH venous",)),
        _v("Venous O2 Pressure", KIND_LAB, "mmHg", (10.0, 300.0)),
        _v("Venous CO2 Pressure", KIND_LAB, "mmHg", (10.0, 150.0)),
        _v("Arterial Base Excess", KIND_LAB, "mmol/L", (-40.0, 40.0), aliases=("Base Excess", "BE")),
        _v("HCO3 (serum)", KIND_LAB, "mmol/L", (2.0, 60.0), aliases=("Bicarbonate", "Standardbikarbonat")),
        _v("Hemoglobin", KIND_LAB, "g/dL", (2.0, 25.0), aliases=("Haemoglobin", "Hb")),
        _v("Hematocrit (serum)", KIND_LAB, "%", (5.0, 70.0), aliases=("Hct", "Haematocrit")),
        _v("Hematocrit (whole blood - calc)", KIND_LAB, "%", (5.0, 70.0)),
        _v("WBC", KIND_LAB, "K/uL", (0.1, 200.0), aliases=("White Blood Cells", "Leukocytes", "Leukozyten")),
        _v("Platelet Count", KIND_LAB, "K/uL", (1.0, 2000.0), aliases=("Thrombocytes", "Thrombozyten")),
        _v("Sodium (serum)", KIND_LAB, "mmol/L", (100.0, 180.0), aliases=("Natrium",)),
        _v("Sodium (whole blood)", KIND_LAB, "mmol/L", (100.0, 180.0)),
        _v("Potassium (serum)", KIND_LAB, "mmol/L", (1.0, 10.0), aliases=("Kalium",)),
        _v("Potassium (whole blood)", KIND_LAB, "mmol/L", (1.0, 10.0)),
        _v("Chloride (serum)", KIND_LAB, "mmol/L", (60.0, 160.0), aliases=("Chlorid",)),
        _v("Chloride (whole blood)", KIND_LAB, "mmol/L", (60.0, 160.0)),
        _v("Magnesium", KIND_LAB, "mg/dL", (0.2, 10.0)),
        _v("Ionized Calcium", KIND_LAB, "mmol/L", (0.2, 3.0), aliases=("Calcium ionized", "Free Calcium")),
        _v("Creatinine (serum)", KIND_LAB, "mg/dL", (0.05, 25.0), aliases=("Kreatinin",)),
        _v("Creatinine (whole blood)", KIND_LAB, "mg/dL", (0.05, 25.0)),
        _v("BUN", KIND_LAB, "mg/dL", (1.0, 250.0), aliases=("Urea", "Harnstoff", "Blood Urea Nitrogen")),
        _v("Glucose (serum)", KIND_LAB, "mg/dL", (10.0, 1500.0), aliases=("Glukose", "Blood Sugar")),
        _v("Glucose (whole blood)", KIND_LAB, "mg/dL", (10.0, 1500.0)),
        _v("Total Bilirubin", KIND_LAB, "mg/dL", (0.05, 60.0), aliases=("Bilirubin total",)),
        _v("Direct Bilirubin", KIND_LAB, "mg/dL", (0.01, 40.0)),
        _v("Albumin", KIND_LAB, "g/dL", (0.5, 7.0)),
        _v("Lactate", KIND_LAB, "mmol/L", (0.1, 30.0), aliases=("Laktat",)),
        _v("Prothrombin time", KIND_LAB, "s", (5.0, 150.0), aliases=("PT", "Quick")),
        _v("PTT", KIND_LAB, "s", (10.0, 250.0), aliases=("aPTT", "partial thromboplastin time")),
        _v("INR", KIND_LAB, "ratio", (0.5, 20.0)),
        # -- ventilation -----------------------------------------------------
        _v("Inspired O2 Fraction", KIND_VENTILATION, "%", (21.0, 100.0), aliases=("FiO2", "FiO2 Set")),
        _v("Tidal Volume", KIND_VENTILATION, "mL", (50.0, 2500.0), aliases=("TV", "Tidal Volume (observed)", "Tidal Volume (set)", "Vt")),
        _v("PEEP Level", KIND_VENTILATION, "cmH2O", (0.0, 40.0), aliases=("PEEP", "PEEP set", "Total PEEP Level")),
        _v("Peak Inspiratory Pressure", KIND_VENTILATION, "cmH2O", (0.0, 80.0), aliases=("PIP", "Peak Insp. Pressure")),
        _v("Plateau Pressure", KIND_VENTILATION, "cmH2O", (0.0, 80.0)),
        _v("Minute Volume", KIND_VENTILATION, "L/min", (0.5, 40.0), aliases=("MV", "Minutenvolumen")),
        _v("Mean Airway Pressure", KIND_VENTILATION, "cmH2O", (0.0, 60.0), aliases=("Pmean",)),
        _v("Dynamic Compliance", KIND_VENTILATION, "mL/cmH2O", (1.0, 200.0), aliases=("Compliance",)),
        _v("PaO2/FiO2 Ratio", KIND_DERIVED, "mmHg", (10.0, 700.0), aliases=("P/F ratio", "PF ratio")),
        _v("Rapid Shallow Breathing Index", KIND_DERIVED, "breaths/min/L", (0.0, 500.0), aliases=("RSBI", "f/Vt")),
        _v("Mechanical Ventilation Duration", KIND_DERIVED, "hours", (0.0, 2000.0), aliases=("vent_duration", "imv_hours")),
        # -- history / context ----------------------------------------------
        _v("readmission_count", KIND_DERIVED, "count", (0.0, 50.0), description="ICU readmissions accumulated so far for this admission."),
        _v("extubation_count", KIND_DERIVED, "count", (0.0, 20.0), description="Prior extubation attempts within this ICU stay."),
    ]
}


# ---------------------------------------------------------------------------
# Per-task canonical state spaces
# ---------------------------------------------------------------------------

# Ordering matters: it fixes the column order of the state table, and therefore
# the meaning of each input unit of the Q-networks. Appending is safe; do not
# reorder without retraining.

_DISCHARGE_STATE_SPACE: Tuple[str, ...] = (
    "age",
    "M",
    "weight",
    "Heart Rate",
    "Arterial O2 pressure",
    "Hemoglobin",
    "Arterial CO2 Pressure",
    "PH (Venous)",
    "Hematocrit (serum)",
    "WBC",
    "Chloride (serum)",
    "Creatinine (serum)",
    "Glucose (serum)",
    "Magnesium",
    "Sodium (serum)",
    "PH (Arterial)",
    "Inspired O2 Fraction",
    "Arterial Base Excess",
    "BUN",
    "Ionized Calcium",
    "Total Bilirubin",
    "Glucose (whole blood)",
    "Potassium (serum)",
    "HCO3 (serum)",
    "Platelet Count",
    "Prothrombin time",
    "PTT",
    "INR",
    "Blood Pressure Systolic",
    "Blood Pressure Diastolic",
    "Blood Pressure Mean",
    "Temperature C",
    "SaO2",
    "GCS Score",
    "RASS",
    "Respiratory Rate",
    "Tidal Volume",
    "readmission_count",
)

_EXTUBATION_STATE_SPACE: Tuple[str, ...] = (
    "age",
    "M",
    "weight",
    "Heart Rate",
    "Blood Pressure Systolic",
    "Blood Pressure Diastolic",
    "Blood Pressure Mean",
    "Temperature C",
    "SaO2",
    "Respiratory Rate",
    "GCS Score",
    "RASS",
    "Arterial O2 pressure",
    "Arterial CO2 Pressure",
    "PH (Arterial)",
    "Arterial Base Excess",
    "HCO3 (serum)",
    "Hemoglobin",
    "Hematocrit (serum)",
    "WBC",
    "Platelet Count",
    "Sodium (serum)",
    "Potassium (serum)",
    "Chloride (serum)",
    "Magnesium",
    "Ionized Calcium",
    "Creatinine (serum)",
    "BUN",
    "Glucose (serum)",
    "Total Bilirubin",
    "Albumin",
    "Lactate",
    "Prothrombin time",
    "PTT",
    "INR",
    "Inspired O2 Fraction",
    "Tidal Volume",
    "PEEP Level",
    "Peak Inspiratory Pressure",
    "Minute Volume",
    "Mean Airway Pressure",
    "Dynamic Compliance",
    "PaO2/FiO2 Ratio",
    "Rapid Shallow Breathing Index",
    "Mechanical Ventilation Duration",
    "extubation_count",
)

#: Canonical, ordered state space per task.
TASK_STATE_SPACE: Dict[str, Tuple[str, ...]] = {
    Task.DISCHARGE: _DISCHARGE_STATE_SPACE,
    Task.EXTUBATION: _EXTUBATION_STATE_SPACE,
}
_CUSTOM_TASK_VARIABLES: Dict[str, Tuple[VariableSpec, ...]] = {}


def task_state_space(task: str) -> List[VariableSpec]:
    """Return the full canonical state space for ``task``, in order."""
    custom_key = str(task).strip().lower().replace("_", "-").replace(" ", "-")
    if custom_key in _CUSTOM_TASK_VARIABLES:
        return list(_CUSTOM_TASK_VARIABLES[custom_key])
    task = Task.normalize(task)
    names = TASK_STATE_SPACE[task]
    missing = [n for n in names if n not in CANONICAL_VARIABLES]
    if missing:
        raise KeyError(
            "Task {0!r} references variables absent from CANONICAL_VARIABLES: "
            "{1}".format(task, ", ".join(missing))
        )
    return [CANONICAL_VARIABLES[n] for n in names]


def register_task_state_space(
    task: str, variables: Sequence[VariableSpec], replace: bool = False
) -> str:
    """Register an approved custom task's ordered state specification."""
    key = str(task).strip().lower().replace("_", "-").replace(" ", "-")
    if not key:
        raise ValueError("Custom task name cannot be empty.")
    if key in TASK_STATE_SPACE and not replace:
        raise ValueError("Refusing to replace built-in task {0!r}.".format(key))
    if key in _CUSTOM_TASK_VARIABLES and not replace:
        raise ValueError("Task state space {0!r} is already registered.".format(key))
    values = tuple(variables)
    if not values:
        raise ValueError("A task state space cannot be empty.")
    names = [value.name for value in values]
    if len(names) != len(set(names)):
        raise ValueError("Task state variable names must be unique.")
    _CUSTOM_TASK_VARIABLES[key] = values
    return key


# ---------------------------------------------------------------------------
# Resolved schema
# ---------------------------------------------------------------------------


@dataclass
class StateSpaceSchema:
    """The state space that a concrete (database, task) pair actually supports.

    ``variables`` is the ordered list that becomes the state-table columns, so
    ``len(schema)`` is the ``state_dim`` to hand to ``RLTraining``.
    """

    task: str
    database: str
    variables: List[VariableSpec] = field(default_factory=list)
    #: Canonical name -> human-readable reason it is not in ``variables``.
    dropped: Dict[str, str] = field(default_factory=dict)
    #: Canonical name -> observed fraction of non-missing rows, when known.
    coverage: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.variables)

    def __iter__(self):
        return iter(self.variables)

    def __contains__(self, name: object) -> bool:
        return any(v.name == name for v in self.variables)

    @property
    def names(self) -> List[str]:
        """Ordered state-table column names."""
        return [v.name for v in self.variables]

    @property
    def state_dim(self) -> int:
        return len(self.variables)

    def by_kind(self, kind: str) -> List[str]:
        return [v.name for v in self.variables if v.kind == kind]

    def spec(self, name: str) -> VariableSpec:
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise KeyError("{0!r} is not part of this state space".format(name))

    @property
    def plausible_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            v.name: v.plausible_range
            for v in self.variables
            if v.plausible_range is not None
        }

    def to_frame(self):
        """Tabular summary, handy for eyeballing in a notebook."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "index": i,
                    "name": v.name,
                    "kind": v.kind,
                    "unit": v.unit,
                    "coverage": self.coverage.get(v.name),
                    "required": v.required,
                }
                for i, v in enumerate(self.variables)
            ]
        )

    def dropped_frame(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {"name": name, "reason": reason, "coverage": self.coverage.get(name)}
                for name, reason in sorted(self.dropped.items())
            ]
        )

    def summary(self) -> str:
        lines = [
            "State space for task={0!r} on database={1!r}".format(self.task, self.database),
            "  kept    : {0} variable(s)".format(len(self.variables)),
            "  dropped : {0} variable(s)".format(len(self.dropped)),
        ]
        if self.dropped:
            lines.append("  dropped variables:")
            for name, reason in sorted(self.dropped.items()):
                lines.append("    - {0}: {1}".format(name, reason))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "task": self.task,
            "database": self.database,
            "state_dim": self.state_dim,
            "variables": [
                {"name": v.name, "kind": v.kind, "unit": v.unit} for v in self.variables
            ],
            "dropped": dict(self.dropped),
            "coverage": {k: round(float(v), 6) for k, v in self.coverage.items()},
        }


class MissingRequiredVariableError(RuntimeError):
    """A variable the task declares as required could not be extracted."""


def resolve_state_space(
    task: str,
    database: str,
    available: Iterable[str],
    coverage: Optional[Mapping[str, float]] = None,
    min_coverage: float = 0.0,
    require: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> StateSpaceSchema:
    """Intersect a task's canonical state space with what a database provides.

    Parameters
    ----------
    task, database:
        Normalised via :class:`~ConMedRL.data.config.Task` / ``Database``.
    available:
        Canonical variable names the adapter managed to extract.
    coverage:
        Canonical name -> fraction of rows observed (``0..1``). Missing entries
        are treated as fully observed, so an adapter that cannot cheaply
        measure coverage still works.
    min_coverage:
        Variables below this observed fraction are dropped, unless listed in
        ``require``.
    require:
        Keep these regardless of coverage. A name here that is not in
        ``available`` is an error, since the caller explicitly asked for it.
    exclude:
        Drop these even if well covered. Excluding a ``required=True`` variable
        is an error.

    Raises
    ------
    MissingRequiredVariableError
        If a ``required=True`` variable is unavailable, or an explicitly
        requested variable is unavailable.
    """
    custom_key = str(task).strip().lower().replace("_", "-").replace(" ", "-")
    task = (
        custom_key
        if custom_key in _CUSTOM_TASK_VARIABLES
        else Task.normalize(task)
    )
    database = Database.normalize(database)

    canonical = task_state_space(task)
    available_set = set(available)
    coverage = dict(coverage or {})
    require_set = set(require)
    exclude_set = set(exclude)

    unknown_request = require_set - {v.name for v in canonical}
    if unknown_request:
        raise MissingRequiredVariableError(
            "Requested variable(s) are not part of the {0} state space: {1}".format(
                task, ", ".join(sorted(unknown_request))
            )
        )
    unavailable_request = require_set - available_set
    if unavailable_request:
        raise MissingRequiredVariableError(
            "Requested variable(s) could not be extracted from {0}: {1}".format(
                database, ", ".join(sorted(unavailable_request))
            )
        )

    kept: List[VariableSpec] = []
    dropped: Dict[str, str] = {}

    for spec in canonical:
        name = spec.name
        observed = float(coverage.get(name, 1.0))

        if name in exclude_set:
            if spec.required:
                raise MissingRequiredVariableError(
                    "{0!r} is required by the {1} task and cannot be excluded.".format(
                        name, task
                    )
                )
            dropped[name] = "excluded by configuration"
            continue

        if name in require_set:
            kept.append(spec)
            continue

        if name not in available_set:
            if spec.required:
                raise MissingRequiredVariableError(
                    "{0!r} is required by the {1} task but is not available in "
                    "{2}. Provide it via source_paths / a custom mapping, or "
                    "choose a different task.".format(name, task, database)
                )
            dropped[name] = "not available in {0}".format(database)
            continue

        if observed < min_coverage:
            if spec.required:
                raise MissingRequiredVariableError(
                    "{0!r} is required by the {1} task but only {2:.1%} of rows "
                    "are observed (min_variable_coverage={3:.1%}).".format(
                        name, task, observed, min_coverage
                    )
                )
            dropped[name] = "coverage {0:.1%} below minimum {1:.1%}".format(
                observed, min_coverage
            )
            continue

        kept.append(spec)

    return StateSpaceSchema(
        task=task,
        database=database,
        variables=kept,
        dropped=dropped,
        coverage={k: float(v) for k, v in coverage.items()},
    )
