"""Mapping a database's variable dictionary onto the canonical state space.

Both supported databases encode physiology behind integer ids and a dictionary
table: MIMIC-IV uses ``d_items.itemid`` / ``label``, SICdb uses
``d_references.ReferenceGlobalID`` / ``ReferenceValue``. Getting from "I want
Heart Rate" to "these ids carry it" is the step the original scripts left to
hand-curated lists pasted into notebooks -- which is why the SICdb scripts had
no mapping at all.

Resolution here is layered, strongest evidence first:

1. **Seed** -- ids verified against the published documentation.
2. **LOINC** -- SICdb v1.0.8 attached LOINC codes to laboratory references, so
   labs can be matched against a standard rather than a name.
3. **Alias** -- exact or normalised match on the canonical name and its aliases,
   with a unit-compatibility check.
4. **Lexical** -- fuzzy match, camel-case aware, glossary-expanded for German.
5. **LLM** -- only for names none of the above resolved.

Every mapping records *how* it was found and how confident that is, and the
whole thing is overridable, because a wrong silent mapping is far worse than a
missing variable.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .config import Database
from .llm import LLMBackend, NullLLMBackend, _lexical_rank, _normalize, _split_camel
from .schema import CANONICAL_VARIABLES, VariableSpec

__all__ = [
    "DictionarySchema",
    "MIMIC_DICTIONARY",
    "SICDB_DICTIONARY",
    "dictionary_schema_for",
    "MappedVariable",
    "VariableMapping",
    "VariableSearch",
    "resolve_variable_mapping",
    "SICDB_SEED_IDS",
    "SICDB_KNOWN_ABSENT",
    "SICDB_UNIT_CONVERSIONS",
    "SICDB_MEASUREMENT_GROUPS",
    "MIMIC_SEED_ITEMIDS",
    "MIMIC_KNOWN_ABSENT",
    "LOINC_HINTS",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dictionary schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DictionarySchema:
    """Column names of one database's variable dictionary."""

    id_column: str
    name_column: str
    unit_column: Optional[str] = None
    #: Column that groups entries by kind (``linksto`` / ``ReferenceName``).
    group_column: Optional[str] = None
    loinc_column: Optional[str] = None
    description_column: Optional[str] = None


MIMIC_DICTIONARY = DictionarySchema(
    id_column="itemid",
    name_column="label",
    unit_column="unitname",
    group_column="linksto",
    loinc_column=None,
    description_column="abbreviation",
)

SICDB_DICTIONARY = DictionarySchema(
    id_column="ReferenceGlobalID",
    name_column="ReferenceValue",
    unit_column="ReferenceUnit",
    group_column="ReferenceName",
    loinc_column="LOINC_code",
    description_column="ReferenceDescription",
)

_DICTIONARY_BY_DATABASE = {
    Database.MIMIC_IV: MIMIC_DICTIONARY,
    Database.SICDB: SICDB_DICTIONARY,
}


def dictionary_schema_for(database: str) -> DictionarySchema:
    return _DICTIONARY_BY_DATABASE[Database.normalize(database)]


# ---------------------------------------------------------------------------
# Verified seed ids
# ---------------------------------------------------------------------------

# SICdb ids verified against a real `d_references` (v1.0.8, 1622 rows) rather
# than inferred from documentation. Tuples are ordered by observed row count in
# `data_float_h`, because the first resolvable id becomes the primary source and
# the frequency ordering is not the one you would guess from the names: the
# `RespiratorSetting` FIO2 (2283) carries ~167k rows while the `SignalFloat`
# FIO2 (727) is effectively unpopulated, and the same holds for PEEP.
SICDB_SEED_IDS: Dict[str, Tuple[int, ...]] = {
    # -- monitor signals, `data_float_h`, ReferenceName == 'SignalFloat' ------
    "Heart Rate": (707, 708, 724),  # ECG, SpO2-derived, arterial-line derived
    "Blood Pressure Systolic": (701, 704),  # arterial line, then cuff
    "Blood Pressure Diastolic": (702, 705),
    "Blood Pressure Mean": (703, 706),
    "Temperature C": (709,),
    "SaO2": (710,),
    "Respiratory Rate": (2274, 719, 2280, 2282),
    "Tidal Volume": (718, 717),  # expiratory preferred over inspiratory
    "Minute Volume": (713, 2019, 2281),
    "Peak Inspiratory Pressure": (715, 3035),  # pPeak
    "Mean Airway Pressure": (2279, 712),  # pMean
    "Dynamic Compliance": (3120,),
    "Central Venous Pressure": (2018,),
    # Ventilator settings live in the same table under 'RespiratorSetting' and
    # are far better populated than their 'SignalFloat' namesakes.
    "Inspired O2 Fraction": (2283, 727, 714),
    "PEEP Level": (2278, 711),
    "Pressure Support": (2284, 2020),
    # -- laboratory, `laboratory` table --------------------------------------
    # '(ZL)' = Zentrallabor (central lab), '(BGA)' = Blutgasanalyse (blood gas).
    # Blood-gas ids are preferred for the arterial variables, central-lab ids
    # for the serum ones, which is what the canonical names actually mean.
    "Creatinine (serum)": (367, 368),  # documented SQL example uses 367
    "BUN": (355,),  # Harnstoff == urea, converted below
    "Sodium (serum)": (469,),
    "Potassium (serum)": (463,),
    "Chloride (serum)": (450,),
    "Magnesium": (468,),  # mmol/L, converted below
    "Ionized Calcium": (452, 655),
    "Glucose (serum)": (348,),
    "Glucose (whole blood)": (656,),
    "Hemoglobin": (289, 288, 658),
    "WBC": (301,),
    "Platelet Count": (314,),
    "Total Bilirubin": (333,),
    "Albumin": (287,),
    "Lactate": (465, 657),
    "PTT": (597,),
    "INR": (3128,),
    "Arterial O2 pressure": (689, 444),
    "Arterial CO2 Pressure": (687, 443),
    "PH (Arterial)": (688, 538),
    "Arterial Base Excess": (668, 449),
    "HCO3 (serum)": (666, 456),
    # Haematocrit exists only on the blood gas panel, never as a central-lab
    # value, so both canonical spellings resolve to the same reference.
    "Hematocrit (serum)": (682,),
    "Hematocrit (whole blood - calc)": (682,),
    # Whole-blood electrolytes come off the same panel.
    "Sodium (whole blood)": (686,),
    "Potassium (whole blood)": (685,),
    "Chloride (whole blood)": (683,),
    # Only the '(BGA) ven' references are genuinely venous. The 'gemV'
    # (mixed-venous) and 'kap' (capillary) siblings carry capillary LOINC
    # codes in SICdb, so they must not be used to satisfy a venous variable.
    "PH (Venous)": (697,),
    "Venous O2 Pressure": (694,),
    # SICdb computes the P/F ratio itself, so it does not have to be derived.
    "PaO2/FiO2 Ratio": (3135,),
    # -- scores, sparse but present ------------------------------------------
    "RASS": (3123,),
}

#: SICdb ids that are deliberately *not* seeded, with the reason. Surfaced in
#: the mapping report so a dropped state variable can be explained without
#: re-reading the reference table.
SICDB_KNOWN_ABSENT: Dict[str, str] = {
    "GCS Score": (
        "SICdb records no Glasgow Coma Scale; sedation depth is captured as "
        "RASS (3123) instead, which is not interchangeable with GCS."
    ),
    "Venous CO2 Pressure": (
        "the blood gas panel has venous pO2 and pH ('(BGA) ven') but no "
        "venous pCO2 reference"
    ),
    "Prothrombin time": (
        "reference 237 is a Quick percentage and 598 is mislabelled with the "
        "thrombin-time LOINC, so neither is comparable to a PT in seconds"
    ),
}

#: Dictionary groups that hold measurements. Everything else in SICdb's
#: `d_references` describes orders and administrative facts -- 622 `Drug`
#: entries, 147 `Fluid`, plus units and discharge codes. Searching those for
#: physiology is actively harmful: 'Ramipril/HCT' and 'HCT + Amilorid' are
#: antihypertensives whose names fuzzy-match haematocrit.
SICDB_MEASUREMENT_GROUPS: Tuple[str, ...] = (
    "SignalFloat",
    "SignalInt",
    "RespiratorSetting",
    "VentilatorConfiguration",
    "Laboratory",
    "Scores",
    "ProcessedFields",
)

#: Kinds that are computed rather than read from a measurement dictionary.
#: These accept seed ids and explicit overrides but are never fuzzy-matched,
#: because a name like 'M' (the male indicator) matches 'Immunglobulin M'.
NON_MEASUREMENT_KINDS: Tuple[str, ...] = ("demographic", "derived")

#: SICdb marks retired references by prefixing the name, e.g.
#: 'ZzzPO2 (BGA) art'. They still carry ids but no new data.
_DEPRECATED_NAME_PREFIXES: Tuple[str, ...] = ("zzz",)

#: Source id -> (scale, offset) applied to raw SICdb values so they arrive in
#: the canonical unit. Without this the values look plausible but are simply
#: the wrong quantity, which no range check would catch.
SICDB_UNIT_CONVERSIONS: Dict[int, Tuple[float, float]] = {
    # 'Harnstoff' is urea (MW 60.06); the canonical BUN counts only the two
    # nitrogens (MW 28.01), so BUN = urea * 28.01/60.06.
    355: (0.4665, 0.0),
    # Magnesium is reported in mmol/L but MIMIC-IV (and the canonical spec)
    # uses mg/dL: mg/dL = mmol/L * 24.305 / 10.
    468: (2.4305, 0.0),
}

# MIMIC-IV itemids taken from the working discharge notebook, so they are known
# to resolve against `d_items`. Only the unambiguous single-concept ones are
# seeded; the redundant duplicates (three flavours of arterial line pressure)
# are left to alias matching, which merges them by canonical name.
MIMIC_SEED_ITEMIDS: Dict[str, Tuple[int, ...]] = {
    "Heart Rate": (220045,),
    "Blood Pressure Systolic": (220179, 220050, 225309),
    "Blood Pressure Diastolic": (220180, 220051, 225310),
    "Blood Pressure Mean": (220181, 220052, 225312),
    "Respiratory Rate": (220210, 224690, 224688, 224689),
    "Temperature C": (223762, 223761),
    "SaO2": (220277, 220227),
    "Inspired O2 Fraction": (223835,),
    "Tidal Volume": (224685, 224684, 224686),
    "PEEP Level": (220339, 224700),
    "Peak Inspiratory Pressure": (224695,),
    "Plateau Pressure": (224696,),  # 228866 is an IABP pressure, not airway
    "Minute Volume": (224687,),  # excludes the 2202xx alarm-limit items
    "Mean Airway Pressure": (224697,),
    "Arterial O2 pressure": (220224,),
    "Arterial CO2 Pressure": (220235,),
    "PH (Arterial)": (223830,),
    "PH (Venous)": (220274,),
    "Arterial Base Excess": (224828,),
    "HCO3 (serum)": (227443,),
    "Hemoglobin": (220228,),
    "Hematocrit (serum)": (220545,),
    "WBC": (220546,),
    "Platelet Count": (227457,),
    "Sodium (serum)": (220645,),
    "Potassium (serum)": (227442,),
    "Chloride (serum)": (220602,),
    "Magnesium": (220635,),
    "Ionized Calcium": (225667,),
    "Creatinine (serum)": (220615,),
    "BUN": (225624,),
    "Glucose (serum)": (220621,),
    "Glucose (whole blood)": (226537,),
    "Total Bilirubin": (225690,),
    "Albumin": (227456,),
    "Lactate": (225668,),
    "Prothrombin time": (227465,),
    "PTT": (227466,),
    "INR": (227467,),
    "GCS Score": (220739, 223900, 223901),
    "RASS": (228096,),
    "weight": (226512, 224639, 226531),
    "height": (226730,),
}

#: The mirror image of :data:`SICDB_KNOWN_ABSENT`. Neither database is a
#: superset of the other, which is why the state space is resolved per run.
MIMIC_KNOWN_ABSENT: Dict[str, str] = {
    "Dynamic Compliance": (
        "the only 'Compliance' item (229661) is recorded in cmH2O/L/seconds, "
        "which is a resistance, not the mL/cmH2O compliance this variable "
        "means; SICdb's DynCompliance (3120) is the comparable measurement"
    ),
    "Rapid Shallow Breathing Index": (
        "not charted; derive it from Spont RR (224422) and Spont Vt (224421)"
    ),
}


# LOINC codes for the laboratory-style canonical variables. SICdb v1.0.8
# attached LOINC to its laboratory references, so these give a standards-based
# match that does not depend on German naming. Treated as evidence, not truth:
# a LOINC hit still has to pass the unit check, and every match is reported.
LOINC_HINTS: Dict[str, Tuple[str, ...]] = {
    "Creatinine (serum)": ("2160-0", "38483-4"),
    # 3091-6 is the urea code SICdb actually uses; the BUN codes are kept for
    # sources that report nitrogen directly.
    "BUN": ("3091-6", "3094-0", "6299-2", "22664-7"),
    "Sodium (serum)": ("2951-2", "2947-0"),
    "Sodium (whole blood)": ("32717-1", "2947-0"),
    "Potassium (serum)": ("2823-3", "6298-4"),
    "Potassium (whole blood)": ("32713-0", "6298-4"),
    "Chloride (serum)": ("2075-0",),
    "Chloride (whole blood)": ("41650-3", "2069-3"),
    "Magnesium": ("2601-3", "19123-9"),
    "Ionized Calcium": ("34581-9", "1994-3", "47598-8"),
    "Glucose (serum)": ("2345-7", "2339-0"),
    "Glucose (whole blood)": ("2339-0", "41653-7"),
    "Hemoglobin": ("718-7", "30313-1"),
    "Hematocrit (serum)": ("4544-3", "20570-8", "32354-3"),
    "Hematocrit (whole blood - calc)": ("32354-3", "20570-8", "4544-3"),
    "WBC": ("6690-2", "26464-8"),
    "Platelet Count": ("777-3", "26515-7"),
    "Total Bilirubin": ("1975-2", "42719-5"),
    "Direct Bilirubin": ("1968-7", "29760-6"),
    "Albumin": ("1751-7", "61151-7"),
    "Lactate": ("2524-7", "2519-7", "32693-4"),
    "Prothrombin time": ("5902-2",),
    "PTT": ("14979-9", "3173-2"),
    "INR": ("6301-6", "34714-6"),
    "Arterial O2 pressure": ("2703-7", "11556-8"),
    "Arterial CO2 Pressure": ("2019-8", "11557-6"),
    "PH (Arterial)": ("2744-1", "11558-4"),
    # Strictly venous codes only. 2745-8 / 2704-5 are capillary and SICdb
    # attaches them to its mixed-venous references, which would otherwise let
    # a capillary sample satisfy a venous variable.
    "PH (Venous)": ("2746-6",),
    "Venous O2 Pressure": ("2705-2",),
    "Venous CO2 Pressure": ("2021-4", "11557-6"),
    "Arterial Base Excess": ("11555-0", "1925-7"),
    "HCO3 (serum)": ("1960-4", "14627-4", "1963-8", "1959-6", "1961-2"),
    "Creatinine (whole blood)": ("38483-4", "2160-0"),
}


# ---------------------------------------------------------------------------
# Unit compatibility
# ---------------------------------------------------------------------------

#: Canonical unit -> spellings that mean the same thing. Used to reject
#: name matches whose units disagree, which is the cheapest available guard
#: against mapping e.g. a set respirator rate onto a measured one.
_UNIT_EQUIVALENTS: Dict[str, Tuple[str, ...]] = {
    "bpm": ("bpm", "/min", "1/min", "min-1", "beats/min", "bpm."),
    "breaths/min": ("breaths/min", "/min", "1/min", "min-1", "bpm"),
    "mmhg": ("mmhg", "mm hg", "torr"),
    "cmh2o": ("cmh2o", "cm h2o", "cmh20", "mbar", "hpa"),
    "degc": ("degc", "c", "°c", "celsius", "deg c"),
    "%": ("%", "percent", "vol%", "pct"),
    "g/dl": ("g/dl", "gm/dl", "gr/dl"),
    "g/l": ("g/l",),
    "mmol/l": ("mmol/l", "meq/l"),
    "mg/dl": ("mg/dl", "mgdl"),
    "k/ul": ("k/ul", "10^3/ul", "cells/ul", "g/l", "t/l.", "/3/cmm", "10*3/ul"),
    "ml": ("ml", "cc", "mls"),
    "l/min": ("l/min", "lpm", "l/m"),
    "s": ("s", "sec", "seconds", "second"),
    "ratio": ("ratio", "none", "", "inr"),
    "points": ("points", "none", "", "score"),
    "years": ("years", "y", "yr", "a"),
    "kg": ("kg", "kgs", "kilogram"),
    "cm": ("cm", "centimeter"),
    "ml/cmh2o": ("ml/cmh2o", "ml/mbar"),
    "mmhg-derived": ("mmhg", "none", ""),
}


def _normalize_unit(unit: Optional[str]) -> str:
    """Fold a source unit string onto a comparable key.

    SICdb's ``d_references`` is mixed-encoding: most fields are UTF-8 but a few
    carry raw cp1252 bytes, so ``°C`` arrives either as ``°C`` or as ``\ufffdC``
    depending on which decoding the caller chose. Both must fold onto ``c``,
    otherwise the temperature unit check silently fails and the variable is
    dropped from the state space.
    """
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return ""
    text = str(unit).strip().lower()
    if text in ("nan", "none", "null", "-"):
        return ""
    text = text.replace("\ufffd", "").replace("°", "").replace("\u00b0", "")
    text = text.replace("\u00b5", "u").replace("\u03bc", "u")  # micro sign
    return re.sub(r"\s+", "", text)


def _units_compatible(canonical_unit: Optional[str], source_unit: Optional[str]) -> Optional[bool]:
    """``True`` compatible, ``False`` contradictory, ``None`` unknown.

    An unknown answer (either side missing a unit) must not veto a match --
    most SICdb signal references carry no unit at all.
    """
    want = _normalize_unit(canonical_unit)
    got = _normalize_unit(source_unit)
    if not want or not got:
        return None

    equivalents = _UNIT_EQUIVALENTS.get(want)
    if equivalents is None:
        return want == got or None
    normalised = {re.sub(r"\s+", "", e) for e in equivalents}
    if got in normalised:
        return True
    # Symmetric check: the source unit may be the canonical key of its own set.
    reverse = _UNIT_EQUIVALENTS.get(got)
    if reverse and want in {re.sub(r"\s+", "", e) for e in reverse}:
        return True
    return False


# ---------------------------------------------------------------------------
# Mapping results
# ---------------------------------------------------------------------------


@dataclass
class MappedVariable:
    """One canonical variable and the source ids that carry it."""

    canonical: str
    source_ids: List[Any] = field(default_factory=list)
    source_names: List[str] = field(default_factory=list)
    source_units: List[str] = field(default_factory=list)
    #: ``seed`` / ``loinc`` / ``alias`` / ``lexical`` / ``llm`` / ``manual``.
    method: str = "unmapped"
    confidence: float = 0.0
    #: Which dictionary group the ids came from (``chartevents``,
    #: ``SignalFloat``, ``Laboratory``, ...).
    groups: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.source_ids)


@dataclass
class VariableMapping:
    """Canonical name -> source ids for one database, with provenance."""

    database: str
    entries: Dict[str, MappedVariable] = field(default_factory=dict)

    def __contains__(self, name: object) -> bool:
        entry = self.entries.get(str(name))
        return entry is not None and entry.resolved

    def __getitem__(self, name: str) -> MappedVariable:
        return self.entries[name]

    def get(self, name: str) -> Optional[MappedVariable]:
        return self.entries.get(name)

    @property
    def resolved_names(self) -> List[str]:
        return [name for name, entry in self.entries.items() if entry.resolved]

    @property
    def unresolved_names(self) -> List[str]:
        return [name for name, entry in self.entries.items() if not entry.resolved]

    def ids_for(self, name: str) -> List[Any]:
        entry = self.entries.get(name)
        return list(entry.source_ids) if entry else []

    def all_ids(self) -> List[Any]:
        seen: List[Any] = []
        for entry in self.entries.values():
            for source_id in entry.source_ids:
                if source_id not in seen:
                    seen.append(source_id)
        return seen

    def id_to_canonical(self) -> Dict[Any, str]:
        """Reverse index for relabelling extracted event rows.

        When two canonical variables claim the same id, the first one wins and
        the collision is logged -- silently double-assigning would put the same
        signal in two state dimensions.
        """
        out: Dict[Any, str] = {}
        for name, entry in self.entries.items():
            for source_id in entry.source_ids:
                if source_id in out and out[source_id] != name:
                    logger.warning(
                        "Source id %s is claimed by both %r and %r; keeping %r.",
                        source_id, out[source_id], name, out[source_id],
                    )
                    continue
                out[source_id] = name
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "canonical": entry.canonical,
                    "resolved": entry.resolved,
                    "method": entry.method,
                    "confidence": entry.confidence,
                    "source_ids": ", ".join(map(str, entry.source_ids)),
                    "source_names": ", ".join(entry.source_names),
                    "source_units": ", ".join(u for u in entry.source_units if u),
                    "groups": ", ".join(sorted(set(entry.groups))),
                    "notes": entry.notes,
                }
                for entry in self.entries.values()
            ]
        )

    def summary(self) -> str:
        resolved = self.resolved_names
        unresolved = self.unresolved_names
        by_method: Dict[str, int] = {}
        for name in resolved:
            method = self.entries[name].method
            by_method[method] = by_method.get(method, 0) + 1

        lines = [
            "Variable mapping for {0}: {1} resolved, {2} unresolved".format(
                self.database, len(resolved), len(unresolved)
            )
        ]
        for method, count in sorted(by_method.items(), key=lambda kv: -kv[1]):
            lines.append("  via {0:<8}: {1}".format(method, count))
        if unresolved:
            lines.append("  unresolved: {0}".format(", ".join(sorted(unresolved))))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database": self.database,
            "entries": {name: asdict(entry) for name, entry in self.entries.items()},
        }

    def save_json(self, path: Any) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False, default=str)
        return path

    @classmethod
    def load_json(cls, path: Any) -> "VariableMapping":
        with open(Path(path), "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        entries = {
            name: MappedVariable(**data) for name, data in payload.get("entries", {}).items()
        }
        return cls(database=payload.get("database", ""), entries=entries)

    def apply_overrides(self, overrides: Mapping[str, Sequence[Any]]) -> "VariableMapping":
        """Force specific ids for specific canonical names.

        The escape hatch for when automatic resolution is wrong: a user who has
        inspected their own ``d_references`` can pin the mapping and the
        pipeline will not second-guess it.
        """
        for name, ids in overrides.items():
            if name not in CANONICAL_VARIABLES:
                raise KeyError(
                    "{0!r} is not a canonical variable; known names are listed "
                    "in ConMedRL.data.CANONICAL_VARIABLES.".format(name)
                )
            self.entries[name] = MappedVariable(
                canonical=name,
                source_ids=list(ids),
                source_names=[],
                source_units=[],
                method="manual",
                confidence=1.0,
                notes="pinned by variable_overrides",
            )
        return self


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class VariableSearch:
    """Keyword search over a database's variable dictionary.

    Replaces the four copy-pasted ``VariableSearch`` classes. The dictionary
    schema is a parameter rather than a hard-coded ``label``/``itemid`` pair,
    so the same object works on MIMIC-IV's ``d_items`` and SICdb's
    ``d_references``.

    >>> search = VariableSearch(d_references, SICDB_DICTIONARY, llm=backend)
    >>> search.search("heart rate", target_languages=["German"])
       source_id      name  unit   group  score
    0        707  HeartRateECG  /min  SignalFloat  0.95
    """

    def __init__(
        self,
        dictionary: pd.DataFrame,
        schema: Optional[DictionarySchema] = None,
        llm: Optional[LLMBackend] = None,
        database: Optional[str] = None,
    ) -> None:
        if schema is None:
            if database is None:
                schema = self._infer_schema(dictionary)
            else:
                schema = dictionary_schema_for(database)

        missing = [
            column
            for column in (schema.id_column, schema.name_column)
            if column not in dictionary.columns
        ]
        if missing:
            raise KeyError(
                "The dictionary is missing column(s) {0}. Present: {1}".format(
                    ", ".join(missing), ", ".join(map(str, dictionary.columns))
                )
            )

        self.dictionary = dictionary
        self.schema = schema
        self.llm = llm or NullLLMBackend()
        self.last_result: Optional[pd.DataFrame] = None

    @staticmethod
    def _infer_schema(dictionary: pd.DataFrame) -> DictionarySchema:
        columns = set(dictionary.columns)
        if {"ReferenceGlobalID", "ReferenceValue"} <= columns:
            return SICDB_DICTIONARY
        if {"itemid", "label"} <= columns:
            return MIMIC_DICTIONARY
        raise KeyError(
            "Could not infer the dictionary schema from columns {0}; pass "
            "schema=DictionarySchema(...) explicitly.".format(sorted(columns))
        )

    # -- helpers --------------------------------------------------------------

    def _names(self) -> List[str]:
        return [
            "" if pd.isna(v) else str(v)
            for v in self.dictionary[self.schema.name_column].tolist()
        ]

    def _row_frame(self, names: Sequence[str], scores: Mapping[str, float]) -> pd.DataFrame:
        name_column = self.schema.name_column
        wanted = set(names)
        subset = self.dictionary[
            self.dictionary[name_column].astype(str).isin(wanted)
        ].copy()

        out = pd.DataFrame(
            {
                "source_id": subset[self.schema.id_column].to_numpy(),
                "name": subset[name_column].astype(str).to_numpy(),
            }
        )
        if self.schema.unit_column and self.schema.unit_column in subset.columns:
            out["unit"] = subset[self.schema.unit_column].to_numpy()
        else:
            out["unit"] = ""
        if self.schema.group_column and self.schema.group_column in subset.columns:
            out["group"] = subset[self.schema.group_column].to_numpy()
        else:
            out["group"] = ""

        out["score"] = out["name"].map(lambda n: scores.get(str(n), 0.0))
        return out.sort_values("score", ascending=False).reset_index(drop=True)

    # -- public API -----------------------------------------------------------

    def search(
        self,
        keyword: str,
        target_languages: Sequence[str] = (),
        top_k: int = 25,
        use_llm_ranking: bool = False,
        min_score: float = 0.0,
    ) -> pd.DataFrame:
        """Rank dictionary entries by relevance to ``keyword``.

        ``target_languages`` translates the query first, which is what makes an
        English query find ``Herzchirurgie`` in SICdb. Translation goes through
        the configured backend, falling back to the built-in glossary.
        """
        names = self._names()

        queries = [str(keyword)]
        for language in target_languages:
            translated = self.llm.translate(keyword, language)
            if translated and _normalize(translated) != _normalize(keyword):
                queries.append(translated)
                logger.info("Translated %r to %s: %r", keyword, language, translated)

        scores: Dict[str, float] = {}
        for query in queries:
            ranked = (
                self.llm.rank_candidates(query, names, top_k=max(top_k, 50))
                if use_llm_ranking
                else _lexical_rank(query, names, max(top_k, 50))
            )
            for name, score in ranked:
                scores[name] = max(scores.get(name, 0.0), float(score))

        # Substring matches are always worth surfacing even if fuzzy scoring
        # ranked them low, e.g. a long compound German name.
        for query in queries:
            needle = _normalize(query)
            if not needle:
                continue
            for name in names:
                if needle in _normalize(_split_camel(name)):
                    scores[name] = max(scores.get(name, 0.0), 0.6)

        kept = {name: score for name, score in scores.items() if score >= min_score}
        result = self._row_frame(list(kept), kept).head(top_k)
        self.last_result = result
        return result

    def search_by_keyword(
        self,
        keyword: str,
        output_column: Optional[str] = None,
        use_regex: bool = False,
        enable_translation: bool = False,
        target_languages: Optional[Sequence[str]] = None,
        translation_service: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_translator: Optional[Any] = None,
        enable_interactive_mode: bool = False,
        top_k: int = 25,
    ) -> List[Any]:
        """Backwards-compatible wrapper returning a list of ids.

        Mirrors the signature of the original ``VariableSearch.search_by_keyword``
        so existing notebooks keep working. ``translation_service`` / ``api_key``
        are accepted but the configured backend is preferred; pass an
        :class:`~ConMedRL.data.llm.LLMConfig` when building the pipeline instead.
        """
        if use_regex:
            pattern = re.compile(str(keyword), re.IGNORECASE)
            mask = (
                self.dictionary[self.schema.name_column]
                .astype(str)
                .apply(lambda value: bool(pattern.search(value)))
            )
            subset = self.dictionary[mask]
            column = output_column or self.schema.id_column
            return subset[column].tolist()

        if api_key and translation_service and not getattr(self.llm, "is_llm", False):
            from .config import LLMConfig
            from .llm import get_llm_backend

            self.llm = get_llm_backend(
                LLMConfig(provider=translation_service, api_key=api_key)
            )

        if custom_translator is not None:
            logger.warning(
                "custom_translator is no longer used; supply an LLMConfig or a "
                "custom LLMBackend subclass instead."
            )
        if enable_interactive_mode:
            logger.info(
                "Interactive search has been replaced by search(); inspect the "
                "returned DataFrame and pin ids via variable_overrides."
            )

        languages = list(target_languages or ()) if enable_translation else []
        result = self.search(keyword, target_languages=languages, top_k=top_k)

        column = output_column or self.schema.id_column
        if column in ("source_id", self.schema.id_column):
            return result["source_id"].tolist()
        if column in ("name", self.schema.name_column):
            return result["name"].tolist()
        return result["source_id"].tolist()

    def by_unit(self, unit: str) -> pd.DataFrame:
        """All dictionary entries recorded in ``unit``.

        Units are a strong, language-independent filter: everything measured in
        ``/min`` in SICdb is a rate, whatever it is called.
        """
        if not self.schema.unit_column or self.schema.unit_column not in self.dictionary.columns:
            raise KeyError("This dictionary has no unit column.")
        wanted = _normalize_unit(unit)
        mask = self.dictionary[self.schema.unit_column].apply(
            lambda value: _normalize_unit(value) == wanted
        )
        return self.dictionary[mask].copy()

    def units(self) -> List[str]:
        if not self.schema.unit_column or self.schema.unit_column not in self.dictionary.columns:
            return []
        values = self.dictionary[self.schema.unit_column].dropna().astype(str).unique()
        return sorted(values)

    def groups(self) -> List[str]:
        if not self.schema.group_column or self.schema.group_column not in self.dictionary.columns:
            return []
        values = self.dictionary[self.schema.group_column].dropna().astype(str).unique()
        return sorted(values)


# ---------------------------------------------------------------------------
# Automatic resolution
# ---------------------------------------------------------------------------


def _seed_for(database: str) -> Mapping[str, Tuple[Any, ...]]:
    database = Database.normalize(database)
    if database == Database.SICDB:
        return SICDB_SEED_IDS
    if database == Database.MIMIC_IV:
        return MIMIC_SEED_ITEMIDS
    return {}


def _alias_candidates(spec: VariableSpec) -> List[str]:
    return [spec.name] + list(spec.aliases)


def resolve_variable_mapping(
    dictionary: pd.DataFrame,
    database: str,
    wanted: Sequence[VariableSpec],
    schema: Optional[DictionarySchema] = None,
    llm: Optional[LLMBackend] = None,
    overrides: Optional[Mapping[str, Sequence[Any]]] = None,
    target_languages: Sequence[str] = (),
    use_llm: bool = False,
    min_lexical_score: float = 0.62,
    allowed_groups: Optional[Sequence[str]] = None,
) -> VariableMapping:
    """Map canonical variables onto dictionary ids, strongest evidence first.

    Parameters
    ----------
    dictionary:
        ``d_items`` (MIMIC-IV) or ``d_references`` (SICdb).
    wanted:
        The canonical variables to look for, typically a task's state space.
    overrides:
        Canonical name -> ids, applied last and never overruled.
    target_languages:
        Languages to translate each canonical name into before lexical
        matching. ``["German"]`` for SICdb.
    use_llm:
        Ask the LLM backend to resolve whatever the deterministic layers could
        not. Off by default so a run is reproducible without network access.
    min_lexical_score:
        Fuzzy-match floor. Deliberately strict: an unresolved variable is
        dropped from the state space and reported, whereas a wrong match
        corrupts a state dimension invisibly.
    allowed_groups:
        Restrict matching to these dictionary groups, e.g. ``["chartevents"]``
        for MIMIC-IV or ``["SignalFloat", "Laboratory"]`` for SICdb.

    Returns
    -------
    VariableMapping
        Inspect ``.to_frame()`` before trusting a run on a new database.
    """
    database = Database.normalize(database)
    schema = schema or dictionary_schema_for(database)
    llm = llm or NullLLMBackend()

    table = dictionary.copy()
    if allowed_groups is None and database == Database.SICDB:
        allowed_groups = SICDB_MEASUREMENT_GROUPS
    if allowed_groups and schema.group_column and schema.group_column in table.columns:
        wanted_groups = {str(g).lower() for g in allowed_groups}
        table = table[table[schema.group_column].astype(str).str.lower().isin(wanted_groups)]
        if table.empty:
            logger.warning(
                "No dictionary rows in group(s) %s; falling back to the whole "
                "dictionary.", ", ".join(allowed_groups),
            )
            table = dictionary.copy()

    if schema.name_column in table.columns:
        lowered = table[schema.name_column].astype(str).str.strip().str.lower()
        retired = lowered.str.startswith(_DEPRECATED_NAME_PREFIXES)
        if retired.any():
            logger.debug("Ignoring %d retired dictionary entry/entries.", int(retired.sum()))
            table = table[~retired]

    id_column = schema.id_column
    name_column = schema.name_column
    unit_column = schema.unit_column if schema.unit_column in table.columns else None
    group_column = schema.group_column if schema.group_column in table.columns else None
    loinc_column = schema.loinc_column if schema.loinc_column and schema.loinc_column in table.columns else None

    # Lookup indices, built once.
    by_id: Dict[Any, Dict[str, Any]] = {}
    by_normalized_name: Dict[str, List[Any]] = {}
    by_loinc: Dict[str, List[Any]] = {}
    names: List[str] = []

    for row in table.itertuples(index=False):
        row_dict = row._asdict()
        source_id = row_dict[id_column]
        raw_name = row_dict[name_column]
        name = "" if pd.isna(raw_name) else str(raw_name)
        unit = str(row_dict.get(unit_column) or "") if unit_column else ""
        group = str(row_dict.get(group_column) or "") if group_column else ""

        by_id[source_id] = {"name": name, "unit": unit, "group": group}
        names.append(name)
        by_normalized_name.setdefault(_normalize(_split_camel(name)), []).append(source_id)

        if loinc_column:
            loinc = row_dict.get(loinc_column)
            if loinc is not None and not pd.isna(loinc):
                by_loinc.setdefault(str(loinc).strip(), []).append(source_id)

    seed = _seed_for(database)
    if database == Database.SICDB:
        known_absent = SICDB_KNOWN_ABSENT
    elif database == Database.MIMIC_IV:
        known_absent = MIMIC_KNOWN_ABSENT
    else:
        known_absent = {}
    mapping = VariableMapping(database=database)

    for spec in wanted:
        entry = MappedVariable(canonical=spec.name)

        # 1. Seed ids, kept only if the dictionary actually contains them.
        seeded = [sid for sid in seed.get(spec.name, ()) if sid in by_id]
        if seeded:
            entry.source_ids = seeded
            entry.method = "seed"
            entry.confidence = 1.0
            missing_seeds = [sid for sid in seed.get(spec.name, ()) if sid not in by_id]
            if missing_seeds:
                entry.notes = "seed id(s) absent from this dictionary version: {0}".format(
                    ", ".join(map(str, missing_seeds))
                )

        # Guessing is only allowed for variables that a measurement dictionary
        # can actually contain. Demographics and derived quantities are
        # computed from the cohort, and a variable documented as absent must
        # not be satisfied by a lookalike -- both would put plausible-looking
        # numbers in the wrong state dimension, which nothing downstream can
        # detect.
        if not entry.resolved:
            if spec.kind in NON_MEASUREMENT_KINDS:
                entry.notes = (
                    "{0} variable: computed from the cohort rather than read "
                    "from the dictionary".format(spec.kind)
                )
                mapping.entries[spec.name] = entry
                continue
            if spec.name in known_absent:
                entry.notes = known_absent[spec.name]
                mapping.entries[spec.name] = entry
                continue

        # 2. LOINC, for laboratory-style variables.
        if not entry.resolved and loinc_column:
            hits: List[Any] = []
            for code in LOINC_HINTS.get(spec.name, ()):
                hits.extend(by_loinc.get(code, ()))
            hits = [sid for sid in dict.fromkeys(hits)]
            if hits:
                compatible = [
                    sid
                    for sid in hits
                    if _units_compatible(spec.unit, by_id[sid]["unit"]) is not False
                ]
                if compatible:
                    entry.source_ids = compatible
                    entry.method = "loinc"
                    entry.confidence = 0.95
                    rejected = set(hits) - set(compatible)
                    if rejected:
                        entry.notes = (
                            "rejected {0} LOINC hit(s) on unit mismatch".format(len(rejected))
                        )

        # 3. Exact / alias name match, unit-checked.
        if not entry.resolved:
            hits = []
            for alias in _alias_candidates(spec):
                hits.extend(by_normalized_name.get(_normalize(_split_camel(alias)), ()))
            hits = [sid for sid in dict.fromkeys(hits)]
            if hits:
                compatible = [
                    sid
                    for sid in hits
                    if _units_compatible(spec.unit, by_id[sid]["unit"]) is not False
                ]
                if compatible:
                    entry.source_ids = compatible
                    entry.method = "alias"
                    entry.confidence = 0.9
                elif hits:
                    entry.notes = "name matched {0} entry/entries but every unit disagreed".format(
                        len(hits)
                    )

        # 4. Fuzzy / translated lexical match.
        if not entry.resolved:
            queries = _alias_candidates(spec)
            for language in target_languages:
                translated = llm.translate(spec.name, language)
                if translated and _normalize(translated) != _normalize(spec.name):
                    queries.append(translated)

            best: Dict[Any, float] = {}
            for query in queries:
                for name, score in _lexical_rank(query, names, top_k=10):
                    if score < min_lexical_score:
                        continue
                    for source_id in by_normalized_name.get(
                        _normalize(_split_camel(name)), ()
                    ):
                        if _units_compatible(spec.unit, by_id[source_id]["unit"]) is False:
                            continue
                        best[source_id] = max(best.get(source_id, 0.0), score)

            if best:
                top = max(best.values())
                # Keep everything near the best score: a canonical variable
                # legitimately maps to several ids (three arterial-line
                # pressures, ECG vs pulse-oximetry heart rate).
                entry.source_ids = [
                    sid for sid, score in best.items() if score >= top - 0.05
                ]
                entry.method = "lexical"
                entry.confidence = round(float(top), 3)

        # 5. LLM, last resort.
        if not entry.resolved and use_llm and getattr(llm, "is_llm", False):
            query = "{0}{1}".format(
                spec.name, " ({0})".format(spec.unit) if spec.unit else ""
            )
            ranked = llm.rank_candidates(query, names, top_k=5)
            hits = []
            for name, score in ranked:
                if score < 0.5:
                    continue
                hits.extend(by_normalized_name.get(_normalize(_split_camel(name)), ()))
            hits = [
                sid
                for sid in dict.fromkeys(hits)
                if _units_compatible(spec.unit, by_id[sid]["unit"]) is not False
            ]
            if hits:
                entry.source_ids = hits
                entry.method = "llm"
                entry.confidence = 0.5
                entry.notes = "resolved by LLM; verify before trusting"

        if entry.resolved:
            entry.source_names = [by_id[sid]["name"] for sid in entry.source_ids]
            entry.source_units = [by_id[sid]["unit"] for sid in entry.source_ids]
            entry.groups = [by_id[sid]["group"] for sid in entry.source_ids]

        mapping.entries[spec.name] = entry

    if overrides:
        mapping.apply_overrides(overrides)

    logger.info("%s", mapping.summary())
    low_confidence = [
        name
        for name, entry in mapping.entries.items()
        if entry.resolved and entry.confidence < 0.9
    ]
    if low_confidence:
        logger.warning(
            "%d variable(s) resolved by fuzzy or LLM matching and should be "
            "reviewed via mapping.to_frame(): %s",
            len(low_confidence), ", ".join(sorted(low_confidence)),
        )
    return mapping
