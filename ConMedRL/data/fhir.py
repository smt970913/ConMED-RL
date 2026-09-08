"""Dependency-light FHIR R4 export for de-identified canonical ICU events.

This module intentionally has no dependency on a FHIR model package.  It emits
plain dictionaries using the FHIR R4 JSON representation, performs fast local
checks, and can optionally invoke the official HL7 validator CLI.

Only the resources emitted by this module are candidates for FHIR conformance.
ConMedRL CSV files, scaled state matrices, tensors, and learned policies are
not FHIR resources and are never described as FHIR compliant here.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

FHIR_VERSION = "4.0.1"
FHIR_FORMAT = "application/fhir+ndjson"
LOINC_SYSTEM = "http://loinc.org"
UCUM_SYSTEM = "http://unitsofmeasure.org"
OBSERVATION_CATEGORY_SYSTEM = (
    "http://terminology.hl7.org/CodeSystem/observation-category"
)
ENCOUNTER_CLASS_SYSTEM = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
PROVENANCE_PARTICIPANT_SYSTEM = (
    "http://terminology.hl7.org/CodeSystem/provenance-participant-type"
)
CANONICAL_CODE_SYSTEM = "https://conmedrl.org/fhir/CodeSystem/canonical-icu-variable"
CANONICAL_PROCEDURE_SYSTEM = (
    "https://conmedrl.org/fhir/CodeSystem/canonical-icu-procedure"
)
RELATIVE_TIME_EXTENSION_URL = (
    "https://conmedrl.org/fhir/StructureDefinition/relative-time-from-icu-admission"
)

_FHIR_ID = re.compile(r"^[A-Za-z0-9\-.]{1,64}$")
_FHIR_REFERENCE = re.compile(
    r"^(Patient|Encounter|Observation|Procedure|ConceptMap|Provenance)/"
    r"([A-Za-z0-9\-.]{1,64})$"
)


@dataclass(frozen=True)
class Terminology:
    """A reviewed canonical-variable mapping to LOINC and UCUM."""

    loinc: str
    display: str
    ucum: Optional[str]
    unit_display: Optional[str] = None
    category: str = "laboratory"
    profile: Optional[str] = None


# Each entry is deliberately a single reviewed code, not a list of guesses.
# Concepts whose specimen/method is ambiguous are left local-only.
VERIFIED_TERMINOLOGY: Dict[str, Terminology] = {
    "Heart Rate": Terminology(
        "8867-4", "Heart rate", "/min", "beats/minute", "vital-signs",
        "http://hl7.org/fhir/StructureDefinition/heartrate",
    ),
    "Respiratory Rate": Terminology(
        "9279-1", "Respiratory rate", "/min", "breaths/minute", "vital-signs",
        "http://hl7.org/fhir/StructureDefinition/resprate",
    ),
    "Temperature C": Terminology(
        "8310-5", "Body temperature", "Cel", "degree Celsius", "vital-signs",
        "http://hl7.org/fhir/StructureDefinition/bodytemp",
    ),
    "SaO2": Terminology(
        "2708-6", "Oxygen saturation in arterial blood", "%", "percent",
        "vital-signs", "http://hl7.org/fhir/StructureDefinition/oxygensat",
    ),
    "weight": Terminology(
        "29463-7", "Body weight", "kg", "kilogram", "vital-signs",
        "http://hl7.org/fhir/StructureDefinition/bodyweight",
    ),
    "height": Terminology(
        "8302-2", "Body height", "cm", "centimeter", "vital-signs",
        "http://hl7.org/fhir/StructureDefinition/bodyheight",
    ),
    "Arterial O2 pressure": Terminology(
        "2703-7", "Oxygen partial pressure in arterial blood", "mm[Hg]", "mmHg"
    ),
    "Arterial CO2 Pressure": Terminology(
        "2019-8", "Carbon dioxide partial pressure in arterial blood",
        "mm[Hg]", "mmHg",
    ),
    "PH (Arterial)": Terminology(
        "2744-1", "pH of arterial blood", "[pH]", "pH"
    ),
    "PH (Venous)": Terminology(
        "2746-6", "pH of venous blood", "[pH]", "pH"
    ),
    "Venous O2 Pressure": Terminology(
        "2705-2", "Oxygen partial pressure in venous blood", "mm[Hg]", "mmHg"
    ),
    "Arterial Base Excess": Terminology(
        "11555-0", "Base excess in arterial blood", "mmol/L", "mmol/L"
    ),
    "HCO3 (serum)": Terminology(
        "1963-8", "Bicarbonate in serum or plasma", "mmol/L", "mmol/L"
    ),
    "Hemoglobin": Terminology(
        "718-7", "Hemoglobin in blood", "g/dL", "g/dL"
    ),
    "Hematocrit (serum)": Terminology(
        "4544-3", "Hematocrit of blood by automated count", "%", "percent"
    ),
    "WBC": Terminology(
        "6690-2", "Leukocytes in blood by automated count", "10*3/uL", "K/uL"
    ),
    "Platelet Count": Terminology(
        "777-3", "Platelets in blood by automated count", "10*3/uL", "K/uL"
    ),
    "Sodium (serum)": Terminology(
        "2951-2", "Sodium in serum or plasma", "mmol/L", "mmol/L"
    ),
    "Potassium (serum)": Terminology(
        "2823-3", "Potassium in serum or plasma", "mmol/L", "mmol/L"
    ),
    "Chloride (serum)": Terminology(
        "2075-0", "Chloride in serum or plasma", "mmol/L", "mmol/L"
    ),
    "Magnesium": Terminology(
        "2601-3", "Magnesium in serum or plasma", "mg/dL", "mg/dL"
    ),
    "Ionized Calcium": Terminology(
        "1994-3", "Calcium.ionized in serum or plasma", "mmol/L", "mmol/L"
    ),
    "Creatinine (serum)": Terminology(
        "2160-0", "Creatinine in serum or plasma", "mg/dL", "mg/dL"
    ),
    "BUN": Terminology(
        "3094-0", "Urea nitrogen in serum or plasma", "mg/dL", "mg/dL"
    ),
    "Glucose (serum)": Terminology(
        "2345-7", "Glucose in serum or plasma", "mg/dL", "mg/dL"
    ),
    "Total Bilirubin": Terminology(
        "1975-2", "Bilirubin.total in serum or plasma", "mg/dL", "mg/dL"
    ),
    "Direct Bilirubin": Terminology(
        "1968-7", "Bilirubin.direct in serum or plasma", "mg/dL", "mg/dL"
    ),
    "Albumin": Terminology(
        "1751-7", "Albumin in serum or plasma", "g/dL", "g/dL"
    ),
    "Lactate": Terminology(
        "2524-7", "Lactate in serum or plasma", "mmol/L", "mmol/L"
    ),
    "Prothrombin time": Terminology(
        "5902-2", "Prothrombin time", "s", "second"
    ),
    "PTT": Terminology(
        "14979-9", "Activated partial thromboplastin time", "s", "second"
    ),
    "INR": Terminology(
        "6301-6", "INR in platelet poor plasma", "1", "ratio"
    ),
}

_BP_TERMINOLOGY: Dict[str, Terminology] = {
    "Blood Pressure Systolic": Terminology(
        "8480-6", "Systolic blood pressure", "mm[Hg]", "mmHg", "vital-signs"
    ),
    "Blood Pressure Diastolic": Terminology(
        "8462-4", "Diastolic blood pressure", "mm[Hg]", "mmHg", "vital-signs"
    ),
    "Blood Pressure Mean": Terminology(
        "8478-0", "Mean blood pressure", "mm[Hg]", "mmHg", "vital-signs"
    ),
}

_UNIT_TO_UCUM: Dict[str, Tuple[str, str]] = {
    "bpm": ("/min", "beats/minute"),
    "breaths/min": ("/min", "breaths/minute"),
    "mmhg": ("mm[Hg]", "mmHg"),
    "degc": ("Cel", "degree Celsius"),
    "%": ("%", "percent"),
    "g/dl": ("g/dL", "g/dL"),
    "mmol/l": ("mmol/L", "mmol/L"),
    "mg/dl": ("mg/dL", "mg/dL"),
    "k/ul": ("10*3/uL", "K/uL"),
    "ml": ("mL", "mL"),
    "l/min": ("L/min", "L/min"),
    "s": ("s", "second"),
    "kg": ("kg", "kilogram"),
    "cm": ("cm", "centimeter"),
    "ratio": ("1", "ratio"),
    "hours": ("h", "hour"),
}

_REQUIRED: Dict[str, Tuple[str, ...]] = {
    "Patient": ("id",),
    "Encounter": ("id", "status", "class", "subject"),
    "Observation": ("id", "status", "code", "subject", "encounter"),
    "Procedure": ("id", "status", "code", "subject", "encounter"),
    "ConceptMap": ("id", "status"),
    "Provenance": ("id", "target", "recorded", "agent"),
}


@dataclass
class FHIRExportResult:
    """Paths, resources, and the machine-readable conformance report."""

    paths: Dict[str, str]
    resources: Dict[str, List[Dict[str, Any]]]
    report: Dict[str, Any]

    @property
    def valid(self) -> bool:
        return bool(self.report.get("exchange_resources_conformant", False))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": dict(self.paths),
            "report": dict(self.report),
        }


def _records(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict(orient="records")
            if isinstance(records, list):
                return [dict(item) for item in records]
        except TypeError:
            pass
    if isinstance(value, Mapping):
        return [dict(value)]
    return [dict(item) for item in value]


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if value.__class__.__name__ in ("NAType", "NaTType"):
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def _finite_number(value: Any) -> Optional[Union[int, float]]:
    if _missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9\-.]+", "-", str(value).strip()).strip("-.")
    return (text or "local")[:64]


def _instant(value: Any) -> Optional[str]:
    if _missing(value) or isinstance(value, (int, float)):
        return None
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text or re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return None
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return text


def _uri(value: Optional[str], fallback: str) -> str:
    if not value:
        return fallback
    parsed = urlparse(str(value))
    if parsed.scheme in ("http", "https", "urn"):
        return str(value)
    return fallback


def _relative_time(offset: Any) -> Dict[str, Any]:
    number = _finite_number(offset)
    if number is None:
        number = 0
    return {
        "url": RELATIVE_TIME_EXTENSION_URL,
        "valueDuration": {
            "value": number,
            "unit": "hour since ICU admission",
            "system": UCUM_SYSTEM,
            "code": "h",
        },
    }


def _category(code: str) -> List[Dict[str, Any]]:
    displays = {
        "vital-signs": "Vital Signs",
        "laboratory": "Laboratory",
        "exam": "Exam",
    }
    return [{
        "coding": [{
            "system": OBSERVATION_CATEGORY_SYSTEM,
            "code": code,
            "display": displays.get(code, code),
        }]
    }]


class FHIRR4Exporter:
    """Map canonical de-identified ICU frames to FHIR R4 NDJSON.

    Parameters
    ----------
    id_salt:
        Secret salt used to pseudonymize patient and encounter identifiers.
        If omitted, a random in-memory salt is generated.  The salt is never
        written to an export.
    local_code_system:
        URI used when source rows carry a ``source_id`` but no
        ``source_system``.  Local clinical codes are retained alongside a
        reviewed LOINC coding when one exists.
    generated_at:
        Optional reproducible Provenance instant.  It describes export time,
        never patient event time.
    """

    resource_types: Tuple[str, ...] = (
        "Patient",
        "Encounter",
        "Observation",
        "Procedure",
        "ConceptMap",
        "Provenance",
    )

    def __init__(
        self,
        id_salt: Optional[Union[str, bytes]] = None,
        local_code_system: str = CANONICAL_CODE_SYSTEM,
        generated_at: Optional[Any] = None,
    ) -> None:
        if id_salt is None:
            self._salt = secrets.token_bytes(32)
        elif isinstance(id_salt, bytes):
            self._salt = id_salt
        else:
            self._salt = str(id_salt).encode("utf-8")
        if not self._salt:
            raise ValueError("id_salt must not be empty")
        self.local_code_system = _uri(local_code_system, CANONICAL_CODE_SYSTEM)
        exported_at = _instant(generated_at)
        self.generated_at = exported_at or datetime.now(timezone.utc).isoformat()

    def _id(self, kind: str, *parts: Any) -> str:
        message = "\x1f".join([kind] + [str(part) for part in parts]).encode("utf-8")
        digest = hmac.new(self._salt, message, hashlib.sha256).hexdigest()[:32]
        prefixes = {
            "patient": "pat",
            "encounter": "enc",
            "observation": "obs",
            "procedure": "prc",
            "conceptmap": "map",
            "provenance": "prv",
        }
        return "{0}-{1}".format(prefixes[kind], digest)

    @staticmethod
    def _subject_key(row: Mapping[str, Any]) -> Any:
        for column in ("subject_id", "patient_id", "PatientID"):
            if column in row and not _missing(row[column]):
                return row[column]
        raise KeyError("Cohort row has no subject_id/patient_id pseudonym source")

    @staticmethod
    def _stay_key(row: Mapping[str, Any]) -> Any:
        for column in ("stay_id", "episode_id", "CaseID"):
            if column in row and not _missing(row[column]):
                return row[column]
        raise KeyError("Row has no stay_id/episode_id")

    def _local_coding(
        self,
        row: Mapping[str, Any],
        canonical: str,
        supplied: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        configured = supplied.get(canonical) if supplied else None
        if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
            configured = configured[0] if configured else None
        if isinstance(configured, Mapping):
            coding = {
                key: configured[key]
                for key in ("system", "code", "display")
                if key in configured and not _missing(configured[key])
            }
            coding["system"] = _uri(
                str(coding.get("system", "")), self.local_code_system
            )
            coding["code"] = str(coding.get("code", _slug(canonical)))
            return coding

        source_code = row.get("source_code", row.get("source_id"))
        if not _missing(source_code):
            coding = {
                "system": _uri(
                    None if _missing(row.get("source_system")) else str(row["source_system"]),
                    self.local_code_system,
                ),
                "code": str(source_code),
            }
            display = row.get("source_display", row.get("source_name"))
            if not _missing(display):
                coding["display"] = str(display)
            return coding
        return {
            "system": self.local_code_system,
            "code": _slug(canonical),
            "display": canonical,
        }

    @staticmethod
    def _standard_coding(term: Terminology) -> Dict[str, str]:
        return {"system": LOINC_SYSTEM, "code": term.loinc, "display": term.display}

    @staticmethod
    def _quantity(value: Any, term: Optional[Terminology], source_unit: Any) -> Dict[str, Any]:
        number = _finite_number(value)
        if number is None:
            raise ValueError("Observation value is not a finite number: {0!r}".format(value))
        unit_key = "" if _missing(source_unit) else str(source_unit).strip().lower()
        if term is not None and term.ucum:
            code = term.ucum
            display = term.unit_display or code
        elif unit_key in _UNIT_TO_UCUM:
            code, display = _UNIT_TO_UCUM[unit_key]
        else:
            # Unknown units remain human-readable text and are not falsely
            # labelled as UCUM.
            return {
                "value": number,
                "unit": "" if _missing(source_unit) else str(source_unit),
            }
        return {
            "value": number,
            "unit": display,
            "system": UCUM_SYSTEM,
            "code": code,
        }

    def _timing(
        self,
        resource: Dict[str, Any],
        row: Mapping[str, Any],
        datetime_field: str,
    ) -> None:
        actual = None
        for column in ("time", "charttime", "event_time", "effectiveDateTime"):
            if column in row:
                actual = _instant(row[column])
                if actual:
                    break
        if actual:
            resource[datetime_field] = actual
            return
        offset = row.get("time_offset_hours", row.get("offset_hours", 0))
        resource.setdefault("extension", []).append(_relative_time(offset))

    def build_resources(
        self,
        cohort: Any,
        observations: Any = None,
        procedures: Any = None,
        local_codings: Optional[Mapping[str, Any]] = None,
        source_name: str = "de-identified canonical ICU data",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build supported R4 resources without writing files.

        ``cohort`` is one row per ICU stay and requires ``stay_id`` and
        ``subject_id`` (accepted aliases are documented by the error messages).
        ``observations`` is canonical long form with ``stay_id``, ``variable``,
        ``value``, and either a real event time or ``time_offset_hours``.
        ``procedures`` uses ``procedure`` or ``action`` plus the same timing
        fields.  DataFrames and iterables of mappings are both accepted.
        """
        cohort_rows = _records(cohort)
        if not cohort_rows:
            raise ValueError("cohort must contain at least one ICU encounter")

        resources: Dict[str, List[Dict[str, Any]]] = {
            resource_type: [] for resource_type in self.resource_types
        }
        patient_ids: Dict[Any, str] = {}
        encounter_ids: Dict[Any, str] = {}
        encounter_subjects: Dict[Any, Any] = {}

        for row in cohort_rows:
            subject_key = self._subject_key(row)
            stay_key = self._stay_key(row)
            if stay_key in encounter_ids:
                raise ValueError("Duplicate ICU stay in cohort: {0!r}".format(stay_key))

            patient_id = patient_ids.get(subject_key)
            if patient_id is None:
                patient_id = self._id("patient", subject_key)
                patient_ids[subject_key] = patient_id
                patient: Dict[str, Any] = {
                    "resourceType": "Patient",
                    "id": patient_id,
                    "meta": {"tag": [{
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationValue",
                        "code": "DEID",
                        "display": "de-identified",
                    }]},
                }
                gender = row.get("gender", row.get("sex"))
                if _missing(gender) and not _missing(row.get("M")):
                    gender = "male" if _finite_number(row.get("M")) == 1 else "female"
                if not _missing(gender):
                    normalized = str(gender).strip().lower()
                    normalized = {"m": "male", "f": "female"}.get(normalized, normalized)
                    patient["gender"] = (
                        normalized
                        if normalized in ("male", "female", "other", "unknown")
                        else "unknown"
                    )
                resources["Patient"].append(patient)

            encounter_id = self._id("encounter", stay_key)
            encounter_ids[stay_key] = encounter_id
            encounter_subjects[stay_key] = subject_key
            encounter: Dict[str, Any] = {
                "resourceType": "Encounter",
                "id": encounter_id,
                "status": (
                    str(row["status"])
                    if row.get("status") in (
                        "planned", "arrived", "triaged", "in-progress", "onleave",
                        "finished", "cancelled", "entered-in-error", "unknown",
                    )
                    else "finished"
                ),
                "class": {
                    "system": ENCOUNTER_CLASS_SYSTEM,
                    "code": "IMP",
                    "display": "inpatient encounter",
                },
                "subject": {"reference": "Patient/{0}".format(patient_id)},
            }
            start = _instant(row.get("intime", row.get("start_time")))
            end = _instant(row.get("outtime", row.get("end_time")))
            if start or end:
                encounter["period"] = {}
                if start:
                    encounter["period"]["start"] = start
                if end:
                    encounter["period"]["end"] = end
            else:
                encounter["extension"] = [_relative_time(0)]
            resources["Encounter"].append(encounter)

        observations_rows = _records(observations)
        bp_groups: Dict[Tuple[Any, str], List[Tuple[int, Dict[str, Any]]]] = {}
        normal_rows: List[Tuple[int, Dict[str, Any]]] = []
        for index, row in enumerate(observations_rows):
            stay_key = self._stay_key(row)
            if stay_key not in encounter_ids:
                raise ValueError(
                    "Observation references stay absent from cohort: {0!r}".format(stay_key)
                )
            variable = str(row.get("variable", "")).strip()
            if not variable:
                raise ValueError("Observation row has no canonical `variable`")
            if variable in _BP_TERMINOLOGY:
                timing_key = (
                    _instant(row.get("time"))
                    or _instant(row.get("charttime"))
                    or str(row.get("time_offset_hours", row.get("offset_hours", 0)))
                )
                bp_groups.setdefault((stay_key, timing_key), []).append((index, row))
            else:
                normal_rows.append((index, row))

        used_mappings: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
        for index, row in normal_rows:
            stay_key = self._stay_key(row)
            variable = str(row["variable"]).strip()
            subject_key = encounter_subjects[stay_key]
            local = self._local_coding(row, variable, local_codings)
            term = VERIFIED_TERMINOLOGY.get(variable)
            codings = [local]
            if term is not None:
                codings.append(self._standard_coding(term))
                used_mappings[(local["system"], local["code"], str(local.get("display", "")))] = (
                    term.loinc,
                    term.display,
                )
            observation: Dict[str, Any] = {
                "resourceType": "Observation",
                "id": self._id(
                    "observation", stay_key, variable,
                    row.get("time_offset_hours", row.get("time", index)), index,
                ),
                "status": "final",
                "code": {"coding": codings, "text": variable},
                "subject": {
                    "reference": "Patient/{0}".format(patient_ids[subject_key])
                },
                "encounter": {
                    "reference": "Encounter/{0}".format(encounter_ids[stay_key])
                },
                "category": [_category(term.category)[0]] if term else _category("exam"),
                "valueQuantity": self._quantity(row.get("value"), term, row.get("unit")),
            }
            if term and term.profile:
                observation["meta"] = {"profile": [term.profile]}
            self._timing(observation, row, "effectiveDateTime")
            resources["Observation"].append(observation)

        for group_index, ((stay_key, timing_key), members) in enumerate(bp_groups.items()):
            subject_key = encounter_subjects[stay_key]
            representative = members[0][1]
            components: List[Dict[str, Any]] = []
            component_variables = {
                str(row["variable"]).strip() for _, row in members
            }
            for _, row in members:
                variable = str(row["variable"]).strip()
                term = _BP_TERMINOLOGY[variable]
                local = self._local_coding(row, variable, local_codings)
                used_mappings[(local["system"], local["code"], str(local.get("display", "")))] = (
                    term.loinc,
                    term.display,
                )
                components.append({
                    "code": {
                        "coding": [local, self._standard_coding(term)],
                        "text": variable,
                    },
                    "valueQuantity": self._quantity(
                        row.get("value"), term, row.get("unit", "mmHg")
                    ),
                })
            observation = {
                "resourceType": "Observation",
                "id": self._id("observation", stay_key, "blood-pressure", timing_key, group_index),
                "status": "final",
                "category": _category("vital-signs"),
                "code": {
                    "coding": [{
                        "system": LOINC_SYSTEM,
                        "code": "85354-9",
                        "display": "Blood pressure panel with all children optional",
                    }],
                    "text": "Blood pressure",
                },
                "subject": {
                    "reference": "Patient/{0}".format(patient_ids[subject_key])
                },
                "encounter": {
                    "reference": "Encounter/{0}".format(encounter_ids[stay_key])
                },
                "component": components,
            }
            # The R4 blood-pressure profile requires both core components.
            # Partial panels remain valid base Observations without claiming
            # profile conformance.
            if {
                "Blood Pressure Systolic",
                "Blood Pressure Diastolic",
            }.issubset(component_variables):
                observation["meta"] = {
                    "profile": ["http://hl7.org/fhir/StructureDefinition/bp"]
                }
            self._timing(observation, representative, "effectiveDateTime")
            resources["Observation"].append(observation)

        for index, row in enumerate(_records(procedures)):
            stay_key = self._stay_key(row)
            if stay_key not in encounter_ids:
                raise ValueError(
                    "Procedure references stay absent from cohort: {0!r}".format(stay_key)
                )
            subject_key = encounter_subjects[stay_key]
            name = str(row.get("procedure", row.get("action", ""))).strip()
            if not name:
                raise ValueError("Procedure row has no `procedure` or `action`")
            local = self._local_coding(row, name, None)
            if local["system"] == self.local_code_system and _missing(row.get("source_id")):
                local["system"] = CANONICAL_PROCEDURE_SYSTEM
            procedure: Dict[str, Any] = {
                "resourceType": "Procedure",
                "id": self._id(
                    "procedure", stay_key, name,
                    row.get("time_offset_hours", row.get("time", index)), index,
                ),
                "status": "completed",
                "code": {"coding": [local], "text": name},
                "subject": {
                    "reference": "Patient/{0}".format(patient_ids[subject_key])
                },
                "encounter": {
                    "reference": "Encounter/{0}".format(encounter_ids[stay_key])
                },
            }
            self._timing(procedure, row, "performedDateTime")
            resources["Procedure"].append(procedure)

        concept_map_id = self._id("conceptmap", source_name)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for (system, code, display), (loinc, loinc_display) in sorted(used_mappings.items()):
            element: Dict[str, Any] = {
                "code": code,
                "target": [{
                    "code": loinc,
                    "display": loinc_display,
                    "equivalence": "equivalent",
                }],
            }
            if display:
                element["display"] = display
            groups.setdefault(system, []).append(element)
        concept_map: Dict[str, Any] = {
            "resourceType": "ConceptMap",
            "id": concept_map_id,
            "status": "active",
            "experimental": False,
            "name": "ConMedRLVerifiedICUTerminologyMap",
            "title": "Verified local ICU terminology to LOINC",
            "description": (
                "Only reviewed mappings used in this export are included. "
                "Unmapped local concepts remain local and are not guessed."
            ),
            "sourceUri": self.local_code_system,
            "targetUri": LOINC_SYSTEM,
        }
        if groups:
            concept_map["group"] = [
                {"source": system, "target": LOINC_SYSTEM, "element": elements}
                for system, elements in sorted(groups.items())
            ]
        resources["ConceptMap"].append(concept_map)

        targets: List[Dict[str, str]] = []
        for resource_type in self.resource_types:
            if resource_type == "Provenance":
                continue
            for resource in resources[resource_type]:
                targets.append({
                    "reference": "{0}/{1}".format(resource_type, resource["id"])
                })
        provenance = {
            "resourceType": "Provenance",
            "id": self._id("provenance", source_name, self.generated_at),
            "target": targets,
            "recorded": self.generated_at,
            "activity": {
                "text": "FHIR R4 mapping of de-identified canonical ICU data"
            },
            "agent": [{
                "type": {
                    "coding": [{
                        "system": PROVENANCE_PARTICIPANT_SYSTEM,
                        "code": "assembler",
                        "display": "Assembler",
                    }]
                },
                "who": {"display": "ConMedRL FHIR R4 exporter"},
            }],
        }
        resources["Provenance"].append(provenance)
        return resources

    def export(
        self,
        cohort: Any,
        observations: Any,
        output_dir: Union[str, Path],
        procedures: Any = None,
        local_codings: Optional[Mapping[str, Any]] = None,
        source_name: str = "de-identified canonical ICU data",
        validator_path: Optional[Union[str, Path]] = None,
        java_command: str = "java",
        validator_timeout: int = 300,
    ) -> FHIRExportResult:
        """Build, validate, and write one NDJSON file per resource type."""
        resources = self.build_resources(
            cohort=cohort,
            observations=observations,
            procedures=procedures,
            local_codings=local_codings,
            source_name=source_name,
        )
        local_report = validate_resources(resources)
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}
        for resource_type in self.resource_types:
            path = directory / "{0}.ndjson".format(resource_type)
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                for resource in resources[resource_type]:
                    handle.write(
                        json.dumps(resource, ensure_ascii=False, separators=(",", ":"))
                    )
                    handle.write("\n")
            paths[resource_type] = str(path)

        official = None
        if validator_path is not None:
            official = run_hl7_validator(
                [paths[name] for name in self.resource_types],
                validator_path=validator_path,
                java_command=java_command,
                timeout=validator_timeout,
            )

        counts = {name: len(resources[name]) for name in self.resource_types}
        mapped = sorted({
            coding["code"]
            for observation in resources["Observation"]
            for coding in observation.get("code", {}).get("coding", [])
            if coding.get("system") == LOINC_SYSTEM
        } | {
            coding["code"]
            for observation in resources["Observation"]
            for component in observation.get("component", [])
            for coding in component.get("code", {}).get("coding", [])
            if coding.get("system") == LOINC_SYSTEM
        })
        has_relative = any(
            extension.get("url") == RELATIVE_TIME_EXTENSION_URL
            for resource_type in ("Encounter", "Observation", "Procedure")
            for resource in resources[resource_type]
            for extension in resource.get("extension", [])
        )
        official_valid = official is None or bool(official.get("valid", False))
        report: Dict[str, Any] = {
            "report_version": "1.0",
            "fhir_release": FHIR_VERSION,
            "format": FHIR_FORMAT,
            "claim_scope": (
                "Conformance applies only to the emitted FHIR exchange resources; "
                "RL CSV files, state matrices, tensors, and models are not FHIR resources."
            ),
            "exchange_resources_conformant": bool(local_report["valid"] and official_valid),
            "validation_level": (
                "official-hl7-and-structural" if official is not None else "structural-only"
            ),
            "counts": counts,
            "files": dict(paths),
            "privacy": {
                "patient_and_encounter_ids_pseudonymized": True,
                "pseudonymization_salt_exported": False,
                "patient_direct_identifier_fields_emitted": [],
            },
            "terminology": {
                "standard_mappings_are_reviewed_only": True,
                "loinc_codes_used": mapped,
                "local_codings_preserved": True,
            },
            "relative_time": {
                "extension_url": RELATIVE_TIME_EXTENSION_URL,
                "used": has_relative,
                "fabricated_patient_dates": False,
            },
            "structural_validation": local_report,
            "official_validator": official,
        }
        report_path = directory / "fhir_conformance_report.json"
        paths["conformance_report"] = str(report_path)
        report["files"] = dict(paths)
        with open(report_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return FHIRExportResult(paths=paths, resources=resources, report=report)


def validate_resources(
    resources: Union[
        Mapping[str, Sequence[Mapping[str, Any]]],
        Iterable[Mapping[str, Any]],
    ]
) -> Dict[str, Any]:
    """Perform fast structural, privacy, terminology, and reference checks."""
    if isinstance(resources, Mapping):
        flattened = [
            dict(resource)
            for values in resources.values()
            for resource in values
        ]
    else:
        flattened = [dict(resource) for resource in resources]

    issues: List[Dict[str, Any]] = []

    def issue(
        severity: str,
        code: str,
        diagnostics: str,
        expression: Optional[str] = None,
    ) -> None:
        item = {
            "severity": severity,
            "code": code,
            "diagnostics": diagnostics,
        }
        if expression:
            item["expression"] = [expression]
        issues.append(item)

    identities: Dict[str, Dict[str, Any]] = {}
    references: List[Tuple[str, str]] = []
    for index, resource in enumerate(flattened):
        resource_type = resource.get("resourceType")
        identity = "{0}[{1}]".format(resource_type or "Resource", index)
        if resource_type not in _REQUIRED:
            issue("error", "not-supported", "Unsupported or missing resourceType", identity)
            continue
        resource_id = resource.get("id")
        if not isinstance(resource_id, str) or not _FHIR_ID.fullmatch(resource_id):
            issue("error", "invalid", "Missing or invalid FHIR id", identity + ".id")
        else:
            full_id = "{0}/{1}".format(resource_type, resource_id)
            if full_id in identities:
                issue("error", "duplicate", "Duplicate resource identity " + full_id, identity)
            identities[full_id] = resource
            identity = full_id
        for field in _REQUIRED[resource_type]:
            if field not in resource or _missing(resource[field]):
                issue("error", "required", "Missing required field " + field, identity + "." + field)

        if resource_type == "Patient":
            for direct in ("identifier", "name", "telecom", "address", "photo", "contact"):
                if direct in resource:
                    issue(
                        "error", "security",
                        "Direct Patient identifier field is forbidden: " + direct,
                        identity + "." + direct,
                    )

        def walk(value: Any, path: str) -> None:
            if isinstance(value, Mapping):
                if isinstance(value.get("reference"), str):
                    references.append((value["reference"], path + ".reference"))
                if value.get("system") == LOINC_SYSTEM:
                    code = value.get("code")
                    verified_codes = {term.loinc for term in VERIFIED_TERMINOLOGY.values()}
                    verified_codes.update(term.loinc for term in _BP_TERMINOLOGY.values())
                    verified_codes.add("85354-9")
                    if code not in verified_codes:
                        issue(
                            "error", "code-invalid",
                            "LOINC code is not in the reviewed exporter mapping: {0}".format(code),
                            path,
                        )
                if value.get("system") == UCUM_SYSTEM and not value.get("code"):
                    issue("error", "required", "UCUM coding requires code", path + ".code")
                for key, child in value.items():
                    walk(child, path + "." + str(key))
            elif isinstance(value, list):
                for child_index, child in enumerate(value):
                    walk(child, "{0}[{1}]".format(path, child_index))

        walk(resource, identity)

        if resource_type == "Observation":
            if "valueQuantity" not in resource and not resource.get("component"):
                issue(
                    "error", "required",
                    "Observation needs valueQuantity or component",
                    identity,
                )
            if "effectiveDateTime" not in resource and not any(
                ext.get("url") == RELATIVE_TIME_EXTENSION_URL
                for ext in resource.get("extension", [])
                if isinstance(ext, Mapping)
            ):
                issue(
                    "warning", "incomplete",
                    "Observation has neither event time nor relative-time extension",
                    identity,
                )
            if resource.get("meta", {}).get("profile") == [
                "http://hl7.org/fhir/StructureDefinition/bp"
            ]:
                component_codes = {
                    coding.get("code")
                    for component in resource.get("component", [])
                    for coding in component.get("code", {}).get("coding", [])
                    if coding.get("system") == LOINC_SYSTEM
                }
                if not {"8480-6", "8462-4"}.issubset(component_codes):
                    issue(
                        "error", "required",
                        "Blood-pressure profile needs a systolic or diastolic component",
                        identity + ".component",
                    )

    for reference, path in references:
        match = _FHIR_REFERENCE.fullmatch(reference)
        if not match:
            issue("error", "invalid", "Unsupported relative reference " + reference, path)
        elif reference not in identities:
            issue("error", "not-found", "Unresolved reference " + reference, path)

    error_count = sum(item["severity"] in ("error", "fatal") for item in issues)
    warning_count = sum(item["severity"] == "warning" for item in issues)
    return {
        "validator": "ConMedRL dependency-light structural validator",
        "valid": error_count == 0,
        "resource_count": len(flattened),
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
    }


def _operation_outcome_issues(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    if payload.get("resourceType") == "OperationOutcome":
        return [
            dict(item) for item in payload.get("issue", []) if isinstance(item, Mapping)
        ]
    issues: List[Dict[str, Any]] = []
    for value in payload.values():
        if isinstance(value, Mapping):
            issues.extend(_operation_outcome_issues(value))
        elif isinstance(value, list):
            for item in value:
                issues.extend(_operation_outcome_issues(item))
    return issues


def run_hl7_validator(
    paths: Sequence[Union[str, Path]],
    validator_path: Union[str, Path],
    java_command: str = "java",
    timeout: int = 300,
) -> Dict[str, Any]:
    """Invoke the official HL7 validator CLI with ``-version 4.0``.

    ``validator_path`` may be the validator JAR or an executable wrapper.  No
    shell is used.  NDJSON is split into temporary single-resource JSON files
    because the core validator consumes one JSON resource per input.
    """
    validator = Path(validator_path)
    if not validator.exists():
        return {
            "invoked": False,
            "valid": False,
            "error": "HL7 validator not found: {0}".format(validator),
            "results": [],
        }
    results: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with tempfile.TemporaryDirectory(prefix="conmedrl-fhir-") as temp_dir:
            inputs: List[Tuple[Optional[int], Path]] = [(None, path)]
            if path.suffix.lower() == ".ndjson":
                inputs = []
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        for line_number, line in enumerate(handle, 1):
                            if not line.strip():
                                continue
                            try:
                                payload = json.loads(line)
                            except ValueError as exc:
                                results.append({
                                    "file": str(path),
                                    "line": line_number,
                                    "valid": False,
                                    "error": "Invalid NDJSON: {0}".format(exc),
                                    "issues": [],
                                })
                                continue
                            item_path = Path(temp_dir) / "resource-{0}.json".format(
                                line_number
                            )
                            with open(item_path, "w", encoding="utf-8") as item_handle:
                                json.dump(payload, item_handle, ensure_ascii=False)
                            inputs.append((line_number, item_path))
                except OSError as exc:
                    results.append({
                        "file": str(path),
                        "valid": False,
                        "error": str(exc),
                        "issues": [],
                    })
                    continue

            for line_number, input_path in inputs:
                suffix = "line-{0}".format(line_number) if line_number else "result"
                output = Path(temp_dir) / "operation-outcome-{0}.json".format(suffix)
                if validator.suffix.lower() == ".jar":
                    command = [
                        java_command, "-jar", str(validator), str(input_path),
                        "-version", "4.0", "-output", str(output),
                    ]
                else:
                    command = [
                        str(validator), str(input_path), "-version", "4.0",
                        "-output", str(output),
                    ]
                try:
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=False,
                    )
                    outcome = None
                    if output.exists():
                        try:
                            with open(output, "r", encoding="utf-8") as handle:
                                outcome = json.load(handle)
                        except (OSError, ValueError):
                            outcome = None
                    issues = _operation_outcome_issues(outcome)
                    severe = [
                        item for item in issues
                        if item.get("severity") in ("fatal", "error")
                    ]
                    item_result: Dict[str, Any] = {
                        "file": str(path),
                        "command": command,
                        "exit_code": completed.returncode,
                        "valid": completed.returncode == 0 and not severe,
                        "issues": issues,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }
                    if line_number is not None:
                        item_result["line"] = line_number
                    results.append(item_result)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    item_result = {
                        "file": str(path),
                        "command": command,
                        "valid": False,
                        "error": str(exc),
                        "issues": [],
                    }
                    if line_number is not None:
                        item_result["line"] = line_number
                    results.append(item_result)
    return {
        "invoked": True,
        "valid": bool(results) and all(item["valid"] for item in results),
        "fhir_version_argument": "4.0",
        "results": results,
    }


def export_fhir_r4(
    cohort: Any,
    observations: Any,
    output_dir: Union[str, Path],
    procedures: Any = None,
    *,
    id_salt: Optional[Union[str, bytes]] = None,
    local_code_system: str = CANONICAL_CODE_SYSTEM,
    local_codings: Optional[Mapping[str, Any]] = None,
    source_name: str = "de-identified canonical ICU data",
    generated_at: Optional[Any] = None,
    validator_path: Optional[Union[str, Path]] = None,
    java_command: str = "java",
    validator_timeout: int = 300,
) -> FHIRExportResult:
    """Convenience API suitable for a later call from :mod:`data.export`."""
    exporter = FHIRR4Exporter(
        id_salt=id_salt,
        local_code_system=local_code_system,
        generated_at=generated_at,
    )
    return exporter.export(
        cohort=cohort,
        observations=observations,
        procedures=procedures,
        output_dir=output_dir,
        local_codings=local_codings,
        source_name=source_name,
        validator_path=validator_path,
        java_command=java_command,
        validator_timeout=validator_timeout,
    )


__all__ = [
    "FHIR_VERSION",
    "FHIR_FORMAT",
    "LOINC_SYSTEM",
    "UCUM_SYSTEM",
    "RELATIVE_TIME_EXTENSION_URL",
    "VERIFIED_TERMINOLOGY",
    "Terminology",
    "FHIRExportResult",
    "FHIRR4Exporter",
    "validate_resources",
    "run_hl7_validator",
    "export_fhir_r4",
]
