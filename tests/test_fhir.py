import json
from pathlib import Path
from unittest.mock import patch

from ConMedRL.data.fhir import (
    LOINC_SYSTEM,
    RELATIVE_TIME_EXTENSION_URL,
    UCUM_SYSTEM,
    export_fhir_r4,
    run_hl7_validator,
    validate_resources,
)


def _synthetic_rows():
    cohort = [
        {
            "subject_id": "raw-patient-123",
            "stay_id": "raw-stay-456",
            "M": 1,
            "los": 2.0,
        }
    ]
    observations = [
        {
            "stay_id": "raw-stay-456",
            "time_offset_hours": 1.5,
            "variable": "Heart Rate",
            "value": 82,
            "unit": "bpm",
            "source_id": "local-hr",
            "source_system": "https://hospital.example/codes/items",
            "source_name": "Monitor HR",
        },
        {
            "stay_id": "raw-stay-456",
            "time_offset_hours": 2,
            "variable": "Blood Pressure Systolic",
            "value": 118,
            "unit": "mmHg",
            "source_id": "local-sbp",
        },
        {
            "stay_id": "raw-stay-456",
            "time_offset_hours": 2,
            "variable": "Blood Pressure Diastolic",
            "value": 67,
            "unit": "mmHg",
            "source_id": "local-dbp",
        },
        {
            "stay_id": "raw-stay-456",
            "time_offset_hours": 3,
            "variable": "Hospital-specific score",
            "value": 4,
            "unit": "local points",
            "source_id": "score-7",
        },
    ]
    procedures = [
        {
            "stay_id": "raw-stay-456",
            "time_offset_hours": 30,
            "procedure": "extubation",
            "source_id": "extubate-local",
        }
    ]
    return cohort, observations, procedures


def test_export_writes_separate_deidentified_r4_ndjson(tmp_path):
    cohort, observations, procedures = _synthetic_rows()
    result = export_fhir_r4(
        cohort,
        observations,
        tmp_path,
        procedures,
        id_salt="unit-test-secret",
        generated_at="2026-01-01T00:00:00+00:00",
    )

    assert result.valid
    assert set(result.paths) == {
        "Patient",
        "Encounter",
        "Observation",
        "Procedure",
        "ConceptMap",
        "Provenance",
        "conformance_report",
    }
    assert result.report["fhir_release"] == "4.0.1"
    assert "RL CSV" in result.report["claim_scope"]
    assert result.report["relative_time"]["fabricated_patient_dates"] is False
    persisted_report = json.loads(
        Path(result.paths["conformance_report"]).read_text(encoding="utf-8")
    )
    assert persisted_report["files"]["conformance_report"].endswith(
        "fhir_conformance_report.json"
    )

    patient = result.resources["Patient"][0]
    encounter = result.resources["Encounter"][0]
    serialized = json.dumps(result.resources)
    assert patient["id"] != "raw-patient-123"
    assert encounter["id"] != "raw-stay-456"
    assert "raw-patient-123" not in serialized
    assert "raw-stay-456" not in serialized
    assert not {"identifier", "name", "telecom", "address"} & set(patient)

    for resource_type in (
        "Patient", "Encounter", "Observation", "Procedure", "ConceptMap", "Provenance"
    ):
        path = Path(result.paths[resource_type])
        assert path.exists()
        for line in path.read_text(encoding="utf-8").splitlines():
            assert json.loads(line)["resourceType"] == resource_type


def test_verified_loinc_local_coding_bp_components_and_relative_time(tmp_path):
    cohort, observations, procedures = _synthetic_rows()
    result = export_fhir_r4(
        cohort,
        observations,
        tmp_path,
        procedures,
        id_salt="unit-test-secret",
    )
    resources = result.resources["Observation"]
    heart_rate = next(item for item in resources if item["code"]["text"] == "Heart Rate")
    heart_codings = heart_rate["code"]["coding"]
    assert any(
        coding.get("system") == LOINC_SYSTEM and coding.get("code") == "8867-4"
        for coding in heart_codings
    )
    assert any(coding.get("code") == "local-hr" for coding in heart_codings)
    assert heart_rate["valueQuantity"]["system"] == UCUM_SYSTEM
    assert heart_rate["valueQuantity"]["code"] == "/min"
    assert "effectiveDateTime" not in heart_rate
    assert heart_rate["extension"][0]["url"] == RELATIVE_TIME_EXTENSION_URL
    assert heart_rate["extension"][0]["valueDuration"]["value"] == 1.5

    blood_pressure = next(item for item in resources if item["code"]["text"] == "Blood pressure")
    assert blood_pressure["meta"]["profile"] == [
        "http://hl7.org/fhir/StructureDefinition/bp"
    ]
    component_codes = {
        coding["code"]
        for component in blood_pressure["component"]
        for coding in component["code"]["coding"]
        if coding["system"] == LOINC_SYSTEM
    }
    assert component_codes == {"8480-6", "8462-4"}

    local_score = next(
        item for item in resources if item["code"]["text"] == "Hospital-specific score"
    )
    assert all(
        coding.get("system") != LOINC_SYSTEM
        for coding in local_score["code"]["coding"]
    )
    assert local_score["valueQuantity"].get("system") != UCUM_SYSTEM

    concept_map = result.resources["ConceptMap"][0]
    mapped_targets = {
        target["code"]
        for group in concept_map["group"]
        for element in group["element"]
        for target in element["target"]
    }
    assert {"8867-4", "8480-6", "8462-4"} <= mapped_targets


def test_real_datetimes_are_used_without_fabricating_offset_dates(tmp_path):
    cohort = [{
        "subject_id": "p1",
        "stay_id": "s1",
        "intime": "2150-01-01T00:00:00",
        "outtime": "2150-01-02T00:00:00",
    }]
    observations = [{
        "stay_id": "s1",
        "time": "2150-01-01T01:30:00",
        "time_offset_hours": 1.5,
        "variable": "Temperature C",
        "value": 37,
        "unit": "degC",
    }]
    result = export_fhir_r4(
        cohort, observations, tmp_path, id_salt="secret"
    )
    encounter = result.resources["Encounter"][0]
    observation = result.resources["Observation"][0]
    assert encounter["period"]["start"] == "2150-01-01T00:00:00"
    assert "extension" not in encounter
    assert observation["effectiveDateTime"] == "2150-01-01T01:30:00"
    assert "extension" not in observation


def test_partial_blood_pressure_does_not_claim_profile_conformance(tmp_path):
    result = export_fhir_r4(
        [{"subject_id": "p1", "stay_id": "s1"}],
        [{
            "stay_id": "s1",
            "time_offset_hours": 1,
            "variable": "Blood Pressure Mean",
            "value": 75,
            "unit": "mmHg",
        }],
        tmp_path,
        id_salt="secret",
    )
    observation = result.resources["Observation"][0]
    assert "meta" not in observation
    assert observation["component"][0]["code"]["coding"][1]["code"] == "8478-0"
    assert result.valid


def test_structural_validator_rejects_identifiers_and_broken_references():
    report = validate_resources([
        {
            "resourceType": "Patient",
            "id": "patient-1",
            "identifier": [{"value": "direct-medical-record-number"}],
        },
        {
            "resourceType": "Encounter",
            "id": "encounter-1",
            "status": "finished",
            "class": {"code": "IMP"},
            "subject": {"reference": "Patient/missing"},
        },
    ])
    assert not report["valid"]
    codes = {issue["code"] for issue in report["issues"]}
    assert "security" in codes
    assert "not-found" in codes


def test_official_validator_invocation_pins_r4_and_parses_outcome(tmp_path):
    validator = tmp_path / "validator_cli.jar"
    validator.write_bytes(b"placeholder")
    resource_file = tmp_path / "Patient.ndjson"
    resource_file.write_text('{"resourceType":"Patient","id":"p1"}\n', encoding="utf-8")

    def fake_run(command, **kwargs):
        output = Path(command[command.index("-output") + 1])
        output.write_text(
            json.dumps({"resourceType": "OperationOutcome", "issue": []}),
            encoding="utf-8",
        )

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return Completed()

    with patch("ConMedRL.data.fhir.subprocess.run", side_effect=fake_run) as called:
        report = run_hl7_validator([resource_file], validator)

    assert report["invoked"]
    assert report["valid"]
    command = called.call_args.args[0]
    assert command[command.index("-version") + 1] == "4.0"
