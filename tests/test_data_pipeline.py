from pathlib import Path

import numpy as np
import pandas as pd

from ConMedRL.data import (
    PreprocessConfig,
    RLDatasetBundle,
    SplitConfig,
    SplitTables,
    StateSpaceSchema,
    VariableSpec,
    build_dataset,
    load_dataset,
    validate_rl_contract,
)
from ConMedRL.data.mdp import split_by_group
from ConMedRL.data.pipeline import _resolve_split_config


def test_default_split_is_patient_level_and_keeps_episodes_intact():
    rows = []
    for subject_id in range(12):
        for episode_number in range(2):
            stay_id = subject_id * 10 + episode_number
            for epoch in range(3):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "stay_id": stay_id,
                        "epoch": epoch,
                    }
                )
    frame = pd.DataFrame(rows)

    splits = split_by_group(
        frame,
        SplitConfig(test_prop=0.4, val_prop=0.5, random_seed=17),
    )

    subject_sets = {
        name: set(tables["subject_id"])
        for name, tables in splits.items()
    }
    assert subject_sets["train"].isdisjoint(subject_sets["val"])
    assert subject_sets["train"].isdisjoint(subject_sets["test"])
    assert subject_sets["val"].isdisjoint(subject_sets["test"])
    assert set.union(*subject_sets.values()) == set(range(12))

    for subject_id in range(12):
        containing_splits = [
            name
            for name, tables in splits.items()
            if subject_id in set(tables["subject_id"])
        ]
        assert len(containing_splits) == 1
        selected = splits[containing_splits[0]]
        subject_rows = selected[selected["subject_id"] == subject_id]
        assert len(subject_rows) == 6
        assert subject_rows["stay_id"].nunique() == 2


def test_extubation_split_keeps_treatments_intact_and_stratifies_failures():
    rows = []
    for treatment_id in range(40):
        failed = treatment_id % 2
        for epoch in range(4):
            rows.append(
                {
                    "subject_id": treatment_id // 2,
                    "stay_id": treatment_id,
                    "epoch": epoch,
                    "extubation_fail": failed,
                }
            )
    frame = pd.DataFrame(rows)

    splits = split_by_group(
        frame,
        SplitConfig(
            test_prop=0.2,
            val_prop=0.5,
            random_seed=19,
            group_key="stay_id",
            stratify_column="extubation_fail",
        ),
    )

    treatment_sets = {
        name: set(tables["stay_id"])
        for name, tables in splits.items()
    }
    assert treatment_sets["train"].isdisjoint(treatment_sets["val"])
    assert treatment_sets["train"].isdisjoint(treatment_sets["test"])
    assert treatment_sets["val"].isdisjoint(treatment_sets["test"])
    for tables in splits.values():
        assert set(tables.groupby("stay_id").size()) == {4}
        assert tables.drop_duplicates("stay_id")["extubation_fail"].mean() == 0.5


def test_pipeline_resolves_task_specific_split_defaults(tmp_path):
    discharge = _resolve_split_config(
        PreprocessConfig(
            database="mimic-iv",
            task="discharge",
            data_dir=tmp_path,
            output_dir=tmp_path / "discharge",
        )
    )
    extubation = _resolve_split_config(
        PreprocessConfig(
            database="mimic-iv",
            task="extubation",
            data_dir=tmp_path,
            output_dir=tmp_path / "extubation",
        )
    )

    assert discharge.group_key == "subject_id"
    assert discharge.stratify_column is None
    assert extubation.group_key == "stay_id"
    assert extubation.stratify_column == "extubation_fail"


def test_contract_validation_uses_configured_trajectory_group(tmp_path):
    config = PreprocessConfig(
        database="mimic-iv",
        task="extubation",
        data_dir=tmp_path,
        split=SplitConfig(group_key="stay_id"),
    )
    schema = StateSpaceSchema(
        task="extubation",
        database="mimic-iv",
        variables=[VariableSpec("age", "demographic")],
    )

    def split(name, stay_id, subject_id):
        outcome = pd.DataFrame(
            {
                "stay_id": [stay_id],
                "subject_id": [subject_id],
                "extubation_action": [1],
                "done": [1],
                "obj_cost": [0.0],
            }
        )
        state = pd.DataFrame({"age": [60.0]})
        return SplitTables(name=name, outcome=outcome, state=state)

    bundle = RLDatasetBundle(
        config=config,
        schema=schema,
        train=split("train", 101, 1),
        val=split("val", 102, 1),
        test=split("test", 103, 2),
        terminal_state=np.zeros(1, dtype=np.float32),
        action_name="extubation_action",
        action_columns=("extubation_action",),
        action_categories=(0, 1),
        report={"split": {"group_key": "stay_id"}},
    )

    assert validate_rl_contract(bundle)["valid"] is True


def _write_sicdb(root: Path, n_patients: int = 10) -> None:
    pd.DataFrame(
        [
            {
                "ReferenceGlobalID": 707,
                "ReferenceValue": "HeartRateECG",
                "ReferenceName": "SignalFloat",
                "ReferenceDescription": "",
                "ReferenceUnit": "/min",
                "LOINC_code": np.nan,
            },
            {
                "ReferenceGlobalID": 710,
                "ReferenceValue": "SPO2",
                "ReferenceName": "SignalFloat",
                "ReferenceDescription": "",
                "ReferenceUnit": "%",
                "LOINC_code": np.nan,
            },
            {
                "ReferenceGlobalID": 3123,
                "ReferenceValue": "RASS - Richmond Agitation-Sedation Scale",
                "ReferenceName": "Scores",
                "ReferenceDescription": "",
                "ReferenceUnit": np.nan,
                "LOINC_code": np.nan,
            },
        ]
    ).to_csv(root / "d_references.csv", index=False)

    cases = []
    signals = []
    for i in range(n_patients):
        case_id = 1000 + i
        cases.append(
            {
                "CaseID": case_id,
                "PatientID": 2000 + i,
                "ICUOffset": 3600,
                "TimeOfStay": 24 * 3600,
                "DischargeState": 2202,
                "OffsetOfDeath": np.nan,
                "Sex": 735 if i % 2 else 736,
                "WeightOnAdmission": 60 + i,
                "HeightOnAdmission": 170,
                "AgeOnAdmission": 40 + i,
                "HospitalUnit": 3,
                "OffsetAfterFirstAdmission": i * 30 * 86400,
            }
        )
        for offset, heart_rate in ((7200, 70 + i), (43200, 75 + i)):
            signals.append(
                {
                    "CaseID": case_id,
                    "DataID": 707,
                    "Offset": offset,
                    "Val": heart_rate,
                }
            )
            signals.append(
                {
                    "CaseID": case_id,
                    "DataID": 710,
                    "Offset": offset,
                    "Val": 96,
                }
            )
    pd.DataFrame(cases).to_csv(root / "cases.csv", index=False)
    pd.DataFrame(signals).to_csv(root / "data_float_h.csv", index=False)
    pd.DataFrame(
        columns=["CaseID", "LaboratoryID", "Offset", "LaboratoryValue"]
    ).to_csv(root / "laboratory.csv", index=False)


def _write_mimic(root: Path, n_patients: int = 10) -> None:
    pd.DataFrame(
        [
            {
                "itemid": 220045,
                "label": "Heart Rate",
                "abbreviation": "HR",
                "linksto": "chartevents",
                "category": "Routine Vital Signs",
                "unitname": "bpm",
            },
            {
                "itemid": 220277,
                "label": "O2 saturation pulseoxymetry",
                "abbreviation": "SpO2",
                "linksto": "chartevents",
                "category": "Respiratory",
                "unitname": "%",
            },
            {
                "itemid": 228096,
                "label": "Richmond-RAS Scale",
                "abbreviation": "RASS",
                "linksto": "chartevents",
                "category": "Neurological",
                "unitname": np.nan,
            },
        ]
    ).to_csv(root / "d_items.csv", index=False)

    stays = []
    patients = []
    admissions = []
    events = []
    for i in range(n_patients):
        subject = 100 + i
        hadm = 200 + i
        stay = 300 + i
        intime = pd.Timestamp("2150-01-01") + pd.Timedelta(days=i * 3)
        outtime = intime + pd.Timedelta(hours=24)
        stays.append(
            {
                "subject_id": subject,
                "hadm_id": hadm,
                "stay_id": stay,
                "first_careunit": "Medical Intensive Care Unit (MICU)",
                "last_careunit": "Medical Intensive Care Unit (MICU)",
                "intime": intime,
                "outtime": outtime,
                "los": 1.0,
            }
        )
        patients.append(
            {
                "subject_id": subject,
                "gender": "M" if i % 2 else "F",
                "anchor_age": 40 + i,
                "anchor_year": 2150,
                "dod": pd.NaT,
            }
        )
        admissions.append(
            {
                "subject_id": subject,
                "hadm_id": hadm,
                "admittime": intime - pd.Timedelta(hours=2),
                "dischtime": outtime + pd.Timedelta(hours=2),
                "deathtime": pd.NaT,
                "hospital_expire_flag": 0,
            }
        )
        for hours, heart_rate in ((3, 70 + i), (15, 75 + i)):
            events.extend(
                [
                    {
                        "stay_id": stay,
                        "itemid": 220045,
                        "charttime": intime + pd.Timedelta(hours=hours),
                        "valuenum": heart_rate,
                    },
                    {
                        "stay_id": stay,
                        "itemid": 220277,
                        "charttime": intime + pd.Timedelta(hours=hours),
                        "valuenum": 97,
                    },
                ]
            )
    pd.DataFrame(stays).to_csv(root / "icustays.csv", index=False)
    pd.DataFrame(patients).to_csv(root / "patients.csv", index=False)
    pd.DataFrame(admissions).to_csv(root / "admissions.csv", index=False)
    pd.DataFrame(events).to_csv(root / "chartevents.csv", index=False)


def test_sicdb_pipeline_exports_and_reloads(tmp_path):
    source = tmp_path / "sicdb"
    output = tmp_path / "processed"
    source.mkdir()
    _write_sicdb(source)

    bundle = build_dataset(
        database="sicdb",
        task="discharge",
        data_dir=source,
        output_dir=output,
        output_formats=("csv", "parquet", "d3rlpy"),
        min_variable_coverage=0.01,
    )

    assert bundle.state_dim >= 4  # age, M, Heart Rate, SaO2
    assert bundle.schema.names[:2] == ["age", "M"]
    assert bundle.action_name == "discharge_action"
    assert bundle.num_constraints == 2
    assert set(bundle.train.state.columns) == set(bundle.schema.names)
    assert bundle.train.state.to_numpy().min() >= 0
    assert bundle.train.state.to_numpy().max() <= 1
    assert bundle.train.outcome["done"].sum() == bundle.train.n_episodes
    assert (output / "sicdb_discharge_manifest.json").exists()
    assert (output / "sicdb_discharge_state_var_table_train.parquet").exists()
    assert (output / "d3rlpy" / "train_mdp_arrays.npz").exists()

    restored = load_dataset(output / "sicdb_discharge_manifest.json")
    assert restored.schema.names == bundle.schema.names
    assert restored.train.state.shape == bundle.train.state.shape
    assert restored.action_name == bundle.action_name

    cached = build_dataset(
        database="sicdb",
        task="discharge",
        data_dir=source,
        output_dir=output,
        output_formats=("csv",),
        min_variable_coverage=0.01,
        write=False,
    )
    assert cached.report["observation_cache"] == "hit"


def test_d3rlpy_marks_actions_terminal_not_timeout(tmp_path):
    source = tmp_path / "sicdb"
    source.mkdir()
    _write_sicdb(source)
    bundle = build_dataset(
        database="sicdb",
        task="discharge",
        data_dir=source,
        output_dir=tmp_path / "out",
        write=False,
    )
    mdp = bundle.to_mdp_dataset("train")
    assert np.array_equal(
        mdp.arrays["terminals"],
        bundle.train.outcome["is_terminal_action"].to_numpy(np.float32),
    )
    assert mdp.arrays["timeouts"].sum() == 0


def test_mimic_pipeline_uses_same_public_api(tmp_path):
    source = tmp_path / "mimic"
    source.mkdir()
    _write_mimic(source)
    bundle = build_dataset(
        database="mimic-iv",
        task="discharge",
        data_dir=source,
        output_dir=tmp_path / "out",
        write=False,
    )
    assert bundle.action_name == "discharge_action"
    assert "Heart Rate" in bundle.schema.names
    assert bundle.train.n_episodes > 0
