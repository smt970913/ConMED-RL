import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ConMedRL.data import (
    ActionSpec,
    CostRule,
    DatasetSpec,
    EventRule,
    ImputationConfig,
    PreprocessConfig,
    TableRoleSpec,
    TaskSpec,
    build_dataset,
    validate_rl_contract,
    write_bundle,
)
from ConMedRL.data.profiler import profile_dataset


def _column(name):
    return {"op": "column", "name": name}


def _literal(value):
    return {"op": "literal", "value": value}


class GenericIntegrationTest(unittest.TestCase):
    def test_approved_strict_continuous_task_reaches_rl_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stays = pd.DataFrame(
                {
                    "person_key": range(1, 9),
                    "encounter_key": range(101, 109),
                    "intime": pd.to_datetime(["2020-01-01"] * 8),
                    "outtime": pd.to_datetime(["2020-01-02"] * 8),
                    "los": [1.0] * 8,
                    "age": [50, 60, 70, 55, 65, 75, 45, 80],
                    "dose": np.linspace(0.1, 0.8, 8),
                    "adverse": [0, 1, 0, 1, 0, 1, 0, 1],
                }
            )
            events = pd.DataFrame(
                [
                    {
                        "encounter_key": stay_id,
                        "charttime": "2020-01-01 {0}:00".format(hour),
                        "value": value,
                    }
                    for stay_id in stays["encounter_key"]
                    for hour, value in ((1, 80.0), (13, 85.0))
                ]
            )
            stays.to_csv(root / "stays.csv", index=False)
            events.to_csv(root / "events.csv", index=False)

            profile = profile_dataset(root, sample_rows=100)
            tables = (
                TableRoleSpec(
                    name="stays",
                    file="stays.csv",
                    role="cohort",
                    columns=tuple(stays.columns),
                    episode_id_column="encounter_key",
                    subject_id_column="person_key",
                    start_time_column="intime",
                    end_time_column="outtime",
                ),
                TableRoleSpec(
                    name="events",
                    file="events.csv",
                    role="observations",
                    columns=("encounter_key", "charttime", "value"),
                    episode_id_column="encounter_key",
                    time_column="charttime",
                    value_column="value",
                    large=True,
                ),
            )
            task = TaskSpec(
                name="dose-control",
                episode_table="stays",
                episode_id_column="encounter_key",
                subject_id_column="person_key",
                timeline_anchor=_column("intime"),
                state_columns=("age", "Heart Rate"),
                action=ActionSpec(
                    name="dose",
                    kind="continuous",
                    columns=("dose",),
                    expression=_column("dose"),
                    bounds=((0.0, 1.0),),
                ),
                objective=CostRule(
                    name="adverse_cost",
                    kind="objective",
                    expression=_column("adverse"),
                ),
                constraints=(
                    CostRule(
                        name="safety_cost",
                        kind="constraint",
                        expression=_column("adverse"),
                        threshold=0.4,
                    ),
                ),
                decision_epoch_hours=12.0,
            )
            event = EventRule(
                name="Heart Rate",
                source_table="events",
                predicate=_literal(True),
                time_expression=_column("charttime"),
                value_expression=_column("value"),
                aggregation="mean",
            )
            fingerprints = {
                name: profile.source_fingerprints[name]
                for name in ("stays.csv", "events.csv")
            }
            spec = DatasetSpec(
                name="synthetic",
                tables=tables,
                tasks=(task,),
                event_rules=(event,),
                source_fingerprints=fingerprints,
                confidence=1.0,
            ).approve(fingerprints)
            config = PreprocessConfig(
                database="generic",
                task="dose-control",
                data_dir=root,
                output_dir=root / "output",
                output_formats=("csv", "fhir"),
                dataset_spec=spec,
                use_cache=False,
                imputation=ImputationConfig(knn_impute=False),
            )
            bundle = build_dataset(config=config, write=False)

            self.assertEqual(bundle.state_dim, 2)
            self.assertEqual(bundle.action_type, "continuous")
            self.assertEqual(bundle.action_columns, ("dose",))
            self.assertEqual(bundle.action_bounds, {"dose": (0.0, 1.0)})
            self.assertEqual(bundle.num_constraints, 1)
            self.assertTrue(validate_rl_contract(bundle)["valid"])
            mdp = bundle.to_mdp_dataset("train")
            self.assertEqual(mdp.arrays["actions"].shape[1], 1)
            self.assertEqual(mdp.arrays["actions"].dtype, np.float32)
            written = write_bundle(bundle)
            self.assertTrue(Path(written["fhir_conformance_report"]).is_file())


if __name__ == "__main__":
    unittest.main()
