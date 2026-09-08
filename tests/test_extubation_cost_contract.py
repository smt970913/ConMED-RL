import unittest

import pandas as pd

from ConMedRL.data.config import CohortConfig
from ConMedRL.data.mdp import assign_costs, task_cost_spec


class ExtubationCostContractTest(unittest.TestCase):
    def test_extubation_has_one_remaining_icu_los_constraint(self):
        cohort = CohortConfig(decision_epoch_hours=12.0, los_threshold=15.0)
        frame = pd.DataFrame(
            {
                "stay_id": [1, 1],
                "time_offset_hours": [12.0, 24.0],
                "los": [2.0, 2.0],
                "extubation_action": [0.0, 1.0],
                "death_in_ICU": [0.0, 0.0],
                "reintubation": [0.0, 1.0],
            }
        )

        result, spec = assign_costs(frame, "extubation", cohort)

        self.assertEqual(spec.objective, "extubation_failure_costs")
        self.assertEqual(spec.num_constraints, 1)
        self.assertEqual(spec.constraints, ("icu_los_costs",))
        self.assertEqual(result["obj_cost"].tolist(), [0.0, 1.0])
        self.assertTrue(
            result["obj_cost"].equals(result["extubation_failure_costs"])
        )
        self.assertAlmostEqual(result.loc[0, "icu_los_costs"], 12.0)
        self.assertAlmostEqual(result.loc[1, "icu_los_costs"], 24.0)
        self.assertTrue(result["con_cost_0"].equals(result["icu_los_costs"]))
        self.assertNotIn("icu_los_costs_scaled", result)
        self.assertNotIn("reintubation_costs", result)
        self.assertNotIn("con_cost_1", result)


if __name__ == "__main__":
    unittest.main()
