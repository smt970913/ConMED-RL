from types import SimpleNamespace

import numpy as np
import pandas as pd

from ConMedRL.data_loader import ValTestDataLoader


def test_select_table_uses_preserved_parent_row_index():
    full_state = pd.DataFrame({"x": [10.0, 11.0, 20.0, 21.0]})
    full_outcome = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2],
            "action": [0.0, 1.0, 0.0, 1.0],
            "done": [0.0, 1.0, 0.0, 1.0],
            "obj_cost": [0.0] * 4,
            "con_cost_0": [0.0] * 4,
            "row_index": [0, 1, 2, 3],
        }
    )
    # Simulates a *_select.csv round-trip: its DataFrame index is 0, while the
    # selected state is parent row 2 and its successor is parent row 3.
    selected_outcome = full_outcome.iloc[[2]].reset_index(drop=True)
    selected_state = full_state.iloc[[2]].reset_index(drop=True)

    loader = ValTestDataLoader(
        SimpleNamespace(memory_capacity=10, device="cpu"),
        selected_outcome,
        selected_state,
        full_outcome,
        full_state,
        np.array([-1.0]),
    )
    loader.data_buffer("action", num_constraint=1)
    _, _, _, _, next_states, _ = loader.buffer_memory.extract()
    assert np.asarray(next_states)[0, 0] == 21.0
