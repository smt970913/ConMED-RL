import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from ConMedRL import (
    ModelArtifactManifest,
    ModelCompatibilityError,
    assert_model_compatible,
    create_model_artifact_manifest,
    dataset_content_hash,
    exact_retrain,
    invalidate_model_artifact,
    set_deterministic_seed,
)
from ConMedRL.conmedrl import ReplayBuffer, save_ocrl_models_and_data
from ConMedRL.conmedrl_continuous import (
    save_ocrl_models_and_data as save_continuous_models_and_data,
)


def _bundle(value=1.0):
    outcome = pd.DataFrame(
        {"action": [0, 1], "obj_cost": [value, 0.0], "stay_id": [1, 1]}
    )
    state = pd.DataFrame({"heart_rate": [0.1, 0.2]})
    split = SimpleNamespace(
        outcome=outcome,
        state=state,
        outcome_select=None,
        state_select=None,
    )
    return SimpleNamespace(
        splits={"train": split, "val": split, "test": split},
        terminal_state=np.array([0.0]),
        schema=SimpleNamespace(
            names=["heart_rate"],
            to_dict=lambda: {"variables": [{"name": "heart_rate"}]},
        ),
        action_name="action",
        action_type="discrete",
        action_columns=("action",),
        action_bounds=None,
        action_categories=(0, 1),
        state_dim=1,
        action_dim=2,
        num_constraints=1,
    )


class ModelUnlearningTest(unittest.TestCase):
    def test_manifest_rejects_stale_or_tampered_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "fqi.pt"
            torch.save({"weight": torch.tensor([1.0])}, artifact)
            bundle = _bundle()
            path = root / "manifest.json"
            create_model_artifact_manifest(
                {"fqi": artifact},
                bundle,
                rl_config={"gamma": 0.9},
                seeds={"python": 7, "numpy": 7, "torch": 7},
            ).write(path)

            loaded = ModelArtifactManifest.read(path)
            self.assertEqual(loaded.status, "active")
            self.assertEqual(loaded.dataset_content_hash, dataset_content_hash(bundle))
            self.assertEqual(loaded.dimensions, {"action_dim": 2, "state_dim": 1})
            self.assertEqual(loaded.constraints, 1)
            self.assertEqual(loaded.rl_config["gamma"], 0.9)
            self.assertLessEqual(
                {"ConMedRL", "python", "numpy", "torch"}, set(loaded.versions)
            )
            assert_model_compatible(path, bundle)

            with self.assertRaisesRegex(ModelCompatibilityError, "stale model"):
                assert_model_compatible(path, _bundle(value=2.0))

            # A cached bundle hash must not hide an in-memory table mutation.
            bundle.content_hash = loaded.dataset_content_hash
            bundle.train = bundle.splits["train"]
            bundle.train.state.iloc[0, 0] = 0.9
            with self.assertRaisesRegex(ModelCompatibilityError, "stale model"):
                assert_model_compatible(path, bundle)

            bundle.train.state.iloc[0, 0] = 0.1
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ModelCompatibilityError, "digest mismatch"):
                assert_model_compatible(path, bundle)

    def test_invalidation_is_audited_and_blocks_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "fqe.pt"
            artifact.write_bytes(b"state-dict")
            bundle = _bundle()
            path = root / "manifest.json"
            create_model_artifact_manifest({"fqe": artifact}, bundle).write(path)

            invalidate_model_artifact(
                path,
                reason="patient deletion request",
                request_digest="request-sha256",
            )
            invalidated = ModelArtifactManifest.read(path)
            self.assertEqual(invalidated.status, "invalidated")
            self.assertEqual(invalidated.audit["reason"], "patient deletion request")
            self.assertEqual(invalidated.audit["request_digest"], "request-sha256")
            with self.assertRaisesRegex(ModelCompatibilityError, "invalidated"):
                assert_model_compatible(path, bundle)

    def test_exact_retrain_has_exact_label_and_invalidates_old_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "old.pt"
            artifact.write_bytes(b"old")
            bundle = _bundle()
            path = root / "manifest.json"
            create_model_artifact_manifest({"fqi": artifact}, bundle).write(path)
            seen = []

            result = exact_retrain(
                lambda retained: seen.append(retained) or "fresh-models",
                bundle,
                seed=13,
                request_digest="delete-request",
                invalidated_manifest=path,
            )
            self.assertEqual(result["method"], "exact_retrain")
            self.assertEqual(result["result"], "fresh-models")
            self.assertEqual(seen, [bundle])
            self.assertEqual(ModelArtifactManifest.read(path).status, "invalidated")

    def test_global_and_replay_seeds_are_deterministic(self):
        set_deterministic_seed(23)
        first = (np.random.rand(), torch.rand(1).item())
        set_deterministic_seed(23)
        second = (np.random.rand(), torch.rand(1).item())
        self.assertEqual(first, second)

        left = ReplayBuffer(10, seed=5)
        right = ReplayBuffer(10, seed=5)
        for index in range(6):
            transition = (index, index, float(index), [0.0], index + 1, False)
            left.push(*transition)
            right.push(*transition)
        self.assertEqual(left.sample(4), right.sample(4))

    def test_existing_save_helper_emits_optional_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = SimpleNamespace(
                state_dim=1,
                action_dim=2,
                cfg=SimpleNamespace(gamma=0.95, batch_size=4),
                FQI_loss=[1.0],
                FQI_est_values=[2.0],
                FQE_loss_obj=[3.0],
                FQE_est_obj_costs=[4.0],
                FQE_loss_con={},
                FQE_est_con_costs={},
                lambda_dict={},
            )
            result = save_ocrl_models_and_data(
                {"weights": torch.tensor([1.0])},
                {"weights": torch.tensor([2.0])},
                [],
                training,
                model_save_path=str(root / "models"),
                data_save_path=str(root / "metrics"),
                save_date=False,
                dataset_content_hash="a" * 64,
                seeds={"python": 11, "numpy": 11, "torch": 11},
            )
            manifest = ModelArtifactManifest.read(result["manifest"])
            self.assertEqual(manifest.dataset_content_hash, "a" * 64)
            self.assertEqual(manifest.dimensions, {"action_dim": 2, "state_dim": 1})
            self.assertEqual(manifest.constraints, 0)
            self.assertEqual(manifest.seeds["torch"], 11)

    def test_continuous_save_helper_emits_optional_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = SimpleNamespace(
                input_dim=3,
                output_dim=2,
                cfg=SimpleNamespace(gamma=0.9),
                Actor_loss=[1.0],
                Critic_loss=[2.0],
                Critic_est_values=[3.0],
                FQE_loss_obj=[4.0],
                FQE_est_obj_costs=[5.0],
                FQE_loss_con={},
                FQE_est_con_costs={},
                lambda_dict={},
            )
            result = save_continuous_models_and_data(
                {"actor": torch.tensor([1.0])},
                {"critic": torch.tensor([2.0])},
                {"fqe": torch.tensor([3.0])},
                [],
                training,
                model_save_path=str(root / "models"),
                data_save_path=str(root / "metrics"),
                save_date=False,
                dataset_content_hash="b" * 64,
                seeds={"python": 17, "numpy": 17, "torch": 17},
            )
            manifest = ModelArtifactManifest.read(result["manifest"])
            self.assertEqual(manifest.dataset_content_hash, "b" * 64)
            self.assertEqual(manifest.dimensions, {"action_dim": 2, "state_dim": 3})
            self.assertEqual(manifest.constraints, 0)


if __name__ == "__main__":
    unittest.main()
