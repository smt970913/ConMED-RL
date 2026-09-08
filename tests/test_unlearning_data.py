import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ConMedRL import (
    ModelCompatibilityError,
    assert_model_compatible,
    create_model_artifact_manifest,
)
from ConMedRL.data import (
    PreprocessConfig,
    build_dataset,
    load_dataset,
    rebuild_dataset_after_withdrawal,
    withdrawal_request_digest,
)
from test_data_pipeline import _write_sicdb


class ExactDataWithdrawalTest(unittest.TestCase):
    def _config(self, source, output, **kwargs):
        return PreprocessConfig(
            database="sicdb",
            task="discharge",
            data_dir=source,
            output_dir=output,
            output_formats=("csv",),
            min_variable_coverage=0.01,
            **kwargs,
        )

    def test_withdrawal_precedes_extraction_cache_and_is_redacted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            _write_sicdb(source, n_patients=12)

            original = build_dataset(self._config(source, output))
            self.assertEqual(original.report["observation_cache"], "miss")
            withdrawn = (2001, "2002")
            rebuilt = build_dataset(
                self._config(
                    source,
                    output,
                    withdrawn_subject_ids=withdrawn,
                )
            )

            self.assertEqual(rebuilt.report["observation_cache"], "miss")
            self.assertEqual(
                rebuilt.report["withdrawal"]["affected_subject_count"], 2
            )
            for tables in rebuilt.splits.values():
                self.assertTrue(
                    set(tables.outcome["subject_id"]).isdisjoint({2001, 2002})
                )

            manifest_path = Path(rebuilt.written_files["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertNotIn("withdrawn_subject_ids", manifest["config"])
            self.assertEqual(manifest["config"]["withdrawal_count"], 2)
            self.assertEqual(
                manifest["config"]["withdrawal_digest"],
                withdrawal_request_digest(withdrawn),
            )

            redacted = PreprocessConfig.from_dict(manifest["config"])
            self.assertEqual(redacted.withdrawn_subject_ids, ())
            loaded_manifest_config = PreprocessConfig.load_json(manifest_path)
            self.assertEqual(loaded_manifest_config.withdrawal_count, 2)
            self.assertEqual(loaded_manifest_config.withdrawn_subject_ids, ())
            with self.assertRaisesRegex(ValueError, "supplied again"):
                build_dataset(redacted, write=False)

    def test_content_hash_detects_tampering_and_legacy_manifest_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            _write_sicdb(source, n_patients=12)
            bundle = build_dataset(self._config(source, output))
            manifest_path = Path(bundle.written_files["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            restored = load_dataset(manifest_path)
            self.assertEqual(restored.content_hash, manifest["content_hash"])
            self.assertTrue(restored.report["contract_validation"]["valid"])

            state_path = Path(manifest["written_files"]["state_var_table_train"])
            state = pd.read_csv(state_path)
            state.iloc[0, 0] = float(state.iloc[0, 0]) + 0.125
            state.to_csv(state_path, index=False)
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                load_dataset(manifest_path)

            manifest.pop("content_hash")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            legacy = load_dataset(manifest_path)
            self.assertTrue(legacy.report["contract_validation"]["valid"])
            self.assertFalse(
                legacy.report["content_hash_validation"]["present_in_manifest"]
            )

    def test_versioned_rebuild_records_lineage_and_safe_cache_purge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            base_output = root / "base"
            next_output = root / "next"
            source.mkdir()
            _write_sicdb(source, n_patients=12)
            parent = build_dataset(self._config(source, base_output))

            source_config = self._config(source, next_output)
            parent_cache = parent.config.cache_dir
            parent_cache.mkdir(parents=True, exist_ok=True)
            derived = (
                parent_cache
                / "sicdb_discharge_observations_prior.parquet"
            )
            unrelated = parent_cache / "keep.txt"
            derived.write_bytes(b"old derived cache")
            unrelated.write_text("source-independent", encoding="utf-8")

            rebuilt = rebuild_dataset_after_withdrawal(
                parent.written_files["manifest"],
                source_config,
                [2003],
                purge_prior_cache=True,
            )
            lineage = rebuilt.report["lineage"]
            self.assertEqual(lineage["generation"], 1)
            self.assertEqual(lineage["request_digest"], withdrawal_request_digest([2003]))
            self.assertEqual(lineage["affected_counts"]["subjects"], 1)
            self.assertEqual(lineage["affected_counts"]["episodes"], 1)
            self.assertGreater(lineage["affected_counts"]["transitions"], 0)
            self.assertGreater(lineage["affected_counts"]["remaining_transitions"], 0)
            self.assertGreaterEqual(lineage["purged_observation_cache_files"], 1)
            self.assertTrue(unrelated.exists())
            self.assertFalse(derived.exists())
            self.assertTrue(rebuilt.config.output_prefix.endswith("_v1"))
            with self.assertRaisesRegex(ValueError, "invalidated"):
                load_dataset(parent.written_files["manifest"])
            audited_parent = load_dataset(
                parent.written_files["manifest"], allow_invalidated=True
            )
            self.assertEqual(audited_parent.content_hash, parent.content_hash)

            model_file = root / "old_fqi.pt"
            model_file.write_bytes(b"old model state")
            model_manifest = root / "old_model_manifest.json"
            create_model_artifact_manifest(
                {"fqi": model_file}, parent
            ).write(model_manifest)
            assert_model_compatible(model_manifest, parent)
            with self.assertRaises(ModelCompatibilityError):
                assert_model_compatible(model_manifest, rebuilt)

    def test_cumulative_withdrawal_requires_and_retains_prior_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _write_sicdb(source, n_patients=12)
            baseline = build_dataset(self._config(source, root / "v0"))

            first = rebuild_dataset_after_withdrawal(
                baseline.written_files["manifest"],
                self._config(source, root / "v1"),
                [2001],
            )
            second = rebuild_dataset_after_withdrawal(
                first.written_files["manifest"],
                self._config(source, root / "v2"),
                [2001, 2002],
                prior_withdrawn_subject_ids=[2001],
            )

            subjects = set().union(
                *[
                    set(tables.outcome["subject_id"])
                    for tables in second.splits.values()
                ]
            )
            self.assertTrue(subjects.isdisjoint({2001, 2002}))
            self.assertEqual(second.report["lineage"]["generation"], 2)
            self.assertEqual(second.config.withdrawal_count, 2)


if __name__ == "__main__":
    unittest.main()
