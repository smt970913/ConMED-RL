"""Reviewed declarative profiles shipped with ConMedRL."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Tuple

__all__ = ["load_nwicu_profile", "profile_approval_hash"]


def profile_approval_hash(dataset_spec: Dict[str, Any], task_spec: Dict[str, Any]) -> str:
    payload = json.dumps(
        {"dataset": dataset_spec, "task": task_spec},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read(name: str) -> Dict[str, Any]:
    with resources.open_text(__name__, name, encoding="utf-8") as stream:
        return json.load(stream)


def load_nwicu_profile(
    task: str, data_dir: Any = None
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Return the reviewed NWICU specs and a content-addressed approval hash.

    Pass ``data_dir`` before execution so local source fingerprints participate
    in approval. Omitting it is useful for inspection only.
    """
    value = str(task).strip().lower().replace("_", "-").replace(" ", "-")
    if value not in ("discharge", "extubation"):
        raise ValueError("NWICU profile supports 'discharge' or 'extubation'.")
    dataset = _read("nwicu_dataset.json")
    task_spec = _read("nwicu_{0}.json".format(value))
    if data_dir is not None:
        from ..profiler import DatasetProfiler

        root = Path(data_dir)
        profiler = DatasetProfiler(sample_rows=1, dictionary_entry_limit=0)
        fingerprints: Dict[str, str] = {}
        for table in dataset["tables"].values():
            relative = str(table["path"])
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError("NWICU profile source is missing: {0}".format(path))
            fingerprints[relative] = profiler._fingerprint(path)
        dataset["source_fingerprints"] = fingerprints
    return dataset, task_spec, profile_approval_hash(dataset, task_spec)
