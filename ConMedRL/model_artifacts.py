"""Model lineage, compatibility checks, and exact-retraining orchestration.

Manifests are JSON metadata only.  This module deliberately does not load model
objects; callers should continue to construct a trusted model architecture and
load a state dict when consuming an artifact.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import numpy as np
import torch


MANIFEST_SCHEMA_VERSION = 1
EXACT_RETRAIN_METHOD = "exact_retrain"


class ModelCompatibilityError(RuntimeError):
    """Raised when a model must not be used with a requested dataset."""


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Return deterministic, JSON-safe metadata without serialising objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
        return "{0}.{1}".format(module, name).strip(".")
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return _jsonable(
            {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        )
    return repr(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def file_sha256(path: Union[str, os.PathLike]) -> str:
    """Compute a streaming SHA-256 digest for an artifact file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _update_frame_hash(digest: "hashlib._Hash", frame: Any) -> None:
    digest.update(_canonical_json(list(frame.columns)))
    digest.update(_canonical_json([str(dtype) for dtype in frame.dtypes]))
    try:
        import pandas as pd

        values = pd.util.hash_pandas_object(frame, index=True, categorize=True)
        digest.update(values.to_numpy(dtype=np.uint64, copy=False).tobytes())
    except (ImportError, TypeError, ValueError):
        digest.update(frame.to_csv(index=True).encode("utf-8"))


def dataset_content_hash(bundle_or_hash: Any) -> str:
    """Resolve or compute the SHA-256 content hash of an RL dataset bundle.

    A 64-character hash may be supplied directly.  Bundle hashes cover all
    split outcome/state tables, decision subsets, terminal state, schema, and
    action metadata.  They therefore change when retained training content
    changes after an unlearning request.
    """
    if isinstance(bundle_or_hash, str):
        value = bundle_or_hash.lower()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("dataset content hash must be a 64-character SHA-256 hex digest")
        return value
    if bundle_or_hash is None:
        raise ValueError("a dataset bundle or content hash is required")

    if isinstance(bundle_or_hash, Mapping):
        for key in ("dataset_content_hash", "content_hash", "dataset_hash"):
            if bundle_or_hash.get(key):
                return dataset_content_hash(str(bundle_or_hash[key]))

    # Finished ConMedRL bundles expose a cached ``content_hash``, but always
    # recompute it here so an in-memory table mutation cannot bypass the model
    # compatibility gate.  The data module's canonical hash is also stable
    # across its CSV round-trip.
    if getattr(bundle_or_hash, "splits", None) is not None:
        try:
            from .data.dataset import compute_dataset_content_hash

            return compute_dataset_content_hash(bundle_or_hash)
        except (AttributeError, TypeError, ValueError):
            # Lightweight bundle-like objects used by third parties can fall
            # back to the generic hasher below.
            pass

    for name in ("dataset_content_hash", "content_hash", "dataset_hash"):
        value = getattr(bundle_or_hash, name, None)
        if value:
            return dataset_content_hash(str(value))

    splits = getattr(bundle_or_hash, "splits", None)
    if splits is None:
        raise TypeError(
            "dataset must be a SHA-256 hash or a bundle exposing train/val/test splits"
        )

    digest = hashlib.sha256()
    digest.update(b"ConMedRL-dataset-content-v1\0")
    for split_name, split in sorted(splits.items()):
        digest.update(split_name.encode("utf-8"))
        for table_name in ("outcome", "state", "outcome_select", "state_select"):
            table = getattr(split, table_name, None)
            digest.update(table_name.encode("utf-8"))
            if table is None:
                digest.update(b"\0")
            else:
                _update_frame_hash(digest, table)
    digest.update(
        np.ascontiguousarray(getattr(bundle_or_hash, "terminal_state")).tobytes()
    )
    digest.update(
        _canonical_json(
            {
                "schema": getattr(bundle_or_hash, "schema", None),
                "action_name": getattr(bundle_or_hash, "action_name", None),
                "action_type": getattr(bundle_or_hash, "action_type", None),
                "action_columns": getattr(bundle_or_hash, "action_columns", None),
                "action_bounds": getattr(bundle_or_hash, "action_bounds", None),
                "action_categories": getattr(bundle_or_hash, "action_categories", None),
                "num_constraints": getattr(bundle_or_hash, "num_constraints", None),
            }
        )
    )
    return digest.hexdigest()


def set_deterministic_seed(seed: int, deterministic_algorithms: bool = True) -> Dict[str, Any]:
    """Seed Python, NumPy, CPU/CUDA torch, and request deterministic torch ops.

    CUDA kernels, drivers, and hardware can still prevent bitwise reproducibility
    across machines or runtime versions.  The returned metadata records that
    limitation for the model manifest.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(
            bool(deterministic_algorithms), warn_only=True
        )
    return {
        "python": seed,
        "numpy": seed,
        "torch": seed,
        "cuda": seed,
        "deterministic_algorithms": bool(deterministic_algorithms),
        "cuda_bitwise_determinism_guaranteed": False,
    }


def _package_version() -> str:
    package = sys.modules.get("ConMedRL")
    if package is not None and getattr(package, "__version__", None):
        return str(package.__version__)
    try:
        return importlib.metadata.version("ConMedRL")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def runtime_versions() -> Dict[str, str]:
    """Versions required to reproduce or audit a training run."""
    return {
        "ConMedRL": _package_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None),
    }


def _bundle_dimensions(bundle: Any) -> Dict[str, Any]:
    if isinstance(bundle, str) or bundle is None:
        return {}
    result = {}
    for name in ("state_dim", "action_dim"):
        value = getattr(bundle, name, None)
        if value is not None:
            result[name] = int(value)
    return result


def _bundle_constraints(bundle: Any) -> Any:
    if isinstance(bundle, str) or bundle is None:
        return None
    value = getattr(bundle, "num_constraints", None)
    return None if value is None else int(value)


@dataclass
class ModelArtifactManifest:
    """JSON-serialisable lineage record for one related model artifact set."""

    dataset_content_hash: str
    artifacts: Dict[str, Dict[str, Any]]
    dimensions: Dict[str, Any] = field(default_factory=dict)
    constraints: Any = None
    rl_config: Dict[str, Any] = field(default_factory=dict)
    seeds: Dict[str, Any] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=runtime_versions)
    status: str = "active"
    audit: Dict[str, Any] = field(
        default_factory=lambda: {"reason": None, "request_digest": None}
    )
    training_method: Optional[str] = None
    created_at: str = field(default_factory=_utc_now)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.dataset_content_hash = dataset_content_hash(self.dataset_content_hash)
        if self.status not in ("active", "invalidated"):
            raise ValueError("manifest status must be 'active' or 'invalidated'")
        self.artifacts = _jsonable(self.artifacts)
        self.dimensions = _jsonable(self.dimensions)
        self.constraints = _jsonable(self.constraints)
        self.rl_config = _jsonable(self.rl_config)
        self.seeds = _jsonable(self.seeds)
        self.versions = _jsonable(self.versions)
        self.audit = _jsonable(self.audit)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelArtifactManifest":
        return cls(**dict(value))

    def write(self, path: Union[str, os.PathLike]) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        return str(path)

    save = write

    @classmethod
    def read(cls, path: Union[str, os.PathLike]) -> "ModelArtifactManifest":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    load = read

    def invalidate(self, reason: str, request_digest: str) -> "ModelArtifactManifest":
        if not str(reason).strip() or not str(request_digest).strip():
            raise ValueError("invalidation reason and request_digest are required")
        self.status = "invalidated"
        self.audit = {
            "reason": str(reason),
            "request_digest": str(request_digest),
            "invalidated_at": _utc_now(),
        }
        return self


def create_model_artifact_manifest(
    artifact_files: Union[Mapping[str, Union[str, os.PathLike]], Iterable[Union[str, os.PathLike]]],
    dataset: Any,
    *,
    dimensions: Optional[Mapping[str, Any]] = None,
    constraints: Any = None,
    rl_config: Any = None,
    seeds: Optional[Mapping[str, Any]] = None,
    training_method: Optional[str] = None,
) -> ModelArtifactManifest:
    """Bind arbitrary FQI/FQE (or actor/critic) files to training lineage."""
    if isinstance(artifact_files, Mapping):
        items = artifact_files.items()
    else:
        items = ((Path(path).stem, path) for path in artifact_files)
    artifacts: Dict[str, Dict[str, Any]] = {}
    for name, raw_path in items:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError("model artifact does not exist: {0}".format(path))
        artifacts[str(name)] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    if not artifacts:
        raise ValueError("at least one model artifact is required")
    resolved_dimensions = dict(dimensions or _bundle_dimensions(dataset))
    resolved_dimensions = {
        name: value for name, value in resolved_dimensions.items() if value is not None
    }
    return ModelArtifactManifest(
        dataset_content_hash=dataset_content_hash(dataset),
        artifacts=artifacts,
        dimensions=resolved_dimensions,
        constraints=constraints if constraints is not None else _bundle_constraints(dataset),
        rl_config=_jsonable(rl_config or {}),
        seeds=dict(seeds or {}),
        training_method=training_method,
    )


ManifestLike = Union[ModelArtifactManifest, str, os.PathLike]


def _as_manifest(manifest: ManifestLike) -> ModelArtifactManifest:
    if isinstance(manifest, ModelArtifactManifest):
        return manifest
    return ModelArtifactManifest.read(manifest)


def invalidate_model_artifact(
    manifest: ManifestLike,
    *,
    reason: str,
    request_digest: str,
    output_path: Optional[Union[str, os.PathLike]] = None,
) -> ModelArtifactManifest:
    """Invalidate a model lineage record while preserving an audit reason."""
    value = _as_manifest(manifest).invalidate(reason, request_digest)
    destination = output_path
    if destination is None and not isinstance(manifest, ModelArtifactManifest):
        destination = manifest
    if destination is not None:
        value.write(destination)
    return value


def assert_model_compatible(
    manifest: ManifestLike,
    dataset: Any,
    *,
    dimensions: Optional[Mapping[str, Any]] = None,
    constraints: Any = None,
    verify_artifacts: bool = True,
) -> ModelArtifactManifest:
    """Refuse invalidated, stale, dimension-incompatible, or modified models."""
    value = _as_manifest(manifest)
    if value.status != "active":
        raise ModelCompatibilityError(
            "model artifact is invalidated: {0}".format(value.audit.get("reason", "no reason recorded"))
        )
    expected_hash = dataset_content_hash(dataset)
    if value.dataset_content_hash != expected_hash:
        raise ModelCompatibilityError(
            "stale model: dataset content hash does not match the retained dataset"
        )

    expected_dimensions = dict(dimensions or _bundle_dimensions(dataset))
    for key, expected in expected_dimensions.items():
        if value.dimensions.get(key) != expected:
            raise ModelCompatibilityError(
                "model dimension {0!r} is {1!r}, expected {2!r}".format(
                    key, value.dimensions.get(key), expected
                )
            )
    expected_constraints = constraints if constraints is not None else _bundle_constraints(dataset)
    if (
        expected_constraints is not None
        and value.constraints != _jsonable(expected_constraints)
    ):
        raise ModelCompatibilityError("model constraint metadata does not match the dataset")

    if verify_artifacts:
        for name, artifact in value.artifacts.items():
            path = Path(artifact["path"])
            if not path.is_file():
                raise ModelCompatibilityError("model artifact is missing: {0}".format(name))
            if file_sha256(path) != artifact["sha256"]:
                raise ModelCompatibilityError("model artifact digest mismatch: {0}".format(name))
    return value


def exact_retrain(
    retrain_callback: Callable[[Any], Any],
    bundle: Any,
    *,
    seed: Optional[int] = None,
    request_digest: Optional[str] = None,
    invalidated_manifest: Optional[ManifestLike] = None,
    invalidation_reason: str = "dataset changed by an unlearning request",
    callback_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrain from scratch on retained data; no approximate method is offered.

    The callback receives the retained bundle and must construct fresh model and
    optimizer state.  When an old manifest is supplied it is invalidated before
    training.  CUDA execution may not be bitwise deterministic across systems.
    """
    if not callable(retrain_callback):
        raise TypeError("retrain_callback must be callable")
    if invalidated_manifest is not None:
        if not request_digest:
            raise ValueError("request_digest is required when invalidating a prior model")
        invalidate_model_artifact(
            invalidated_manifest,
            reason=invalidation_reason,
            request_digest=request_digest,
        )
    seed_metadata = set_deterministic_seed(seed) if seed is not None else {}
    result = retrain_callback(bundle, **dict(callback_kwargs or {}))
    return {
        "method": EXACT_RETRAIN_METHOD,
        "dataset_content_hash": dataset_content_hash(bundle),
        "request_digest": request_digest,
        "seeds": seed_metadata,
        "result": result,
    }


# Short aliases make the intended gate/read/write operations discoverable.
write_model_artifact_manifest = ModelArtifactManifest.write
load_model_artifact_manifest = ModelArtifactManifest.read


__all__ = [
    "EXACT_RETRAIN_METHOD",
    "MANIFEST_SCHEMA_VERSION",
    "ModelArtifactManifest",
    "ModelCompatibilityError",
    "assert_model_compatible",
    "create_model_artifact_manifest",
    "dataset_content_hash",
    "exact_retrain",
    "file_sha256",
    "invalidate_model_artifact",
    "load_model_artifact_manifest",
    "runtime_versions",
    "set_deterministic_seed",
    "write_model_artifact_manifest",
]
