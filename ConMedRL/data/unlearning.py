"""Exact patient-withdrawal helpers for reproducible dataset rebuilds.

Withdrawal is deliberately implemented as a source rebuild, never as row
editing or approximate model unlearning.  Raw subject identifiers are accepted
in memory only; persisted configuration and lineage contain a stable digest and
count.
"""

from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

UNLEARNING_API_VERSION = "1.0"

__all__ = [
    "UNLEARNING_API_VERSION",
    "canonical_subject_id",
    "canonical_withdrawal_ids",
    "withdrawal_request_digest",
    "invalidate_dataset_manifest",
    "purge_observation_cache",
    "rebuild_dataset_after_withdrawal",
]


def canonical_subject_id(value: Any) -> str:
    """Return a deterministic comparison token for one non-null subject ID."""
    if value is None:
        raise ValueError("Withdrawal subject IDs cannot contain null values.")
    try:
        is_null = bool(value != value)  # NaN/NaT, including NumPy variants
    except (TypeError, ValueError):
        is_null = False
    if is_null:
        raise ValueError("Withdrawal subject IDs cannot contain null values.")

    text = str(value).strip()
    if not text:
        raise ValueError("Withdrawal subject IDs cannot be empty.")
    try:
        number = Decimal(text)
    except InvalidOperation:
        return "s:" + text
    if not number.is_finite():
        raise ValueError("Withdrawal subject IDs must be finite.")
    if number == number.to_integral_value():
        return "n:" + str(number.quantize(Decimal(1)))
    return "n:" + format(number.normalize(), "f")


def canonical_withdrawal_ids(subject_ids: Iterable[Any]) -> Tuple[str, ...]:
    """Canonicalise, de-duplicate, and sort a withdrawal request."""
    return tuple(sorted({canonical_subject_id(value) for value in subject_ids}))


def withdrawal_request_digest(subject_ids: Iterable[Any]) -> str:
    """SHA-256 digest of the canonical complete withdrawal-ID set."""
    canonical = canonical_withdrawal_ids(subject_ids)
    payload = json.dumps(
        list(canonical), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def purge_observation_cache(config: Any) -> int:
    """Delete only derived observation caches for ``config``.

    The operation is intentionally narrow: it never recursively removes a
    directory and cannot touch source files or exported dataset artefacts.
    """
    cache_dir = Path(config.cache_dir)
    if not cache_dir.is_dir():
        return 0
    stem = "{0}_{1}_observations_".format(
        config.database.replace("-", "_"), config.task
    )
    removed = 0
    for path in cache_dir.glob(stem + "*.parquet"):
        if path.is_file() and path.parent.resolve() == cache_dir.resolve():
            path.unlink()
            removed += 1
    return removed


def invalidate_dataset_manifest(
    manifest_path: Union[str, Path],
    *,
    request_digest: str,
    successor_manifest: Union[str, Path],
) -> Path:
    """Atomically mark a superseded dataset/scaler manifest unusable."""
    path = Path(manifest_path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not str(request_digest).strip():
        raise ValueError("request_digest is required to invalidate a dataset.")
    payload["status"] = "invalidated"
    payload["invalidation"] = {
        "reason": "patient withdrawal requires exact dataset and model rebuild",
        "request_digest": str(request_digest),
        "successor_manifest": str(successor_manifest),
        "manifest_digest_before_invalidation": _manifest_digest(path),
    }
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)
    return path


def _load_source_config(source_config: Any) -> Any:
    from .config import PreprocessConfig

    if isinstance(source_config, PreprocessConfig):
        return copy.deepcopy(source_config)
    if isinstance(source_config, (str, Path)):
        return PreprocessConfig.load_json(source_config)
    if isinstance(source_config, Mapping):
        return PreprocessConfig.from_dict(source_config)
    raise TypeError(
        "source_config must be a PreprocessConfig, mapping, or JSON path."
    )


def _manifest_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebuild_dataset_after_withdrawal(
    prior_manifest_path: Union[str, Path],
    source_config: Any,
    withdrawn_subject_ids: Sequence[Any],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    output_prefix: Optional[str] = None,
    purge_prior_cache: bool = False,
    prior_withdrawn_subject_ids: Optional[Sequence[Any]] = None,
    variable_overrides: Optional[Mapping[str, Sequence[Any]]] = None,
    write: bool = True,
) -> Any:
    """Build the next exact dataset version after patient withdrawal.

    ``withdrawn_subject_ids`` is the complete cumulative set, including IDs
    from earlier withdrawal generations.  A redacted prior manifest cannot
    supply those IDs, so callers must provide them again.
    """
    from .pipeline import build_dataset

    manifest_path = Path(prior_manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        parent = json.load(fh)
    config = _load_source_config(source_config)

    parent_config = dict(parent.get("config") or {})
    for key in ("database", "task"):
        expected = parent_config.get(key)
        if expected is not None and getattr(config, key) != expected:
            raise ValueError(
                "source_config {0}={1!r} does not match parent manifest {2!r}."
                .format(key, getattr(config, key), expected)
            )

    canonical = canonical_withdrawal_ids(withdrawn_subject_ids)
    if not canonical:
        raise ValueError("An exact withdrawal rebuild requires at least one subject ID.")
    parent_count = int(parent_config.get("withdrawal_count", 0) or 0)
    parent_request_digest = parent_config.get("withdrawal_digest")
    if len(canonical) < parent_count:
        raise ValueError(
            "The complete cumulative withdrawal set must contain at least the "
            "{0} ID(s) recorded by the parent manifest.".format(parent_count)
        )
    if parent_count:
        current_digest = withdrawal_request_digest(withdrawn_subject_ids)
        if len(canonical) == parent_count and current_digest == parent_request_digest:
            prior_canonical = canonical
        else:
            if prior_withdrawn_subject_ids is None:
                raise ValueError(
                    "The parent manifest is redacted. Supply "
                    "prior_withdrawn_subject_ids again to verify a cumulative "
                    "withdrawal rebuild."
                )
            prior_canonical = canonical_withdrawal_ids(
                prior_withdrawn_subject_ids
            )
            if (
                len(prior_canonical) != parent_count
                or withdrawal_request_digest(prior_withdrawn_subject_ids)
                != parent_request_digest
            ):
                raise ValueError(
                    "prior_withdrawn_subject_ids do not match the parent "
                    "manifest's count and digest."
                )
            if not set(prior_canonical).issubset(canonical):
                raise ValueError(
                    "The cumulative withdrawal set omits an ID from the parent request."
                )

    config.withdrawn_subject_ids = tuple(withdrawn_subject_ids)
    config.withdrawal_count = len(canonical)
    config.withdrawal_digest = withdrawal_request_digest(withdrawn_subject_ids)
    if output_dir is not None:
        config.output_dir = Path(output_dir).expanduser()

    parent_lineage = dict(parent.get("report", {}).get("lineage") or {})
    generation = int(parent_lineage.get("generation", 0)) + 1
    if output_prefix is not None:
        config.output_prefix = output_prefix
    else:
        root_prefix = str(parent_lineage.get("root_output_prefix") or config.output_prefix)
        config.output_prefix = "{0}_v{1}".format(root_prefix, generation)

    purged = 0
    if purge_prior_cache:
        from .config import PreprocessConfig

        prior_config = PreprocessConfig.from_dict(parent_config)
        purged = purge_observation_cache(prior_config)
    lineage: Dict[str, Any] = {
        "api_version": UNLEARNING_API_VERSION,
        "generation": generation,
        "root_output_prefix": str(
            parent_lineage.get("root_output_prefix")
            or parent_config.get("output_prefix")
            or config.output_prefix
        ),
        "parent_manifest": str(manifest_path),
        "parent_manifest_digest": _manifest_digest(manifest_path),
        "parent_request_digest": parent_request_digest,
        "request_digest": config.withdrawal_digest,
        "withdrawal_request_digest": config.withdrawal_digest,
        "withdrawal_count": config.withdrawal_count,
        "purged_observation_cache_files": purged,
        "parent_transition_count": sum(
            int(split.get("rows", 0))
            for split in parent.get("splits", ())
            if isinstance(split, Mapping)
        ),
    }
    rebuilt = build_dataset(
        config=config,
        variable_overrides=variable_overrides,
        write=write,
        lineage=lineage,
    )
    if write:
        invalidate_dataset_manifest(
            manifest_path,
            request_digest=config.withdrawal_digest,
            successor_manifest=rebuilt.written_files["manifest"],
        )
    return rebuilt
