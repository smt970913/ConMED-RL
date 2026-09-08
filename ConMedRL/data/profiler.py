"""Privacy-bounded local profiling for unknown ICU datasets.

Profiling reads a bounded local sample to infer data types and aggregates, but
the payload exposed to an LLM contains only file/schema metadata, aggregate
counts and rows from confidently identified *dictionary* files.  Clinical
patient rows are never retained in, or returned by, :meth:`to_llm_payload`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

__all__ = [
    "ColumnProfile",
    "FileProfile",
    "DatasetProfile",
    "DatasetProfiler",
    "profile_dataset",
]

PathLike = Union[str, "os.PathLike[str]"]
_SOURCE_SUFFIXES = (".csv", ".csv.gz", ".parquet")
_DICTIONARY_FILE_RE = re.compile(
    r"(^|[/\\])(?:d_(?:items|labitems|references?|codes?)|"
    r"(?:clinical_)?dictionary|lookup|codebook)(?:[._-]|$)",
    re.IGNORECASE,
)
_ID_NAMES = ("itemid", "item_id", "dataid", "fieldid", "code", "id")
_LABEL_NAMES = (
    "label",
    "name",
    "description",
    "referencevalue",
    "referencename",
    "title",
)
_UNIT_NAMES = ("unit", "unitname", "referenceunit", "valueuom")


def _json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(float(value)) else None
    if hasattr(value, "item"):
        try:
            return _json_scalar(value.item())
        except (TypeError, ValueError):
            pass
    text = str(value)
    return text[:500]


@dataclass(frozen=True)
class ColumnProfile:
    """Schema and bounded aggregate facts for one column."""

    name: str
    dtype: str
    non_null_fraction: Optional[float] = None
    distinct_count_sample: Optional[int] = None
    unique_fraction_sample: Optional[float] = None
    likely_identifier: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "non_null_fraction": self.non_null_fraction,
            "distinct_count_sample": self.distinct_count_sample,
            "unique_fraction_sample": self.unique_fraction_sample,
            "likely_identifier": self.likely_identifier,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ColumnProfile":
        known = {
            "name",
            "dtype",
            "non_null_fraction",
            "distinct_count_sample",
            "unique_fraction_sample",
            "likely_identifier",
        }
        unknown = set(payload) - known
        if unknown:
            raise ValueError("Unknown ColumnProfile fields: {0}".format(", ".join(sorted(unknown))))
        return cls(**dict(payload))


@dataclass(frozen=True)
class FileProfile:
    """Metadata-only description of one supported local source file."""

    file: str
    format: str
    size_bytes: int
    fingerprint: str
    columns: Tuple[ColumnProfile, ...] = ()
    sampled_rows: int = 0
    row_count: Optional[int] = None
    dictionary_entries: Tuple[Dict[str, Any], ...] = ()
    dictionary_columns: Dict[str, str] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()

    @property
    def column_names(self) -> Tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def is_dictionary(self) -> bool:
        return bool(self.dictionary_columns)

    def to_dict(self, include_dictionary_entries: bool = True) -> Dict[str, Any]:
        return {
            "file": self.file,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "fingerprint": self.fingerprint,
            "columns": [column.to_dict() for column in self.columns],
            "sampled_rows": self.sampled_rows,
            "row_count": self.row_count,
            "dictionary_entries": (
                [dict(entry) for entry in self.dictionary_entries]
                if include_dictionary_entries
                else []
            ),
            "dictionary_columns": dict(self.dictionary_columns),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FileProfile":
        known = {
            "file",
            "format",
            "size_bytes",
            "fingerprint",
            "columns",
            "sampled_rows",
            "row_count",
            "dictionary_entries",
            "dictionary_columns",
            "warnings",
        }
        unknown = set(payload) - known
        if unknown:
            raise ValueError("Unknown FileProfile fields: {0}".format(", ".join(sorted(unknown))))
        data = dict(payload)
        data["columns"] = tuple(ColumnProfile.from_dict(item) for item in data.get("columns", ()))
        data["dictionary_entries"] = tuple(dict(item) for item in data.get("dictionary_entries", ()))
        data["dictionary_columns"] = dict(data.get("dictionary_columns", {}))
        data["warnings"] = tuple(data.get("warnings", ()))
        return cls(**data)


@dataclass(frozen=True)
class DatasetProfile:
    """Serializable metadata snapshot used as the sole LLM planning input."""

    root_name: str
    files: Tuple[FileProfile, ...]
    profile_version: str = "1.0"
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.profile_version != "1.0":
            raise ValueError("Unsupported profile version {0!r}".format(self.profile_version))
        names = [item.file for item in self.files]
        if len(set(names)) != len(names):
            raise ValueError("DatasetProfile file paths must be unique")

    @property
    def source_fingerprints(self) -> Dict[str, str]:
        return {item.file: item.fingerprint for item in self.files}

    @property
    def offerings(self) -> Dict[str, Tuple[str, ...]]:
        return {item.file: item.column_names for item in self.files}

    def file(self, relative_path: str) -> FileProfile:
        for item in self.files:
            if item.file == relative_path:
                return item
        raise KeyError("Profile does not offer file {0!r}".format(relative_path))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "root_name": self.root_name,
            "files": [item.to_dict() for item in self.files],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetProfile":
        known = {"profile_version", "root_name", "files", "warnings"}
        unknown = set(payload) - known
        if unknown:
            raise ValueError("Unknown DatasetProfile fields: {0}".format(", ".join(sorted(unknown))))
        return cls(
            profile_version=str(payload.get("profile_version", "1.0")),
            root_name=str(payload["root_name"]),
            files=tuple(FileProfile.from_dict(item) for item in payload["files"]),
            warnings=tuple(str(item) for item in payload.get("warnings", ())),
        )

    def to_llm_payload(self) -> Dict[str, Any]:
        """Return metadata safe for a remote planning model.

        This is intentionally constructed field-by-field rather than by
        removing keys from a richer profile.  There is therefore no code path
        through which sampled patient records can accidentally be serialized.
        Dictionary entries are included only when both the filename and schema
        identify a lookup table.
        """
        files: List[Dict[str, Any]] = []
        for item in self.files:
            files.append(
                {
                    "file": item.file,
                    "format": item.format,
                    "size_bytes": item.size_bytes,
                    "columns": [column.to_dict() for column in item.columns],
                    "sampled_rows_for_aggregates": item.sampled_rows,
                    "row_count": item.row_count,
                    "dictionary_columns": dict(item.dictionary_columns),
                    "dictionary_entries": (
                        [dict(entry) for entry in item.dictionary_entries]
                        if item.is_dictionary
                        else []
                    ),
                    "warnings": list(item.warnings),
                }
            )
        return {
            "profile_version": self.profile_version,
            "dataset_name": self.root_name,
            "privacy": {
                "patient_rows_included": False,
                "content": "filenames, schemas, aggregate counts, and dictionary rows only",
            },
            "files": files,
            "warnings": list(self.warnings),
        }

    def llm_payload_json(self) -> str:
        return json.dumps(
            self.to_llm_payload(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )


class DatasetProfiler:
    """Scan CSV/CSV.GZ/Parquet sources without retaining patient records."""

    def __init__(
        self,
        sample_rows: int = 2000,
        dictionary_entry_limit: int = 500,
        full_hash_limit_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if sample_rows < 1:
            raise ValueError("sample_rows must be positive")
        if dictionary_entry_limit < 0:
            raise ValueError("dictionary_entry_limit must be non-negative")
        self.sample_rows = int(sample_rows)
        self.dictionary_entry_limit = int(dictionary_entry_limit)
        self.full_hash_limit_bytes = int(full_hash_limit_bytes)

    def profile(self, data_dir: PathLike) -> DatasetProfile:
        root = Path(data_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError("Dataset directory does not exist: {0}".format(root))
        profiles: List[FileProfile] = []
        warnings: List[str] = []
        candidates = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.name.lower().endswith(_SOURCE_SUFFIXES)
        ]
        for path in sorted(candidates, key=lambda value: value.as_posix().lower()):
            try:
                resolved = path.resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                warnings.append("Skipped source outside dataset root: {0}".format(path.name))
                continue
            try:
                profiles.append(self._profile_file(root, resolved))
            except (OSError, ValueError, ImportError) as exc:
                relative = resolved.relative_to(root).as_posix()
                # Exception text from third-party parsers is not sent onward:
                # some parsers include an offending source line in an error.
                error_name = type(exc).__name__
                warnings.append("Could not profile {0}: {1}".format(relative, error_name))
                profiles.append(
                    FileProfile(
                        file=relative,
                        format=_format_for(resolved),
                        size_bytes=resolved.stat().st_size,
                        fingerprint=self._fingerprint(resolved),
                        warnings=("Schema inspection failed: {0}".format(error_name),),
                    )
                )
        if not profiles:
            warnings.append("No .csv, .csv.gz or .parquet files were found.")
        return DatasetProfile(
            root_name=root.name,
            files=tuple(profiles),
            warnings=tuple(warnings),
        )

    def _profile_file(self, root: Path, path: Path) -> FileProfile:
        relative = path.relative_to(root).as_posix()
        file_format = _format_for(path)
        warnings: List[str] = []
        row_count: Optional[int] = None
        if file_format == "parquet":
            frame, row_count = self._read_parquet_sample(path)
        else:
            frame = self._read_csv_sample(path)
        columns = tuple(self._column_profile(name, frame[name]) for name in frame.columns)
        dictionary_columns = self._dictionary_schema(relative, tuple(frame.columns))
        entries: Tuple[Dict[str, Any], ...] = ()
        if dictionary_columns and self.dictionary_entry_limit:
            selected = list(dict.fromkeys(dictionary_columns.values()))
            entries = tuple(
                {
                    logical_name: _json_scalar(row[source_name])
                    for logical_name, source_name in dictionary_columns.items()
                }
                for _, row in frame[selected].head(self.dictionary_entry_limit).iterrows()
            )
            if len(frame) >= self.sample_rows:
                warnings.append(
                    "Dictionary entries are limited to the first {0} locally sampled rows.".format(
                        min(self.dictionary_entry_limit, self.sample_rows)
                    )
                )
        return FileProfile(
            file=relative,
            format=file_format,
            size_bytes=path.stat().st_size,
            fingerprint=self._fingerprint(path),
            columns=columns,
            sampled_rows=len(frame),
            row_count=row_count,
            dictionary_entries=entries,
            dictionary_columns=dictionary_columns,
            warnings=tuple(warnings),
        )

    def _read_csv_sample(self, path: Path) -> pd.DataFrame:
        kwargs: Dict[str, Any] = {
            "nrows": self.sample_rows,
            "low_memory": False,
            "encoding": "utf-8-sig",
        }
        try:
            return pd.read_csv(path, **kwargs)
        except UnicodeDecodeError:
            kwargs["encoding"] = "latin-1"
            return pd.read_csv(path, **kwargs)

    def _read_parquet_sample(self, path: Path) -> Tuple[pd.DataFrame, Optional[int]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Parquet profiling requires pyarrow") from exc
        parquet = pq.ParquetFile(path)
        row_count = int(parquet.metadata.num_rows)
        batches = parquet.iter_batches(batch_size=self.sample_rows)
        try:
            frame = next(batches).to_pandas()
        except StopIteration:
            frame = pd.DataFrame(columns=parquet.schema.names)
        return frame, row_count

    @staticmethod
    def _column_profile(name: Any, values: pd.Series) -> ColumnProfile:
        count = len(values)
        non_null = int(values.notna().sum())
        distinct = int(values.nunique(dropna=True))
        denominator = max(non_null, 1)
        normalized = str(name).strip().lower()
        likely_identifier = (
            normalized in _ID_NAMES
            or normalized.endswith("_id")
            or normalized.endswith("id")
        ) and distinct / denominator >= 0.5
        return ColumnProfile(
            name=str(name),
            dtype=str(values.dtype),
            non_null_fraction=round(non_null / max(count, 1), 6),
            distinct_count_sample=distinct,
            unique_fraction_sample=round(distinct / denominator, 6),
            likely_identifier=likely_identifier,
        )

    @staticmethod
    def _dictionary_schema(relative_path: str, columns: Sequence[Any]) -> Dict[str, str]:
        # Filename evidence is mandatory.  A patient table with generic
        # ``id``/``name`` columns must never be promoted to dictionary rows.
        if not _DICTIONARY_FILE_RE.search(relative_path):
            return {}
        by_lower = {str(column).strip().lower(): str(column) for column in columns}

        def choose(candidates: Sequence[str]) -> Optional[str]:
            for candidate in candidates:
                if candidate in by_lower:
                    return by_lower[candidate]
            for lower, original in by_lower.items():
                if any(lower.endswith(candidate) for candidate in candidates):
                    return original
            return None

        identifier = choose(_ID_NAMES)
        label = choose(_LABEL_NAMES)
        if not identifier or not label or identifier == label:
            return {}
        result = {"id": identifier, "name": label}
        unit = choose(_UNIT_NAMES)
        if unit and unit not in result.values():
            result["unit"] = unit
        for logical, candidates in (
            ("group", ("group", "category", "linksto", "referencename")),
            ("code", ("loinc", "loinc_code", "snomed", "standard_code")),
            ("description", ("description", "abbreviation", "referencedescription")),
        ):
            column = choose(candidates)
            if column and column not in result.values():
                result[logical] = column
        return result

    def _fingerprint(self, path: Path) -> str:
        """Content-aware source identity without loading the file into memory."""
        stat = path.stat()
        digest = hashlib.sha256()
        digest.update(b"conmedrl-source-v1\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        if stat.st_size <= self.full_hash_limit_bytes:
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            # Multi-hundred-GB clinical tables should not be fully re-read just
            # to draft a plan.  Size + mtime + first/last 1 MiB catches normal
            # replacements/appends while keeping profiling bounded.
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
            with open(path, "rb") as handle:
                digest.update(handle.read(1024 * 1024))
                handle.seek(max(0, stat.st_size - 1024 * 1024))
                digest.update(handle.read(1024 * 1024))
        return digest.hexdigest()


def _format_for(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv.gz"):
        return "csv.gz"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".parquet"):
        return "parquet"
    raise ValueError("Unsupported source format: {0}".format(path))


def profile_dataset(
    data_dir: PathLike,
    sample_rows: int = 2000,
    dictionary_entry_limit: int = 500,
) -> DatasetProfile:
    """Profile a local dataset for deterministic or LLM-assisted planning."""
    return DatasetProfiler(
        sample_rows=sample_rows,
        dictionary_entry_limit=dictionary_entry_limit,
    ).profile(data_dir)
