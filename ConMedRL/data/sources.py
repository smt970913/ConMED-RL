"""Locating and loading raw source tables.

Users point the pipeline at a directory and it finds the tables, rather than
requiring eight absolute paths the way the original ``ICUDataInput`` did. Both
supported databases ship as gzipped CSV, but their layouts differ: MIMIC-IV
splits tables across ``hosp/`` and ``icu/`` subdirectories, while SICdb is flat.

Only the tables a task actually needs are read. The original code loaded
``inputevents``, ``outputevents`` and ``d_labitems`` on every run and never
referenced them again -- on MIMIC-IV that is tens of gigabytes of pointless
I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .config import Database

__all__ = [
    "TableSpec",
    "MIMIC_IV_TABLES",
    "SICDB_TABLES",
    "SourceRegistry",
    "MissingTableError",
]

logger = logging.getLogger(__name__)


class MissingTableError(FileNotFoundError):
    """A required source table could not be found under the data directory."""


@dataclass(frozen=True)
class TableSpec:
    """Where a logical table lives and how big it is likely to be.

    Attributes
    ----------
    stems:
        Accepted file basenames without extension, in preference order.
    subdirs:
        Subdirectories of the data root to search, in addition to the root.
    large:
        Hint that the table should be streamed (Dask / chunked) rather than
        read whole. ``chartevents`` and ``data_float_h`` do not fit in memory
        on a typical workstation.
    key_columns:
        Columns the pipeline relies on. Used to fail loudly and early with a
        useful message instead of a ``KeyError`` deep inside a transform.
    """

    stems: Tuple[str, ...]
    subdirs: Tuple[str, ...] = ()
    large: bool = False
    key_columns: Tuple[str, ...] = ()
    description: str = ""


# ---------------------------------------------------------------------------
# MIMIC-IV
# ---------------------------------------------------------------------------

# MIMIC-IV v2.x / v3.x place clinical tables under `hosp/` and ICU tables under
# `icu/`. Older mirrors and hand-assembled folders keep everything flat, so
# every spec searches the root too.
MIMIC_IV_TABLES: Dict[str, TableSpec] = {
    "icustays": TableSpec(
        stems=("icustays",),
        subdirs=("icu",),
        key_columns=("subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit"),
        description="One row per ICU stay; the cohort backbone.",
    ),
    "patients": TableSpec(
        stems=("patients",),
        subdirs=("hosp",),
        key_columns=("subject_id", "gender", "anchor_age", "dod"),
        description="Demographics and date of death.",
    ),
    "admissions": TableSpec(
        stems=("admissions",),
        subdirs=("hosp",),
        key_columns=("subject_id", "hadm_id", "admittime", "dischtime", "deathtime"),
        description="Hospital admissions, used for in-hospital death timing.",
    ),
    "d_items": TableSpec(
        stems=("d_items",),
        subdirs=("icu",),
        key_columns=("itemid", "label", "linksto"),
        description="ICU item dictionary.",
    ),
    "d_labitems": TableSpec(
        stems=("d_labitems",),
        subdirs=("hosp",),
        key_columns=("itemid", "label"),
        description="Laboratory item dictionary.",
    ),
    "chartevents": TableSpec(
        stems=("chartevents",),
        subdirs=("icu",),
        large=True,
        key_columns=("subject_id", "stay_id", "itemid", "charttime", "valuenum"),
        description="Charted observations; the bulk of the physiology.",
    ),
    "labevents": TableSpec(
        stems=("labevents",),
        subdirs=("hosp",),
        large=True,
        key_columns=("subject_id", "itemid", "charttime", "valuenum"),
        description="Laboratory results.",
    ),
    "procedureevents": TableSpec(
        stems=("procedureevents",),
        subdirs=("icu",),
        key_columns=("subject_id", "stay_id", "itemid", "starttime", "endtime"),
        description="Procedures with a duration; defines the ventilation cohort.",
    ),
    "outputevents": TableSpec(
        stems=("outputevents",),
        subdirs=("icu",),
        key_columns=("stay_id", "itemid", "charttime", "value"),
    ),
    "inputevents": TableSpec(
        stems=("inputevents",),
        subdirs=("icu",),
        large=True,
        key_columns=("stay_id", "itemid", "starttime", "endtime"),
    ),
    "datetimeevents": TableSpec(
        stems=("datetimeevents",),
        subdirs=("icu",),
        key_columns=("stay_id", "itemid", "charttime"),
    ),
}


# ---------------------------------------------------------------------------
# SICdb
# ---------------------------------------------------------------------------

# SICdb ships eight flat gzipped CSVs. Schemas per the official documentation
# (https://www.sicdb.com/Documentation/SICdb_Documentation); every `Offset` is
# seconds since PDMS admission, *not* since ICU admission -- see
# `cases.ICUOffset`.
SICDB_TABLES: Dict[str, TableSpec] = {
    "cases": TableSpec(
        stems=("cases",),
        key_columns=("CaseID", "PatientID", "TimeOfStay", "ICUOffset", "AgeOnAdmission"),
        description="One row per admission; demographics, mortality, discharge type.",
    ),
    "d_references": TableSpec(
        stems=("d_references", "d_reference"),
        key_columns=("ReferenceGlobalID", "ReferenceValue", "ReferenceName"),
        description="Dictionary for every encoded field, including units and LOINC.",
    ),
    "data_float_h": TableSpec(
        stems=("data_float_h", "data_float_hourly"),
        large=True,
        key_columns=("CaseID", "DataID", "Offset", "Val"),
        description="Hourly-aggregated signal data; the bulk of the physiology.",
    ),
    "laboratory": TableSpec(
        stems=("laboratory",),
        large=True,
        # The published schema table names this column `DrugID` while the SQL
        # examples on the same page use `LaboratoryID`. Neither is required
        # here; `SicdbSchema` picks whichever is present.
        key_columns=("CaseID", "Offset", "LaboratoryValue"),
        description="Laboratory results, including some pre-admission labs.",
    ),
    "medication": TableSpec(
        stems=("medication",),
        large=True,
        key_columns=("CaseID", "DrugID", "Offset", "Amount"),
        description="Administered drugs with start/end offsets.",
    ),
    "data_range": TableSpec(
        stems=("data_range",),
        key_columns=("CaseID", "DataID", "Offset", "OffsetEnd"),
        description=(
            "Items with a start and an end offset -- lines, drainages, and "
            "airway/ventilation periods. Defines the extubation cohort."
        ),
    ),
    "data_ref": TableSpec(
        stems=("data_ref",),
        key_columns=("CaseID", "FieldID", "RefID"),
        description="Nominal per-admission data (preconditions, premedication).",
    ),
    "unitlog": TableSpec(
        stems=("unitlog",),
        key_columns=("CaseID", "Offset", "HospitalUnit"),
        description=(
            "Ward transfers. Needed to time ICU discharge, since `cases` only "
            "records the last unit."
        ),
    ),
}


_TABLES_BY_DATABASE: Dict[str, Dict[str, TableSpec]] = {
    Database.MIMIC_IV: MIMIC_IV_TABLES,
    Database.SICDB: SICDB_TABLES,
}

#: Extensions tried for each stem, in preference order. Parquet first because
#: it is both smaller and far faster to read column-selectively.
_EXTENSIONS: Tuple[str, ...] = (
    ".parquet",
    ".csv.gz",
    ".csv",
    ".csv.zip",
    ".csv.bz2",
    ".csv.xz",
    ".tsv",
    ".tsv.gz",
)


@dataclass
class SourceRegistry:
    """Resolves logical table names to files and loads them on demand.

    >>> reg = SourceRegistry(Path("/data/mimic-iv/3.1"), "mimic-iv")
    >>> reg.resolve("icustays")
    PosixPath('/data/mimic-iv/3.1/icu/icustays.csv.gz')
    >>> stays = reg.load("icustays", usecols=["stay_id", "intime"])
    """

    data_dir: Path
    database: str
    #: Logical name -> explicit path, bypassing the search.
    overrides: Mapping[str, Path] = field(default_factory=dict)
    compression: Optional[str] = "infer"
    #: Injected catalog used by approved generic-dataset plans.
    table_specs: Optional[Mapping[str, TableSpec]] = None
    _resolved: Dict[str, Path] = field(default_factory=dict, init=False, repr=False)
    _cache: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.database = Database.normalize(self.database)
        self.data_dir = Path(self.data_dir).expanduser()
        self.overrides = {str(k): Path(v).expanduser() for k, v in dict(self.overrides).items()}
        if self.table_specs is not None:
            self.table_specs = dict(self.table_specs)

        if not self.data_dir.is_dir() and not self.overrides:
            raise MissingTableError(
                "Data directory does not exist: {0}".format(self.data_dir)
            )

    # -- introspection --------------------------------------------------------

    @property
    def specs(self) -> Dict[str, TableSpec]:
        if self.table_specs is not None:
            return dict(self.table_specs)
        return _TABLES_BY_DATABASE[self.database]

    def spec(self, name: str) -> TableSpec:
        try:
            return self.specs[name]
        except KeyError:
            raise KeyError(
                "{0!r} is not a known {1} table. Known tables: {2}".format(
                    name, self.database, ", ".join(sorted(self.specs))
                )
            )

    def resolve(self, name: str, required: bool = True) -> Optional[Path]:
        """Find the file backing logical table ``name``."""
        if name in self._resolved:
            return self._resolved[name]

        if name in self.overrides:
            path = self.overrides[name]
            if not path.is_file():
                raise MissingTableError(
                    "source_paths[{0!r}] points at a missing file: {1}".format(name, path)
                )
            self._resolved[name] = path
            return path

        spec = self.spec(name)
        search_dirs: List[Path] = [self.data_dir]
        for subdir in spec.subdirs:
            search_dirs.append(self.data_dir / subdir)
        # Some mirrors nest one level deeper, e.g. `<root>/mimiciv/3.1/icu`.
        for subdir in spec.subdirs:
            search_dirs.extend(sorted(self.data_dir.glob("*/{0}".format(subdir))))

        for directory in search_dirs:
            if not directory.is_dir():
                continue
            for stem in spec.stems:
                for ext in _EXTENSIONS:
                    candidate = directory / "{0}{1}".format(stem, ext)
                    if candidate.is_file():
                        self._resolved[name] = candidate
                        logger.debug("Resolved %s -> %s", name, candidate)
                        return candidate

        if required:
            searched = "\n  ".join(str(d) for d in search_dirs if d.is_dir())
            raise MissingTableError(
                "Could not find the {0} table {1!r} (expected one of {2} with "
                "an extension in {3}).\nSearched:\n  {4}\n"
                "Pass an explicit path via "
                "PreprocessConfig(source_paths={{{1!r}: '/path/to/file'}}).".format(
                    self.database,
                    name,
                    " / ".join(spec.stems),
                    ", ".join(_EXTENSIONS),
                    searched or str(self.data_dir),
                )
            )
        return None

    def available(self) -> Dict[str, Optional[str]]:
        """Map every known table to its resolved path, or ``None`` if absent."""
        out: Dict[str, Optional[str]] = {}
        for name in sorted(self.specs):
            try:
                path = self.resolve(name, required=False)
            except MissingTableError:
                path = None
            out[name] = str(path) if path else None
        return out

    def require(self, names: Iterable[str]) -> None:
        """Resolve several tables up front so a run fails before doing work."""
        missing: List[str] = []
        for name in names:
            try:
                self.resolve(name, required=True)
            except MissingTableError as exc:
                missing.append(str(exc).splitlines()[0])
        if missing:
            raise MissingTableError(
                "Missing {0} required source table(s):\n  {1}".format(
                    len(missing), "\n  ".join(missing)
                )
            )

    # -- loading --------------------------------------------------------------

    def load(
        self,
        name: str,
        usecols: Optional[Sequence[str]] = None,
        dtype: Optional[Mapping[str, Any]] = None,
        parse_dates: Optional[Sequence[str]] = None,
        cache: bool = False,
        **read_kwargs: Any,
    ) -> pd.DataFrame:
        """Read a table into a DataFrame.

        ``usecols`` is honoured leniently: columns absent from the file are
        skipped with a warning rather than raising, because MIMIC-IV renames a
        few columns between minor versions (``cgid`` / ``caregiver_id``).
        """
        if cache and name in self._cache:
            return self._cache[name]

        path = self.resolve(name, required=True)
        spec = self.spec(name)

        if spec.large and usecols is None:
            logger.warning(
                "Reading the whole of %s (%s) into memory; pass usecols= or use "
                "iter_chunks()/load_dask() if this is the full table.", name, path,
            )

        frame = self._read(path, usecols, dtype, parse_dates, **read_kwargs)

        missing_keys = [c for c in spec.key_columns if c not in frame.columns]
        if missing_keys and usecols is None:
            logger.warning(
                "%s is missing expected column(s): %s. The file may be from a "
                "different database version.", name, ", ".join(missing_keys),
            )

        if cache:
            self._cache[name] = frame
        return frame

    def _read(
        self,
        path: Path,
        usecols: Optional[Sequence[str]],
        dtype: Optional[Mapping[str, Any]],
        parse_dates: Optional[Sequence[str]],
        **read_kwargs: Any,
    ) -> pd.DataFrame:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path, columns=list(usecols) if usecols else None)
            if parse_dates:
                for column in parse_dates:
                    if column in frame.columns:
                        frame[column] = pd.to_datetime(frame[column], errors="coerce")
            return frame

        kwargs: Dict[str, Any] = {"low_memory": False}
        if self.compression not in (None, "infer"):
            kwargs["compression"] = self.compression
        kwargs.update(read_kwargs)

        if usecols is not None:
            header = pd.read_csv(path, nrows=0, **{
                k: v for k, v in kwargs.items() if k not in ("nrows", "usecols")
            })
            present = [c for c in usecols if c in header.columns]
            absent = [c for c in usecols if c not in header.columns]
            if absent:
                logger.warning(
                    "%s has no column(s) %s; continuing without them.",
                    path.name, ", ".join(absent),
                )
            if not present:
                raise MissingTableError(
                    "None of the requested columns {0} exist in {1}. Present: "
                    "{2}".format(list(usecols), path, list(header.columns))
                )
            kwargs["usecols"] = present
            if dtype:
                dtype = {k: v for k, v in dtype.items() if k in present}
            if parse_dates:
                parse_dates = [c for c in parse_dates if c in present]

        if dtype:
            kwargs["dtype"] = dict(dtype)
        if parse_dates:
            kwargs["parse_dates"] = list(parse_dates)

        return pd.read_csv(path, **kwargs)

    def iter_chunks(
        self,
        name: str,
        chunk_size: int = 1_000_000,
        usecols: Optional[Sequence[str]] = None,
        dtype: Optional[Mapping[str, Any]] = None,
        **read_kwargs: Any,
    ) -> Iterable[pd.DataFrame]:
        """Stream a large table in row chunks.

        The fallback path when Dask is unavailable, and the one used for
        ``chartevents`` / ``data_float_h`` filtering. Memory stays bounded by
        ``chunk_size`` regardless of file size.
        """
        path = self.resolve(name, required=True)

        if path.suffix == ".parquet":
            # Parquet has no chunksize; read row groups instead.
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path)
            columns = list(usecols) if usecols else None
            for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
                yield batch.to_pandas()
            return

        kwargs: Dict[str, Any] = {"chunksize": chunk_size, "low_memory": False}
        if self.compression not in (None, "infer"):
            kwargs["compression"] = self.compression
        if usecols is not None:
            header = pd.read_csv(path, nrows=0)
            present = [c for c in usecols if c in header.columns]
            if not present:
                raise MissingTableError(
                    "None of the requested columns {0} exist in {1}.".format(
                        list(usecols), path
                    )
                )
            kwargs["usecols"] = present
            if dtype:
                dtype = {k: v for k, v in dtype.items() if k in present}
        if dtype:
            kwargs["dtype"] = dict(dtype)
        kwargs.update(read_kwargs)

        for chunk in pd.read_csv(path, **kwargs):
            yield chunk

    def load_dask(
        self,
        name: str,
        usecols: Optional[Sequence[str]] = None,
        dtype: Optional[Mapping[str, Any]] = None,
        blocksize: Optional[str] = "256MB",
        **read_kwargs: Any,
    ) -> Any:
        """Return a Dask DataFrame for a large table.

        Gzipped CSV cannot be split, so Dask reads it as a single partition and
        offers no parallelism -- :meth:`iter_chunks` is usually the better
        choice there. Dask pays off on ``.csv`` or ``.parquet`` inputs.
        """
        try:
            import dask.dataframe as dd
        except ImportError as exc:
            raise ImportError(
                "load_dask needs dask: pip install 'dask[complete]'"
            ) from exc

        path = self.resolve(name, required=True)

        if path.suffix == ".parquet":
            return dd.read_parquet(path, columns=list(usecols) if usecols else None)

        kwargs: Dict[str, Any] = {}
        if path.name.endswith((".gz", ".bz2", ".xz", ".zip")):
            # Compressed text is not splittable; forcing one block avoids
            # Dask's "blocksize with compression" error.
            kwargs["blocksize"] = None
            logger.info(
                "%s is compressed, so Dask cannot split it; consider "
                "iter_chunks() or converting to Parquet.", path.name,
            )
        elif blocksize:
            kwargs["blocksize"] = blocksize

        if usecols is not None:
            kwargs["usecols"] = list(usecols)
        if dtype:
            kwargs["dtype"] = dict(dtype)
        kwargs.update(read_kwargs)

        return dd.read_csv(path, **kwargs)

    def clear_cache(self) -> None:
        self._cache.clear()
