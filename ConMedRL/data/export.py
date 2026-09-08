"""Writing a finished dataset to disk in the formats the user asked for.

CSV is always written because the ConMedRL data loaders read the outcome and
state tables from CSV, and because it is the format the published example
notebooks use. Everything else is additive.

File naming follows ``{prefix}_{artefact}_{split}.{ext}``, e.g.
``mimic_iv_discharge_outcome_table_train.csv``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import OutputFormat, PreprocessConfig
from .dataset import ROW_INDEX_COLUMN, RLDatasetBundle, SplitTables

__all__ = ["write_bundle", "available_formats"]

logger = logging.getLogger(__name__)


def available_formats() -> Dict[str, bool]:
    """Which output formats this installation can actually produce."""
    status = {OutputFormat.CSV: True}
    status[OutputFormat.FHIR] = True

    try:
        import pyarrow  # noqa: F401

        status[OutputFormat.PARQUET] = True
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            status[OutputFormat.PARQUET] = True
        except ImportError:
            status[OutputFormat.PARQUET] = False

    try:
        import d3rlpy  # noqa: F401

        status[OutputFormat.D3RLPY] = True
    except ImportError:
        status[OutputFormat.D3RLPY] = False

    return status


def _write_fhir(bundle: RLDatasetBundle, directory: Path) -> Dict[str, str]:
    """Map unscaled decision-epoch data to de-identified FHIR R4 resources."""
    from .fhir import export_fhir_r4

    cohort_parts: List[pd.DataFrame] = []
    observation_parts: List[pd.DataFrame] = []
    procedure_parts: List[pd.DataFrame] = []
    units = {variable.name: variable.unit for variable in bundle.schema.variables}
    demographic_names = {
        variable.name
        for variable in bundle.schema.variables
        if variable.kind == "demographic"
    }
    for split_name, tables in bundle.splits.items():
        outcome = tables.outcome
        cohort_columns = [
            column
            for column in (
                "subject_id", "stay_id", "intime", "outtime", "los", "M", "gender"
            )
            if column in outcome
        ]
        cohort_parts.append(
            outcome[cohort_columns].drop_duplicates("stay_id").assign(split=split_name)
        )

        raw_state = bundle.raw_state_tables.get(split_name)
        if raw_state is not None and len(raw_state):
            metadata_columns = [
                column
                for column in ("stay_id", "time", "time_offset_hours")
                if column in outcome
            ]
            values = pd.concat(
                [
                    outcome[metadata_columns].reset_index(drop=True),
                    raw_state.reset_index(drop=True),
                ],
                axis=1,
            )
            demographics = [
                column for column in raw_state.columns if column in demographic_names
            ]
            if demographics:
                first_demographics = (
                    values[["stay_id"] + demographics]
                    .groupby("stay_id", as_index=False)
                    .first()
                )
                cohort_parts[-1] = cohort_parts[-1].merge(
                    first_demographics, on="stay_id", how="left"
                )
            observation_columns = [
                column for column in raw_state.columns if column not in demographic_names
            ]
            long = values.melt(
                id_vars=metadata_columns,
                value_vars=observation_columns,
                var_name="variable",
                value_name="value",
            ).dropna(subset=["value"])
            long["unit"] = long["variable"].map(units)
            observation_parts.append(long)

        if "is_terminal_action" in outcome:
            mask = outcome["is_terminal_action"].fillna(0).astype(float) > 0
            if mask.any():
                columns = [
                    column
                    for column in ("stay_id", "time", "time_offset_hours")
                    if column in outcome
                ]
                procedure_parts.append(
                    outcome.loc[mask, columns].assign(procedure=bundle.action_name)
                )

    cohort = pd.concat(cohort_parts, ignore_index=True).drop_duplicates("stay_id")
    observations = (
        pd.concat(observation_parts, ignore_index=True)
        if observation_parts
        else pd.DataFrame(columns=["stay_id", "variable", "value"])
    )
    procedures = (
        pd.concat(procedure_parts, ignore_index=True)
        if procedure_parts
        else None
    )
    salt = os.environ.get(bundle.config.fhir.id_salt_env)
    result = export_fhir_r4(
        cohort=cohort,
        observations=observations,
        procedures=procedures,
        output_dir=directory,
        id_salt=salt,
        source_name="{0} decision-epoch exchange".format(bundle.config.output_prefix),
        validator_path=bundle.config.fhir.validator_path,
    )
    return {
        "fhir_{0}".format(name.lower()): path
        for name, path in result.paths.items()
    }


def _split_artefacts(split: SplitTables) -> List[Tuple[str, pd.DataFrame]]:
    """Logical artefact name -> table, for one split."""
    items = [
        ("outcome_table_{0}".format(split.name), split.outcome),
        ("state_var_table_{0}".format(split.name), split.state),
    ]
    if split.outcome_select is not None:
        items.append(
            ("outcome_table_{0}_select".format(split.name), split.outcome_select)
        )
    if split.state_select is not None:
        items.append(
            ("state_var_table_{0}_select".format(split.name), split.state_select)
        )
    return items


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    # index=False keeps the files interchangeable with the published examples.
    # Row alignment survives anyway: the outcome tables carry an explicit
    # `row_index` column, and the select subsets carry the parent split's
    # positions in it, so a CSV round-trip cannot silently break next-state
    # lookups the way a dropped pandas index would.
    table.to_csv(path, index=False)


def _write_parquet(table: pd.DataFrame, path: Path) -> None:
    table.to_parquet(path, index=False)


def _write_d3rlpy(bundle: RLDatasetBundle, directory: Path) -> Dict[str, str]:
    """Persist MDPDatasets per split, with an ``.npz`` fallback.

    d3rlpy's own serialisation differs across major versions and its v2 format
    is not readable by v1. The ``.npz`` of raw arrays is always written so the
    transitions remain recoverable regardless of which d3rlpy the reader has.
    """
    written: Dict[str, str] = {}
    directory.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        try:
            mdp = bundle.to_mdp_dataset(split=split_name)
        except Exception as exc:  # noqa: BLE001 - optional artefact
            logger.warning("Could not build an MDPDataset for %s: %s", split_name, exc)
            continue

        npz_path = directory / "{0}_mdp_arrays.npz".format(split_name)
        np.savez_compressed(npz_path, **mdp.arrays)
        written["d3rlpy_arrays_{0}".format(split_name)] = str(npz_path)

        for label, dataset in [("objective", mdp.objective)] + [
            ("constraint_{0}".format(i), ds) for i, ds in enumerate(mdp.constraints)
        ]:
            dump = getattr(dataset, "dump", None)
            if dump is None:
                continue
            target = directory / "{0}_{1}.h5".format(split_name, label)
            try:
                dump(str(target))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "d3rlpy could not dump %s/%s (%s); the .npz arrays still "
                    "contain the transitions.", split_name, label, exc,
                )
                continue
            written["d3rlpy_{0}_{1}".format(split_name, label)] = str(target)

    return written


def _write_scaler(scaler: Any, path: Path) -> Optional[str]:
    """Persist the fitted scaler so inference can reproduce the scaling."""
    try:
        import joblib
    except ImportError:
        logger.warning("joblib is unavailable; the fitted scaler was not saved.")
        return None
    try:
        joblib.dump(scaler, path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save the scaler to %s: %s", path, exc)
        return None
    return str(path)


def write_bundle(
    bundle: RLDatasetBundle,
    config: Optional[PreprocessConfig] = None,
) -> Dict[str, str]:
    """Write ``bundle`` to ``config.output_dir`` and return the paths written.

    Keys are logical artefact names (``"outcome_table_train"``,
    ``"schema"``, ``"d3rlpy_train_objective"``, ...) so callers can look up a
    file without reconstructing its name.
    """
    config = config or bundle.config
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.output_prefix

    formats = tuple(config.output_formats)
    supported = available_formats()
    for fmt in formats:
        if not supported.get(fmt, False):
            logger.warning(
                "Output format %r was requested but its dependency is not "
                "installed; skipping it.", fmt,
            )

    written: Dict[str, str] = {}

    tabular_writers = []
    if OutputFormat.CSV in formats:
        tabular_writers.append(("csv", _write_csv))
    if OutputFormat.PARQUET in formats and supported.get(OutputFormat.PARQUET):
        tabular_writers.append(("parquet", _write_parquet))

    for split in (bundle.train, bundle.val, bundle.test):
        for artefact, table in _split_artefacts(split):
            for ext, writer in tabular_writers:
                path = out_dir / "{0}_{1}.{2}".format(prefix, artefact, ext)
                writer(table, path)
                key = artefact if ext == "csv" else "{0}__{1}".format(artefact, ext)
                written[key] = str(path)

    # Unscaled state tables, when the pipeline kept them: needed to refit a
    # different scaler later without re-running extraction.
    for name, table in bundle.raw_state_tables.items():
        for ext, writer in tabular_writers:
            path = out_dir / "{0}_state_var_table_{1}_unscaled.{2}".format(prefix, name, ext)
            writer(table, path)
            key = "state_var_table_{0}_unscaled".format(name)
            written[key if ext == "csv" else "{0}__{1}".format(key, ext)] = str(path)

    # Terminal state: needed by both data loaders.
    terminal_path = out_dir / "{0}_terminal_state.csv".format(prefix)
    pd.DataFrame(
        [bundle.terminal_state], columns=bundle.schema.names
    ).to_csv(terminal_path, index=False)
    written["terminal_state"] = str(terminal_path)

    # Resolved schema, including what was dropped and why.
    schema_path = out_dir / "{0}_state_space_schema.json".format(prefix)
    with open(schema_path, "w", encoding="utf-8") as fh:
        json.dump(bundle.schema.to_dict(), fh, indent=2, ensure_ascii=False)
    written["schema"] = str(schema_path)

    # Full run manifest, so a dataset on disk explains itself.
    manifest_path = out_dir / "{0}_manifest.json".format(prefix)
    manifest = bundle.to_dict()
    # Defence in depth: raw withdrawal IDs are memory-only even if a custom
    # config implementation accidentally includes the private request field.
    manifest.get("config", {}).pop("withdrawn_subject_ids", None)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    written["manifest"] = str(manifest_path)

    if bundle.scaler is not None:
        scaler_path = _write_scaler(bundle.scaler, out_dir / "{0}_scaler.joblib".format(prefix))
        if scaler_path:
            written["scaler"] = scaler_path

    if OutputFormat.D3RLPY in formats and supported.get(OutputFormat.D3RLPY):
        written.update(_write_d3rlpy(bundle, out_dir / "d3rlpy"))
    if OutputFormat.FHIR in formats:
        written.update(_write_fhir(bundle, out_dir / "fhir_r4"))

    bundle.written_files.update(written)

    # Rewrite the manifest so it lists itself and everything else.
    manifest["written_files"] = dict(written)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)

    logger.info("Wrote %d artefact(s) to %s", len(written), out_dir)
    return written
