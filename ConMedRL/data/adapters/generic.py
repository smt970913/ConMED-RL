"""Declarative adapter for MIMIC-shaped ICU datasets.

Unlike the built-in adapters, this class contains no site itemids.  All paths,
columns, variable ids, units and clinical event roles come from an approved
``DatasetSpec``/``TaskSpec`` mapping.  A new site therefore adds a reviewed
JSON document, not executable Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..base import EPISODE_KEY, TIME_OFFSET_COLUMN, CohortFrame, ObservationFrame
from ..config import Task
from ..schema import CANONICAL_VARIABLES, VariableSpec
from ..sources import SourceRegistry, TableSpec
from ..variables import DictionarySchema, MappedVariable, VariableMapping
from ..profiles import profile_approval_hash
from ..profiler import DatasetProfiler
from ..specs import DatasetSpec, SafeRuleEvaluator
from .mimic import MimicIVAdapter, SECONDS_PER_HOUR

__all__ = ["GenericMimicAdapter"]


def _plain(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    raise TypeError("Expected a mapping or object with to_dict(), got {0}".format(type(value)))


def _normalise_tables(spec: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = spec.get("tables", {})
    if isinstance(raw, Mapping):
        return {str(name): _plain(value) for name, value in raw.items()}
    result: Dict[str, Dict[str, Any]] = {}
    for item in raw or ():
        value = _plain(item)
        role = value.get("name") or value.get("role")
        if role:
            result[str(role)] = value
    return result


class GenericMimicAdapter(MimicIVAdapter):
    """Interpret an approved MIMIC-like dataset specification."""

    database = "generic"

    def __init__(self, config: Any, llm: Any = None) -> None:
        strict: Optional[DatasetSpec] = None
        if isinstance(config.dataset_spec, DatasetSpec):
            strict = config.dataset_spec
        else:
            candidate = _plain(config.dataset_spec)
            if "version" in candidate and "tasks" in candidate:
                strict = DatasetSpec.from_dict(candidate)
        self._strict_spec = strict
        if strict is not None:
            strict.assert_approved()
            if config.approved_plan_hash != strict.approval_hash:
                raise ValueError("approved_plan_hash does not match DatasetSpec approval.")
            current: Dict[str, str] = {}
            profiler = DatasetProfiler(sample_rows=1, dictionary_entry_limit=0)
            for relative in strict.source_fingerprints:
                path = Path(relative)
                if not path.is_absolute():
                    path = Path(config.data_dir) / path
                if not path.exists():
                    raise FileNotFoundError("Approved source file is missing: {0}".format(path))
                current[relative] = profiler._fingerprint(path)
            strict.assert_approved(current)
            self.dataset_spec = strict.to_dict()
            selected = next(
                (
                    task
                    for task in strict.tasks
                    if task.name.strip().lower().replace("_", "-")
                    == str(config.task).strip().lower().replace("_", "-")
                ),
                None,
            )
            if selected is None:
                raise ValueError(
                    "Approved DatasetSpec has no task {0!r}.".format(config.task)
                )
            self.task_spec = selected.to_dict()
        else:
            self.dataset_spec = _plain(config.dataset_spec)
            self.task_spec = _plain(config.task_spec)
            fingerprints = dict(self.dataset_spec.get("source_fingerprints") or {})
            if not fingerprints:
                raise ValueError(
                    "Reviewed generic profiles require local source fingerprints. "
                    "Reload the profile with data_dir=... before execution."
                )
            expected_hash = profile_approval_hash(self.dataset_spec, self.task_spec)
            if config.approved_plan_hash != expected_hash:
                raise ValueError(
                    "approved_plan_hash does not match the dataset/task specifications. "
                    "Re-review and approve the current plan before execution."
                )
            profiler = DatasetProfiler(sample_rows=1, dictionary_entry_limit=0)
            current = {}
            for relative in fingerprints:
                path = Path(relative)
                if not path.is_absolute():
                    path = Path(config.data_dir) / path
                if not path.is_file():
                    raise FileNotFoundError(
                        "Approved source file is missing: {0}".format(path)
                    )
                current[relative] = profiler._fingerprint(path)
            if current != fingerprints:
                raise ValueError(
                    "Local source fingerprints changed; reload and review the profile."
                )
        self._tables = _normalise_tables(self.dataset_spec)
        for dictionary in self.dataset_spec.get("dictionaries", ()):
            value = _plain(dictionary)
            self._tables[str(value["name"])] = {
                "path": value["file"],
                "role": "dictionary",
                "id_column": value["id_column"],
                "name_column": value["name_column"],
                "unit_column": value.get("unit_column"),
                "group_column": value.get("group_column"),
                "description_column": value.get("description_column"),
            }

        catalog: Dict[str, TableSpec] = {}
        overrides: Dict[str, Path] = dict(config.source_paths)
        for role, table in self._tables.items():
            path_value = table.get("path") or table.get("file")
            if path_value:
                path = Path(path_value)
                if not path.is_absolute():
                    path = config.data_dir / path
                overrides[role] = path
                stem = path.name
                for suffix in (".gz", ".bz2", ".xz", ".zip", ".csv", ".parquet", ".tsv"):
                    if stem.lower().endswith(suffix):
                        stem = stem[: -len(suffix)]
            else:
                stems = table.get("stems") or (role,)
                stem = str(stems[0])
            subdirs = tuple(table.get("subdirs") or ())
            columns_value = table.get("columns")
            columns = (
                _plain(columns_value)
                if isinstance(columns_value, Mapping)
                else {}
            )
            required = tuple(
                table.get("key_columns")
                or (
                    list(columns_value)
                    if isinstance(columns_value, (list, tuple))
                    else [source for source in columns.values() if source]
                )
            )
            catalog[role] = TableSpec(
                stems=tuple(table.get("stems") or (stem,)),
                subdirs=subdirs,
                large=bool(table.get("large", role in ("chartevents", "labevents"))),
                key_columns=required,
                description=str(table.get("description", "")),
            )
        sources = SourceRegistry(
            data_dir=config.data_dir,
            database="generic",
            overrides=overrides,
            compression=config.source_compression,
            table_specs=catalog,
        )
        super(GenericMimicAdapter, self).__init__(config, sources=sources, llm=llm)

    @property
    def dictionary_schema(self) -> DictionarySchema:
        dictionaries = self.dataset_spec.get("dictionaries") or ()
        dictionary = (
            _plain(dictionaries[0])
            if dictionaries
            else _plain(self.dataset_spec.get("dictionary"))
        )
        return DictionarySchema(
            id_column=str(dictionary.get("id_column", "itemid")),
            name_column=str(dictionary.get("name_column", "label")),
            unit_column=dictionary.get("unit_column", "unitname"),
            group_column=dictionary.get("group_column", "_source_table"),
            loinc_column=dictionary.get("loinc_column"),
            description_column=dictionary.get("description_column", "abbreviation"),
        )

    def wanted_variables(self) -> List[VariableSpec]:
        states = (
            self.task_spec.get("states")
            or self.task_spec.get("state_variables")
            or self.task_spec.get("state_columns")
            or ()
        )
        variables: List[VariableSpec] = []
        for item in states:
            if isinstance(item, str):
                variables.append(
                    CANONICAL_VARIABLES.get(item, VariableSpec(item, "unknown"))
                )
                continue
            value = _plain(item)
            name = str(value.get("name") or value.get("canonical"))
            if not name:
                continue
            variables.append(
                CANONICAL_VARIABLES.get(
                    name,
                    VariableSpec(
                        name=name,
                        kind=str(value.get("kind", "unknown")),
                        unit=value.get("unit"),
                        plausible_range=(
                            tuple(value["plausible_range"])
                            if value.get("plausible_range") is not None
                            else None
                        ),
                        required=bool(value.get("required", False)),
                        aliases=tuple(value.get("aliases") or ()),
                        description=str(value.get("description", "")),
                    ),
                )
            )
        if variables:
            excluded = set(self.config.exclude_variables)
            return [value for value in variables if value.name not in excluded]
        return super(GenericMimicAdapter, self).wanted_variables()

    def required_tables(self) -> List[str]:
        requested = self.dataset_spec.get("required_tables")
        if requested:
            return [str(value) for value in requested]
        roles = ["icustays", "patients", "admissions"]
        if self.task_spec.get("episode_table"):
            roles.append(str(self.task_spec["episode_table"]))
        roles.extend(self._observation_roles())
        roles.extend(self._dictionary_roles())
        events = _plain(self.task_spec.get("events"))
        for value in events.values():
            role = _plain(value).get("table")
            if role:
                roles.append(str(role))
        action = _plain(self.task_spec.get("action"))
        if action.get("table"):
            roles.append(str(action["table"]))
        for label in self.task_spec.get("labels") or ():
            role = _plain(label).get("table")
            if role:
                roles.append(str(role))
        for rule in self.dataset_spec.get("event_rules") or ():
            role = _plain(rule).get("source_table")
            if role:
                roles.append(str(role))
        return list(dict.fromkeys(role for role in roles if role in self._tables))

    def _dictionary_roles(self) -> List[str]:
        if self.dataset_spec.get("dictionaries"):
            return [
                str(_plain(value)["name"])
                for value in self.dataset_spec["dictionaries"]
            ]
        configured = self.dataset_spec.get("dictionary_tables")
        if configured:
            return [str(value) for value in configured]
        return [
            role
            for role in ("d_items", "d_labitems", "dictionary")
            if role in self._tables
        ]

    def _observation_roles(self) -> List[str]:
        configured = self.dataset_spec.get("observation_tables")
        if configured:
            return [str(value) for value in configured]
        selected = [
            name
            for name, table in self._tables.items()
            if str(table.get("role", "")).lower() in ("observations", "observation")
        ]
        if selected:
            return selected
        return [role for role in ("chartevents", "labevents") if role in self._tables]

    def load_dictionary(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        schema = self.dictionary_schema
        for role in self._dictionary_roles():
            frame = self.sources.load(role, cache=True).copy()
            frame["_source_table"] = role
            if schema.unit_column and schema.unit_column not in frame:
                frame[schema.unit_column] = ""
            if schema.group_column and schema.group_column not in frame:
                frame[schema.group_column] = role
            frames.append(frame)
        if not frames:
            raise ValueError("The dataset specification declares no dictionary table.")
        return pd.concat(frames, ignore_index=True, sort=False)

    def resolve_variables(
        self, overrides: Optional[Mapping[str, Sequence[Any]]] = None
    ) -> VariableMapping:
        if self._mapping is not None:
            return self._mapping
        declared = self.dataset_spec.get("variable_mappings") or ()
        entries: Dict[str, MappedVariable] = {}
        for item in declared:
            value = _plain(item)
            name = str(value.get("canonical") or value.get("name") or "")
            if not name:
                continue
            source_ids = list(value.get("source_ids") or ())
            entries[name] = MappedVariable(
                canonical=name,
                source_ids=source_ids,
                source_names=list(value.get("source_names") or ()),
                source_units=list(value.get("source_units") or ()),
                groups=[str(value.get("table", ""))] * len(source_ids),
                method=str(value.get("method", "approved-plan")),
                confidence=float(value.get("confidence", 1.0)),
                notes=str(value.get("notes", "")),
            )
        mapping = VariableMapping(database="generic", entries=entries)
        if self._strict_spec is not None:
            wanted_names = {item.name for item in self.wanted_variables()}
            for rule in self.dataset_spec.get("event_rules") or ():
                value = _plain(rule)
                name = str(value.get("name", ""))
                if name in wanted_names and name not in mapping.entries:
                    mapping.entries[name] = MappedVariable(
                        canonical=name,
                        source_ids=["declarative:{0}".format(name)],
                        source_names=[name],
                        groups=[str(value.get("source_table", ""))],
                        method="approved-event-rule",
                        confidence=float(self.dataset_spec.get("confidence", 0.0)),
                    )
        if overrides:
            mapping.apply_overrides(overrides)
        missing = [value.name for value in self.wanted_variables() if value.name not in entries]
        if missing:
            # Deterministic/LLM mapping remains available, but its result must
            # already have been reviewed and copied into variable_mappings.
            for name in missing:
                mapping.entries[name] = MappedVariable(
                    canonical=name,
                    notes="unresolved in approved dataset plan",
                )
        self._mapping = mapping
        return mapping

    def build_cohort(self) -> CohortFrame:
        if self._strict_spec is not None:
            return self._build_strict_cohort()
        cohort = super(GenericMimicAdapter, self).build_cohort()
        return self._attach_declared_event_flags(cohort)

    def _build_strict_cohort(self) -> CohortFrame:
        role = str(self.task_spec["episode_table"])
        table = self._tables[role]
        cohort = self.sources.load(role, cache=True).copy()
        episode_column = str(self.task_spec["episode_id_column"])
        subject_column = str(self.task_spec["subject_id_column"])
        cohort = cohort.rename(
            columns={episode_column: EPISODE_KEY, subject_column: "subject_id"}
        )
        context = {name: cohort[name] for name in cohort.columns}
        anchor = SafeRuleEvaluator().evaluate(
            self.task_spec["timeline_anchor"], context
        )
        cohort["intime"] = pd.to_datetime(anchor, errors="coerce")
        end_column = table.get("end_time_column")
        start_column = table.get("start_time_column")
        if start_column and start_column in cohort:
            cohort["intime"] = pd.to_datetime(cohort[start_column], errors="coerce")
        if end_column and end_column in cohort:
            cohort["outtime"] = pd.to_datetime(cohort[end_column], errors="coerce")
        elif "outtime" in cohort:
            cohort["outtime"] = pd.to_datetime(cohort["outtime"], errors="coerce")
        if "los" not in cohort:
            if "outtime" not in cohort:
                raise ValueError(
                    "Strict episode table requires los or an end_time_column."
                )
            cohort["los"] = (
                cohort["outtime"] - cohort["intime"]
            ).dt.total_seconds() / 86400.0
        cohort["los"] = pd.to_numeric(cohort["los"], errors="coerce")
        cohort = self._attach_strict_event_features(cohort)

        before = len(cohort)
        for expression in self.task_spec.get("cohort_filters") or ():
            context = {name: cohort[name] for name in cohort.columns}
            mask = SafeRuleEvaluator().evaluate(expression, context)
            cohort = cohort[np.asarray(mask, dtype=bool)]
        cohort = cohort[
            cohort[EPISODE_KEY].notna()
            & cohort["subject_id"].notna()
            & cohort["los"].notna()
            & (cohort["los"] > 0)
        ]
        self._log_filter("strict declarative cohort", before, len(cohort))
        return cohort.drop_duplicates(EPISODE_KEY).reset_index(drop=True)

    def _attach_strict_event_features(
        self, cohort: pd.DataFrame
    ) -> pd.DataFrame:
        rules = [_plain(item) for item in self.dataset_spec.get("event_rules") or ()]
        state_names = {value.name for value in self.wanted_variables()}
        feature_rules = [rule for rule in rules if rule.get("name") not in state_names]
        if not feature_rules:
            return cohort
        evaluator = SafeRuleEvaluator()
        frames: Dict[str, pd.DataFrame] = {}
        table_cache: Dict[str, pd.DataFrame] = {}
        for rule in rules:
            role = str(rule["source_table"])
            if role not in table_cache:
                table = self._tables[role]
                columns = list(table.get("columns") or ())
                table_cache[role] = self.sources.load(
                    role, cache=True, usecols=columns or None
                )
            source = table_cache[role]
            table = self._tables[role]
            episode_column = str(table.get("episode_id_column", "stay_id"))
            source = source[source[episode_column].isin(set(cohort[EPISODE_KEY]))]
            context = {name: source[name] for name in source.columns}
            mask = np.asarray(evaluator.evaluate(rule["predicate"], context), dtype=bool)
            if mask.ndim == 0:
                mask = np.full(len(source), mask.item(), dtype=bool)
            chosen = source[mask].copy()
            if chosen.empty:
                frames[str(rule["name"])] = pd.DataFrame(
                    columns=[EPISODE_KEY, "event_time", "event_value"]
                )
                continue
            selected_context = {name: chosen[name] for name in chosen.columns}
            event_time = evaluator.evaluate(
                rule["time_expression"], selected_context
            )
            value_expression = rule.get("value_expression")
            event_value = (
                evaluator.evaluate(value_expression, selected_context)
                if value_expression is not None
                else np.ones(len(chosen))
            )
            value_array = np.asarray(event_value)
            if value_array.ndim == 0:
                value_array = np.full(len(chosen), value_array.item())
            time_array = np.asarray(event_time)
            if time_array.ndim == 0:
                time_array = np.full(len(chosen), time_array.item())
            frames[str(rule["name"])] = pd.DataFrame(
                {
                    EPISODE_KEY: chosen[episode_column].to_numpy(),
                    "event_time": time_array,
                    "event_value": value_array,
                }
            )

        anchors = cohort.set_index(EPISODE_KEY)["intime"]
        for rule in feature_rules:
            name = str(rule["name"])
            events = frames[name].copy()
            if rule.get("operation") == "pair":
                starts = frames.get(str(rule.get("pair_with")))
                if starts is None:
                    raise ValueError(
                        "Pair event {0!r} references unavailable event {1!r}.".format(
                            name, rule.get("pair_with")
                        )
                    )
                pairs = starts.merge(
                    events, on=EPISODE_KEY, suffixes=("_start", "_end")
                )
                start_time = pd.to_datetime(
                    pairs["event_time_start"], errors="coerce"
                )
                end_time = pd.to_datetime(pairs["event_time_end"], errors="coerce")
                delta = (end_time - start_time).dt.total_seconds() / SECONDS_PER_HOUR
                valid = delta > 0
                if rule.get("window_hours") is not None:
                    valid &= delta <= float(rule["window_hours"])
                events = pairs.loc[
                    valid, [EPISODE_KEY, "event_time_end", "event_value_end"]
                ].rename(
                    columns={
                        "event_time_end": "event_time",
                        "event_value_end": "event_value",
                    }
                )
            aggregation = str(rule.get("aggregation", "first")).lower()
            if aggregation == "none":
                aggregation = "first"
            if aggregation == "count":
                feature = events.groupby(EPISODE_KEY).size().astype(float)
            elif rule.get("value_expression") is not None:
                feature = events.groupby(EPISODE_KEY)["event_value"].agg(aggregation)
            else:
                times = pd.to_datetime(events["event_time"], errors="coerce")
                offsets = (
                    times - events[EPISODE_KEY].map(anchors)
                ).dt.total_seconds() / SECONDS_PER_HOUR
                timed = events.assign(_offset=offsets)
                feature = timed.groupby(EPISODE_KEY)["_offset"].agg(aggregation)
            cohort[name] = cohort[EPISODE_KEY].map(feature)
        return cohort

    def _label_ventilation(self, cohort: pd.DataFrame) -> pd.DataFrame:
        events = _plain(self.task_spec.get("events"))
        if not events:
            return super(GenericMimicAdapter, self)._label_ventilation(cohort)
        procedures = self.sources.load(
            str(_plain(events.get("extubation") or events.get("intubation")).get(
                "table", "procedureevents"
            )),
            cache=True,
            parse_dates=["starttime", "endtime"],
        )
        stay_column = str(self._tables.get("procedureevents", {}).get(
            "stay_id_column", "stay_id"
        ))
        id_column = str(self._tables.get("procedureevents", {}).get(
            "id_column", "itemid"
        ))
        time_column = str(self._tables.get("procedureevents", {}).get(
            "time_column", "starttime"
        ))

        def event_rows(name: str) -> pd.DataFrame:
            rule = _plain(events.get(name))
            ids = set(rule.get("source_ids") or ())
            values = procedures[procedures[id_column].isin(ids)].copy()
            return values.sort_values([stay_column, time_column])

        intubation = pd.concat(
            [event_rows("intubation"), event_rows("ventilation")],
            ignore_index=True,
        ).drop_duplicates()
        extubation = event_rows("extubation")
        first_int = intubation.groupby(stay_column)[time_column].min()
        pairs = extubation.merge(
            first_int.rename("intubation_time"),
            left_on=stay_column,
            right_index=True,
            how="inner",
        )
        pairs = pairs[pairs[time_column] > pairs["intubation_time"]]
        first_ext = pairs.groupby(stay_column)[time_column].min()

        cohort = cohort.merge(
            first_int.rename("intubation_time"),
            left_on="stay_id",
            right_index=True,
            how="left",
        )
        cohort = cohort.merge(
            first_ext.rename("extubation_time"),
            left_on="stay_id",
            right_index=True,
            how="left",
        )
        window = pd.Timedelta(days=self.config.cohort.reintubation_observation_days)
        later = intubation.merge(
            first_ext.rename("extubation_time"),
            left_on=stay_column,
            right_index=True,
            how="inner",
        )
        gap = later[time_column] - later["extubation_time"]
        reintubated = set(
            later.loc[(gap > pd.Timedelta(0)) & (gap <= window), stay_column]
        )
        cohort["reintubation"] = cohort["stay_id"].isin(reintubated).astype(int)
        cohort["extubation_fail"] = cohort["reintubation"]
        for event_name, flag in (
            ("tracheostomy", "has_tracheostomy"),
            ("non_invasive_ventilation", "had_niv"),
            ("unplanned_extubation", "unplanned_extubation"),
        ):
            rows = event_rows(event_name) if event_name in events else procedures.iloc[0:0]
            cohort[flag] = cohort["stay_id"].isin(set(rows[stay_column])).astype(int)
        cohort["ventilation_hours"] = (
            cohort["extubation_time"] - cohort["intubation_time"]
        ).dt.total_seconds() / SECONDS_PER_HOUR
        cohort["intubation_hours"] = (
            cohort["intubation_time"] - cohort["intime"]
        ).dt.total_seconds() / SECONDS_PER_HOUR
        cohort["extubation_hours"] = (
            cohort["extubation_time"] - cohort["intime"]
        ).dt.total_seconds() / SECONDS_PER_HOUR
        self.report["event_pairing"] = "point-event"
        return cohort

    def _attach_declared_event_flags(self, cohort: pd.DataFrame) -> pd.DataFrame:
        for item in self.task_spec.get("labels") or ():
            rule = _plain(item)
            name = str(rule.get("name") or "")
            role = str(rule.get("table") or "")
            if not name or not role:
                continue
            events = self.sources.load(role).copy()
            table = self._tables.get(role, {})
            id_column = rule.get("source_id_column") or table.get("id_column")
            source_ids = set(rule.get("source_ids") or ())
            if id_column and source_ids:
                events = events[events[str(id_column)].isin(source_ids)]
            code_column = rule.get("code_column")
            codes = set(rule.get("codes") or ())
            if code_column and codes:
                events = events[events[str(code_column)].astype(str).isin(codes)]

            join_key = str(
                rule.get("join_key")
                or (
                    "stay_id"
                    if "stay_id" in events
                    else "hadm_id"
                    if "hadm_id" in events
                    else "subject_id"
                )
            )
            event_time = rule.get("time_column") or table.get("time_column")
            base = cohort[
                ["stay_id", "subject_id", "hadm_id", "intime", "outtime"]
            ].copy()
            joined = base.merge(events, on=join_key, how="left")
            if event_time and str(event_time) in joined:
                times = pd.to_datetime(joined[str(event_time)], errors="coerce")
                reference = str(rule.get("reference_column", "outtime"))
                reference_time = pd.to_datetime(joined[reference], errors="coerce")
                hours = (times - reference_time).dt.total_seconds() / SECONDS_PER_HOUR
                lower = float(rule.get("window_start_hours", 0.0))
                upper = float(rule.get("window_end_hours", 0.0))
                joined = joined[(hours >= lower) & (hours <= upper)]
            counts = joined.groupby("stay_id", sort=False).size()
            aggregation = str(rule.get("aggregation", "any")).lower()
            if aggregation == "count":
                values = counts.astype(float)
            else:
                values = (counts > 0).astype(float)
            cohort[name] = cohort["stay_id"].map(values).fillna(0.0)
        return cohort

    def extract_observations(
        self, cohort: CohortFrame, mapping: VariableMapping
    ) -> ObservationFrame:
        if self._strict_spec is not None:
            return self._extract_strict_observations(cohort)
        collected: List[pd.DataFrame] = []
        intimes = cohort.set_index("stay_id")["intime"]
        conversions = {
            str(rule.get("source_id")): (
                float(rule.get("scale", 1.0)),
                float(rule.get("offset", 0.0)),
            )
            for rule in (
                _plain(item)
                for item in (self.dataset_spec.get("unit_rules") or ())
            )
            if rule.get("source_id") is not None
        }

        for role in self._observation_roles():
            table = self._tables[role]
            id_column = str(table.get("id_column", "itemid"))
            value_column = str(table.get("value_column", "valuenum"))
            time_column = str(table.get("time_column", "charttime"))
            stay_column = str(table.get("stay_id_column", "stay_id"))
            hadm_column = str(table.get("hadm_id_column", "hadm_id"))
            reverse: Dict[Any, str] = {}
            for name, entry in mapping.entries.items():
                groups = set(entry.groups)
                if not groups or role in groups:
                    reverse.update({source_id: name for source_id in entry.source_ids})
            if not reverse:
                continue
            wanted_ids = set(reverse)
            usecols = [id_column, value_column, time_column]
            if stay_column:
                usecols.append(stay_column)
            if hadm_column and hadm_column not in usecols:
                usecols.append(hadm_column)
            seen = kept = 0
            for chunk in self.sources.iter_chunks(
                role,
                chunk_size=self.config.chunk_size,
                usecols=list(dict.fromkeys(usecols)),
                parse_dates=[time_column],
            ):
                seen += len(chunk)
                block = chunk[
                    chunk[id_column].isin(wanted_ids) & chunk[value_column].notna()
                ].copy()
                if stay_column not in block.columns or block[stay_column].isna().all():
                    links = cohort[
                        ["stay_id", "hadm_id", "intime", "outtime"]
                    ].drop_duplicates()
                    block = block.merge(
                        links, left_on=hadm_column, right_on="hadm_id", how="inner"
                    )
                    block = block[
                        (block[time_column] >= block["intime"])
                        & (block[time_column] <= block["outtime"])
                    ]
                    block[stay_column] = block["stay_id"]
                else:
                    block = block[block[stay_column].isin(set(cohort["stay_id"]))]
                if block.empty:
                    continue
                block = block.rename(
                    columns={
                        stay_column: EPISODE_KEY,
                        id_column: "source_id",
                        value_column: "value",
                    }
                )
                block["variable"] = block["source_id"].map(reverse)
                block[TIME_OFFSET_COLUMN] = (
                    block[time_column] - block[EPISODE_KEY].map(intimes)
                ).dt.total_seconds() / SECONDS_PER_HOUR
                for source_id, (scale, offset) in conversions.items():
                    mask = block["source_id"].astype(str) == source_id
                    block.loc[mask, "value"] = (
                        pd.to_numeric(block.loc[mask, "value"], errors="coerce")
                        * scale
                        + offset
                    )
                collected.append(
                    block[
                        [EPISODE_KEY, TIME_OFFSET_COLUMN, "variable", "value", "source_id"]
                    ]
                )
                kept += len(block)
            self.report["{0}_rows_scanned".format(role)] = seen
            self.report["{0}_rows_kept".format(role)] = kept
        if not collected:
            raise ValueError("No observations matched the approved variable mappings.")
        result = pd.concat(collected, ignore_index=True)
        result["value"] = pd.to_numeric(result["value"], errors="coerce")
        return result.dropna(subset=["value"])

    def _extract_strict_observations(
        self, cohort: CohortFrame
    ) -> ObservationFrame:
        evaluator = SafeRuleEvaluator()
        wanted = {value.name for value in self.wanted_variables()}
        rules_by_table: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.dataset_spec.get("event_rules") or ():
            rule = _plain(item)
            if rule.get("name") in wanted and rule.get("value_expression") is not None:
                rules_by_table.setdefault(str(rule["source_table"]), []).append(rule)
        if not rules_by_table:
            return pd.DataFrame(
                columns=[
                    EPISODE_KEY,
                    TIME_OFFSET_COLUMN,
                    "variable",
                    "value",
                    "source_id",
                ]
            )

        intimes = cohort.set_index(EPISODE_KEY)["intime"]
        keep = set(cohort[EPISODE_KEY])
        unit_rules = [
            _plain(item) for item in (self.dataset_spec.get("unit_rules") or ())
        ]
        collected: List[pd.DataFrame] = []
        for role, rules in rules_by_table.items():
            table = self._tables[role]
            episode_column = str(table.get("episode_id_column", "stay_id"))
            selected = list(table.get("columns") or ())
            rows_seen = rows_kept = 0
            for chunk in self.sources.iter_chunks(
                role,
                chunk_size=self.config.chunk_size,
                usecols=selected or None,
            ):
                rows_seen += len(chunk)
                block = chunk[chunk[episode_column].isin(keep)].copy()
                if block.empty:
                    continue
                context = {name: block[name] for name in block.columns}
                for rule in rules:
                    mask = evaluator.evaluate(rule["predicate"], context)
                    mask_array = np.asarray(mask, dtype=bool)
                    if mask_array.ndim == 0:
                        mask_array = np.full(len(block), mask_array.item(), dtype=bool)
                    chosen = block[mask_array].copy()
                    if chosen.empty:
                        continue
                    chosen_context = {name: chosen[name] for name in chosen.columns}
                    values = evaluator.evaluate(rule["value_expression"], chosen_context)
                    times = evaluator.evaluate(rule["time_expression"], chosen_context)
                    if pd.api.types.is_numeric_dtype(times):
                        offsets = pd.to_numeric(times, errors="coerce")
                    elif pd.api.types.is_datetime64_any_dtype(times) or isinstance(
                        times, (pd.Timestamp,)
                    ):
                        offsets = (
                            pd.to_datetime(times, errors="coerce")
                            - chosen[episode_column].map(intimes)
                        ).dt.total_seconds() / SECONDS_PER_HOUR
                    else:
                        parsed = pd.to_datetime(times, errors="coerce")
                        if parsed.notna().any():
                            offsets = (
                                parsed - chosen[episode_column].map(intimes)
                            ).dt.total_seconds() / SECONDS_PER_HOUR
                        else:
                            offsets = pd.to_numeric(times, errors="coerce")
                    value_array = np.asarray(values)
                    if value_array.ndim == 0:
                        value_array = np.full(len(chosen), value_array.item())
                    offset_array = np.asarray(offsets)
                    if offset_array.ndim == 0:
                        offset_array = np.full(len(chosen), offset_array.item())
                    output = pd.DataFrame(
                        {
                            EPISODE_KEY: chosen[episode_column].to_numpy(),
                            TIME_OFFSET_COLUMN: offset_array,
                            "variable": str(rule["name"]),
                            "value": value_array,
                            "source_id": str(rule["name"]),
                        },
                        index=chosen.index,
                    )
                    for conversion in unit_rules:
                        if (
                            str(conversion.get("source_table")) == role
                            and str(conversion.get("name")) == str(rule["name"])
                        ):
                            applies = np.ones(len(chosen), dtype=bool)
                            if conversion.get("when") is not None:
                                applies = np.asarray(
                                    evaluator.evaluate(
                                        conversion["when"], chosen_context
                                    ),
                                    dtype=bool,
                                )
                            output.loc[applies, "value"] = (
                                pd.to_numeric(
                                    output.loc[applies, "value"], errors="coerce"
                                )
                                * float(conversion.get("scale", 1.0))
                                + float(conversion.get("offset", 0.0))
                            )
                    collected.append(output)
                    rows_kept += len(output)
            self.report["{0}_rows_scanned".format(role)] = rows_seen
            self.report["{0}_rows_kept".format(role)] = rows_kept
        if not collected:
            raise ValueError("No rows matched the approved state event rules.")
        return pd.concat(collected, ignore_index=True).dropna(
            subset=[TIME_OFFSET_COLUMN, "value"]
        )

    def action_events(self, cohort: CohortFrame) -> pd.DataFrame:
        if self.config.task == Task.EXTUBATION:
            return pd.DataFrame(
                {
                    EPISODE_KEY: cohort[EPISODE_KEY].to_numpy(),
                    TIME_OFFSET_COLUMN: cohort["extubation_hours"].to_numpy(dtype=float),
                }
            )
        if self.config.task == Task.DISCHARGE:
            return super(GenericMimicAdapter, self).action_events(cohort)

        action = _plain(self.task_spec.get("action"))
        if self._strict_spec is not None and not action.get("table"):
            return pd.DataFrame(columns=[EPISODE_KEY, TIME_OFFSET_COLUMN])
        table_role = action.get("table")
        if not table_role:
            raise ValueError("Custom tasks must declare action.table.")
        table = self._tables[str(table_role)]
        frame = self.sources.load(str(table_role))
        stay_column = str(table.get("stay_id_column", "stay_id"))
        time_column = str(action.get("time_column") or table.get("time_column", "charttime"))
        frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
        frame = frame[frame[stay_column].isin(set(cohort["stay_id"]))].copy()
        source_id_column = action.get("source_id_column") or table.get("id_column")
        source_ids = action.get("source_ids") or ()
        if source_id_column and source_ids:
            frame = frame[frame[str(source_id_column)].isin(set(source_ids))]
        frame[TIME_OFFSET_COLUMN] = (
            frame[time_column] - frame[stay_column].map(
                cohort.set_index("stay_id")["intime"]
            )
        ).dt.total_seconds() / SECONDS_PER_HOUR
        action_columns = list(action.get("columns") or ())
        value_column = action.get("value_column")
        if value_column and not action_columns:
            action_columns = [str(action.get("name", "action"))]
            frame[action_columns[0]] = pd.to_numeric(
                frame[str(value_column)], errors="coerce"
            )
        return frame.rename(columns={stay_column: EPISODE_KEY})[
            [EPISODE_KEY, TIME_OFFSET_COLUMN] + action_columns
        ]
