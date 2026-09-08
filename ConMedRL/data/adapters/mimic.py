"""MIMIC-IV adapter for the discharge and extubation tasks.

A rewrite rather than a port. The original scripts carried four bugs that all
produced quiet, plausible-looking output:

* the readmission accumulator was cleared inside its own loop, so every
  ``discharge_fail`` / ``readmission`` label came out zero;
* the panel builder indexed ``icustays`` positionally (``iloc[i]``) while
  iterating over unique ``stay_id`` values, so admission times were attached to
  the wrong stay whenever the two orders diverged;
* ``compute_qsofa`` read ``'GCS score'`` while the column was ``'GCS Score'``;
* ``qSOFA_safe_action_space`` read ``'qSOFA'`` while the column was ``'qsofa'``.

Everything here is expressed as merges and vectorised comparisons, which is
also what makes it fast enough to run on the full 3.3 GB ``chartevents``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from ..base import (
    EPISODE_KEY,
    TIME_OFFSET_COLUMN,
    BaseAdapter,
    CohortFrame,
    ObservationFrame,
)
from ..config import Task
from ..variables import VariableMapping

__all__ = ["MimicIVAdapter", "MIMIC_PROCEDURE_ITEMIDS"]

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600.0


class MIMIC_PROCEDURE_ITEMIDS:
    """``procedureevents`` itemids that define the ventilation cohort."""

    INVASIVE_VENTILATION = 225792
    NON_INVASIVE_VENTILATION = 225794
    TRACHEOSTOMY = 225448
    UNPLANNED_EXTUBATION_PATIENT = 225468
    UNPLANNED_EXTUBATION_NON_PATIENT = 225477

    UNPLANNED = (UNPLANNED_EXTUBATION_PATIENT, UNPLANNED_EXTUBATION_NON_PATIENT)


class MimicIVAdapter(BaseAdapter):
    """Build a cohort and physiology table from a local MIMIC-IV download."""

    database = "mimic-iv"
    # `d_items.linksto` names the table an item lives in; physiology is all in
    # chartevents, and searching the rest invites matches on order categories.
    dictionary_groups = ("chartevents",)

    def required_tables(self) -> List[str]:
        tables = ["icustays", "patients", "admissions", "d_items", "chartevents"]
        if self.config.task == Task.EXTUBATION:
            tables.append("procedureevents")
        return tables

    def load_dictionary(self) -> pd.DataFrame:
        return self.sources.load("d_items", cache=True)

    # -- cohort ---------------------------------------------------------------

    def build_cohort(self) -> CohortFrame:
        icustays = self.sources.load(
            "icustays", cache=True, parse_dates=["intime", "outtime"]
        )
        before = len(icustays)
        self.report["source_icustays"] = before

        cohort = icustays.rename(columns={"subject_id": "subject_id"}).copy()
        cohort["los"] = pd.to_numeric(cohort["los"], errors="coerce")

        cohort = self._attach_patients(cohort)
        cohort = self._attach_admissions(cohort)
        cohort = self._label_readmissions(cohort)

        if self.config.task == Task.EXTUBATION:
            cohort = self._label_ventilation(cohort)

        cohort = self._apply_inclusion_criteria(cohort)
        self._log_filter("cohort total", before, len(cohort))
        return cohort.reset_index(drop=True)

    def _attach_patients(self, cohort: pd.DataFrame) -> pd.DataFrame:
        patients = self.sources.load(
            "patients",
            cache=True,
            usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
            parse_dates=["dod"],
        )
        cohort = cohort.merge(patients, on="subject_id", how="left")

        cohort["M"] = (cohort["gender"].astype(str).str.upper() == "M").astype(float)
        cohort.loc[~cohort["gender"].astype(str).str.upper().isin(["M", "F"]), "M"] = np.nan

        # MIMIC-IV ages are anchored: `anchor_age` is the age in `anchor_year`,
        # and admission years are shifted per patient. Adding the elapsed years
        # since the anchor recovers the age at this ICU admission.
        elapsed_years = cohort["intime"].dt.year - cohort["anchor_year"]
        cohort["age"] = cohort["anchor_age"] + elapsed_years
        cohort.loc[cohort["age"] <= 0, "age"] = np.nan
        return cohort

    def _attach_admissions(self, cohort: pd.DataFrame) -> pd.DataFrame:
        admissions = self.sources.load(
            "admissions",
            cache=True,
            usecols=[
                "subject_id", "hadm_id", "admittime", "dischtime",
                "deathtime", "hospital_expire_flag",
            ],
            parse_dates=["admittime", "dischtime", "deathtime"],
        )
        cohort = cohort.merge(admissions, on=["subject_id", "hadm_id"], how="left")

        death_time = cohort["deathtime"].fillna(cohort["dod"])
        cohort["death"] = death_time.notna().astype(int)
        cohort["death_in_ICU"] = (
            death_time.notna()
            & (death_time >= cohort["intime"])
            & (death_time <= cohort["outtime"])
        ).astype(int)

        horizon = pd.Timedelta(days=self.config.cohort.death_observation_days)
        cohort["death_out_ICU"] = (
            death_time.notna()
            & (death_time > cohort["outtime"])
            & (death_time <= cohort["outtime"] + horizon)
        ).astype(int)
        return cohort

    def _label_readmissions(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Flag a subsequent ICU admission within the configured window.

        Vectorised replacement for the loop whose accumulator was reset on
        every iteration, which is why the original labels were all zero.
        """
        window = pd.Timedelta(
            days=self.config.cohort.readmission_observation_days
        )

        ordered = cohort.sort_values(["subject_id", "intime"])
        grouped = ordered.groupby("subject_id", sort=False)

        ordered["readmission_count"] = grouped.cumcount()
        gap = grouped["intime"].shift(-1) - ordered["outtime"]

        ordered["readmission"] = ((gap >= pd.Timedelta(0)) & (gap <= window)).astype(int)
        ordered["discharge_fail"] = ordered["readmission"]
        ordered["days_to_readmission"] = gap.dt.total_seconds() / 86400.0

        self.report["readmissions_flagged"] = int(ordered["readmission"].sum())
        repeated_patients = len(ordered) - ordered["subject_id"].nunique()
        if repeated_patients > 0 and ordered["readmission"].sum() == 0:
            logger.warning(
                "No readmissions detected across %d stays; the discharge task "
                "would have a constant constraint cost.", len(ordered),
            )
        return ordered.sort_index()

    def _label_ventilation(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Derive intubation, extubation and reintubation from procedureevents."""
        procedures = self.sources.load(
            "procedureevents",
            cache=True,
            usecols=["stay_id", "itemid", "starttime", "endtime", "patientweight"],
            parse_dates=["starttime", "endtime"],
        )

        invasive = procedures[
            procedures["itemid"] == MIMIC_PROCEDURE_ITEMIDS.INVASIVE_VENTILATION
        ].copy()
        self.report["invasive_ventilation_events"] = int(len(invasive))

        first = (
            invasive.sort_values(["stay_id", "starttime"])
            .groupby("stay_id", as_index=False)
            .agg(
                intubation_time=("starttime", "first"),
                extubation_time=("endtime", "first"),
                ventilation_events=("starttime", "size"),
            )
        )
        cohort = cohort.merge(first, on="stay_id", how="left")

        # Reintubation: a later invasive period beginning within the failure
        # window of the first one ending.
        window = pd.Timedelta(
            days=self.config.cohort.reintubation_observation_days
        )
        later = invasive.merge(
            first[["stay_id", "extubation_time"]], on="stay_id", how="left"
        )
        gap = later["starttime"] - later["extubation_time"]
        reintubated = later.loc[
            (gap > pd.Timedelta(0)) & (gap <= window), "stay_id"
        ].unique()
        cohort["reintubation"] = cohort["stay_id"].isin(set(reintubated)).astype(int)
        cohort["extubation_fail"] = cohort["reintubation"]
        self.report["reintubations_flagged"] = int(cohort["reintubation"].sum())

        for itemid, flag in (
            (MIMIC_PROCEDURE_ITEMIDS.TRACHEOSTOMY, "has_tracheostomy"),
            (MIMIC_PROCEDURE_ITEMIDS.NON_INVASIVE_VENTILATION, "had_niv"),
        ):
            stays = set(procedures.loc[procedures["itemid"] == itemid, "stay_id"])
            cohort[flag] = cohort["stay_id"].isin(stays).astype(int)

        unplanned = set(
            procedures.loc[
                procedures["itemid"].isin(MIMIC_PROCEDURE_ITEMIDS.UNPLANNED), "stay_id"
            ]
        )
        cohort["unplanned_extubation"] = cohort["stay_id"].isin(unplanned).astype(int)

        cohort["ventilation_hours"] = (
            cohort["extubation_time"] - cohort["intubation_time"]
        ).dt.total_seconds() / SECONDS_PER_HOUR
        cohort["intubation_hours"] = (
            cohort["intubation_time"] - cohort["intime"]
        ).dt.total_seconds() / SECONDS_PER_HOUR
        cohort["extubation_hours"] = (
            cohort["extubation_time"] - cohort["intime"]
        ).dt.total_seconds() / SECONDS_PER_HOUR

        # `patientweight` on the ventilation order is often the only recorded
        # weight for short stays.
        weights = (
            procedures.dropna(subset=["patientweight"])
            .groupby("stay_id", as_index=False)["patientweight"]
            .median()
            .rename(columns={"patientweight": "weight"})
        )
        cohort = cohort.merge(weights, on="stay_id", how="left")
        return cohort

    def _apply_inclusion_criteria(self, cohort: pd.DataFrame) -> pd.DataFrame:
        criteria = self.config.cohort

        step = len(cohort)
        cohort = cohort[cohort["los"] > 0]
        self._log_filter("positive length of stay", step, len(cohort))

        if criteria.min_age is not None:
            step = len(cohort)
            cohort = cohort[cohort["age"].isna() | (cohort["age"] >= criteria.min_age)]
            self._log_filter("age >= %s" % criteria.min_age, step, len(cohort))

        if criteria.icu_units:
            step = len(cohort)
            cohort = cohort[cohort["first_careunit"].isin(criteria.icu_units)]
            self._log_filter("ICU unit restriction", step, len(cohort))

        if criteria.los_threshold is not None:
            step = len(cohort)
            cohort = cohort[cohort["los"] <= criteria.los_threshold]
            self._log_filter(
                "length of stay <= %s d" % criteria.los_threshold, step, len(cohort)
            )

        if criteria.readmission_count_threshold is not None:
            step = len(cohort)
            cohort = cohort[
                cohort["readmission_count"] < criteria.readmission_count_threshold
            ]
            self._log_filter("readmission count cap", step, len(cohort))

        if self.config.task == Task.EXTUBATION:
            cohort = self._apply_extubation_criteria(cohort)

        return cohort

    def _apply_extubation_criteria(self, cohort: pd.DataFrame) -> pd.DataFrame:
        criteria = self.config.cohort

        step = len(cohort)
        cohort = cohort[cohort["intubation_time"].notna()]
        self._log_filter("invasively ventilated", step, len(cohort))

        step = len(cohort)
        cohort = cohort[cohort["extubation_time"].notna()]
        self._log_filter("ventilation period closed", step, len(cohort))

        step = len(cohort)
        cohort = cohort[cohort["has_tracheostomy"] == 0]
        self._log_filter("no tracheostomy", step, len(cohort))

        # Self-extubation is not the decision being modelled: the clinician
        # never chose to extubate, so the transition carries no policy signal.
        step = len(cohort)
        cohort = cohort[cohort["unplanned_extubation"] == 0]
        self._log_filter("no unplanned extubation", step, len(cohort))

        step = len(cohort)
        cohort = cohort[cohort["ventilation_hours"] > 0]
        self._log_filter("positive ventilation duration", step, len(cohort))

        if criteria.min_ventilation_hours is not None:
            step = len(cohort)
            cohort = cohort[
                cohort["ventilation_hours"] >= criteria.min_ventilation_hours
            ]
            self._log_filter(
                "ventilated >= %s h" % criteria.min_ventilation_hours, step, len(cohort)
            )
        return cohort

    # -- observations ---------------------------------------------------------

    def extract_observations(
        self, cohort: CohortFrame, mapping: VariableMapping
    ) -> ObservationFrame:
        """Stream ``chartevents``, keeping only cohort stays and mapped itemids."""
        keep_stays = set(cohort[EPISODE_KEY])
        intimes = cohort.set_index(EPISODE_KEY)["intime"]
        reverse = mapping.id_to_canonical()
        if not reverse:
            raise ValueError(
                "No canonical variable resolved to a MIMIC-IV itemid; check "
                "the mapping report before continuing."
            )
        wanted_ids = set(reverse)

        collected: List[pd.DataFrame] = []
        rows_seen = 0
        for chunk in self.sources.iter_chunks(
            "chartevents",
            chunk_size=self.config.chunk_size,
            usecols=["stay_id", "itemid", "charttime", "valuenum"],
            parse_dates=["charttime"],
        ):
            rows_seen += len(chunk)
            keep = chunk[
                chunk["stay_id"].isin(keep_stays)
                & chunk["itemid"].isin(wanted_ids)
                & chunk["valuenum"].notna()
            ]
            if len(keep):
                collected.append(keep)

        self.report["chartevents_rows_scanned"] = rows_seen
        if not collected:
            raise ValueError(
                "No chartevents rows matched the cohort and the resolved "
                "itemids."
            )

        events = pd.concat(collected, ignore_index=True)
        events = events.rename(columns={"itemid": "source_id", "valuenum": "value"})
        events["variable"] = events["source_id"].map(reverse)
        events = self._normalise_units(events)

        events[TIME_OFFSET_COLUMN] = (
            events["charttime"] - events[EPISODE_KEY].map(intimes)
        ).dt.total_seconds() / SECONDS_PER_HOUR

        before = len(events)
        events = events[events[TIME_OFFSET_COLUMN] >= 0]
        if before - len(events):
            self.report["observations_before_icu"] = int(before - len(events))

        self.report["chartevent_rows_kept"] = int(len(events))
        return events[
            [EPISODE_KEY, TIME_OFFSET_COLUMN, "variable", "value", "source_id"]
        ].reset_index(drop=True)

    @staticmethod
    def _normalise_units(events: pd.DataFrame) -> pd.DataFrame:
        """Fold the Fahrenheit temperature item onto Celsius.

        MIMIC-IV charts temperature under two itemids with different units.
        Both map to the canonical ``Temperature C``, so without this the two
        populations would be averaged together into a meaningless number.
        """
        fahrenheit = events["source_id"] == 223761
        if fahrenheit.any():
            events.loc[fahrenheit, "value"] = (
                events.loc[fahrenheit, "value"] - 32.0
            ) * 5.0 / 9.0
        fio2 = events["variable"] == "Inspired O2 Fraction"
        as_fraction = fio2 & (events["value"] > 0) & (events["value"] <= 1.5)
        if as_fraction.any():
            events.loc[as_fraction, "value"] *= 100.0
        return events

    # -- actions --------------------------------------------------------------

    def action_events(self, cohort: CohortFrame) -> pd.DataFrame:
        if self.config.task == Task.EXTUBATION:
            action_hours = cohort["extubation_hours"].to_numpy(dtype=float)
        else:
            action_hours = (
                (cohort["outtime"] - cohort["intime"]).dt.total_seconds()
                / SECONDS_PER_HOUR
            ).to_numpy(dtype=float)

        return pd.DataFrame(
            {
                EPISODE_KEY: cohort[EPISODE_KEY].to_numpy(),
                TIME_OFFSET_COLUMN: action_hours,
            }
        )
