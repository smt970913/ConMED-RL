"""SICdb (Salzburg Intensive Care database) adapter.

Written against the real schema rather than adapted from the MIMIC-IV code,
because the two databases share almost nothing structurally:

============  ==================================  ============================
concept       MIMIC-IV                            SICdb
============  ==================================  ============================
episode       ``icustays.stay_id``                ``cases.CaseID``
patient       ``subject_id``                      ``PatientID``
time          wall-clock ``charttime``            integer seconds from PDMS
                                                  admission
physiology    ``chartevents`` (long, per event)   ``data_float_h`` (already
                                                  aggregated to the hour)
labs          ``chartevents`` itemids             ``laboratory`` table with its
                                                  own ``LaboratoryID`` space
intubation    ``procedureevents`` itemid 225792   ``data_range`` DataID 720,
                                                  with start *and* end offsets
============  ==================================  ============================

The time axis needs care. SICdb offsets are seconds since the *PDMS* admission,
which for surgical patients begins in theatre, before the ICU. ``cases.ICUOffset``
is when the patient actually reached the ICU, so this adapter reports
``(Offset - ICUOffset) / 3600`` and genuinely negative values -- pre-ICU
theatre measurements and labs drawn on the ward -- are dropped rather than
folded into the first epoch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from ..base import (
    EPISODE_KEY,
    TIME_OFFSET_COLUMN,
    AdapterResult,
    BaseAdapter,
    CohortFrame,
    ObservationFrame,
)
from ..config import Task
from ..variables import (
    SICDB_MEASUREMENT_GROUPS,
    SICDB_UNIT_CONVERSIONS,
    VariableMapping,
)

__all__ = ["SICdbAdapter", "SICDB_REFERENCES"]

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0


class SICDB_REFERENCES:
    """Reference ids read from a real ``d_references`` (v1.0.8).

    Grouped as a namespace rather than scattered as literals so that a SICdb
    version bump has one place to check.
    """

    # cases.Sex
    SEX_MALE = 735
    SEX_FEMALE = 736

    # cases.DischargeState
    DISCHARGE_ALIVE = 2202  # 'lebend'
    DISCHARGE_DEAD = 2215  # 'verstorben'
    DISCHARGE_UNKNOWN = 2212

    # cases.HospitalDischargeType
    HOSPITAL_DECEASED = 2028
    HOSPITAL_DEATH_CASE = 3130  # 'Sterbefall'

    # data_range.DataID -- airway devices, carrying Offset and OffsetEnd
    ENDOTRACHEAL_TUBE = 720
    TRACHEAL_CANNULA = 3041

    # cases.HospitalUnit -- the two wards that perform invasive ventilation.
    # INIC (2) and INID (5) are intermediate care; INID supports non-invasive
    # ventilation only, so neither belongs in an extubation cohort.
    ICU_UNITS = (3, 4)  # CWIN, INBD
    INTERMEDIATE_UNITS = (2, 5)  # INIC, INID

    #: ``HospitalDischargeDay`` uses -1 where the value is unknown.
    MISSING_SENTINEL = -1


class SICdbAdapter(BaseAdapter):
    """Build a cohort and physiology table from a local SICdb download."""

    database = "sicdb"
    dictionary_groups = SICDB_MEASUREMENT_GROUPS
    target_languages = ("German",)

    def required_tables(self) -> List[str]:
        tables = ["cases", "d_references", "data_float_h", "laboratory"]
        if self.config.task == Task.EXTUBATION:
            tables.append("data_range")
        return tables

    def load_dictionary(self) -> pd.DataFrame:
        # `d_references` is mixed-encoding: the bulk is UTF-8 but a handful of
        # fields carry raw cp1252 bytes. Decoding as UTF-8 with replacement
        # keeps the German umlauts (which matter for name matching) intact and
        # damages only the degree sign, which unit normalisation strips anyway.
        # Decoding as cp1252 would do the reverse and mangle every umlaut.
        return self.sources.load(
            "d_references", cache=True, encoding="utf-8", encoding_errors="replace"
        )

    # -- cohort ---------------------------------------------------------------

    def build_cohort(self) -> CohortFrame:
        cases = self.sources.load("cases", cache=True)
        before = len(cases)
        self.report["source_cases"] = before

        cohort = cases.rename(
            columns={"CaseID": EPISODE_KEY, "PatientID": "subject_id"}
        ).copy()

        # Time zero and duration, both in canonical units.
        cohort["icu_offset_seconds"] = cohort["ICUOffset"].astype(float)
        cohort["los"] = cohort["TimeOfStay"].astype(float) / SECONDS_PER_DAY

        cohort = self._label_demographics(cohort)
        cohort = self._label_mortality(cohort)
        cohort = self._label_readmissions(cohort)

        if self.config.task == Task.EXTUBATION:
            cohort = self._label_ventilation(cohort)

        cohort = self._apply_inclusion_criteria(cohort)
        self._log_filter("cohort total", before, len(cohort))
        return cohort.reset_index(drop=True)

    def _label_demographics(self, cohort: pd.DataFrame) -> pd.DataFrame:
        cohort["M"] = (cohort["Sex"] == SICDB_REFERENCES.SEX_MALE).astype(float)
        cohort.loc[
            ~cohort["Sex"].isin([SICDB_REFERENCES.SEX_MALE, SICDB_REFERENCES.SEX_FEMALE]),
            "M",
        ] = np.nan

        cohort["age"] = pd.to_numeric(cohort["AgeOnAdmission"], errors="coerce")
        cohort["weight"] = pd.to_numeric(cohort["WeightOnAdmission"], errors="coerce")
        cohort["height"] = pd.to_numeric(cohort["HeightOnAdmission"], errors="coerce")

        # Zero is SICdb's "not recorded" for the anthropometrics, and a 0 kg
        # patient would otherwise survive every plausibility check downstream
        # because the range test only sees a number.
        for column in ("weight", "height", "age"):
            cohort.loc[cohort[column] <= 0, column] = np.nan
        return cohort

    def _label_mortality(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Death inside the ICU stay versus after it.

        ``OffsetOfDeath`` is on the PDMS clock like every other offset, so it
        is comparable with ``ICUOffset + TimeOfStay``.
        """
        died = cohort["DischargeState"] == SICDB_REFERENCES.DISCHARGE_DEAD
        death_offset = pd.to_numeric(cohort["OffsetOfDeath"], errors="coerce")
        icu_end = cohort["icu_offset_seconds"] + cohort["TimeOfStay"].astype(float)

        cohort["death"] = died.astype(int)
        cohort["death_in_ICU"] = (died & (death_offset <= icu_end)).astype(int)

        horizon_seconds = self.config.cohort.death_observation_days * SECONDS_PER_DAY
        cohort["death_out_ICU"] = (
            died
            & (death_offset > icu_end)
            & (death_offset <= icu_end + horizon_seconds)
        ).astype(int)

        # A death flagged without a usable offset cannot be placed in time; the
        # in/out split is unknown, so only the overall flag is trustworthy.
        unplaceable = int((died & death_offset.isna()).sum())
        if unplaceable:
            self.report["deaths_without_offset"] = unplaceable
            logger.warning(
                "%d case(s) are marked deceased but carry no OffsetOfDeath; "
                "they count toward `death` but not toward death_in_ICU.",
                unplaceable,
            )
        return cohort

    def _label_readmissions(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Rank a patient's ICU stays and flag readmission within the window.

        The original code reset its accumulator inside the loop, so every
        readmission flag came out zero. This is a vectorised replacement whose
        result is checked in the report.
        """
        window_seconds = (
            self.config.cohort.readmission_observation_days * SECONDS_PER_DAY
        )

        # `OffsetAfterFirstAdmission` places every case of a patient on one
        # timeline, which is the only way to order them: CaseID is not
        # chronological.
        ordered = cohort.sort_values(["subject_id", "OffsetAfterFirstAdmission"])
        grouped = ordered.groupby("subject_id", sort=False)

        ordered["readmission_count"] = grouped.cumcount()
        stay_end = (
            ordered["OffsetAfterFirstAdmission"]
            + ordered["icu_offset_seconds"]
            + ordered["TimeOfStay"].astype(float)
        )
        icu_start = (
            ordered["OffsetAfterFirstAdmission"] + ordered["icu_offset_seconds"]
        )
        next_start = icu_start.groupby(ordered["subject_id"], sort=False).shift(-1)
        gap = next_start - stay_end

        ordered["readmission"] = ((gap >= 0) & (gap <= window_seconds)).astype(int)
        # A stay that is followed by a prompt readmission is a failed discharge.
        ordered["discharge_fail"] = ordered["readmission"]
        ordered["days_to_readmission"] = gap / SECONDS_PER_DAY

        self.report["readmissions_flagged"] = int(ordered["readmission"].sum())
        repeated_patients = len(ordered) - ordered["subject_id"].nunique()
        if repeated_patients > 0 and ordered["readmission"].sum() == 0:
            logger.warning(
                "No readmissions were detected. With %d cases across %d "
                "patients this is unlikely and suggests the ordering key is "
                "wrong for this SICdb version.",
                len(ordered), ordered["subject_id"].nunique(),
            )
        return ordered.sort_index()

    def _label_ventilation(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Attach invasive ventilation periods from ``data_range``.

        ``data_range`` is the only place SICdb records device start and end, so
        it -- not a procedure code -- defines intubation and extubation.
        """
        ranges = self.sources.load("data_range", cache=True)
        airway = ranges[
            ranges["DataID"].isin(
                [SICDB_REFERENCES.ENDOTRACHEAL_TUBE, SICDB_REFERENCES.TRACHEAL_CANNULA]
            )
        ].copy()

        tube = airway[airway["DataID"] == SICDB_REFERENCES.ENDOTRACHEAL_TUBE]
        trach = airway[airway["DataID"] == SICDB_REFERENCES.TRACHEAL_CANNULA]
        self.report["intubation_periods"] = int(len(tube))
        self.report["tracheostomy_periods"] = int(len(trach))

        # One row per case: first intubation and its end.
        first = (
            tube.sort_values(["CaseID", "Offset"])
            .groupby("CaseID", as_index=False)
            .agg(
                intubation_offset=("Offset", "first"),
                extubation_offset=("OffsetEnd", "first"),
                intubation_periods=("Offset", "size"),
            )
        )
        cohort = cohort.merge(
            first, left_on=EPISODE_KEY, right_on="CaseID", how="left"
        ).drop(columns=["CaseID"], errors="ignore")

        cohort["has_tracheostomy"] = (
            cohort[EPISODE_KEY].isin(set(trach["CaseID"])).astype(int)
        )

        # Hours relative to ICU admission, matching the observation clock.
        for column in ("intubation_offset", "extubation_offset"):
            cohort[column.replace("_offset", "_hours")] = (
                cohort[column] - cohort["icu_offset_seconds"]
            ) / SECONDS_PER_HOUR

        cohort["ventilation_hours"] = (
            cohort["extubation_offset"] - cohort["intubation_offset"]
        ) / SECONDS_PER_HOUR

        # Reintubation: a later tube period starting within the failure window
        # of the first one ending.
        window_seconds = (
            self.config.cohort.reintubation_observation_days * SECONDS_PER_DAY
        )
        later = tube.merge(
            first[["CaseID", "extubation_offset"]], on="CaseID", how="left"
        )
        gap = later["Offset"] - later["extubation_offset"]
        reintubated = later.loc[(gap > 0) & (gap <= window_seconds), "CaseID"].unique()

        cohort["reintubation"] = cohort[EPISODE_KEY].isin(set(reintubated)).astype(int)
        cohort["extubation_fail"] = cohort["reintubation"]
        self.report["reintubations_flagged"] = int(cohort["reintubation"].sum())
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
        cohort = cohort[cohort["intubation_offset"].notna()]
        self._log_filter("invasively ventilated", step, len(cohort))

        step = len(cohort)
        cohort = cohort[cohort["extubation_offset"].notna()]
        self._log_filter("ventilation period closed", step, len(cohort))

        # A tracheostomy is a different weaning problem with a different
        # decision structure, so those stays do not belong in this cohort.
        step = len(cohort)
        cohort = cohort[cohort["has_tracheostomy"] == 0]
        self._log_filter("no tracheostomy", step, len(cohort))

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

        if criteria.icu_units:
            step = len(cohort)
            cohort = cohort[cohort["HospitalUnit"].isin(criteria.icu_units)]
            self._log_filter("ICU unit restriction", step, len(cohort))

        return cohort

    # -- observations ---------------------------------------------------------

    def extract_observations(
        self, cohort: CohortFrame, mapping: VariableMapping
    ) -> ObservationFrame:
        """Read physiology from ``data_float_h`` and ``laboratory``.

        Both tables are streamed and filtered per chunk: ``data_float_h`` is
        ~2 GB gzipped and would not fit comfortably in memory unfiltered.
        """
        keep_cases = set(cohort[EPISODE_KEY])
        offsets = cohort.set_index(EPISODE_KEY)["icu_offset_seconds"]
        reverse = mapping.id_to_canonical()
        if not reverse:
            raise ValueError(
                "No canonical variable resolved to a SICdb reference id; "
                "check the mapping report before continuing."
            )
        wanted_ids = set(reverse)

        frames = [
            self._extract_signals(keep_cases, wanted_ids, reverse),
            self._extract_labs(keep_cases, wanted_ids, reverse),
        ]
        observations = pd.concat([f for f in frames if len(f)], ignore_index=True)
        if observations.empty:
            raise ValueError(
                "No observations survived filtering. The cohort and the "
                "measurement tables may not overlap."
            )

        observations = self._apply_unit_conversions(observations)

        observations[TIME_OFFSET_COLUMN] = (
            observations["Offset"] - observations[EPISODE_KEY].map(offsets)
        ) / SECONDS_PER_HOUR

        before = len(observations)
        observations = observations[observations[TIME_OFFSET_COLUMN] >= 0]
        pre_icu = before - len(observations)
        if pre_icu:
            self.report["observations_before_icu"] = int(pre_icu)
            logger.info(
                "Dropped %d measurement(s) recorded before ICU admission "
                "(theatre and ward samples).", pre_icu,
            )

        return observations[
            [EPISODE_KEY, TIME_OFFSET_COLUMN, "variable", "value", "source_id"]
        ].reset_index(drop=True)

    def _extract_signals(
        self, keep_cases: Set[int], wanted_ids: Set[Any], reverse: Dict[Any, str]
    ) -> pd.DataFrame:
        """Stream ``data_float_h``, keeping only wanted cases and DataIDs.

        ``rawdata`` holds the sub-hourly waveform blob and is many times the
        size of the rest of the row, so it is never read.
        """
        collected: List[pd.DataFrame] = []
        rows_seen = 0
        for chunk in self.sources.iter_chunks(
            "data_float_h",
            chunk_size=self.config.chunk_size,
            usecols=["CaseID", "DataID", "Offset", "Val"],
        ):
            rows_seen += len(chunk)
            keep = chunk[
                chunk["CaseID"].isin(keep_cases) & chunk["DataID"].isin(wanted_ids)
            ]
            if len(keep):
                collected.append(keep)

        self.report["data_float_h_rows_scanned"] = rows_seen
        if not collected:
            return pd.DataFrame()

        signals = pd.concat(collected, ignore_index=True)
        signals = signals.rename(
            columns={"CaseID": EPISODE_KEY, "DataID": "source_id", "Val": "value"}
        )
        signals["variable"] = signals["source_id"].map(reverse)
        self.report["signal_rows"] = int(len(signals))
        return signals

    def _extract_labs(
        self, keep_cases: Set[int], wanted_ids: Set[Any], reverse: Dict[Any, str]
    ) -> pd.DataFrame:
        collected: List[pd.DataFrame] = []
        for chunk in self.sources.iter_chunks(
            "laboratory",
            chunk_size=self.config.chunk_size,
            usecols=["CaseID", "LaboratoryID", "Offset", "LaboratoryValue"],
        ):
            keep = chunk[
                chunk["CaseID"].isin(keep_cases)
                & chunk["LaboratoryID"].isin(wanted_ids)
            ]
            if len(keep):
                collected.append(keep)

        if not collected:
            return pd.DataFrame()

        labs = pd.concat(collected, ignore_index=True)
        labs = labs.rename(
            columns={
                "CaseID": EPISODE_KEY,
                "LaboratoryID": "source_id",
                "LaboratoryValue": "value",
            }
        )
        labs["variable"] = labs["source_id"].map(reverse)
        self.report["laboratory_rows"] = int(len(labs))
        return labs

    @staticmethod
    def _apply_unit_conversions(observations: pd.DataFrame) -> pd.DataFrame:
        """Bring the handful of divergent units onto the canonical scale.

        SICdb reports urea where the canonical variable is BUN, and magnesium
        in mmol/L where the canonical variable is mg/dL. Both look entirely
        plausible unconverted, so no range check downstream would notice.
        """
        for source_id, (scale, offset) in SICDB_UNIT_CONVERSIONS.items():
            rows = observations["source_id"] == source_id
            if rows.any():
                observations.loc[rows, "value"] = (
                    observations.loc[rows, "value"] * scale + offset
                )
        fio2 = observations["variable"] == "Inspired O2 Fraction"
        as_fraction = (
            fio2
            & (observations["value"] > 0)
            & (observations["value"] <= 1.5)
        )
        observations.loc[as_fraction, "value"] *= 100.0
        return observations

    # -- actions --------------------------------------------------------------

    def action_events(self, cohort: CohortFrame) -> pd.DataFrame:
        """When the terminal decision was taken, in hours since ICU admission."""
        if self.config.task == Task.EXTUBATION:
            action_hours = cohort["extubation_hours"]
        else:
            # Discharge happens at the end of the recorded stay.
            action_hours = cohort["los"] * 24.0

        return pd.DataFrame(
            {
                EPISODE_KEY: cohort[EPISODE_KEY].to_numpy(),
                TIME_OFFSET_COLUMN: np.asarray(action_hours, dtype=float),
            }
        )
