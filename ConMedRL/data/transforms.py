"""Outlier removal, imputation and scaling, with train-only fitting.

Every transform here follows scikit-learn's fit/transform split, and the
pipeline fits them on the *training stays only*. That ordering matters: the
original preprocessing scripts imputed with KNN and fitted a MinMaxScaler over
the whole cohort before splitting, which lets test-set physiology influence
training features. Nothing downstream reports that as an error -- it just
inflates offline evaluation.

Two other corrections are baked in here:

* Carry-forward and interpolation run **within one ICU stay**, never across the
  frame. A global ``ffill`` copies the last patient's last measurement into the
  next patient's first row.
* Implausible values are cleared against per-variable physiological bounds
  before any statistical outlier rule, so a unit mix-up (mmol/L charted as
  mg/dL) cannot widen the IQR enough to hide itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import ImputationConfig, OutlierConfig
from .schema import StateSpaceSchema

__all__ = [
    "clip_to_plausible_range",
    "OutlierFilter",
    "GroupedFiller",
    "StateImputer",
    "StateScaler",
    "compute_coverage",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def compute_coverage(
    frame: pd.DataFrame, columns: Optional[Sequence[str]] = None
) -> Dict[str, float]:
    """Fraction of non-missing values per column, in ``[0, 1]``.

    Feeds :func:`~ConMedRL.data.schema.resolve_state_space`, which is how the
    state space shrinks to fit a database that records fewer signals.
    """
    if len(frame) == 0:
        return {str(c): 0.0 for c in (columns or frame.columns)}
    columns = list(columns) if columns is not None else list(frame.columns)
    out: Dict[str, float] = {}
    for column in columns:
        if column not in frame.columns:
            out[str(column)] = 0.0
        else:
            out[str(column)] = float(frame[column].notna().mean())
    return out


# ---------------------------------------------------------------------------
# Physiological range clipping
# ---------------------------------------------------------------------------


def clip_to_plausible_range(
    frame: pd.DataFrame,
    ranges: Mapping[str, Tuple[float, float]],
    inplace: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Set values outside ``(low, high)`` to ``NaN`` so imputation replaces them.

    Values are *cleared*, not clamped: a heart rate charted as 900 is a
    recording error, and clamping it to 300 would invent a tachycardic patient
    where the honest answer is "unknown".

    Returns the frame and a count of cleared values per column.
    """
    target = frame if inplace else frame.copy()
    cleared: Dict[str, int] = {}

    for column, bounds in ranges.items():
        if column not in target.columns or bounds is None:
            continue
        low, high = bounds
        values = pd.to_numeric(target[column], errors="coerce")
        bad = values.notna() & ((values < low) | (values > high))
        count = int(bad.sum())
        if count:
            values = values.mask(bad)
            cleared[column] = count
        target[column] = values

    if cleared:
        logger.info(
            "Cleared %d implausible value(s) across %d variable(s)",
            sum(cleared.values()), len(cleared),
        )
    return target, cleared


# ---------------------------------------------------------------------------
# Statistical outlier filtering
# ---------------------------------------------------------------------------


@dataclass
class OutlierFilter:
    """Clear statistical outliers using bounds learned from training rows only.

    ``method="range"`` relies purely on the schema's physiological bounds and
    needs no fitting; ``"iqr"`` and ``"zscore"`` learn per-variable cut-offs.
    """

    config: OutlierConfig
    #: Column -> ``(low, high)`` learned during :meth:`fit`.
    bounds_: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, frame: pd.DataFrame, columns: Sequence[str]) -> "OutlierFilter":
        self.bounds_ = {}
        method = self.config.method

        if method in ("none", "range"):
            self.fitted_ = True
            return self

        for column in columns:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue

            if method == "iqr":
                q1, q3 = values.quantile(0.25), values.quantile(0.75)
                iqr = q3 - q1
                if not np.isfinite(iqr) or iqr <= 0:
                    continue
                margin = self.config.iqr_factor * iqr
                self.bounds_[column] = (float(q1 - margin), float(q3 + margin))
            elif method == "zscore":
                mean, std = values.mean(), values.std()
                if not np.isfinite(std) or std <= 0:
                    continue
                margin = self.config.z_threshold * std
                self.bounds_[column] = (float(mean - margin), float(mean + margin))

        self.fitted_ = True
        logger.info(
            "Fitted %s outlier bounds for %d variable(s)", method, len(self.bounds_)
        )
        return self

    def transform(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        if not self.fitted_:
            raise RuntimeError("Call OutlierFilter.fit before transform.")
        if self.config.method == "none" or not self.bounds_:
            return frame, {}
        return clip_to_plausible_range(frame, self.bounds_)

    def fit_transform(
        self, frame: pd.DataFrame, columns: Sequence[str]
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        return self.fit(frame, columns).transform(frame)


# ---------------------------------------------------------------------------
# Within-stay filling
# ---------------------------------------------------------------------------


@dataclass
class GroupedFiller:
    """Carry-forward and interpolation confined to each ICU stay.

    Requires the frame to be sorted by ``(group_key, time)``. Order is
    load-bearing: carry-forward on shuffled rows propagates values backwards in
    time.
    """

    group_key: str = "stay_id"
    forward_fill: bool = True
    backward_fill: bool = False
    interpolate: bool = False

    def transform(
        self, frame: pd.DataFrame, columns: Sequence[str]
    ) -> pd.DataFrame:
        columns = [c for c in columns if c in frame.columns]
        if not columns:
            return frame
        if self.group_key not in frame.columns:
            raise KeyError(
                "GroupedFiller needs the grouping column {0!r}; filling across "
                "patients would leak one stay's physiology into the "
                "next.".format(self.group_key)
            )

        result = frame.copy()
        for column in columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

        grouped = result.groupby(self.group_key, sort=False, group_keys=False)

        if self.interpolate:
            # Interpolate first, so carry-forward only handles the leading and
            # trailing gaps that interpolation cannot bridge.
            result[columns] = grouped[list(columns)].apply(
                lambda block: block.interpolate(method="linear", limit_area="inside")
            )
            grouped = result.groupby(self.group_key, sort=False, group_keys=False)

        if self.forward_fill:
            result[columns] = grouped[list(columns)].ffill()
            grouped = result.groupby(self.group_key, sort=False, group_keys=False)

        if self.backward_fill:
            # This is intentionally opt-in: it copies a future measurement
            # into an earlier decision state and is therefore unsuitable for
            # prospective policy evaluation.
            result[columns] = grouped[list(columns)].bfill()

        return result


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


@dataclass
class StateImputer:
    """Classify columns by missingness, then impute what remains.

    Columns are bucketed exactly as the original pipeline did -- "too sparse",
    "moderately missing", "nearly complete" -- but the KNN model is fitted on
    training rows only and reused for validation and test.
    """

    config: ImputationConfig
    #: Column -> bucket name (``"sparse"`` / ``"moderate"`` / ``"dense"``).
    buckets_: Dict[str, str] = field(default_factory=dict)
    #: Median of each column on the training rows, used as a last resort.
    medians_: Dict[str, float] = field(default_factory=dict)
    knn_columns_: List[str] = field(default_factory=list)
    _knn: Optional[Any] = None
    _knn_scaler: Optional[Any] = None
    fitted_: bool = False

    def classify(
        self, frame: pd.DataFrame, columns: Sequence[str]
    ) -> Dict[str, List[str]]:
        """Bucket ``columns`` by their missing fraction."""
        drop_threshold = self.config.missing_threshold_drop
        knn_threshold = self.config.missing_threshold_knn

        buckets: Dict[str, List[str]] = {"sparse": [], "moderate": [], "dense": []}
        for column in columns:
            if column not in frame.columns:
                buckets["sparse"].append(column)
                continue
            missing = float(frame[column].isna().mean())
            if missing > drop_threshold:
                bucket = "sparse"
            elif missing > knn_threshold:
                bucket = "moderate"
            else:
                bucket = "dense"
            buckets[bucket].append(column)
            self.buckets_[column] = bucket
        return buckets

    def fit(self, frame: pd.DataFrame, columns: Sequence[str]) -> "StateImputer":
        columns = [c for c in columns if c in frame.columns]
        self.classify(frame, columns)

        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        self.medians_ = {
            column: (
                float(numeric[column].median())
                if numeric[column].notna().any()
                else 0.0
            )
            for column in columns
        }

        if not self.config.knn_impute:
            self.fitted_ = True
            return self

        # KNN needs enough observed structure to be meaningful; restrict it to
        # columns that are not overwhelmingly missing.
        self.knn_columns_ = [
            c for c in columns if self.buckets_.get(c) in ("moderate", "dense")
        ]
        if len(self.knn_columns_) < 2 or numeric[self.knn_columns_].dropna().empty:
            logger.info("Not enough observed structure for KNN; using medians.")
            self.knn_columns_ = []
            self.fitted_ = True
            return self

        try:
            from sklearn.impute import KNNImputer
            from sklearn.preprocessing import MinMaxScaler
        except ImportError:
            logger.warning("scikit-learn is unavailable; falling back to medians.")
            self.knn_columns_ = []
            self.fitted_ = True
            return self

        # Distances must be computed on comparable scales, otherwise a variable
        # measured in thousands dominates every neighbourhood.
        self._knn_scaler = MinMaxScaler()
        fit_block = numeric[self.knn_columns_]
        max_rows = int(self.config.knn_fit_max_rows)
        if len(fit_block) > max_rows:
            # A deterministic reference sample bounds KNN's O(query*reference)
            # distance cost without learning from validation/test patients.
            fit_block = fit_block.sample(
                n=max_rows, random_state=0, replace=False
            ).sort_index()
        scaled = self._knn_scaler.fit_transform(fit_block)

        self._knn = KNNImputer(
            n_neighbors=self.config.knn_neighbors, weights="uniform"
        )
        self._knn.fit(scaled)

        self.fitted_ = True
        logger.info(
            "Fitted KNN imputer (k=%d) over %d variable(s), %d reference row(s)",
            self.config.knn_neighbors, len(self.knn_columns_), len(fit_block),
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call StateImputer.fit before transform.")

        result = frame.copy()
        columns = [c for c in self.medians_ if c in result.columns]
        for column in columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

        if self._knn is not None and self.knn_columns_:
            present = [c for c in self.knn_columns_ if c in result.columns]
            if present == self.knn_columns_:
                result[self.knn_columns_] = self._knn_chunked(
                    result[self.knn_columns_]
                )
            else:
                logger.warning(
                    "Skipping KNN imputation: expected %d fitted column(s) but "
                    "found %d.", len(self.knn_columns_), len(present),
                )

        # Median fill for whatever KNN did not cover (sparse columns, and any
        # row whose neighbours were themselves all missing).
        for column in columns:
            if result[column].isna().any():
                result[column] = result[column].fillna(self.medians_.get(column, 0.0))

        return result

    def _knn_chunked(self, block: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted KNN imputer in row chunks to bound peak memory."""
        missing = block.isna().any(axis=1)
        if not missing.any():
            return block.copy()
        query = block.loc[missing]
        scaled = self._knn_scaler.transform(query)
        chunk = max(1, int(self.config.knn_chunk_size))
        pieces: List[np.ndarray] = []
        for start in range(0, scaled.shape[0], chunk):
            pieces.append(self._knn.transform(scaled[start : start + chunk]))
        imputed_scaled = np.vstack(pieces) if pieces else scaled
        restored = self._knn_scaler.inverse_transform(imputed_scaled)
        result = block.copy()
        result.loc[missing, :] = pd.DataFrame(
            restored, index=query.index, columns=block.columns
        )
        return result

    def fit_transform(
        self, frame: pd.DataFrame, columns: Sequence[str]
    ) -> pd.DataFrame:
        return self.fit(frame, columns).transform(frame)

    @property
    def sparse_columns(self) -> List[str]:
        return [c for c, bucket in self.buckets_.items() if bucket == "sparse"]


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


@dataclass
class StateScaler:
    """Min-max scaling of the state variables, fitted on training rows only.

    The fitted object is exported alongside the dataset because the decision
    support application has to scale a live patient's vitals with exactly the
    same bounds the policy was trained on.
    """

    columns: List[str] = field(default_factory=list)
    scaler: Optional[Any] = None
    clip: bool = True

    def fit(self, frame: pd.DataFrame, columns: Sequence[str]) -> "StateScaler":
        from sklearn.preprocessing import MinMaxScaler

        self.columns = [c for c in columns if c in frame.columns]
        self.scaler = MinMaxScaler()
        self.scaler.fit(frame[self.columns].apply(pd.to_numeric, errors="coerce"))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise RuntimeError("Call StateScaler.fit before transform.")
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise KeyError(
                "Cannot scale: column(s) missing from the frame: {0}".format(
                    ", ".join(missing)
                )
            )

        block = frame[self.columns].apply(pd.to_numeric, errors="coerce")
        scaled = self.scaler.transform(block)
        if self.clip:
            # Validation and test rows can fall outside the training range;
            # clipping keeps every state inside the [0, 1] box the networks and
            # the terminal state assume.
            scaled = np.clip(scaled, 0.0, 1.0)
        return pd.DataFrame(scaled, index=frame.index, columns=self.columns)

    def fit_transform(
        self, frame: pd.DataFrame, columns: Sequence[str]
    ) -> pd.DataFrame:
        return self.fit(frame, columns).transform(frame)
