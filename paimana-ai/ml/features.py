"""
PAIMANA-AI :: Feature engineering
=================================
Single source of truth for how a raw CUF row becomes a model matrix.
Used identically by ml/train.py (fit) and backend/app.py (serve), which is what
prevents train/serve skew.

Two feature SETS are defined so that Technical Dimension (c) of the problem
statement - "how much of the predictive performance is attributable to existing
CUF fields vs additional variables not presently captured in the CUF" - can be
answered by an honest ablation:

    FEATURESET["cuf"]      -> only fields that exist in the PAIMANA Common Upload Form
    FEATURESET["extended"] -> CUF + 7 external signals that MoSPI does NOT collect today
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
CUF_NUMERIC = [
    "original_cost_cr", "original_duration_m", "physical_progress_pct",
    "financial_progress_pct", "elapsed_schedule_pct", "land_acquired_pct",
    "award_lag_months", "num_cost_revisions", "num_packages",
    # engineered from CUF fields only
    "log_cost", "cost_per_month", "progress_gap", "spend_efficiency",
    "sanction_year", "sanction_age_months", "schedule_burn_ratio",
]

CUF_CATEGORICAL = [
    "ministry", "sector", "state", "agency_type", "funding_mode",
    "contract_type", "forest_clearance", "env_clearance",
    "monitoring_frequency", "reason_for_delay",
]

EXTENDED_NUMERIC = [
    "ext_monsoon_stoppage_days", "ext_steel_price_index", "ext_cement_price_index",
    "ext_litigation_cases", "ext_contractor_stress", "ext_row_availability_pct",
    "ext_local_unrest_days",
]

FEATURESET = {
    "cuf":      {"num": CUF_NUMERIC, "cat": CUF_CATEGORICAL},
    "extended": {"num": CUF_NUMERIC + EXTENDED_NUMERIC, "cat": CUF_CATEGORICAL},
}

ASOF = pd.Timestamp("2026-04-30")


# --------------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from raw CUF columns. Pure / idempotent."""
    d = df.copy()

    sanction = pd.to_datetime(d["date_of_sanction"])
    d["sanction_year"] = sanction.dt.year
    d["sanction_age_months"] = ((ASOF - sanction).dt.days / 30.44).round(1)

    d["log_cost"] = np.log1p(d["original_cost_cr"])
    d["cost_per_month"] = (d["original_cost_cr"] / d["original_duration_m"].clip(lower=1)).round(2)

    # The single most diagnostic early-warning feature: physical progress is
    # behind the elapsed share of the sanctioned schedule.
    d["progress_gap"] = (d["physical_progress_pct"] - d["elapsed_schedule_pct"]).round(2)

    # Rupees spent per unit of physical progress delivered (>1 = burning fast)
    d["spend_efficiency"] = (d["financial_progress_pct"] /
                             d["physical_progress_pct"].clip(lower=0.5)).round(3)

    d["schedule_burn_ratio"] = (d["elapsed_schedule_pct"] /
                                d["physical_progress_pct"].clip(lower=0.5)).round(3)
    return d


def make_matrix(df: pd.DataFrame, featureset: str = "extended"):
    """Return (X, feature_names). Categoricals are pandas `category` dtype,
    which LightGBM consumes natively (no one-hot explosion)."""
    spec = FEATURESET[featureset]
    d = engineer(df)
    X = d[spec["num"] + spec["cat"]].copy()
    for c in spec["cat"]:
        X[c] = X[c].astype("category")
    return X, list(X.columns)


def align_categories(X: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """At serve time, force the category levels to exactly match training."""
    for col, cats in reference.items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col], categories=cats)
    return X


# --------------------------------------------------------------------------
def risk_label(cost_overrun_pct, time_overrun_months) -> np.ndarray:
    """
    Four-band supervised risk label used by the Project Risk Scoring Framework.
    Thresholds follow MoSPI flash-report reporting conventions.
    """
    c = np.asarray(cost_overrun_pct, dtype=float)
    t = np.asarray(time_overrun_months, dtype=float)
    out = np.full(len(c), "Low", dtype=object)
    out[(c > 5)  | (t > 3)]   = "Medium"
    out[(c > 20) | (t > 12)]  = "High"
    out[(c > 45) | (t > 30)]  = "Critical"
    return out


RISK_ORDER = ["Low", "Medium", "High", "Critical"]
