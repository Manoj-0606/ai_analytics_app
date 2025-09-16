# app/services.py
"""
Data loading + KPI calculation + quality checks.

Functions:
- load_data(): returns (df, warnings) where warnings is a list of strings describing issues
- calculate_kpis(): returns dict with KPIs and 'warnings' list (so API can surface them)
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd

LOG = logging.getLogger("app.services")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    # basic console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOG.addHandler(ch)


DEFAULT_PATH = Path("data/cloud_spend.csv")

# columns we expect in the richer schema
EXPECTED_COLUMNS = [
    "month",         # YYYY-MM string
    "service",       # service name
    "cost",          # numeric cost
    "account_id",    # optional
    "subscription",  # optional
    "resource_id",   # optional
    "region",        # optional
    "tags",          # optional: comma-separated or JSON-like string
]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected columns present; create empty defaults if missing."""
    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    # normalize types: cost -> numeric
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0).astype(float)
    # ensure month and service are strings
    df["month"] = df["month"].astype(str).fillna("")
    df["service"] = df["service"].astype(str).fillna("")
    return df[EXPECTED_COLUMNS]  # keep consistent column order


def _quality_checks(df: pd.DataFrame) -> List[str]:
    """
    Run quality checks and return list of warning strings.
    Checks implemented:
      1) Missing values in critical columns (month, service, cost)
      2) Duplicate rows (exact duplicates)
      3) Negative costs or suspicious zero costs counts
    """
    warnings: List[str] = []

    # 1) missing / blank values
    missing_month = df["month"].isna() | (df["month"].astype(str).str.strip() == "")
    missing_service = df["service"].isna() | (df["service"].astype(str).str.strip() == "")
    missing_cost = df["cost"].isna()

    n_missing_month = int(missing_month.sum())
    n_missing_service = int(missing_service.sum())
    n_missing_cost = int(missing_cost.sum())

    if n_missing_month:
        warnings.append(f"{n_missing_month} rows missing 'month' value.")
    if n_missing_service:
        warnings.append(f"{n_missing_service} rows missing 'service' value.")
    if n_missing_cost:
        warnings.append(f"{n_missing_cost} rows with invalid 'cost' value (coerced to NaN).")

    # 2) duplicate rows
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        warnings.append(f"{n_duplicates} duplicate rows found.")

    # 3) negative costs
    n_negative = int((df["cost"] < 0).sum())
    if n_negative:
        warnings.append(f"{n_negative} rows with negative cost detected.")

    # 4) zero-cost suspicious rows (informational)
    n_zero = int((df["cost"] == 0).sum())
    if n_zero:
        warnings.append(f"{n_zero} rows with zero cost (may indicate idle/unbilled resources).")

    # 5) tags missing (optional guidance)
    n_missing_tags = int(df["tags"].isna().sum())
    if n_missing_tags == len(df):
        warnings.append("All rows missing 'tags' column values. Consider adding tags for better analytics.")
    elif n_missing_tags > 0:
        warnings.append(f"{n_missing_tags} rows missing tags.")

    # Log warnings
    if warnings:
        for w in warnings:
            LOG.warning(w)
    else:
        LOG.info("Quality checks passed: no issues found.")

    return warnings


def load_data(path: str | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load CSV data from `path` (defaults to data/cloud_spend.csv).
    Returns: (df, warnings)
    - df: pandas.DataFrame with ensured columns and normalized types
    - warnings: list[str] describing detected data quality issues
    """
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        LOG.warning(f"Data file not found at {p}. Returning empty DataFrame with expected schema.")
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        df = _ensure_columns(df)
        warnings = ["Data file not found; empty dataframe returned."]
        return df, warnings

    try:
        df = pd.read_csv(p)
    except Exception as e:
        LOG.error(f"Failed to read CSV {p}: {e}")
        # return an empty dataframe but include error in warnings
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        df = _ensure_columns(df)
        return df, [f"Failed to read CSV: {e}"]

    # Ensure columns & types
    df = _ensure_columns(df)

    # Run quality checks
    warnings = _quality_checks(df)

    return df, warnings


def calculate_kpis(path: str | None = None) -> Dict:
    """
    Compute KPIs from loaded data.
    Returns a dict:
    {
      "total_spend": int,
      "highest_service": str,
      "lowest_service": str,
      "monthly_trend": [int, ...],
      "service_totals": {service: cost, ...},
      "warnings": [...],  # quality warnings
    }
    """
    df, warnings = load_data(path)

    # total spend
    total_spend = int(df["cost"].sum())

    # group by service and compute totals (safe for empty df)
    service_totals = {}
    if not df.empty and df["service"].notna().any():
        grouped = df.groupby("service")["cost"].sum()
        # convert to python types
        service_totals = {str(k): float(v) for k, v in grouped.items()}
        try:
            highest_service = grouped.idxmax()
            lowest_service = grouped.idxmin()
        except Exception:
            highest_service = None
            lowest_service = None
    else:
        highest_service = None
        lowest_service = None

    # monthly trend: ensure months sorted lexicographically (YYYY-MM works)
    monthly = []
    if not df.empty and df["month"].notna().any():
        monthly_sums = df.groupby("month")["cost"].sum().sort_index()
        monthly = [int(v) for v in monthly_sums.tolist()]

    result = {
        "total_spend": total_spend,
        "highest_service": highest_service or "N/A",
        "lowest_service": lowest_service or "N/A",
        "monthly_trend": monthly,
        "service_totals": service_totals,
        "warnings": warnings,
    }
    return result

def detect_idle_resources(path: str | None = None, idle_months: int = 2, min_monthly_saving: float = 1.0) -> Dict:
    """
    Heuristic detection for idle / underutilized resources.

    - Loads data (or uses provided path)
    - Requires 'resource_id' column to be present & not all-null
    - Finds resources that had non-zero historical costs but have zero cost for the last `idle_months`
    - Estimates monthly saving as the historical average monthly cost (prior to the idle window)

    Returns:
      {
        "idle_resources": [ { resource_id, owner, last_seen_month, est_monthly_saving, history: {month: cost,...} }, ... ],
        "warnings": [...]
      }
    """
    df, warnings = load_data(path)

    if df.empty:
        return {"idle_resources": [], "warnings": warnings + ["Empty dataframe; no resources to analyze."]}

    # ensure resource_id column exists
    if "resource_id" not in df.columns or df["resource_id"].isna().all():
        return {"idle_resources": [], "warnings": warnings + ["No resource_id column present or all resource_id values are missing."]}

    # pivot per resource_id x month
    pivot = df.pivot_table(index="resource_id", columns="month", values="cost", aggfunc="sum").fillna(0)
    if pivot.shape[1] < idle_months + 1:
        return {"idle_resources": [], "warnings": warnings + [f"Not enough months of data to detect idle resources (need > {idle_months})."]}

    # sort months lexicographically (YYYY-MM)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # identify last N months and prior months
    all_months = list(pivot.columns)
    last_months = all_months[-idle_months:]
    prior_months = all_months[:-idle_months]
    if not prior_months:
        return {"idle_resources": [], "warnings": warnings + [f"No prior months available before the last {idle_months} months."]}

    idle_list = []
    for resource_id, row in pivot.iterrows():
        # check all zero in last N months
        last_vals = row[last_months].values
        if all([float(v) == 0.0 for v in last_vals]):
            # compute prior average monthly cost (only if it had any cost historically)
            prior_vals = row[prior_months].values
            prior_sum = float(prior_vals.sum())
            prior_month_count = len(prior_vals)
            if prior_sum <= 0:
                # never had cost historically — skip (cannot estimate savings)
                continue
            prior_avg = prior_sum / prior_month_count
            # only consider if prior_avg above threshold
            if prior_avg >= float(min_monthly_saving):
                # attempt to fetch owner/env/tags from original df for context
                sample_rows = df[df["resource_id"] == resource_id].to_dict(orient="records")
                owner = None
                env = None
                tags = None
                if sample_rows:
                    sr = sample_rows[0]
                    owner = sr.get("owner") or sr.get("tags") or None
                    env = sr.get("env") or None
                    tags = sr.get("tags") or None

                # build a small history sample (recent months)
                recent_months = all_months[-(idle_months + min(6, len(prior_months))):]
                history_sample = {m: float(row[m]) for m in recent_months}

                idle_list.append({
                    "resource_id": resource_id,
                    "owner": owner,
                    "env": env,
                    "last_months_zero": last_months,
                    "prior_months_avg": round(prior_avg, 2),
                    "estimated_monthly_saving": round(prior_avg, 2),
                    "history_sample": history_sample
                })

    if not idle_list:
        return {"idle_resources": [], "warnings": warnings + ["No idle resources detected with the current criteria."]}

    return {"idle_resources": idle_list, "warnings": warnings}
