# app/services.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd

LOG = logging.getLogger("app.services")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOG.addHandler(ch)

DEFAULT_PATH = Path("data/cloud_spend.csv")

EXPECTED_COLUMNS = [
    "month", "service", "cost", "account_id", "subscription",
    "resource_id", "region", "tags",
]

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0).astype(float)
    df["month"] = df["month"].astype(str).fillna("")
    df["service"] = df["service"].astype(str).fillna("")
    return df[EXPECTED_COLUMNS]

def _quality_checks(df: pd.DataFrame) -> List[str]:
    warnings: List[str] = []

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

    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        warnings.append(f"{n_duplicates} duplicate rows found.")

    n_negative = int((df["cost"] < 0).sum())
    if n_negative:
        warnings.append(f"{n_negative} rows with negative cost detected.")

    n_zero = int((df["cost"] == 0).sum())
    if n_zero:
        warnings.append(f"{n_zero} rows with zero cost (may indicate idle/unbilled resources).")

    n_missing_tags = int(df["tags"].isna().sum())
    if n_missing_tags == len(df):
        warnings.append("All rows missing 'tags' column values.")
    elif n_missing_tags > 0:
        warnings.append(f"{n_missing_tags} rows missing tags.")

    if warnings:
        for w in warnings:
            LOG.warning(w)
    else:
        LOG.info("Quality checks passed.")

    return warnings

def load_data(path: str | None = None) -> Tuple[pd.DataFrame, List[str]]:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        LOG.warning(f"Data file not found at {p}. Returning empty DataFrame.")
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        df = _ensure_columns(df)
        warnings = ["Data file not found; empty dataframe returned."]
        return df, warnings

    try:
        df = pd.read_csv(p)
    except Exception as e:
        LOG.error(f"Failed to read CSV {p}: {e}")
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        df = _ensure_columns(df)
        return df, [f"Failed to read CSV: {e}"]

    df = _ensure_columns(df)
    warnings = _quality_checks(df)
    return df, warnings

def calculate_kpis(path: str | None = None) -> Dict:
    df, warnings = load_data(path)
    total_spend = int(df["cost"].sum())

    service_totals = {}
    if not df.empty and df["service"].notna().any():
        grouped = df.groupby("service")["cost"].sum()
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
    df, warnings = load_data(path)
    if df.empty:
        return {"idle_resources": [], "warnings": warnings + ["Empty dataframe; no resources to analyze."]}

    if "resource_id" not in df.columns or df["resource_id"].isna().all():
        return {"idle_resources": [], "warnings": warnings + ["No resource_id column present or all resource_id values are missing."]}

    pivot = df.pivot_table(index="resource_id", columns="month", values="cost", aggfunc="sum").fillna(0)
    if pivot.shape[1] < idle_months + 1:
        return {"idle_resources": [], "warnings": warnings + [f"Not enough months of data to detect idle resources (need > {idle_months})."]}

    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    all_months = list(pivot.columns)
    last_months = all_months[-idle_months:]
    prior_months = all_months[:-idle_months]
    if not prior_months:
        return {"idle_resources": [], "warnings": warnings + [f"No prior months available before the last {idle_months} months."]}

    idle_list = []
    for resource_id, row in pivot.iterrows():
        last_vals = row[last_months].values
        if all([float(v) == 0.0 for v in last_vals]):
            prior_vals = row[prior_months].values
            prior_sum = float(prior_vals.sum())
            prior_month_count = len(prior_vals)
            if prior_sum <= 0:
                continue
            prior_avg = prior_sum / prior_month_count
            if prior_avg >= float(min_monthly_saving):
                sample_rows = df[df["resource_id"] == resource_id].to_dict(orient="records")
                owner = None
                env = None
                tags = None
                if sample_rows:
                    sr = sample_rows[0]
                    owner = sr.get("owner") or sr.get("tags") or None
                    env = sr.get("env") or None
                    tags = sr.get("tags") or None

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
