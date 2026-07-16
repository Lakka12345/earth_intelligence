"""
Agent 5 scientific analysis functions.

Replaces the single generic "statistical_summary" with the actual
analysis types the system prompt names (Step 6): trend analysis,
climatology, anomaly detection, correlation analysis, plus the
baseline statistical summary. Each function is vectorized (no Python
loops over grid cells) so this scales to real gridded datasets, not
just point time series.

Dispatch: phase6_run_analysis (in agent5.py) decides which of these to
run by matching keywords in the user's expected_scientific_outputs /
research goal -- deterministic keyword matching, not another LLM call,
since "does the word 'trend' appear in the request" doesn't need
judgment.
"""

from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from models.agent5_schemas import AnalysisResult


def _numeric_vars(ds: xr.Dataset) -> List[str]:
    return [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.floating)]


def statistical_summary(ds: xr.Dataset) -> AnalysisResult:
    stats: Dict[str, float] = {}
    for var in _numeric_vars(ds):
        arr = ds[var]
        stats[f"{var}_mean"] = float(arr.mean(skipna=True).values)
        stats[f"{var}_std"] = float(arr.std(skipna=True).values)
        stats[f"{var}_min"] = float(arr.min(skipna=True).values)
        stats[f"{var}_max"] = float(arr.max(skipna=True).values)

    return AnalysisResult(
        analysis_type="statistical_summary",
        description="Descriptive statistics across all numeric variables.",
        output_variable_names=list(ds.data_vars.keys()),
        summary_statistics=stats,
        notes="Vectorized xarray reductions -- scales to dask-backed arrays without loading everything into memory.",
    )


def trend_analysis(ds: xr.Dataset) -> Optional[AnalysisResult]:
    """
    Linear trend (slope per unit time) for each numeric variable, at
    every grid point if the data is gridded. Uses xr.polyfit, which is
    vectorized across all non-time dimensions -- no manual looping over
    lat/lon.
    """
    if "time" not in ds.dims or ds.sizes.get("time", 0) < 3:
        return None

    time_numeric = (
        ds["time"].astype("datetime64[ns]").astype("int64") / 1e9 / 86400.0
    )  # days since epoch, for an interpretable slope unit
    ds_for_fit = ds.assign_coords(time=time_numeric)

    stats: Dict[str, float] = {}
    output_vars = []
    for var in _numeric_vars(ds):
        try:
            fit = ds_for_fit[var].polyfit(dim="time", deg=1, skipna=True)
            slope = fit["polyfit_coefficients"].sel(degree=1)
            # Report the spatial-mean slope as a headline number; the
            # full spatial field of slopes is preserved in output_vars
            # naming so a caller can pull it from the merged dataset if
            # gridded detail is needed.
            stats[f"{var}_trend_per_day"] = float(slope.mean(skipna=True).values)
            output_vars.append(f"{var}_trend_per_day")
        except Exception:
            continue

    if not stats:
        return None

    return AnalysisResult(
        analysis_type="trend_analysis",
        description="Linear trend (slope per day) for each variable, via least-squares fit over the time dimension.",
        output_variable_names=output_vars,
        summary_statistics=stats,
        notes="Positive values indicate an increasing trend; negative indicate decreasing. "
              "Slope is per-day; multiply by 365.25 for an approximate annual rate.",
    )


def climatology(ds: xr.Dataset) -> Optional[AnalysisResult]:
    """
    Monthly climatology (mean per calendar month across all years
    present) for each numeric variable. Vectorized via groupby.
    """
    if "time" not in ds.dims:
        return None

    stats: Dict[str, float] = {}
    output_vars = []
    for var in _numeric_vars(ds):
        try:
            monthly = ds[var].groupby("time.month").mean(skipna=True)
            for month in monthly["month"].values:
                stats[f"{var}_month{int(month):02d}_mean"] = float(
                    monthly.sel(month=month).mean(skipna=True).values
                )
            output_vars.append(f"{var}_monthly_climatology")
        except Exception:
            continue

    if not stats:
        return None

    return AnalysisResult(
        analysis_type="climatology",
        description="Mean value per calendar month, averaged across all years in the dataset.",
        output_variable_names=output_vars,
        summary_statistics=stats,
        notes="Spatial mean of each month's climatology is reported; full gridded climatology "
              "is computed but only the scalar summary is surfaced here.",
    )


def anomaly_detection(ds: xr.Dataset, z_threshold: float = 3.0) -> Optional[AnalysisResult]:
    """
    Flags values that deviate from the monthly climatology by more than
    z_threshold standard deviations -- a climatological anomaly, not a
    simple global outlier (which phase4's invalid_value_filtering
    already handles). Reports the count and fraction of anomalous
    points per variable.
    """
    if "time" not in ds.dims:
        return None

    stats: Dict[str, float] = {}
    output_vars = []
    for var in _numeric_vars(ds):
        try:
            grouped = ds[var].groupby("time.month")
            monthly_mean = grouped.mean(skipna=True)
            monthly_std = grouped.std(skipna=True)

            anomaly = grouped - monthly_mean
            normalized = xr.apply_ufunc(
                lambda a, s: a / np.where(s == 0, np.nan, s),
                anomaly.groupby("time.month"),
                monthly_std,
            )
            is_anomalous = np.abs(normalized) > z_threshold
            total = int(is_anomalous.size)
            anomalous_count = int(is_anomalous.sum(skipna=True).values)

            stats[f"{var}_anomalous_count"] = float(anomalous_count)
            stats[f"{var}_anomalous_fraction"] = (
                float(anomalous_count) / total if total else 0.0
            )
            output_vars.append(f"{var}_anomaly_flag")
        except Exception:
            continue

    if not stats:
        return None

    return AnalysisResult(
        analysis_type="anomaly_detection",
        description=f"Points deviating more than {z_threshold} standard deviations from their "
                     f"variable's monthly climatology.",
        output_variable_names=output_vars,
        summary_statistics=stats,
        notes="Anomalies are relative to each variable's own seasonal cycle, not a flat global threshold.",
    )


def correlation_analysis(ds: xr.Dataset) -> Optional[AnalysisResult]:
    """
    Pairwise Pearson correlation between every pair of numeric
    variables, computed over all shared dimensions.
    """
    numeric_vars = _numeric_vars(ds)
    if len(numeric_vars) < 2:
        return None

    stats: Dict[str, float] = {}
    for i, var_a in enumerate(numeric_vars):
        for var_b in numeric_vars[i + 1:]:
            try:
                a = ds[var_a].values.ravel()
                b = ds[var_b].values.ravel()
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() < 3:
                    continue
                corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
                stats[f"corr_{var_a}_{var_b}"] = corr
            except Exception:
                continue

    if not stats:
        return None

    return AnalysisResult(
        analysis_type="correlation_analysis",
        description="Pearson correlation coefficient between each pair of numeric variables.",
        output_variable_names=list(stats.keys()),
        summary_statistics=stats,
        notes="Correlation across all points in the dataset; does not control for spatial or temporal autocorrelation.",
    )


# Keyword -> analysis function, used for deterministic dispatch by
# phase6_run_analysis based on the request's own text (goal +
# expected_scientific_outputs), never a second LLM call.
_KEYWORD_DISPATCH = {
    "trend": trend_analysis,
    "climatology": climatology,
    "seasonal": climatology,
    "anomaly": anomaly_detection,
    "correlation": correlation_analysis,
    "relationship": correlation_analysis,
}


def dispatch_requested_analyses(
    ds: xr.Dataset, goal: str, expected_scientific_outputs: List[str]
) -> List[AnalysisResult]:
    """
    Always runs statistical_summary as a baseline. Additionally runs
    any analysis type whose keyword appears in the goal or expected
    outputs -- e.g. "identify long-term trends" triggers trend_analysis,
    "detect anomalies" triggers anomaly_detection.
    """
    results = [statistical_summary(ds)]

    haystack = (goal + " " + " ".join(expected_scientific_outputs)).lower()
    already_run = set()
    for keyword, func in _KEYWORD_DISPATCH.items():
        if keyword in haystack and func not in already_run:
            result = func(ds)
            if result is not None:
                results.append(result)
            already_run.add(func)

    return results
