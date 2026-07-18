"""
Agent 5 preprocessing executors -- the steps that had no implementation
in the first build. Kept in their own module (rather than growing
agent5.py further) since this is where new step types get added as
they come up.

Each executor takes an xr.Dataset (and any step-specific args) and
returns (new_dataset, human_readable_summary), matching the calling
convention already used by the executors in agent5.py, so both sets of
executors are interchangeable in _STEP_EXECUTORS.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr


# ------------------------------------------------------------------ #
# coordinate_normalization                                             #
# ------------------------------------------------------------------ #

def apply_coordinate_normalization(ds: xr.Dataset) -> Tuple[xr.Dataset, str]:
    """
    Wraps longitude to a single consistent convention (-180..180) and
    sorts lat/lon ascending. Mismatched longitude conventions (0..360
    vs -180..180) are a common, silent cause of merge/alignment
    failures between sources from different providers.
    """
    notes = []
    lon_name = next((n for n in ("lon", "longitude") if n in ds.coords), None)
    lat_name = next((n for n in ("lat", "latitude") if n in ds.coords), None)

    if lon_name is not None:
        lon_vals = ds[lon_name].values
        if np.nanmax(lon_vals) > 180:
            ds = ds.assign_coords({lon_name: ((ds[lon_name] + 180) % 360) - 180})
            ds = ds.sortby(lon_name)
            notes.append(f"Wrapped '{lon_name}' from 0..360 to -180..180 and re-sorted.")

    if lat_name is not None:
        lat_vals = ds[lat_name].values
        if lat_vals.size > 1 and lat_vals[0] > lat_vals[-1]:
            ds = ds.sortby(lat_name)
            notes.append(f"Sorted '{lat_name}' ascending.")

    return ds, "; ".join(notes) if notes else "Coordinates already normalized -- no change needed."


# ------------------------------------------------------------------ #
# crs_transformation                                                   #
# ------------------------------------------------------------------ #

def apply_crs_transformation(ds: xr.Dataset, target_crs: str = "EPSG:4326") -> Tuple[xr.Dataset, str]:
    """
    Reprojects a dataset to target_crs using rioxarray. Only applies if
    the dataset has CRS-aware spatial dims (i.e. rioxarray recognizes
    it) -- datasets without a defined CRS are left unchanged with a
    clear note, rather than guessing a source CRS.
    """
    try:
        import rioxarray  # noqa: F401 -- registers .rio accessor
    except ImportError:
        return ds, "rioxarray not installed -- CRS transformation skipped. Install with: pip install rioxarray"

    try:
        current_crs = ds.rio.crs
    except Exception:
        current_crs = None

    if current_crs is None:
        return ds, "No CRS metadata found on this dataset -- cannot safely reproject. Left unchanged."

    if str(current_crs) == target_crs:
        return ds, f"Already in {target_crs} -- no reprojection needed."

    reprojected = ds.rio.reproject(target_crs)
    return reprojected, f"Reprojected from {current_crs} to {target_crs}."


# ------------------------------------------------------------------ #
# resampling                                                           #
# ------------------------------------------------------------------ #

def apply_resampling(
    ds: xr.Dataset, target_frequency: Optional[str] = None
) -> Tuple[xr.Dataset, str]:
    """
    Resamples the time dimension to a target frequency (e.g. 'D', 'M').
    Distinct from time_alignment in agent5.py (which forces uniform
    spacing) in that this is meant for genuine up/downsampling requests
    -- e.g. daily source data resampled to monthly for a climatology.
    """
    if "time" not in ds.dims:
        return ds, "No time dimension present -- resampling skipped."

    freq = target_frequency or "D"
    resampled = ds.resample(time=freq).mean(skipna=True)
    return resampled, f"Resampled time dimension to '{freq}' frequency (mean aggregation)."


# ------------------------------------------------------------------ #
# interpolation (general -- distinct from missing_value_handling's     #
# narrower gap-filling use in agent5.py)                                #
# ------------------------------------------------------------------ #

def apply_interpolation(
    ds: xr.Dataset, target_coords: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[xr.Dataset, str]:
    """
    Interpolates the dataset onto a new coordinate grid (spatial and/or
    temporal). If target_coords is None, this is a no-op with a clear
    note -- interpolation without a target grid has nothing to do.
    """
    if not target_coords:
        return ds, "No target coordinate grid supplied -- interpolation skipped (nothing to interpolate onto)."

    interpolated = ds.interp(**target_coords, method="linear")
    return interpolated, f"Interpolated onto new coordinates: {list(target_coords.keys())}."


# ------------------------------------------------------------------ #
# spatial_alignment                                                    #
# ------------------------------------------------------------------ #

def apply_spatial_alignment(
    ds: xr.Dataset, reference: xr.Dataset
) -> Tuple[xr.Dataset, str]:
    """
    Aligns ds onto reference's spatial grid via nearest-neighbor
    reindexing -- the standard prerequisite for merging two gridded
    sources with different native resolutions.
    """
    lat_name = next((n for n in ("lat", "latitude") if n in ds.coords and n in reference.coords), None)
    lon_name = next((n for n in ("lon", "longitude") if n in ds.coords and n in reference.coords), None)

    if lat_name is None or lon_name is None:
        return ds, "Could not find matching lat/lon coordinate names between source and reference -- skipped."

    aligned = ds.reindex(
        {lat_name: reference[lat_name], lon_name: reference[lon_name]}, method="nearest"
    )
    return aligned, f"Reindexed onto reference grid via nearest-neighbor on '{lat_name}'/'{lon_name}'."


# ------------------------------------------------------------------ #
# variable_standardization                                             #
# ------------------------------------------------------------------ #

# Common provider-specific variable name aliases -> CF standard names.
# Deliberately small and explicit rather than a fuzzy-matching scheme,
# since silently renaming the wrong variable is worse than leaving an
# unrecognized name alone.
_STANDARD_NAME_ALIASES = {
    "sst": "sea_surface_temperature",
    "temp": "sea_water_temperature",
    "sal": "sea_water_salinity",
    "sss": "sea_surface_salinity",
    "ssh": "sea_surface_height",
    "u10": "eastward_wind",
    "v10": "northward_wind",
    "precip": "precipitation_amount",
    "rr": "precipitation_amount",
    "slp": "air_pressure_at_sea_level",
}


def apply_variable_standardization(ds: xr.Dataset) -> Tuple[xr.Dataset, str]:
    renamed = {}
    for var in list(ds.data_vars):
        key = var.lower().strip()
        if key in _STANDARD_NAME_ALIASES and _STANDARD_NAME_ALIASES[key] != var:
            renamed[var] = _STANDARD_NAME_ALIASES[key]

    if not renamed:
        return ds, "No recognized provider-specific variable aliases found -- names left as-is."

    ds = ds.rename(renamed)
    for old, new in renamed.items():
        ds[new].attrs["standard_name"] = new
        ds[new].attrs["original_variable_name"] = old

    return ds, f"Renamed to CF standard names: {renamed}."


# ------------------------------------------------------------------ #
# derived_variable_computation                                         #
# ------------------------------------------------------------------ #

# Registry of derivable variables: (output_name, required_inputs, fn).
# Extend this as new derived quantities are needed -- deliberately
# explicit rather than a generic formula-eval, so every derivation is
# auditable and dimensionally sound by construction.
def _wind_speed(ds: xr.Dataset) -> xr.DataArray:
    return np.sqrt(ds["eastward_wind"] ** 2 + ds["northward_wind"] ** 2)


_DERIVED_VARIABLE_REGISTRY = {
    "wind_speed": (["eastward_wind", "northward_wind"], _wind_speed),
}


def apply_derived_variable_computation(
    ds: xr.Dataset, requested_variables: Optional[List[str]] = None
) -> Tuple[xr.Dataset, str]:
    requested_variables = requested_variables or list(_DERIVED_VARIABLE_REGISTRY.keys())
    computed = []

    for var_name in requested_variables:
        entry = _DERIVED_VARIABLE_REGISTRY.get(var_name)
        if entry is None:
            continue
        required_inputs, fn = entry
        if not all(inp in ds.data_vars for inp in required_inputs):
            continue
        ds[var_name] = fn(ds)
        computed.append(var_name)

    if not computed:
        return ds, "No derivable variables had all required inputs present -- nothing computed."

    return ds, f"Computed derived variable(s): {', '.join(computed)}."


def apply_feature_extraction(
    ds: xr.Dataset, features: Optional[List[str]] = None
) -> Tuple[xr.Dataset, str]:
    """
    Extracts simple scalar/reduced features (e.g. spatial mean time
    series from a gridded field) as new variables. Conservative first
    implementation: spatial-mean time series for every numeric
    variable that has lat/lon dims, which is the most common feature
    an analysis step downstream actually needs.
    """
    lat_name = next((n for n in ("lat", "latitude") if n in ds.dims), None)
    lon_name = next((n for n in ("lon", "longitude") if n in ds.dims), None)

    if lat_name is None or lon_name is None:
        return ds, "No spatial dimensions found -- feature extraction skipped."

    extracted = []
    for var in list(ds.data_vars):
        if np.issubdtype(ds[var].dtype, np.floating):
            new_name = f"{var}_spatial_mean"
            ds[new_name] = ds[var].mean(dim=[lat_name, lon_name], skipna=True)
            extracted.append(new_name)

    if not extracted:
        return ds, "No numeric spatial variables found -- nothing extracted."

    return ds, f"Extracted spatial-mean time series for: {', '.join(extracted)}."
