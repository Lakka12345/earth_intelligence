"""
Agent 5 unit conversion.

Replaces the previous no-op placeholder. Uses `pint` (a proper
dimensional-analysis library) as the conversion engine so conversions
are checked for dimensional consistency rather than hand-rolled with a
lookup table that could silently accept a nonsensical conversion (e.g.
converting a temperature to a speed).

TARGET UNITS: Agent 5 standardizes each variable to the
preferred_unit_or_representation the user's RetrievalRequest already
specifies per-measurement (see models/retrieval_request.py). If no
preferred unit is given for a variable, it's left as retrieved and
flagged in the processing log -- never silently guessed.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

try:
    import pint

    _UREG = pint.UnitRegistry()
    _UREG.define("degrees_north = degree")
    _UREG.define("degrees_east = degree")
    _PINT_AVAILABLE = True
except ImportError:
    _PINT_AVAILABLE = False
    _UREG = None

# Common CF/oceanographic unit aliases that pint doesn't recognize
# out of the box. Mapped to a pint-parseable equivalent before parsing.
_UNIT_ALIASES = {
    "degc": "degC",
    "deg_c": "degC",
    "celsius": "degC",
    "degrees_celsius": "degC",
    "degk": "kelvin",
    "deg_k": "kelvin",
    "kelvin": "kelvin",
    "psu": "dimensionless",     # practical salinity unit -- no true SI equivalent
    "1": "dimensionless",
    "m s-1": "meter / second",
    "m/s": "meter / second",
    "cm s-1": "centimeter / second",
    "kg m-3": "kilogram / meter ** 3",
    "kg/m3": "kilogram / meter ** 3",
    "hpa": "hectopascal",
    "mbar": "millibar",
    "w m-2": "watt / meter ** 2",
    "mm": "millimeter",
    "km": "kilometer",
}

# PSU has no pint equivalent (it's a practical, not absolute, scale) --
# treated as a no-op "conversion" (dimensionless-to-dimensionless) so a
# request for PSU never silently fails just because pint can't parse it.
_NON_CONVERTIBLE_UNITS = {"psu", "practical salinity unit", "1", "dimensionless"}


def _normalize_unit_string(unit_str: str) -> str:
    key = unit_str.strip().lower()
    return _UNIT_ALIASES.get(key, unit_str)


def units_are_convertible(from_unit: str, to_unit: str) -> bool:
    if not _PINT_AVAILABLE:
        return from_unit.strip().lower() == to_unit.strip().lower()

    from_key = from_unit.strip().lower()
    to_key = to_unit.strip().lower()
    if from_key in _NON_CONVERTIBLE_UNITS or to_key in _NON_CONVERTIBLE_UNITS:
        return from_key == to_key

    try:
        q = _UREG.Quantity(1.0, _normalize_unit_string(from_unit))
        q.to(_normalize_unit_string(to_unit))
        return True
    except Exception:
        return False


def convert_array(
    values: np.ndarray, from_unit: str, to_unit: str
) -> Tuple[np.ndarray, str]:
    """
    Converts a numpy array from from_unit to to_unit. Returns
    (converted_values, note). Raises ValueError if the units are not
    dimensionally compatible -- callers must catch this and treat it
    as a validation issue, never silently pass through mismatched data.
    """
    from_key = from_unit.strip().lower()
    to_key = to_unit.strip().lower()

    if from_key == to_key:
        return values, f"Already in target unit '{to_unit}'; no conversion needed."

    if from_key in _NON_CONVERTIBLE_UNITS or to_key in _NON_CONVERTIBLE_UNITS:
        if from_key == to_key:
            return values, "No conversion needed."
        raise ValueError(
            f"'{from_unit}' has no defined physical conversion to '{to_unit}' "
            f"(e.g. PSU is a practical, not absolute, scale)."
        )

    if not _PINT_AVAILABLE:
        raise ValueError(
            "pint is not installed -- cannot safely convert "
            f"'{from_unit}' to '{to_unit}'. Install with: pip install pint"
        )

    try:
        q = _UREG.Quantity(values, _normalize_unit_string(from_unit))
        converted = q.to(_normalize_unit_string(to_unit))
        return np.asarray(converted.magnitude), f"Converted '{from_unit}' -> '{to_unit}' via pint."
    except pint.errors.DimensionalityError as exc:
        raise ValueError(
            f"'{from_unit}' and '{to_unit}' are not dimensionally compatible: {exc}"
        )


def convert_dataset_variable(
    ds: xr.Dataset, variable: str, from_unit: str, to_unit: str
) -> Tuple[xr.Dataset, str]:
    """Converts one variable in-place (returns a new Dataset) and updates
    its 'units' attribute to the new unit so downstream steps and the
    profiling record stay consistent with what's actually in the array."""
    converted_values, note = convert_array(ds[variable].values, from_unit, to_unit)
    ds = ds.copy()
    ds[variable] = xr.DataArray(
        converted_values, dims=ds[variable].dims, coords=ds[variable].coords, attrs=dict(ds[variable].attrs)
    )
    ds[variable].attrs["units"] = to_unit
    return ds, note


def resolve_target_units(
    measurements, variables_in_dataset: list
) -> Dict[str, str]:
    """
    Builds {variable_name_lower: preferred_unit} from the user's
    RetrievalRequest measurements, restricted to variables actually
    present in this dataset. Variables with no preferred unit specified
    are simply absent from the returned dict (left as-is upstream).
    """
    target: Dict[str, str] = {}
    present_lower = {v.lower(): v for v in variables_in_dataset}
    for m in measurements:
        var_key = m.variable_measured.lower().strip()
        if var_key in present_lower and m.preferred_unit_or_representation:
            target[present_lower[var_key]] = m.preferred_unit_or_representation
    return target
