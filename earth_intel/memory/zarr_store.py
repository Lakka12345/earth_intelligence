"""
Zarr storage layer for Agent 5.

WHY ZARR HERE: Agent 5 ingests heterogeneous formats (NetCDF, GRIB,
GeoTIFF, CSV, HDF5 -- see DownloadFormat in discovery_schemas.py) but
should hand back ONE consistent, chunked, self-describing format for
every cleaned/standardized/merged output, regardless of what format it
started as. Zarr was chosen because it works natively with xarray
(which is already the working representation for gridded data inside
agent5.py), supports partial/lazy reads for large datasets, and needs
no compiled format-specific reader on the consuming side.

This module does not implement any of the actual preprocessing (unit
conversion, resampling, etc.) -- it is purely the read/write boundary
for the canonical on-disk representation. Preprocessing logic lives in
agent5.py.
"""

import os
from typing import Optional

import xarray as xr

from agent5_config import OUTPUT_STORE_DIR


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_STORE_DIR, exist_ok=True)


def write_dataset(ds: xr.Dataset, name: str, mode: str = "w") -> str:
    """
    Writes an xarray Dataset to the canonical Zarr store and returns
    the path it was written to. `name` should be a short, filesystem-
    safe identifier (e.g. a source_id or a merged-output name) -- this
    function does not sanitize it further.
    """
    _ensure_output_dir()
    path = os.path.join(OUTPUT_STORE_DIR, f"{name}.zarr")
    ds.to_zarr(path, mode=mode)
    return path


def read_dataset(path: str) -> xr.Dataset:
    """Opens a previously-written Zarr store lazily (dask-backed)."""
    return xr.open_zarr(path)


def dataset_exists(name: str) -> Optional[str]:
    path = os.path.join(OUTPUT_STORE_DIR, f"{name}.zarr")
    return path if os.path.isdir(path) else None
