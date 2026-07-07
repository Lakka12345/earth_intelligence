"""
Agent 4 — Download Manager.

Handles location/format choice, then executes the actual fetch per
approved source: subset-at-source first (locked-in decision), falling
back to full download when a connector doesn't support subsetting.

HONEST SCOPE NOTE: local trimming of a fully-downloaded file (to
recover the storage savings subsetting would have given) is only
implemented here for NetCDF via xarray, since that's the dominant
format among the oceanographic/climate sources this system targets.
Other formats fall back to keeping the full file as-is -- extend
_trim_local_file for additional formats as needed, don't assume it's
handled for everything.

HONEST SCOPE NOTE 2: bounding-box/time-range subsetting parameters
require actual coordinates, not the free-text location strings
RetrievalRequest currently carries ("chennai", "last 5 years", etc.).
Geocoding/date-parsing to turn those into real (lon/lat) and ISO date
tuples is a separate piece of work, not yet wired here -- until it is,
fetch_request.bounding_box / time_range will be None and connectors
that need them for subsetting will fall back to full download. This is
called out explicitly rather than silently downloading full data and
claiming it was subsetted.
"""

import os
from typing import List, Optional

from agents.agent4_connectors.base import Credentials, FetchRequest
from agents.agent4_connectors.registry import get_connector
from models.agent4_schemas import DownloadFormat, DownloadLocationMode, DownloadManifestEntry, FetchMethod
from models.website_analysis_schemas import SourceSnapshot

DEFAULT_MANAGED_FOLDER = os.path.join(os.getcwd(), "data")


def ask_download_location() -> str:
    print("\nWhere should downloaded data go?")
    print("  1. Managed project data/ folder (recommended, auto-organized)")
    print("  2. A custom path you specify")
    print("  3. Ask me each time per source")
    choice = input("Choice [1/2/3]: ").strip()

    if choice == "2":
        path = input("Enter the full folder path: ").strip()
        os.makedirs(path, exist_ok=True)
        return path
    if choice == "3":
        return DownloadLocationMode.ask_each_time.value
    os.makedirs(DEFAULT_MANAGED_FOLDER, exist_ok=True)
    return DEFAULT_MANAGED_FOLDER


def ask_download_format() -> DownloadFormat:
    print("\nWhat format should the data be downloaded in?")
    print("  1. Native format (recommended -- smallest, no conversion loss)")
    print("  2. CSV")
    print("  3. NetCDF")
    print("  4. Parquet")
    print("  5. GeoTIFF")
    mapping = {"1": DownloadFormat.native, "2": DownloadFormat.csv, "3": DownloadFormat.netcdf,
               "4": DownloadFormat.parquet, "5": DownloadFormat.geotiff}
    choice = input("Choice [1-5]: ").strip()
    fmt = mapping.get(choice, DownloadFormat.native)
    if fmt != DownloadFormat.native:
        print(f"  Note: converting to {fmt.value} may increase file size versus the source's native format "
              f"(e.g. NetCDF -> CSV often bloats significantly). Conversion is not yet implemented in this "
              f"version -- data will be saved in native format regardless, and this is flagged so it's not silent.")
    return fmt


def _trim_local_file(local_path: str, variables: List[str]) -> bool:
    """Best-effort local trim for NetCDF via xarray. Returns True if
    trimming succeeded (raw file reduced in place), False otherwise --
    caller keeps the untrimmed file rather than losing data."""
    if not local_path.endswith((".nc", ".nc4")):
        return False
    try:
        import xarray as xr
        ds = xr.open_dataset(local_path)
        if variables:
            available = [v for v in variables if v in ds.variables]
            if available:
                ds = ds[available]
        tmp_path = local_path + ".trimmed"
        ds.to_netcdf(tmp_path)
        ds.close()
        os.replace(tmp_path, local_path)
        return True
    except Exception as exc:
        print(f"  [Download Manager] Local trim skipped (non-fatal): {exc}")
        return False


def download_source(
    snapshot: SourceSnapshot,
    variables: List[str],
    location: str,
    credentials: Optional[Credentials],
    bounding_box=None,
    time_range=None,
) -> DownloadManifestEntry:
    connector = get_connector(snapshot)
    dest_dir = location if location != DownloadLocationMode.ask_each_time.value else input(f"  Download path for {snapshot.name}: ").strip()
    os.makedirs(dest_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in snapshot.name)[:80]
    dest_path = os.path.join(dest_dir, f"{safe_name}_{snapshot.source_id}.dat")

    fetch_request = FetchRequest(variables=variables, dest_path=dest_path, bounding_box=bounding_box, time_range=time_range)

    try:
        local_path = connector.fetch_subset(snapshot, fetch_request, credentials)
        return DownloadManifestEntry(
            source_id=snapshot.source_id, source_name=snapshot.name, local_path=local_path,
            fetch_method=FetchMethod.server_side_subset, variables_included=variables, success=True,
            size_bytes=os.path.getsize(local_path) if os.path.exists(local_path) else None,
        )
    except NotImplementedError:
        print(f"  {connector.name} connector doesn't support subsetting for this source -- "
              f"downloading full file and attempting a local trim.")
    except Exception as exc:
        return DownloadManifestEntry(source_id=snapshot.source_id, source_name=snapshot.name,
                                      fetch_method=FetchMethod.server_side_subset, success=False, error=str(exc))

    try:
        local_path = connector.fetch_full(snapshot, fetch_request, credentials)
        trimmed = _trim_local_file(local_path, variables)
        return DownloadManifestEntry(
            source_id=snapshot.source_id, source_name=snapshot.name, local_path=local_path,
            fetch_method=FetchMethod.full_download_then_trimmed if trimmed else FetchMethod.full_download_untrimmed,
            variables_included=variables, success=True,
            size_bytes=os.path.getsize(local_path) if os.path.exists(local_path) else None,
        )
    except Exception as exc:
        return DownloadManifestEntry(source_id=snapshot.source_id, source_name=snapshot.name,
                                      fetch_method=FetchMethod.full_download_untrimmed, success=False, error=str(exc))
