import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectors.base_connector import FetchRequest
from connectors.connector_factory import get_connector
from models.agent4_schemas import DatasetDescriptor
from models.website_analysis_schemas import SourceSnapshot


def snapshot(source_id, name, url, api_type="unknown", variables=None):
    return SourceSnapshot(
        source_id=source_id,
        name=name,
        url=url,
        api_type=api_type,
        dataset_type="gridded",
        variables_available=variables or [],
    )


CASES = [
    ("NASA", snapshot("nasa", "NASA EarthData", "https://earthdata.nasa.gov", "earthaccess"), "nasa_earthdata"),
    ("NOAA", snapshot("noaa", "NOAA NCEI", "https://www.ncei.noaa.gov", "rest"), "noaa"),
    ("Copernicus", snapshot("copernicus", "Copernicus", "https://cds.climate.copernicus.eu", "cdsapi"), "copernicus_cds"),
    ("Copernicus CDS", snapshot("cds", "Copernicus Climate Data Store ERA5", "https://cds.climate.copernicus.eu", "cdsapi"), "copernicus_cds"),
    ("Copernicus Marine", snapshot("cmems", "Copernicus Marine CMEMS", "https://data.marine.copernicus.eu", "copernicusmarine"), "copernicus_marine"),
    ("Planetary Computer", snapshot("pc", "Microsoft Planetary Computer STAC", "https://planetarycomputer.microsoft.com/api/stac/v1", "stac"), "planetary_computer"),
    ("Open-Meteo", snapshot("openmeteo", "Open-Meteo", "https://api.open-meteo.com", "rest"), "open_meteo"),
    ("Earth Engine", snapshot("gee", "Google Earth Engine", "https://earthengine.google.com", "earthengine-api"), "google_earth_engine"),
    ("GHRSST", snapshot("ghrsst", "GHRSST MUR Sea Surface Temperature", "https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1", "cmr"), "ghrsst"),
    ("DataONE", snapshot("dataone", "DataONE Coordinating Node", "https://cn.dataone.org/cn/v2/query/solr", "solr"), "dataone"),
    ("INCOIS", snapshot("incois", "INCOIS Ocean Data", "https://incois.gov.in/portal/datainfo/datainfo.jsp", "rest"), "incois"),
    ("Unknown", snapshot("unknown", "Unknown Provider", "https://example.com/data.csv", "unknown"), "generic_http"),
]


def main():
    failures = []
    for label, source, expected in CASES:
        connector = get_connector(source)
        datasets = list(connector.discover_datasets(source, {"variables": source.variables_available}) or [])
        metadata = connector.probe_metadata(source, FetchRequest(variables=source.variables_available))
        print(f"{label}: {connector.name}")
        if connector.name != expected:
            failures.append(f"{label}: expected {expected}, got {connector.name}")
        if not datasets:
            failures.append(f"{label}: no dataset descriptors returned")
        elif not isinstance(datasets[0], DatasetDescriptor):
            failures.append(f"{label}: first discovery result is not DatasetDescriptor")
        elif expected != "generic_http":
            descriptor = datasets[0]
            required = {
                "dataset_id": descriptor.dataset_id,
                "api_endpoint": descriptor.api_endpoint,
                "metadata_endpoint": descriptor.metadata_endpoint,
                "download_endpoint": descriptor.download_endpoint,
                "supported_variables": descriptor.supported_variables,
                "spatial_coverage": descriptor.spatial_coverage,
                "temporal_coverage": descriptor.temporal_coverage,
                "supported_formats": descriptor.supported_formats,
            }
            missing = [field for field, value in required.items() if not value or value == "Unknown"]
            if missing:
                failures.append(f"{label}: descriptor missing {', '.join(missing)}")
            if metadata.dataset_id in (None, "", source.source_id):
                failures.append(f"{label}: metadata probe did not return a provider dataset id")

    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("\nAll connector phase 2 checks passed.")


if __name__ == "__main__":
    main()
