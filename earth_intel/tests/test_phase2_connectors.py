from connectors.connector_factory import get_connector
from models.agent4_schemas import DatasetDescriptor
from models.website_analysis_schemas import SourceSnapshot


def _snapshot(source_id: str, name: str, url: str, api_type: str = "unknown", variables=None):
    return SourceSnapshot(
        source_id=source_id,
        name=name,
        url=url,
        api_type=api_type,
        dataset_type="gridded",
        variables_available=variables or [],
    )


def test_nasa_connector_returns_dataset_descriptor():
    connector = get_connector(_snapshot("nasa", "NASA EarthData", "https://earthdata.nasa.gov", "earthaccess", ["temperature anomaly"]))
    datasets = connector.discover_datasets(_snapshot("nasa", "NASA EarthData", "https://earthdata.nasa.gov", "earthaccess", ["temperature anomaly"]), {"variables": ["temperature anomaly"]})
    assert connector.name == "nasa_earthdata"
    assert datasets
    assert isinstance(datasets[0], DatasetDescriptor)
    assert datasets[0].dataset_id


def test_copernicus_connector_returns_dataset_descriptor():
    snapshot = _snapshot("era5", "Copernicus Climate Data Store ERA5", "https://cds.climate.copernicus.eu", "cdsapi", ["temperature"])
    connector = get_connector(snapshot)
    datasets = connector.discover_datasets(snapshot, {"variables": ["temperature"]})
    assert connector.name == "copernicus_cds"
    assert datasets[0].dataset_id == "reanalysis-era5-single-levels"


def test_open_meteo_connector_returns_dataset_descriptor():
    snapshot = _snapshot("openmeteo", "Open-Meteo", "https://api.open-meteo.com", "rest", ["precipitation"])
    connector = get_connector(snapshot)
    datasets = connector.discover_datasets(snapshot, {"variables": ["precipitation"]})
    assert connector.name == "open_meteo"
    assert datasets
    assert datasets[0].provider == "Open-Meteo"


def test_generic_connector_still_works_as_fallback():
    snapshot = _snapshot("unknown", "Unknown Provider", "https://example.com/data.csv", "unknown", ["temperature"])
    connector = get_connector(snapshot)
    datasets = connector.discover_datasets(snapshot, {"variables": ["temperature"]})
    assert connector.name == "generic_http"
    assert datasets[0].dataset_id == "unknown"


def test_factory_selects_provider_before_generic():
    nasa = _snapshot("nasa", "NASA EarthData", "https://earthdata.nasa.gov", "earthaccess")
    generic = _snapshot("generic", "Plain HTTP", "https://example.com/file.nc")
    assert get_connector(nasa).name == "nasa_earthdata"
    assert get_connector(generic).name == "generic_http"
