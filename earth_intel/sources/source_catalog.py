"""
Source catalog — hardcoded known data sources.

CHANGED: Every entry now carries the new access classification fields:
  - access_type   : AccessType enum  ("free" | "registration" | "api_key" | "paid")
  - requires_login: bool
  - requires_payment: bool
  - price_estimate: str | None
  - login_url     : str | None
  - api_docs      : str | None
  - api_type      : APIType enum  ("REST" | "OPeNDAP" | "ERDDAP" | ...)
  - discovery_origin: "catalog"  (always for this file)

These fields are read by Agent 4 to classify access and decide whether
a human approval gate is needed before retrieval.

OLD field removed: requires_auth (renamed to requires_login everywhere)
"""

from typing import Dict, List

from models.discovery_schemas import AccessType, APIType, CandidateSource, DownloadFormat


KNOWN_SOURCES: List[Dict] = [

    # ---------------------------------------------------------------- #
    # Ocean — SST, Salinity, Currents, Chlorophyll                      #
    # ---------------------------------------------------------------- #
    {
        "source_id": "noaa_erddap_coastwatch",
        "name": "NOAA CoastWatch ERDDAP",
        "url": "https://coastwatch.pfeg.noaa.gov/erddap",
        "dataset_type": "ocean",
        "variables_available": [
            "sea_surface_temperature",
            "chlorophyll_a",
            "sea_surface_salinity",
            "ocean_currents",
            "sea_level_anomaly",
        ],
        "spatial_coverage": "Global",
        "temporal_coverage": "1981-present",
        "temporal_resolution": "Daily, Monthly",
        "spatial_resolution": "0.25 degree to 0.01 degree depending on dataset",
        "available_formats": [DownloadFormat.netcdf, DownloadFormat.csv],
        "access_type": AccessType.free,
        "requires_login": False,
        "requires_payment": False,
        "price_estimate": None,
        "login_url": None,
        "api_docs": "https://coastwatch.pfeg.noaa.gov/erddap/information.html",
        "api_type": APIType.erddap,
        "discovery_origin": "catalog",
        "metadata_url": "https://coastwatch.pfeg.noaa.gov/erddap/info/index.html",
        "description": (
            "NOAA CoastWatch ERDDAP serves satellite-derived ocean data "
            "including SST, chlorophyll, and currents globally."
        ),
        "catalog_authority_score": 0.95,
        "catalog_scientific_acceptance": 0.95,
        "catalog_historical_reliability": 0.90,
    },

    {
        "source_id": "noaa_erddap_ioos",
        "name": "NOAA IOOS ERDDAP",
        "url": "https://erddap.ioos.us/erddap",
        "dataset_type": "ocean",
        "variables_available": [
            "sea_surface_temperature",
            "wave_height",
            "wave_period",
            "water_level",
            "salinity",
            "ocean_currents",
        ],
        "spatial_coverage": "US coastal and global",
        "temporal_coverage": "Varies by dataset — mostly 2000-present",
        "temporal_resolution": "Hourly, Daily",
        "spatial_resolution": "Station-based and gridded",
        "available_formats": [DownloadFormat.netcdf, DownloadFormat.csv],
        "access_type": AccessType.free,
        "requires_login": False,
        "requires_payment": False,
        "price_estimate": None,
        "login_url": None,
        "api_docs": "https://erddap.ioos.us/erddap/information.html",
        "api_type": APIType.erddap,
        "discovery_origin": "catalog",
        "metadata_url": "https://erddap.ioos.us/erddap/info/index.html",
        "description": (
            "NOAA Integrated Ocean Observing System ERDDAP — "
            "real-time and historical ocean observations."
        ),
        "catalog_authority_score": 0.90,
        "catalog_scientific_acceptance": 0.88,
        "catalog_historical_reliability": 0.85,
    },

    {
        "source_id": "incois_data_portal",
        "name": "INCOIS Ocean Data Portal",
        "url": "https://incois.gov.in/portal/datainfo/drform.jsp",
        "dataset_type": "ocean",
        "variables_available": [
            "sea_surface_temperature",
            "salinity",
            "ocean_currents",
            "wave_height",
            "potential_fishing_zone",
            "storm_surge",
        ],
        "spatial_coverage": "Indian Ocean, Bay of Bengal, Arabian Sea",
        "temporal_coverage": "1990-present",
        "temporal_resolution": "Daily, Weekly",
        "spatial_resolution": "0.1 degree",
        "available_formats": [DownloadFormat.netcdf, DownloadFormat.csv],
        "access_type": AccessType.free,
        "requires_login": False,
        "requires_payment": False,
        "price_estimate": None,
        "login_url": None,
        "api_docs": "https://incois.gov.in/portal/datainfo/sst.jsp",
        "api_type": APIType.rest,
        "discovery_origin": "catalog",
        "metadata_url": "https://incois.gov.in/portal/datainfo/sst.jsp",
        "description": (
            "Indian National Centre for Ocean Information Services — "
            "primary source for Indian Ocean, Bay of Bengal, and Arabian Sea data."
        ),
        "catalog_authority_score": 0.95,
        "catalog_scientific_acceptance": 0.92,
        "catalog_historical_reliability": 0.88,
    },

    {
        "source_id": "noaa_tides_currents",
        "name": "NOAA Tides and Currents API",
        "url": "https://api.tidesandcurrents.noaa.gov/api/prod/",
        "dataset_type": "ocean",
        "variables_available": [
            "water_level",
            "tidal_predictions",
            "ocean_currents",
            "salinity",
            "water_temperature",
            "wind",
            "air_pressure",
        ],
        "spatial_coverage": "US coastal stations, global tidal predictions",
        "temporal_coverage": "1850-present for some stations",
        "temporal_resolution": "6-minute, Hourly, Daily",
        "spatial_resolution": "Station-based",
        "available_formats": [DownloadFormat.json, DownloadFormat.csv],
        "access_type": AccessType.free,
        "requires_login": False,
        "requires_payment": False,
        "price_estimate": None,
        "login_url": None,
        "api_docs": "https://api.tidesandcurrents.noaa.gov/api/prod/",
        "api_type": APIType.rest,
        "discovery_origin": "catalog",
        "metadata_url": "https://api.tidesandcurrents.noaa.gov/api/prod/",
        "description": (
            "NOAA CO-OPS tidal and current observations. "
            "No API key required. JSON/CSV responses."
        ),
        "catalog_authority_score": 0.92,
        "catalog_scientific_acceptance": 0.90,
        "catalog_historical_reliability": 0.92,
    },

    # ---------------------------------------------------------------- #
    # Weather / Atmospheric                                              #
    # ---------------------------------------------------------------- #
    {
        "source_id": "open_meteo",
        "name": "Open-Meteo Weather API",
        "url": "https://api.open-meteo.com/v1/forecast",
        "dataset_type": "weather",
        "variables_available": [
            "temperature_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "surface_pressure",
            "relative_humidity",
            "wave_height",
            "wave_direction",
            "wave_period",
            "soil_moisture",
        ],
        "spatial_coverage": "Global",
        "temporal_coverage": "1940-present (historical via archive endpoint)",
        "temporal_resolution": "Hourly, Daily",
        "spatial_resolution": "0.1 degree (~11km)",
        "available_formats": [DownloadFormat.json],
        "access_type": AccessType.free,
        "requires_login": False,
        "requires_payment": False,
        "price_estimate": None,
        "login_url": None,
        "api_docs": "https://open-meteo.com/en/docs",
        "api_type": APIType.rest,
        "discovery_origin": "catalog",
        "metadata_url": "https://open-meteo.com/en/docs",
        "description": (
            "Completely free weather API. No API key, no registration. "
            "Covers forecasts, historical reanalysis, and marine data."
        ),
        "catalog_authority_score": 0.80,
        "catalog_scientific_acceptance": 0.78,
        "catalog_historical_reliability": 0.82,
    },

    {
        "source_id": "open_meteo_marine",
        "name": "Open-Meteo Marine API",
        "url": "https://marine-api.open-meteo.com/v1/marine",
        "dataset_type": "ocean",
        "variables_available": [
            "wave_height",
            "wave_direction",
            "wave_period",
            "swell_wave_height",
            "swell_wave_direction",
            "wind_wave_height",
        ],
        "spatial_coverage": "Global ocean",
        "temporal_coverage": "2016-present",
        "temporal_resolution": "Hourly",
        "spatial_resolution": "0.2 degree",
        "available_formats": [DownloadFormat.json],
        "access_type": AccessType.free,
        "requires_login": False,
        "requires_payment": False,
        "price_estimate": None,
        "login_url": None,
        "api_docs": "https://open-meteo.com/en/docs/marine-weather-api",
        "api_type": APIType.rest,
        "discovery_origin": "catalog",
        "metadata_url": "https://open-meteo.com/en/docs/marine-weather-api",
        "description": (
            "Open-Meteo marine endpoint. Wave heights, swell, "
            "wind-wave forecasts. No registration required."
        ),
        "catalog_authority_score": 0.78,
        "catalog_scientific_acceptance": 0.75,
        "catalog_historical_reliability": 0.78,
    },

    {
        "source_id": "noaa_gfs_nomads",
        "name": "NOAA GFS via NOMADS",
        "url": "https://nomads.ncep.noaa.gov/dods/gfs_0p25",
        "dataset_type": "weather",
        "variables_available": [
            "temperature",
            "wind_u",
            "wind_v",
            "precipitation",
            "pressure",
            "humidity",
            "geopotential_height",
        ],
        "spatial_coverage": "Global",
        "temporal_coverage": "Last 10 days + 16-day forecast",
        "temporal_resolution": "3-hourly",
        "spatial_resolution": "0.25 degree (~28km)",
        "available_formats": [DownloadFormat.grib, DownloadFormat.netcdf],
        "access_type": AccessType.free,
        "requires_login": False,
        "requires_payment": False,
        "price_estimate": None,
        "login_url": None,
        "api_docs": "https://nomads.ncep.noaa.gov/",
        "api_type": APIType.opendap,
        "discovery_origin": "catalog",
        "metadata_url": "https://nomads.ncep.noaa.gov/",
        "description": (
            "NOAA Global Forecast System via NOMADS OPeNDAP. "
            "No registration. Global atmospheric forecasts."
        ),
        "catalog_authority_score": 0.92,
        "catalog_scientific_acceptance": 0.90,
        "catalog_historical_reliability": 0.88,
    },

    # ---------------------------------------------------------------- #
    # Registration-required sources                                     #
    # ---------------------------------------------------------------- #
    {
        "source_id": "nasa_earthdata",
        "name": "NASA EarthData (Earthdata Login required)",
        "url": "https://search.earthdata.nasa.gov/",
        "dataset_type": "satellite",
        "variables_available": [
            "sea_surface_temperature",
            "chlorophyll_a",
            "ocean_color",
            "land_surface_temperature",
            "precipitation",
            "soil_moisture",
            "ice_cover",
            "ndvi",
        ],
        "spatial_coverage": "Global",
        "temporal_coverage": "1970-present depending on dataset",
        "temporal_resolution": "Daily, Monthly",
        "spatial_resolution": "250m to 25km depending on product",
        "available_formats": [DownloadFormat.netcdf, DownloadFormat.hdf5, DownloadFormat.geotiff],
        "access_type": AccessType.registration,
        "requires_login": True,
        "requires_payment": False,
        "price_estimate": "Free after registration",
        "login_url": "https://urs.earthdata.nasa.gov/users/new",
        "api_docs": "https://cmr.earthdata.nasa.gov/search/",
        "api_type": APIType.rest,
        "discovery_origin": "catalog",
        "metadata_url": "https://search.earthdata.nasa.gov/",
        "description": (
            "NASA's primary data portal. Requires free Earthdata account. "
            "Access to MODIS, VIIRS, SMAP, GPM, and hundreds of other datasets."
        ),
        "catalog_authority_score": 0.98,
        "catalog_scientific_acceptance": 0.98,
        "catalog_historical_reliability": 0.95,
    },

    {
        "source_id": "copernicus_marine",
        "name": "Copernicus Marine Service (CMEMS)",
        "url": "https://marine.copernicus.eu/",
        "dataset_type": "ocean",
        "variables_available": [
            "sea_surface_temperature",
            "salinity",
            "ocean_currents",
            "sea_level_anomaly",
            "chlorophyll_a",
            "wave_height",
            "mixed_layer_depth",
        ],
        "spatial_coverage": "Global and regional (Atlantic, Mediterranean, Arctic, etc.)",
        "temporal_coverage": "1993-present",
        "temporal_resolution": "Daily, Monthly",
        "spatial_resolution": "0.083 degree to 0.25 degree",
        "available_formats": [DownloadFormat.netcdf],
        "access_type": AccessType.registration,
        "requires_login": True,
        "requires_payment": False,
        "price_estimate": "Free after registration",
        "login_url": "https://marine.copernicus.eu/register-copernicus-marine-service",
        "api_docs": "https://help.marine.copernicus.eu/en/articles/7949409-copernicus-marine-toolbox-introduction",
        "api_type": APIType.rest,
        "discovery_origin": "catalog",
        "metadata_url": "https://data.marine.copernicus.eu/products",
        "description": (
            "EU Copernicus Marine Environment Monitoring Service. "
            "High-quality ocean analysis and forecast products. Free registration."
        ),
        "catalog_authority_score": 0.97,
        "catalog_scientific_acceptance": 0.97,
        "catalog_historical_reliability": 0.93,
    },

    # ---------------------------------------------------------------- #
    # Disaster / Cyclone / Storm Surge                                  #
    # ---------------------------------------------------------------- #
    {
        "source_id": "incois_storm_surge",
        "name": "INCOIS Storm Surge and Cyclone Alerts",
        "url": "https://incois.gov.in/portal/osf/osf.jsp",
        "dataset_type": "disaster",
        "variables_available": [
            "storm_surge_height",
            "cyclone_track",
            "cyclone_intensity",
            "inundation_forecast",
            "coastal_alert_level",
        ],
        "spatial_coverage": "Indian coastline, Bay of Bengal, Arabian Sea",
        "temporal_coverage": "Real-time during cyclone events",
        "temporal_resolution": "3-hourly during events",
        "spatial_resolution": "~3km coastal",
        "available_formats": [DownloadFormat.json, DownloadFormat.csv],
        "access_type": AccessType.free,
        "requires_login": False,
        "requires_payment": False,
        "price_estimate": None,
        "login_url": None,
        "api_docs": "https://incois.gov.in/portal/osf/osf.jsp",
        "api_type": APIType.rest,
        "discovery_origin": "catalog",
        "metadata_url": "https://incois.gov.in/portal/osf/osf.jsp",
        "description": (
            "INCOIS Ocean State Forecast — storm surge, cyclone alerts, "
            "and inundation forecasts for Indian coastal regions."
        ),
        "catalog_authority_score": 0.95,
        "catalog_scientific_acceptance": 0.93,
        "catalog_historical_reliability": 0.90,
    },

    {
        "source_id": "imd_cyclone",
        "name": "IMD Cyclone Track and Warnings",
        "url": "https://mausam.imd.gov.in/imd_latest/contents/cyclone.php",
        "dataset_type": "disaster",
        "variables_available": [
            "cyclone_track",
            "cyclone_intensity",
            "landfall_forecast",
            "wind_speed",
            "rainfall_forecast",
        ],
        "spatial_coverage": "Indian subcontinent, Bay of Bengal, Arabian Sea",
        "temporal_coverage": "Real-time",
        "temporal_resolution": "6-hourly",
        "spatial_resolution": "Point-based track",
        "available_formats": [DownloadFormat.json],
        "access_type": AccessType.free,
        "requires_login": False,
        "requires_payment": False,
        "price_estimate": None,
        "login_url": None,
        "api_docs": "https://mausam.imd.gov.in/",
        "api_type": APIType.rest,
        "discovery_origin": "catalog",
        "metadata_url": "https://mausam.imd.gov.in/",
        "description": (
            "India Meteorological Department cyclone warnings and track forecasts. "
            "Primary authoritative source for Indian cyclone data."
        ),
        "catalog_authority_score": 0.95,
        "catalog_scientific_acceptance": 0.92,
        "catalog_historical_reliability": 0.90,
    },

    # ---------------------------------------------------------------- #
    # GIS / Geospatial                                                  #
    # ---------------------------------------------------------------- #
    {
        "source_id": "bhuvan_isro",
        "name": "Bhuvan — ISRO Geoportal",
        "url": "https://bhuvan-app1.nrsc.gov.in/bhuvan/wms",
        "dataset_type": "gis",
        "variables_available": [
            "land_use_land_cover",
            "elevation_dem",
            "coastal_boundary",
            "flood_inundation",
            "vegetation_index",
            "district_boundaries",
        ],
        "spatial_coverage": "India",
        "temporal_coverage": "2000-present",
        "temporal_resolution": "Annual, Event-based",
        "spatial_resolution": "30m to 250m depending on layer",
        "available_formats": [DownloadFormat.geotiff, DownloadFormat.shapefile],
        "access_type": AccessType.free,
        "requires_login": False,
        "requires_payment": False,
        "price_estimate": None,
        "login_url": None,
        "api_docs": "https://bhuvan.nrsc.gov.in/",
        "api_type": APIType.wms_wfs,
        "discovery_origin": "catalog",
        "metadata_url": "https://bhuvan.nrsc.gov.in/",
        "description": (
            "ISRO's National Remote Sensing Centre geoportal. "
            "Indian GIS data — land cover, elevation, boundaries, flood maps."
        ),
        "catalog_authority_score": 0.90,
        "catalog_scientific_acceptance": 0.88,
        "catalog_historical_reliability": 0.85,
    },

    {
        "source_id": "noaa_etopo",
        "name": "NOAA ETOPO Global Relief",
        "url": "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer",
        "dataset_type": "gis",
        "variables_available": [
            "bathymetry",
            "elevation",
            "seafloor_topography",
        ],
        "spatial_coverage": "Global",
        "temporal_coverage": "Static (updated periodically)",
        "temporal_resolution": "Static",
        "spatial_resolution": "1 arc-minute (~1.8km)",
        "available_formats": [DownloadFormat.netcdf, DownloadFormat.geotiff],
        "access_type": AccessType.free,
        "requires_login": False,
        "requires_payment": False,
        "price_estimate": None,
        "login_url": None,
        "api_docs": "https://www.ngdc.noaa.gov/mgg/global/",
        "api_type": APIType.rest,
        "discovery_origin": "catalog",
        "metadata_url": "https://www.ngdc.noaa.gov/mgg/global/",
        "description": (
            "NOAA global seafloor and land elevation model. "
            "Essential for tsunami, storm surge, and coastal inundation analysis."
        ),
        "catalog_authority_score": 0.92,
        "catalog_scientific_acceptance": 0.93,
        "catalog_historical_reliability": 0.95,
    },
]


# ------------------------------------------------------------------ #
# Dataset type keyword mapping                                         #
# ------------------------------------------------------------------ #

DATASET_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "hydrology": [
        "river", "streamflow", "discharge", "runoff", "groundwater",
        "reservoir", "basin", "catchment", "hydrology", "hydrological",
        "water level", "water resources", "irrigation", "drainage",
    ],
    "ocean": [
        "sst", "sea surface temperature", "salinity", "currents",
        "chlorophyll", "wave", "ocean", "marine", "coastal", "tidal",
        "water level", "sea level",
    ],
    "weather": [
        "weather", "atmospheric", "wind", "precipitation", "rainfall",
        "temperature", "pressure", "humidity", "forecast",
    ],
    "disaster": [
        "cyclone", "storm surge", "flood", "tsunami", "inundation",
        "disaster", "hazard", "alert", "warning", "landfall",
    ],
    "gis": [
        "gis", "geospatial", "land use", "elevation", "dem",
        "boundary", "shapefile", "coastal boundary", "bathymetry",
    ],
    "satellite": [
        "satellite", "remote sensing", "imagery", "raster",
        "multispectral", "sar", "ndvi", "vegetation",
    ],
}


def get_candidates_for_request(
    dataset_types: List[str],
    variables_needed: List[str],
) -> List[CandidateSource]:
    """
    Pure Python — no LLM.
    Returns CandidateSource objects matching requested types/variables.
    Unchanged logic; now produces CandidateSources with all new fields.
    """
    matched_source_ids = set()
    candidates = []

    types_lower = [t.lower().strip() for t in dataset_types]
    variables_lower = [v.lower().strip() for v in variables_needed]

    for entry in KNOWN_SOURCES:
        source_type = entry["dataset_type"].lower()
        source_vars = [v.lower() for v in entry["variables_available"]]

        matched = False

        if source_type in types_lower:
            matched = True

        if not matched:
            for requested_type in types_lower:
                keywords = DATASET_TYPE_KEYWORDS.get(requested_type, [])
                for kw in keywords:
                    if kw in source_type or any(kw in sv for sv in source_vars):
                        matched = True
                        break
                if matched:
                    break

        if not matched:
            for var in variables_lower:
                for sv in source_vars:
                    if var in sv or sv in var:
                        matched = True
                        break
                if matched:
                    break

        if matched and entry["source_id"] not in matched_source_ids:
            matched_source_ids.add(entry["source_id"])
            candidates.append(CandidateSource(**entry))

    return candidates
