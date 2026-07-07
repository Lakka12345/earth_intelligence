"""
providers.py — Master provider registry for Agent 3.

This file defines WHAT Agent 3 knows about data providers:
  - Who has the data
  - What kind of access is required
  - Which discoverer to use to find specific datasets
  - Human approval gates for login and payment

Agent 4 NEVER reads this file directly.
Agent 3 selects a provider, resolves datasets via the appropriate
discoverer, and hands Agent 4 a fully resolved CandidateSource.

Access tiers:
  Tier 1 — Free, no registration     (AccessType.free)
  Tier 2 — Free, registration needed (AccessType.registration)
  Tier 3 — Browse only               (AccessType.free, browse_only=True)
  Tier 4 — Paid / non-commercial     (AccessType.paid)

Discoverer tags:
  "erddap"      → ERDDAPDiscoverer probes this base_url
  "stac"        → STACDiscoverer probes this base_url
  "ckan"        → CKANDiscoverer probes this base_url
  "thredds"     → THREDDSDiscoverer probes this base_url
  "cmr"         → NASA CMR REST API
  "wcs_wms"     → OGC WCS/WMS endpoint
  "rest"        → Generic REST API
  "ftp"         → FTP bulk download
  "web"         → HTML / scrape (fallback)

Human approval gates (enforced by Agent 3 Phase 8 and Agent 4):
  requires_login    → Agent 3 asks user before handing to Agent 4.
                      Agent 4 asks for credentials before any request.
  requires_payment  → Agent 3 shows cost estimate and asks for EXPLICIT
                      confirmation. Agent 4 NEVER initiates payment
                      without a second human confirmation.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import re
from models.discovery_schemas import AccessType, APIType


@dataclass
class ProviderDefinition:
    """
    Describes a single data provider.
    Used by Agent 3 to route discovery and by the human gate logic.
    Agent 4 only sees the resolved CandidateSource, not this object.
    """
    source_id: str
    name: str
    base_url: str
    access_type: AccessType
    discoverer_type: str            # "erddap" | "stac" | "ckan" | "thredds" | "cmr" | "rest" | ...
    description: str
    dataset_types: List[str]        # e.g. ["ocean", "weather", "satellite"]
    variables_hint: List[str]       # hint variables — discoverer refines at runtime

    login_url: Optional[str] = None
    pricing_url: Optional[str] = None
    api_docs_url: Optional[str] = None
    requires_login: bool = False
    requires_payment: bool = False
    browse_only: bool = False       # True = Tier 3, no programmatic download
    price_estimate: Optional[str] = None
    authority_score: float = 0.80
    scientific_acceptance: float = 0.80
    historical_reliability: float = 0.80
    regions: List[str] = field(default_factory=list)   # [] = global


# ======================================================================
# TIER 1 — FREE, NO REGISTRATION
# ======================================================================

TIER1_FREE: List[ProviderDefinition] = [

    # ── NOAA family ────────────────────────────────────────────────────

    ProviderDefinition(
        source_id="noaa_erddap_coastwatch",
        name="NOAA CoastWatch ERDDAP",
        base_url="https://coastwatch.pfeg.noaa.gov/erddap",
        access_type=AccessType.free,
        discoverer_type="erddap",
        description=(
            "NOAA CoastWatch ERDDAP — satellite-derived ocean data including SST, "
            "chlorophyll-a, salinity, currents, and sea level anomaly. Global coverage. "
            "Direct download in NetCDF, CSV, JSON."
        ),
        dataset_types=["ocean", "satellite"],
        variables_hint=["sea_surface_temperature", "chlorophyll_a", "salinity",
                        "ocean_currents", "sea_level_anomaly"],
        api_docs_url="https://coastwatch.pfeg.noaa.gov/erddap/information.html",
        authority_score=0.97,
        scientific_acceptance=0.97,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="noaa_erddap_ioos",
        name="NOAA IOOS ERDDAP",
        base_url="https://erddap.ioos.us/erddap",
        access_type=AccessType.free,
        discoverer_type="erddap",
        description=(
            "NOAA Integrated Ocean Observing System ERDDAP. US coastal and global "
            "observations — SST, wave height, water level, salinity, currents."
        ),
        dataset_types=["ocean", "coastal"],
        variables_hint=["sea_surface_temperature", "wave_height", "wave_period",
                        "water_level", "salinity", "ocean_currents"],
        api_docs_url="https://erddap.ioos.us/erddap/information.html",
        authority_score=0.93,
        scientific_acceptance=0.92,
        historical_reliability=0.90,
    ),

    ProviderDefinition(
        source_id="noaa_nomads",
        name="NOAA NOMADS",
        base_url="https://nomads.ncep.noaa.gov",
        access_type=AccessType.free,
        discoverer_type="thredds",
        description=(
            "NOAA National Operational Model Archive and Distribution System. "
            "Numerical weather prediction model outputs — GFS, NAM, HRRR, WaveWatch III."
        ),
        dataset_types=["weather", "atmosphere", "wave"],
        variables_hint=["wind_speed", "pressure", "temperature", "precipitation",
                        "wave_height", "wave_period"],
        api_docs_url="https://nomads.ncep.noaa.gov/",
        authority_score=0.95,
        scientific_acceptance=0.95,
        historical_reliability=0.92,
    ),

    ProviderDefinition(
        source_id="noaa_ncei",
        name="NOAA NCEI",
        base_url="https://www.ncei.noaa.gov",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "NOAA National Centers for Environmental Information — climate and ocean "
            "archives. World Ocean Atlas, World Ocean Database, and GHRSST SST products."
        ),
        dataset_types=["ocean", "climate", "atmosphere"],
        variables_hint=["sea_surface_temperature", "salinity", "ocean_temperature",
                        "oxygen", "climate_normals"],
        api_docs_url="https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation",
        authority_score=0.97,
        scientific_acceptance=0.97,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="noaa_ndbc",
        name="NOAA NDBC Buoys",
        base_url="https://www.ndbc.noaa.gov",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "NOAA National Data Buoy Center — real-time and historical buoy, wave, "
            "and meteorological data from hundreds of stations globally."
        ),
        dataset_types=["ocean", "weather", "wave"],
        variables_hint=["wave_height", "wave_period", "wind_speed", "sea_surface_temperature",
                        "air_temperature", "pressure"],
        api_docs_url="https://www.ndbc.noaa.gov/data/",
        authority_score=0.95,
        scientific_acceptance=0.95,
        historical_reliability=0.93,
    ),

    ProviderDefinition(
        source_id="noaa_dart",
        name="NOAA DART Tsunami Buoys",
        base_url="https://www.ngdc.noaa.gov/hazard/dart",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Deep-ocean Assessment and Reporting of Tsunamis — real-time and archived "
            "bottom pressure and surface measurements from DART buoy network."
        ),
        dataset_types=["ocean", "disaster", "tsunami"],
        variables_hint=["sea_level", "bottom_pressure", "tsunami_wave"],
        api_docs_url="https://www.ngdc.noaa.gov/hazard/dart/",
        authority_score=0.97,
        scientific_acceptance=0.97,
        historical_reliability=0.94,
    ),

    ProviderDefinition(
        source_id="noaa_wavewatch3",
        name="NOAA WaveWatch III",
        base_url="https://polar.ncep.noaa.gov/waves",
        access_type=AccessType.free,
        discoverer_type="thredds",
        description=(
            "Global and regional wave model output — significant wave height, peak period, "
            "mean direction, wind wave and swell components."
        ),
        dataset_types=["wave", "ocean"],
        variables_hint=["wave_height", "wave_period", "wave_direction", "swell"],
        api_docs_url="https://polar.ncep.noaa.gov/waves/",
        authority_score=0.95,
        scientific_acceptance=0.94,
        historical_reliability=0.92,
    ),

    ProviderDefinition(
        source_id="noaa_tao_triton",
        name="TAO/TRITON/PIRATA/IndOOS Buoy Arrays",
        base_url="https://www.pmel.noaa.gov/tao",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Tropical moored buoy arrays for Pacific (TAO/TRITON), Atlantic (PIRATA), "
            "and Indian Ocean (IndOOS). Real-time temperature, salinity, currents, wind."
        ),
        dataset_types=["ocean", "climate"],
        variables_hint=["sea_surface_temperature", "salinity", "ocean_currents",
                        "wind_speed", "thermocline_depth"],
        api_docs_url="https://www.pmel.noaa.gov/tao/drupal/disdel/",
        authority_score=0.95,
        scientific_acceptance=0.96,
        historical_reliability=0.93,
    ),

    ProviderDefinition(
        source_id="noaa_icoads",
        name="ICOADS",
        base_url="https://icoads.noaa.gov",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "International Comprehensive Ocean-Atmosphere Data Set — 455 million+ surface "
            "marine observations from ships, buoys, and coastal platforms spanning 1662–2014."
        ),
        dataset_types=["ocean", "climate", "atmosphere"],
        variables_hint=["sea_surface_temperature", "wind", "pressure",
                        "air_temperature", "humidity"],
        api_docs_url="https://icoads.noaa.gov/data.icoads.html",
        authority_score=0.94,
        scientific_acceptance=0.96,
        historical_reliability=0.93,
    ),

    ProviderDefinition(
        source_id="noaa_wod",
        name="World Ocean Database (WOD)",
        base_url="https://www.ncei.noaa.gov/products/world-ocean-database",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "World's largest uniformly formatted, quality-controlled public ocean profile "
            "database. Temperature, salinity, oxygen, nutrients, plankton, sediments."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "oxygen", "nutrients", "chlorophyll"],
        api_docs_url="https://www.ncei.noaa.gov/access/world-ocean-database-select/bin/dbsearch.pl",
        authority_score=0.97,
        scientific_acceptance=0.97,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="noaa_ghrsst",
        name="GHRSST (Group for High-Resolution SST)",
        base_url="https://www.ghrsst.org",
        access_type=AccessType.free,
        discoverer_type="erddap",
        description=(
            "Daily global SST analyses from multi-satellite blending. Available via NCEI "
            "and NASA PO.DAAC. High-resolution (0.01-0.25 degree) L2P, L3, L4 products."
        ),
        dataset_types=["ocean", "satellite"],
        variables_hint=["sea_surface_temperature", "sst_anomaly"],
        api_docs_url="https://www.ghrsst.org/ghrsst-data-services/",
        authority_score=0.97,
        scientific_acceptance=0.97,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="noaa_gloss",
        name="GLOSS (Global Sea Level Observing System)",
        base_url="https://www.gloss-sealevel.info",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Worldwide tide gauge records and sea level time series from the GLOSS network "
            "of ~300 stations. Long-term sea level change and coastal hazard monitoring."
        ),
        dataset_types=["ocean", "coastal"],
        variables_hint=["sea_level", "tide_gauge", "sea_level_rise"],
        api_docs_url="https://www.gloss-sealevel.info/data",
        authority_score=0.95,
        scientific_acceptance=0.96,
        historical_reliability=0.94,
    ),

    ProviderDefinition(
        source_id="noaa_socat",
        name="SOCAT (Surface Ocean CO₂ Atlas)",
        base_url="https://www.socat.info",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Global surface ocean CO₂ and fCO₂ (fugacity of CO₂) measurements from "
            "ships, moorings, and drifters. Used for ocean carbon cycle research."
        ),
        dataset_types=["ocean", "carbon"],
        variables_hint=["co2", "fco2", "ocean_carbon", "pco2"],
        api_docs_url="https://www.socat.info/index.php/data-access-2/",
        authority_score=0.95,
        scientific_acceptance=0.97,
        historical_reliability=0.93,
    ),

    # ── ECMWF ──────────────────────────────────────────────────────────

    ProviderDefinition(
        source_id="ecmwf_open_data",
        name="ECMWF Open Data",
        base_url="https://www.ecmwf.int/en/forecasts/datasets/open-data",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "ECMWF Open Data — entire real-time catalogue open to all as of October 2025. "
            "Medium-range and ensemble forecasts in GRIB2. No login required."
        ),
        dataset_types=["weather", "atmosphere", "climate"],
        variables_hint=["wind_speed", "pressure", "temperature", "precipitation",
                        "humidity", "wave_height"],
        api_docs_url="https://confluence.ecmwf.int/display/DAC/ECMWF+open+data+real-time+API",
        authority_score=0.97,
        scientific_acceptance=0.98,
        historical_reliability=0.97,
    ),

    # ── USGS ───────────────────────────────────────────────────────────

    ProviderDefinition(
        source_id="usgs_earthquake",
        name="USGS Earthquake API",
        base_url="https://earthquake.usgs.gov/fdsnws",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "USGS FDSN earthquake API — seismic event data, magnitude, location, depth. "
            "Real-time and historical. Fully open REST API."
        ),
        dataset_types=["seismology", "disaster"],
        variables_hint=["earthquake_magnitude", "seismicity", "ground_motion"],
        api_docs_url="https://earthquake.usgs.gov/fdsnws/event/1/",
        authority_score=0.98,
        scientific_acceptance=0.98,
        historical_reliability=0.97,
    ),

    # ── Seismology ─────────────────────────────────────────────────────

    ProviderDefinition(
        source_id="iris_earthscope_sage",
        name="EarthScope SAGE / IRIS",
        base_url="https://service.iris.edu",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "IRIS/EarthScope SAGE — global seismological data. Waveforms, station metadata, "
            "earthquake data. Free and open access via FDSN web services."
        ),
        dataset_types=["seismology"],
        variables_hint=["seismic_waveform", "earthquake_magnitude", "ground_motion",
                        "station_metadata"],
        api_docs_url="https://service.iris.edu/",
        authority_score=0.97,
        scientific_acceptance=0.97,
        historical_reliability=0.96,
    ),

    ProviderDefinition(
        source_id="fdsn",
        name="FDSN (Federation of Digital Seismograph Networks)",
        base_url="https://www.fdsn.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Global federation of digital seismograph networks. All members provide free "
            "and open broadband seismic data. Routes to national seismological institutes."
        ),
        dataset_types=["seismology"],
        variables_hint=["seismic_waveform", "earthquake_magnitude", "broadband_seismic"],
        api_docs_url="https://www.fdsn.org/webservices/",
        authority_score=0.96,
        scientific_acceptance=0.97,
        historical_reliability=0.95,
    ),

    # ── Open-Meteo ─────────────────────────────────────────────────────

    ProviderDefinition(
        source_id="open_meteo",
        name="Open-Meteo",
        base_url="https://api.open-meteo.com",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Open-source weather API. Free for non-commercial use. No API key required. "
            "Forecasts and 80 years of historical data for any location worldwide."
        ),
        dataset_types=["weather", "climate"],
        variables_hint=["temperature", "wind_speed", "precipitation", "humidity",
                        "pressure", "solar_radiation"],
        api_docs_url="https://open-meteo.com/en/docs",
        authority_score=0.85,
        scientific_acceptance=0.83,
        historical_reliability=0.87,
    ),

    # ── NASA public (no-login) products ────────────────────────────────

    ProviderDefinition(
        source_id="nasa_gibs",
        name="NASA GIBS",
        base_url="https://gibs.earthdata.nasa.gov",
        access_type=AccessType.free,
        discoverer_type="wcs_wms",
        description=(
            "NASA Global Imagery Browse Services — satellite imagery tiles for browse. "
            "Raw science data available via NASA Earthdata (requires registration)."
        ),
        dataset_types=["satellite", "ocean", "atmosphere"],
        variables_hint=["sea_surface_temperature", "chlorophyll", "aerosol",
                        "land_cover", "fire", "snow_ice"],
        api_docs_url="https://nasa-gibs.github.io/gibs-api-docs/",
        browse_only=True,
        authority_score=0.95,
        scientific_acceptance=0.95,
        historical_reliability=0.97,
    ),

    # ── Argo (public mirror) ────────────────────────────────────────────

    ProviderDefinition(
        source_id="argo_gdac_usgodae",
        name="Argo GDAC Mirror (USGODAE)",
        base_url="https://www.usgodae.org/argo",
        access_type=AccessType.free,
        discoverer_type="ftp",
        description=(
            "Global ocean profiling float data. Temperature and salinity profiles from "
            "~4000 floats worldwide. No login for public mirror."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "oxygen", "pressure"],
        api_docs_url="https://argo.ucsd.edu/data/documentation/",
        authority_score=0.97,
        scientific_acceptance=0.98,
        historical_reliability=0.95,
    ),

    # ── Geospatial / terrain ────────────────────────────────────────────

    ProviderDefinition(
        source_id="gebco",
        name="GEBCO (Global Ocean Bathymetry)",
        base_url="https://www.gebco.net",
        access_type=AccessType.free,
        discoverer_type="wcs_wms",
        description=(
            "GEBCO (General Bathymetric Chart of the Oceans) — global ocean bathymetry "
            "at 15 arc-second resolution. Free download of full grid."
        ),
        dataset_types=["bathymetry", "ocean"],
        variables_hint=["ocean_depth", "bathymetry", "seafloor_topography"],
        api_docs_url="https://www.gebco.net/data_and_products/",
        authority_score=0.97,
        scientific_acceptance=0.97,
        historical_reliability=0.96,
    ),

    ProviderDefinition(
        source_id="natural_earth",
        name="Natural Earth",
        base_url="https://www.naturalearthdata.com",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Free GIS vector and raster data for cartography. Physical, cultural, and raster "
            "data at 1:10m, 1:50m, and 1:110m scales. Public domain."
        ),
        dataset_types=["geospatial", "land"],
        variables_hint=["coastline", "ocean_mask", "land_cover", "administrative_boundaries"],
        api_docs_url="https://www.naturalearthdata.com/downloads/",
        authority_score=0.90,
        scientific_acceptance=0.90,
        historical_reliability=0.95,
    ),

    # ── Disaster ───────────────────────────────────────────────────────

    ProviderDefinition(
        source_id="gdacs",
        name="GDACS (Global Disaster Alerts)",
        base_url="https://www.gdacs.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Global Disaster Alerts and Coordination System — real-time and historical "
            "alerts for earthquakes, cyclones, floods, volcanoes, and tsunamis."
        ),
        dataset_types=["disaster"],
        variables_hint=["cyclone_track", "flood_extent", "earthquake_impact",
                        "tsunami_alert", "volcanic_activity"],
        api_docs_url="https://www.gdacs.org/About/Help_api.aspx",
        authority_score=0.90,
        scientific_acceptance=0.88,
        historical_reliability=0.87,
    ),

    ProviderDefinition(
        source_id="reliefweb",
        name="ReliefWeb",
        base_url="https://reliefweb.int",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "UN OCHA's humanitarian information service — disaster reports, situation "
            "reports, and humanitarian data. Global coverage."
        ),
        dataset_types=["disaster", "humanitarian"],
        variables_hint=["disaster_report", "cyclone", "flood", "earthquake"],
        api_docs_url="https://reliefweb.int/help/api",
        authority_score=0.88,
        scientific_acceptance=0.85,
        historical_reliability=0.88,
    ),

    # ── Biodiversity ───────────────────────────────────────────────────

    ProviderDefinition(
        source_id="obis",
        name="OBIS (Ocean Biodiversity Information System)",
        base_url="https://api.obis.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Ocean Biodiversity Information System — free species occurrence data from the "
            "global ocean. 100+ million records. Used by IPBES and IUCN."
        ),
        dataset_types=["biodiversity", "ocean"],
        variables_hint=["species_occurrence", "marine_biodiversity", "fish",
                        "coral", "plankton"],
        api_docs_url="https://api.obis.org/",
        authority_score=0.93,
        scientific_acceptance=0.95,
        historical_reliability=0.93,
    ),

    ProviderDefinition(
        source_id="gbif",
        name="GBIF (Global Biodiversity Information Facility)",
        base_url="https://api.gbif.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Free and open global biodiversity data — species occurrences, checklists, "
            "sampling events. 2+ billion records. Used by IPCC, IPBES, IUCN Red List."
        ),
        dataset_types=["biodiversity"],
        variables_hint=["species_occurrence", "biodiversity", "ecosystem"],
        api_docs_url="https://www.gbif.org/developer/summary",
        authority_score=0.95,
        scientific_acceptance=0.96,
        historical_reliability=0.95,
    ),

    # ── Data repositories ──────────────────────────────────────────────

    ProviderDefinition(
        source_id="pangaea",
        name="PANGAEA",
        base_url="https://www.pangaea.de",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Earth and environmental science data publisher — 419,000+ datasets with "
            "25 billion+ measurements. Long-term archive with DOIs. Most open access."
        ),
        dataset_types=["ocean", "climate", "geology", "atmosphere"],
        variables_hint=["temperature", "salinity", "sediment", "geochemistry",
                        "carbon", "bathymetry"],
        api_docs_url="https://www.pangaea.de/about/www.pangaea.de/services/api.php",
        authority_score=0.95,
        scientific_acceptance=0.97,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="zenodo",
        name="Zenodo",
        base_url="https://zenodo.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Open repository by CERN — free, assigns DOIs, hosts datasets, papers, and "
            "software. General purpose, widely used in earth science."
        ),
        dataset_types=["multidisciplinary"],
        variables_hint=[],
        api_docs_url="https://developers.zenodo.org/",
        authority_score=0.88,
        scientific_acceptance=0.88,
        historical_reliability=0.92,
    ),

    ProviderDefinition(
        source_id="dataone",
        name="DataONE",
        base_url="https://cn.dataone.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Data Observation Network for Earth — 770,000+ datasets across 50+ member "
            "repositories. Integrated search and API. Includes ocean, climate, ecology."
        ),
        dataset_types=["ocean", "climate", "ecology"],
        variables_hint=["temperature", "salinity", "species", "carbon"],
        api_docs_url="https://releases.dataone.org/online/api-documentation-v2.0.1/",
        authority_score=0.90,
        scientific_acceptance=0.92,
        historical_reliability=0.90,
    ),

    ProviderDefinition(
        source_id="emodnet",
        name="EMODnet (European Marine Observation and Data Network)",
        base_url="https://emodnet.ec.europa.eu",
        access_type=AccessType.free,
        discoverer_type="wcs_wms",
        description=(
            "Seven-theme European marine data network — bathymetry, geology, chemistry, "
            "biology, physics, seabed habitats, human activities. All products free."
        ),
        dataset_types=["ocean", "bathymetry", "geology", "biology"],
        variables_hint=["bathymetry", "sea_surface_temperature", "salinity",
                        "species_distribution", "seabed_habitat"],
        api_docs_url="https://emodnet.ec.europa.eu/geonetwork/srv/eng/catalog.search",
        authority_score=0.94,
        scientific_acceptance=0.95,
        historical_reliability=0.93,
        regions=["European waters", "Atlantic", "Mediterranean", "Baltic", "Arctic"],
    ),

    ProviderDefinition(
        source_id="ocean_sites",
        name="OceanSITES",
        base_url="https://www.ocean-sites.org",
        access_type=AccessType.free,
        discoverer_type="erddap",
        description=(
            "International long-term open-ocean mooring stations. Meteorology, physical "
            "oceanography, biogeochemistry, carbon cycle, geophysics. Free via NDBC/GDAC."
        ),
        dataset_types=["ocean", "climate"],
        variables_hint=["temperature", "salinity", "oxygen", "carbon", "current"],
        api_docs_url="https://www.ocean-sites.org/data.html",
        authority_score=0.94,
        scientific_acceptance=0.95,
        historical_reliability=0.93,
    ),

    ProviderDefinition(
        source_id="argo_gdac_global",
        name="Argo Float Data (global, all GDACs)",
        base_url="https://argo.ucsd.edu",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Global ocean profiling floats — temperature, salinity, oxygen from ~4000 "
            "active floats. Real-time and delayed-mode profiles."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "oxygen", "biogeochemistry"],
        api_docs_url="https://argo.ucsd.edu/data/data-from-gdacs/",
        authority_score=0.97,
        scientific_acceptance=0.98,
        historical_reliability=0.96,
    ),

    ProviderDefinition(
        source_id="ooi",
        name="OOI (Ocean Observatories Initiative)",
        base_url="https://oceanobservatories.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Physical, chemical, geological, and biological ocean observations. Open to "
            "anyone via Data Explorer. Continuous data from fixed and mobile platforms."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "oxygen", "pH", "current",
                        "acoustic_doppler"],
        api_docs_url="https://oceanobservatories.org/data/",
        authority_score=0.93,
        scientific_acceptance=0.95,
        historical_reliability=0.90,
    ),

    ProviderDefinition(
        source_id="cchdo",
        name="CCHDO (CLIVAR and Carbon Hydrographic Data Office)",
        base_url="https://cchdo.ucsd.edu",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "High-quality global CTD and hydrographic data from GO-SHIP, WOCE, and CLIVAR. "
            "Full-depth ocean sections with temperature, salinity, nutrients, oxygen, carbon."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "oxygen", "nutrients",
                        "dissolved_inorganic_carbon", "alkalinity"],
        api_docs_url="https://cchdo.ucsd.edu/search",
        authority_score=0.96,
        scientific_acceptance=0.97,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="bco_dmo",
        name="BCO-DMO (Biological and Chemical Oceanography Data Management Office)",
        base_url="https://www.bco-dmo.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Free and open biological and chemical oceanography research data. "
            "NSF-funded. Cruise-based and time-series datasets."
        ),
        dataset_types=["ocean", "biology", "chemistry"],
        variables_hint=["chlorophyll", "nutrients", "plankton", "productivity",
                        "biogeochemistry"],
        api_docs_url="https://www.bco-dmo.org/how-get-data",
        authority_score=0.93,
        scientific_acceptance=0.95,
        historical_reliability=0.93,
    ),

    ProviderDefinition(
        source_id="mgds",
        name="MGDS (Marine Geoscience Data System)",
        base_url="https://www.marine-geo.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Free public access to marine geophysical data from global oceans — multibeam "
            "bathymetry, seismic, gravity, magnetics, and sample data."
        ),
        dataset_types=["geology", "bathymetry", "geophysics"],
        variables_hint=["bathymetry", "seismic", "gravity", "magnetics", "seafloor"],
        api_docs_url="https://www.marine-geo.org/portals/mgdl/",
        authority_score=0.93,
        scientific_acceptance=0.94,
        historical_reliability=0.93,
    ),

    ProviderDefinition(
        source_id="glodap",
        name="GLODAP (Global Ocean Data Analysis Project)",
        base_url="https://www.glodap.info",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Global ocean carbon chemistry — internally consistent data product for "
            "ocean carbon cycle research. Covers all ocean basins."
        ),
        dataset_types=["ocean", "carbon"],
        variables_hint=["dissolved_inorganic_carbon", "alkalinity", "ph",
                        "oxygen", "nutrients"],
        api_docs_url="https://www.glodap.info/index.php/data-access/",
        authority_score=0.96,
        scientific_acceptance=0.97,
        historical_reliability=0.96,
    ),

    ProviderDefinition(
        source_id="imos_aodn",
        name="IMOS / AODN (Australian Ocean Data Network)",
        base_url="https://portal.aodn.org.au",
        access_type=AccessType.free,
        discoverer_type="erddap",
        description=(
            "All Australian marine and climate science data openly and freely available. "
            "Ships, autonomous vehicles, moorings, biological and physical observations."
        ),
        dataset_types=["ocean", "climate", "biology"],
        variables_hint=["temperature", "salinity", "current", "wave", "chlorophyll"],
        api_docs_url="https://help.aodn.org.au/web-services/",
        authority_score=0.93,
        scientific_acceptance=0.94,
        historical_reliability=0.93,
        regions=["Australia", "Southern Ocean", "Indian Ocean", "Pacific"],
    ),

    ProviderDefinition(
        source_id="seadatanet",
        name="SeaDataNet",
        base_url="https://www.seadatanet.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Pan-European standardized marine data system covering 35 countries. "
            "Metadata discovery and data access for physical, chemical, biological data."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "current", "nutrients"],
        api_docs_url="https://www.seadatanet.org/Tools",
        authority_score=0.93,
        scientific_acceptance=0.94,
        historical_reliability=0.92,
        regions=["European waters", "Mediterranean", "Baltic", "Arctic", "Atlantic"],
    ),

    ProviderDefinition(
        source_id="jodc",
        name="JODC (Japan Oceanographic Data Center)",
        base_url="https://www.jodc.go.jp",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Japan's national marine data bank — hydrographic, meteorological, fixed "
            "station, and depth data. UNESCO/IOC IODE National Oceanographic Data Centre."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "current", "tide"],
        api_docs_url="https://www.jodc.go.jp/jodcweb/JDOSS/index.html",
        authority_score=0.92,
        scientific_acceptance=0.94,
        historical_reliability=0.93,
        regions=["Japan", "Northwest Pacific", "Indian Ocean"],
    ),

    ProviderDefinition(
        source_id="seanoe",
        name="SEANOE (Sea Scientific Open Data Edition)",
        base_url="https://www.seanoe.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "French publisher of open marine scientific data — physical oceanography, "
            "marine biology, marine geology. Creative Commons licensed, free."
        ),
        dataset_types=["ocean", "biology", "geology"],
        variables_hint=["temperature", "salinity", "current", "species"],
        api_docs_url="https://www.seanoe.org/html/help-access-data.htm",
        authority_score=0.90,
        scientific_acceptance=0.93,
        historical_reliability=0.91,
        regions=["Global", "Atlantic", "Mediterranean"],
    ),

    ProviderDefinition(
        source_id="r2r",
        name="R2R (Rolling Deck to Repository)",
        base_url="https://www.rvdata.us",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Underway data from US research vessel cruises — navigation, meteorology, "
            "bathymetry, and more. Free, no login required."
        ),
        dataset_types=["ocean", "weather"],
        variables_hint=["navigation", "meteorology", "bathymetry", "temperature"],
        api_docs_url="https://www.rvdata.us/about/getting-started",
        authority_score=0.90,
        scientific_acceptance=0.92,
        historical_reliability=0.91,
    ),

    ProviderDefinition(
        source_id="nsidc",
        name="NSIDC (National Snow and Ice Data Center)",
        base_url="https://nsidc.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Snow, ice, and cryosphere data. Most data free. Some high-resolution "
            "products require NASA Earthdata login."
        ),
        dataset_types=["cryosphere", "climate"],
        variables_hint=["sea_ice_extent", "snow_cover", "ice_thickness",
                        "permafrost", "glacial_mass"],
        api_docs_url="https://nsidc.org/data",
        authority_score=0.96,
        scientific_acceptance=0.97,
        historical_reliability=0.96,
    ),

    ProviderDefinition(
        source_id="marine_regions",
        name="Marine Regions",
        base_url="https://www.marineregions.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Geographic boundaries and gazetteer for marine areas — EEZ, high seas, "
            "ocean basins, FAO fishing areas. Free vector download."
        ),
        dataset_types=["geospatial", "ocean"],
        variables_hint=["eez", "ocean_boundary", "marine_zone", "shipping_route"],
        api_docs_url="https://www.marineregions.org/gazetteer.php?p=webservices",
        authority_score=0.93,
        scientific_acceptance=0.95,
        historical_reliability=0.95,
    ),

    # ── India-specific (public) ─────────────────────────────────────────

    ProviderDefinition(
        source_id="imd",
        name="IMD (India Meteorological Department)",
        base_url="https://mausam.imd.gov.in",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Indian meteorological data — cyclone tracks, historical rainfall, weather "
            "forecasts. Most data freely accessible. India and Indian Ocean focus."
        ),
        dataset_types=["weather", "disaster", "climate"],
        variables_hint=["cyclone_track", "rainfall", "temperature",
                        "monsoon", "wind_speed"],
        api_docs_url="https://mausam.imd.gov.in/",
        authority_score=0.93,
        scientific_acceptance=0.93,
        historical_reliability=0.92,
        regions=["India", "Indian Ocean", "Bay of Bengal", "Arabian Sea"],
    ),

    ProviderDefinition(
        source_id="data_gov_in",
        name="Open Government Data Platform India (data.gov.in)",
        base_url="https://data.gov.in",
        access_type=AccessType.free,
        discoverer_type="ckan",
        description=(
            "Indian government open data — environment, fisheries, disaster, coastal, "
            "agriculture, and climate datasets. CKAN-based portal with API access."
        ),
        dataset_types=["multidisciplinary", "ocean", "disaster", "climate"],
        variables_hint=["fisheries", "coastal", "environment", "disaster"],
        api_docs_url="https://data.gov.in/help/how-use-datasets-using-apis",
        authority_score=0.85,
        scientific_acceptance=0.83,
        historical_reliability=0.80,
        regions=["India"],
    ),

    # ── Fisheries ──────────────────────────────────────────────────────

    ProviderDefinition(
        source_id="fao_fisheries",
        name="FAO Fisheries Statistics",
        base_url="https://www.fao.org/fishery",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Global catch and aquaculture statistics — production, trade, consumption, "
            "fleet data from 1950. Free data download via FishStatJ and API."
        ),
        dataset_types=["fisheries", "ocean"],
        variables_hint=["fish_catch", "aquaculture", "fishing_effort", "stock"],
        api_docs_url="https://www.fao.org/fishery/en/statistics/software/fishstatj",
        authority_score=0.95,
        scientific_acceptance=0.95,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="sea_around_us",
        name="Sea Around Us",
        base_url="https://www.seaaroundus.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Reconstructed fisheries data by EEZ — catch reconstruction, ecosystem, "
            "biodiversity, and economics of marine fisheries."
        ),
        dataset_types=["fisheries", "ocean"],
        variables_hint=["fish_catch", "eez_fisheries", "fishing_effort",
                        "marine_biodiversity"],
        api_docs_url="https://www.seaaroundus.org/data/#/fao",
        authority_score=0.90,
        scientific_acceptance=0.92,
        historical_reliability=0.90,
    ),

    ProviderDefinition(
        source_id="ices_data",
        name="ICES Data Centre",
        base_url="https://www.ices.dk/data",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Fisheries, hydrographic, and ecosystem data for the North Atlantic and "
            "adjacent seas. International Council for the Exploration of the Sea."
        ),
        dataset_types=["fisheries", "ocean", "ecosystem"],
        variables_hint=["fish_stock", "temperature", "salinity", "plankton",
                        "fishing_effort"],
        api_docs_url="https://www.ices.dk/data/tools/Pages/ICES-Data-API.aspx",
        authority_score=0.94,
        scientific_acceptance=0.95,
        historical_reliability=0.94,
        regions=["North Atlantic", "North Sea", "Baltic", "Arctic"],
    ),

    ProviderDefinition(
        source_id="earthchem",
        name="EarthChem Library",
        base_url="https://www.earthchem.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description="Geochemical data — petrology, geochemistry, geochronology. Open access.",
        dataset_types=["geology", "geochemistry"],
        variables_hint=["geochemistry", "isotopes", "mineral_composition"],
        api_docs_url="https://www.earthchem.org/resources/tools/",
        authority_score=0.92,
        scientific_acceptance=0.94,
        historical_reliability=0.93,
    ),

    ProviderDefinition(
        source_id="deltares",
        name="Deltares Open Data",
        base_url="https://www.deltares.nl/en/open-data",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Coastal and flood modeling data from Deltares — storm surge, flood hazard, "
            "coastal erosion, and delta management datasets."
        ),
        dataset_types=["coastal", "disaster", "ocean"],
        variables_hint=["storm_surge", "flood", "coastal_erosion", "wave"],
        api_docs_url="https://www.deltares.nl/en/open-data",
        authority_score=0.92,
        scientific_acceptance=0.93,
        historical_reliability=0.92,
    ),

    ProviderDefinition(
        source_id="ioos_data_catalog",
        name="IOOS Data Catalog",
        base_url="https://www.ioos.us",
        access_type=AccessType.free,
        discoverer_type="erddap",
        description=(
            "Thousands of datasets from 11 US regional ocean observing associations. "
            "Integrated data access for coastal and ocean observations."
        ),
        dataset_types=["ocean", "coastal"],
        variables_hint=["temperature", "salinity", "current", "wave", "tide"],
        api_docs_url="https://www.ioos.us/data/",
        authority_score=0.93,
        scientific_acceptance=0.93,
        historical_reliability=0.91,
    ),

    ProviderDefinition(
        source_id="wmo_climate_catalogue",
        name="WMO Climate Data Catalogue",
        base_url="https://climatedata-catalogue.wmo.int",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Climate datasets assessed using WMO Stewardship Maturity Matrix. Open "
            "browse and download. Authoritative climate datasets endorsed by WMO."
        ),
        dataset_types=["climate"],
        variables_hint=["temperature", "precipitation", "sea_level", "ice"],
        api_docs_url="https://climatedata-catalogue.wmo.int/",
        authority_score=0.95,
        scientific_acceptance=0.96,
        historical_reliability=0.95,
    ),

]


# ======================================================================
# TIER 2 — FREE, REGISTRATION REQUIRED
# ======================================================================

TIER2_REGISTRATION: List[ProviderDefinition] = [

    ProviderDefinition(
        source_id="incois_portal",
        name="INCOIS Ocean Data Portal",
        base_url="https://incois.gov.in",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "India's central marine data repository — National Oceanographic Data Centre "
            "and National Argo Data Centre for the Indian Ocean. Ocean state forecasts, "
            "potential fishing zones, tsunami warnings, wave forecasts, oil spill advisories."
        ),
        dataset_types=["ocean", "disaster", "fisheries"],
        variables_hint=["sea_surface_temperature", "wave_height", "tsunami_alert",
                        "potential_fishing_zone", "storm_surge", "salinity"],
        login_url="https://incois.gov.in/portal/member/login.jsp",
        api_docs_url="https://incois.gov.in/portal/datainfo/driodatainfo.jsp",
        requires_login=True,
        authority_score=0.96,
        scientific_acceptance=0.95,
        historical_reliability=0.93,
        regions=["Indian Ocean", "Bay of Bengal", "Arabian Sea", "India coastal"],
    ),

    ProviderDefinition(
        source_id="incois_digital_ocean",
        name="Digital Ocean (INCOIS)",
        base_url="https://do.incois.gov.in",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "INCOIS dedicated ocean data management platform. Free account gives access "
            "to Indian Ocean in-situ and satellite products."
        ),
        dataset_types=["ocean"],
        variables_hint=["sea_surface_temperature", "salinity", "current",
                        "chlorophyll", "wave"],
        login_url="https://do.incois.gov.in",
        requires_login=True,
        authority_score=0.94,
        scientific_acceptance=0.94,
        historical_reliability=0.91,
        regions=["Indian Ocean", "Bay of Bengal", "Arabian Sea"],
    ),

    ProviderDefinition(
        source_id="nasa_earthdata",
        name="NASA Earthdata / PO.DAAC / OceanColor",
        base_url="https://earthdata.nasa.gov",
        access_type=AccessType.registration,
        discoverer_type="cmr",
        description=(
            "All NASA EOSDIS data is free. One Earthdata login covers all components — "
            "PO.DAAC (ocean), OceanColor, NSIDC, ASDC, LP DAAC, and CMR catalog API."
        ),
        dataset_types=["ocean", "atmosphere", "land", "cryosphere", "satellite"],
        variables_hint=["sea_surface_temperature", "chlorophyll", "salinity",
                        "sea_surface_height", "ocean_wind", "ice"],
        login_url="https://urs.earthdata.nasa.gov/",
        api_docs_url="https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html",
        requires_login=True,
        authority_score=0.98,
        scientific_acceptance=0.98,
        historical_reliability=0.97,
    ),

    ProviderDefinition(
        source_id="copernicus_marine",
        name="Copernicus Marine Service (CMEMS)",
        base_url="https://marine.copernicus.eu",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "Ocean physical, biogeochemical, and sea ice products from Copernicus. "
            "Global and regional forecasts, reanalysis, near-real-time data. Free account."
        ),
        dataset_types=["ocean", "climate"],
        variables_hint=["sea_surface_temperature", "salinity", "current",
                        "sea_level", "wave", "sea_ice", "chlorophyll"],
        login_url="https://marine.copernicus.eu/user-corner/user-registration",
        api_docs_url="https://marine.copernicus.eu/services/user-learning-services",
        requires_login=True,
        authority_score=0.97,
        scientific_acceptance=0.97,
        historical_reliability=0.96,
    ),

    ProviderDefinition(
        source_id="copernicus_data_space",
        name="Copernicus Data Space (Sentinel Hub)",
        base_url="https://dataspace.copernicus.eu",
        access_type=AccessType.registration,
        discoverer_type="stac",
        description=(
            "Free Sentinel-1, 2, 3 satellite data — SAR, optical, altimetry. Account "
            "needed for download. STAC catalog for discovery."
        ),
        dataset_types=["satellite", "ocean", "land"],
        variables_hint=["sar", "optical", "land_cover", "ocean_color",
                        "sea_surface_temperature", "altimetry"],
        login_url="https://identity.dataspace.copernicus.eu/",
        api_docs_url="https://documentation.dataspace.copernicus.eu/APIs/OData.html",
        requires_login=True,
        authority_score=0.97,
        scientific_acceptance=0.97,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="copernicus_cds_era5",
        name="Copernicus Climate Data Store (ERA5)",
        base_url="https://cds.climate.copernicus.eu",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "ERA5 global climate reanalysis (1940-present) and CMIP6 climate projections. "
            "Hourly, 31km resolution. Free Copernicus CDS account required."
        ),
        dataset_types=["climate", "weather", "atmosphere"],
        variables_hint=["temperature", "wind_speed", "precipitation", "humidity",
                        "pressure", "sea_surface_temperature", "wave"],
        login_url="https://cds.climate.copernicus.eu/user/register",
        api_docs_url="https://cds.climate.copernicus.eu/api-how-to",
        requires_login=True,
        authority_score=0.98,
        scientific_acceptance=0.98,
        historical_reliability=0.97,
    ),

    ProviderDefinition(
        source_id="copernicus_land",
        name="Copernicus Land Monitoring Service",
        base_url="https://land.copernicus.eu",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "Land cover, vegetation indices (NDVI), ground motion, urban atlas, and "
            "crop monitoring products. Free Copernicus account."
        ),
        dataset_types=["land", "satellite"],
        variables_hint=["land_cover", "vegetation_index", "ndvi", "ground_motion"],
        login_url="https://land.copernicus.eu/en/user-corner/my-account",
        requires_login=True,
        authority_score=0.96,
        scientific_acceptance=0.96,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="copernicus_atmosphere",
        name="Copernicus Atmosphere Monitoring Service (CAMS)",
        base_url="https://atmosphere.copernicus.eu",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "Air quality, atmospheric composition, aerosol, ozone, and greenhouse gas "
            "analysis and forecasts. Free CAMS account."
        ),
        dataset_types=["atmosphere", "climate"],
        variables_hint=["air_quality", "aerosol", "ozone", "co2", "methane",
                        "pm25", "no2"],
        login_url="https://ads.atmosphere.copernicus.eu/user/register",
        api_docs_url="https://ads.atmosphere.copernicus.eu/api-how-to",
        requires_login=True,
        authority_score=0.97,
        scientific_acceptance=0.97,
        historical_reliability=0.96,
    ),

    ProviderDefinition(
        source_id="bhuvan_nrsc",
        name="Bhuvan / NRSC (ISRO)",
        base_url="https://bhuvan.nrsc.gov.in",
        access_type=AccessType.registration,
        discoverer_type="wcs_wms",
        description=(
            "Indian Remote Sensing satellite data including ocean, atmospheric, and "
            "cryospheric products. ResourceSat, OceanSat, CartoDEM, and more. Free registration."
        ),
        dataset_types=["satellite", "ocean", "land", "atmosphere"],
        variables_hint=["ocean_color", "sea_surface_temperature", "land_use",
                        "chlorophyll", "vegetation", "dem"],
        login_url="https://bhuvan.nrsc.gov.in/bhuvan_links.php#",
        api_docs_url="https://bhuvan.nrsc.gov.in/bhuvan_links.php",
        requires_login=True,
        authority_score=0.92,
        scientific_acceptance=0.91,
        historical_reliability=0.88,
        regions=["India", "Indian Ocean", "South Asia"],
    ),

    ProviderDefinition(
        source_id="bhoonidhi_nrsc",
        name="Bhoonidhi (NRSC/ISRO)",
        base_url="https://bhoonidhi.nrsc.gov.in",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "NRSC/ISRO satellite data portal including NISAR mission data. "
            "API available (contact needed). Free registration."
        ),
        dataset_types=["satellite", "land", "ocean"],
        variables_hint=["sar", "optical", "land_surface", "terrain"],
        login_url="https://bhoonidhi.nrsc.gov.in/bhoonidhi/login.html",
        api_docs_url="https://bhoonidhi.nrsc.gov.in/bhoonidhi/",
        requires_login=True,
        authority_score=0.91,
        scientific_acceptance=0.90,
        historical_reliability=0.87,
        regions=["India", "South Asia"],
    ),

    ProviderDefinition(
        source_id="hycom",
        name="HYCOM (Hybrid Coordinate Ocean Model)",
        base_url="https://www.hycom.org",
        access_type=AccessType.registration,
        discoverer_type="thredds",
        description=(
            "Ocean circulation model data — global and regional runs. Temperature, "
            "salinity, velocity, sea surface height. Free account for bulk access."
        ),
        dataset_types=["ocean", "climate"],
        variables_hint=["temperature", "salinity", "ocean_currents",
                        "sea_surface_height", "mixed_layer"],
        login_url="https://www.hycom.org/dataserver",
        api_docs_url="https://www.hycom.org/dataserver",
        requires_login=True,
        authority_score=0.94,
        scientific_acceptance=0.94,
        historical_reliability=0.92,
    ),

    ProviderDefinition(
        source_id="global_fishing_watch",
        name="Global Fishing Watch",
        base_url="https://globalfishingwatch.org",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "AIS vessel tracking data — fishing effort, vessel presence, apparent "
            "fishing hours globally. Free research account."
        ),
        dataset_types=["fisheries", "ocean"],
        variables_hint=["fishing_effort", "vessel_tracking", "ais",
                        "fishing_hours"],
        login_url="https://globalfishingwatch.org/api-login/",
        api_docs_url="https://globalfishingwatch.org/our-apis/",
        requires_login=True,
        authority_score=0.90,
        scientific_acceptance=0.91,
        historical_reliability=0.89,
    ),

    ProviderDefinition(
        source_id="opentopography",
        name="OpenTopography",
        base_url="https://opentopography.org",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "High-resolution terrain and LiDAR data — SRTM, ALOS, CopernicusDEM, and "
            "community-contributed LiDAR point clouds. Free account."
        ),
        dataset_types=["topography", "land"],
        variables_hint=["elevation", "dem", "lidar", "terrain", "topography"],
        login_url="https://portal.opentopography.org/login",
        api_docs_url="https://opentopography.org/developers",
        requires_login=True,
        authority_score=0.94,
        scientific_acceptance=0.95,
        historical_reliability=0.94,
    ),

    ProviderDefinition(
        source_id="esgf",
        name="ESGF (Earth System Grid Federation)",
        base_url="https://esgf.llnl.gov",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "CMIP5/CMIP6 climate model output — global climate model simulations for "
            "IPCC assessment reports. Free, account needed per node."
        ),
        dataset_types=["climate"],
        variables_hint=["temperature", "precipitation", "sea_level", "ocean_heat",
                        "sea_ice"],
        login_url="https://esgf.llnl.gov/user_info.html",
        api_docs_url="https://esgf.llnl.gov/search_docs.html",
        requires_login=True,
        authority_score=0.97,
        scientific_acceptance=0.98,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="iucn_red_list",
        name="IUCN Red List",
        base_url="https://www.iucnredlist.org",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "Global species assessments for 172,600+ species with spatial distribution "
            "data. Free account to download via REST API."
        ),
        dataset_types=["biodiversity"],
        variables_hint=["species_distribution", "conservation_status",
                        "threat_level", "range_map"],
        login_url="https://www.iucnredlist.org/users/sign_up",
        api_docs_url="https://apiv3.iucnredlist.org/api/v3/docs",
        requires_login=True,
        authority_score=0.97,
        scientific_acceptance=0.97,
        historical_reliability=0.96,
    ),

    ProviderDefinition(
        source_id="bodc",
        name="BODC (British Oceanographic Data Centre)",
        base_url="https://www.bodc.ac.uk",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "UK and global marine data — CTD profiles, time series, underway, and "
            "model data. Free registration for some datasets."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "current", "oxygen", "nutrients"],
        login_url="https://www.bodc.ac.uk/about/registration/",
        api_docs_url="https://www.bodc.ac.uk/data/online_delivery/",
        requires_login=True,
        authority_score=0.94,
        scientific_acceptance=0.95,
        historical_reliability=0.94,
        regions=["UK", "North Atlantic", "Arctic"],
    ),

    ProviderDefinition(
        source_id="noaa_nodd",
        name="NOAA Open Data Dissemination (NODD)",
        base_url="https://www.noaa.gov/nodd",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "NOAA datasets on AWS, Google Cloud, and Azure — GFS, Global Drifter "
            "Program, NCEP/NCAR Reanalysis, Climate Data Records. Some need NOAA account."
        ),
        dataset_types=["weather", "ocean", "climate"],
        variables_hint=["temperature", "wind", "pressure", "drifter", "reanalysis"],
        login_url="https://www.noaa.gov/nodd",
        api_docs_url="https://www.noaa.gov/nodd",
        requires_login=True,
        authority_score=0.96,
        scientific_acceptance=0.96,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="geoss_portal",
        name="GEOSS Portal",
        base_url="https://www.geoportal.org",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "Index of thousands of Earth observation datasets from 100+ organizations. "
            "Group on Earth Observations metadata portal. Free login for full features."
        ),
        dataset_types=["multidisciplinary"],
        variables_hint=["earth_observation", "satellite", "climate", "ocean"],
        login_url="https://www.geoportal.org/",
        api_docs_url="https://www.geoportal.org/api",
        requires_login=True,
        authority_score=0.88,
        scientific_acceptance=0.88,
        historical_reliability=0.85,
    ),

    ProviderDefinition(
        source_id="planetary_computer",
        name="Microsoft Planetary Computer",
        base_url="https://planetarycomputer.microsoft.com",
        access_type=AccessType.registration,
        discoverer_type="stac",
        description=(
            "Microsoft-hosted STAC catalog with petabytes of Earth science data — "
            "Landsat, Sentinel, MODIS, ERA5, NAIP. Free account for API access."
        ),
        dataset_types=["satellite", "climate", "land", "ocean"],
        variables_hint=["land_cover", "vegetation", "temperature", "snow_ice",
                        "ocean_color", "sar"],
        login_url="https://planetarycomputer.microsoft.com/account/request",
        api_docs_url="https://planetarycomputer.microsoft.com/docs/overview/about",
        requires_login=True,
        authority_score=0.93,
        scientific_acceptance=0.93,
        historical_reliability=0.93,
    ),

]


# ======================================================================
# TIER 4 — PAID / NON-COMMERCIAL RESTRICTED
# ======================================================================

TIER4_PAID: List[ProviderDefinition] = [

    ProviderDefinition(
        source_id="planet_labs",
        name="Planet Labs",
        base_url="https://www.planet.com",
        access_type=AccessType.paid,
        discoverer_type="stac",
        description=(
            "Daily 3-5m optical satellite imagery of the entire Earth. Planet STAC API "
            "for discovery. Commercial subscription required. Education/research plans available."
        ),
        dataset_types=["satellite", "land", "ocean"],
        variables_hint=["optical_imagery", "land_change", "vegetation", "coastal"],
        login_url="https://www.planet.com/login/",
        pricing_url="https://www.planet.com/pricing/",
        api_docs_url="https://developers.planet.com/",
        requires_login=True,
        requires_payment=True,
        price_estimate="Contact Planet for research pricing. Commercial from ~$5,000/year.",
        authority_score=0.92,
        scientific_acceptance=0.90,
        historical_reliability=0.90,
    ),

    ProviderDefinition(
        source_id="maxar",
        name="Maxar WorldView",
        base_url="https://www.maxar.com",
        access_type=AccessType.paid,
        discoverer_type="rest",
        description=(
            "Very high-resolution (30cm) optical satellite imagery. Commercial license "
            "required. Government and research programs may have special access."
        ),
        dataset_types=["satellite", "land"],
        variables_hint=["high_resolution_imagery", "coastal_mapping", "disaster_assessment"],
        login_url="https://www.maxar.com/contact",
        pricing_url="https://www.maxar.com/products/satellite-imagery",
        requires_login=True,
        requires_payment=True,
        price_estimate="Per-scene pricing — typically $10–$25/km². Contact Maxar for research pricing.",
        authority_score=0.92,
        scientific_acceptance=0.88,
        historical_reliability=0.88,
    ),

    ProviderDefinition(
        source_id="tomorrow_io",
        name="Tomorrow.io",
        base_url="https://www.tomorrow.io",
        access_type=AccessType.paid,
        discoverer_type="rest",
        description=(
            "Commercial weather intelligence API — hyperlocal forecasts, historical weather, "
            "severe weather insights. Free tier available (limited). Paid for production use."
        ),
        dataset_types=["weather"],
        variables_hint=["temperature", "precipitation", "wind", "humidity",
                        "severe_weather"],
        login_url="https://app.tomorrow.io/",
        pricing_url="https://www.tomorrow.io/weather-api/pricing/",
        api_docs_url="https://docs.tomorrow.io/reference/api-overview",
        requires_login=True,
        requires_payment=True,
        price_estimate="Free tier: 500 calls/day. Paid from ~$99/month.",
        authority_score=0.82,
        scientific_acceptance=0.78,
        historical_reliability=0.82,
    ),

    ProviderDefinition(
        source_id="spire_global",
        name="Spire Global",
        base_url="https://spire.com",
        access_type=AccessType.paid,
        discoverer_type="rest",
        description=(
            "Commercial satellite data — GNSS-RO weather soundings, maritime AIS, "
            "aviation tracking. Commercial license required."
        ),
        dataset_types=["weather", "ocean", "atmosphere"],
        variables_hint=["temperature_profile", "humidity_profile", "ais",
                        "vessel_tracking"],
        login_url="https://spire.com/contact-us/",
        pricing_url="https://spire.com/maritime/",
        requires_login=True,
        requires_payment=True,
        price_estimate="Contact Spire for pricing. Research access negotiable.",
        authority_score=0.85,
        scientific_acceptance=0.82,
        historical_reliability=0.83,
    ),

    ProviderDefinition(
        source_id="descartes_labs",
        name="Descartes Labs",
        base_url="https://www.descarteslabs.com",
        access_type=AccessType.paid,
        discoverer_type="rest",
        description=(
            "Geospatial analytics platform — access to satellite data, models, and "
            "AI-ready datasets. Commercial platform."
        ),
        dataset_types=["satellite", "land", "agriculture"],
        variables_hint=["vegetation", "land_use", "crop_monitoring",
                        "change_detection"],
        login_url="https://www.descarteslabs.com/contact/",
        pricing_url="https://www.descarteslabs.com/",
        requires_login=True,
        requires_payment=True,
        price_estimate="Contact Descartes Labs for pricing.",
        authority_score=0.83,
        scientific_acceptance=0.80,
        historical_reliability=0.80,
    ),

    ProviderDefinition(
        source_id="google_earth_engine",
        name="Google Earth Engine",
        base_url="https://earthengine.google.com",
        access_type=AccessType.paid,
        discoverer_type="rest",
        description=(
            "900+ Earth observation datasets with cloud compute. Free for academic and "
            "nonprofit use. Paid for commercial/operational use. Requires application."
        ),
        dataset_types=["satellite", "climate", "land", "ocean"],
        variables_hint=["land_cover", "vegetation", "sea_surface_temperature",
                        "precipitation", "snow_ice"],
        login_url="https://signup.earthengine.google.com/",
        pricing_url="https://earthengine.google.com/commercial/",
        api_docs_url="https://developers.google.com/earth-engine/",
        requires_login=True,
        requires_payment=True,
        price_estimate="Free for non-commercial research. Commercial pricing by application.",
        authority_score=0.93,
        scientific_acceptance=0.92,
        historical_reliability=0.92,
    ),

    # ── Added from sources.docx gap check ────────────────────────────────

    ProviderDefinition(
        source_id="janes_ihs",
        name="Jane's / IHS Jane's",
        base_url="https://www.janes.com",
        access_type=AccessType.paid,
        discoverer_type="web",
        description=(
            "Defense and aerospace intelligence. Paid subscription required."
        ),
        dataset_types=["defense"],
        variables_hint=["defense_intelligence", "aerospace"],
        pricing_url="https://www.janes.com/contact-us",
        requires_payment=True,
        price_estimate="Contact Jane's for subscription pricing.",
        authority_score=0.85,
        scientific_acceptance=0.75,
        historical_reliability=0.85,
    ),

    ProviderDefinition(
        source_id="eumetsat",
        name="EUMETSAT (select products)",
        base_url="https://user.eumetsat.int",
        access_type=AccessType.paid,
        discoverer_type="rest",
        description=(
            "Most Sentinel products are free; some high-resolution and "
            "near-real-time products require a license."
        ),
        dataset_types=["satellite", "atmosphere", "ocean"],
        variables_hint=["satellite_imagery", "sea_surface_temperature",
                        "atmospheric_composition"],
        login_url="https://user.eumetsat.int/",
        pricing_url="https://www.eumetsat.int/data-licensing",
        requires_login=True,
        requires_payment=True,
        price_estimate="Most products free; select high-res/NRT products require a license.",
        authority_score=0.92,
        scientific_acceptance=0.93,
        historical_reliability=0.92,
    ),

    ProviderDefinition(
        source_id="waqi_bulk_api",
        name="World Air Quality Index (bulk API)",
        base_url="https://aqicn.org",
        access_type=AccessType.paid,
        discoverer_type="rest",
        description=(
            "Bulk/commercial API access to the World Air Quality Index "
            "is paid. Free token covers limited individual-station queries."
        ),
        dataset_types=["atmosphere"],
        variables_hint=["air_quality_index", "pm2_5", "pm10", "ozone", "no2"],
        api_docs_url="https://aqicn.org/api/",
        pricing_url="https://aqicn.org/data-platform/token/",
        requires_payment=True,
        price_estimate="Free for limited individual queries; bulk/commercial access is paid.",
        authority_score=0.78,
        scientific_acceptance=0.75,
        historical_reliability=0.82,
    ),

]


# ======================================================================
# TIER 1 — ADDITIONAL FREE PROVIDERS (from doc, not in original list)
# ======================================================================

TIER1_ADDITIONAL: List[ProviderDefinition] = [

    ProviderDefinition(
        source_id="world_ocean_atlas",
        name="World Ocean Atlas (WOA)",
        base_url="https://www.ncei.noaa.gov/products/world-ocean-atlas",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Climatological ocean temperature, salinity, dissolved oxygen — "
            "most widely used free reference dataset for ocean baselines."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "dissolved_oxygen", "nutrients",
                        "apparent_oxygen_utilization"],
        api_docs_url="https://www.ncei.noaa.gov/products/world-ocean-atlas",
        authority_score=0.97,
        scientific_acceptance=0.97,
        historical_reliability=0.98,
    ),

    ProviderDefinition(
        source_id="noaa_socat",
        name="SOCAT — Surface Ocean CO₂ Atlas",
        base_url="https://www.socat.info",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Quality-controlled surface ocean CO₂ atlas — 33M+ observations. "
            "Freely available for download."
        ),
        dataset_types=["ocean"],
        variables_hint=["co2", "pco2", "fugacity_co2", "carbon"],
        authority_score=0.94,
        scientific_acceptance=0.97,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="noaa_gtspp",
        name="GTSPP — Global Temp/Salinity Profile Programme",
        base_url="https://www.ncei.noaa.gov",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Real-time and delayed-mode ocean temperature and salinity profiles "
            "from the Global Temperature and Salinity Profile Programme."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "ocean_profiles"],
        authority_score=0.92,
        scientific_acceptance=0.93,
        historical_reliability=0.90,
    ),

    ProviderDefinition(
        source_id="tao_triton",
        name="TAO/TRITON/PIRATA/IndOOS Moored Buoys",
        base_url="https://www.pmel.noaa.gov/tao",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Tropical moored buoy arrays in Pacific, Atlantic, and Indian Oceans — "
            "real-time temperature, salinity, currents, wind."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "ocean_currents", "wind",
                        "relative_humidity"],
        api_docs_url="https://www.pmel.noaa.gov/tao/drupal/data/",
        authority_score=0.95,
        scientific_acceptance=0.95,
        historical_reliability=0.92,
    ),

    ProviderDefinition(
        source_id="noaa_gloss",
        name="GLOSS — Global Sea Level Observing System",
        base_url="https://www.gloss-sealevel.info",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Worldwide tide gauge records and sea level time series — "
            "global network operated under UNESCO/IOC."
        ),
        dataset_types=["ocean"],
        variables_hint=["sea_level", "tidal_records", "mean_sea_level"],
        authority_score=0.93,
        scientific_acceptance=0.95,
        historical_reliability=0.93,
    ),

    ProviderDefinition(
        source_id="glodap",
        name="GLODAP — Global Ocean Carbon Chemistry",
        base_url="https://www.glodap.info",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Internally consistent quality-controlled global ocean carbon chemistry — "
            "DIC, alkalinity, oxygen, nutrients, pH."
        ),
        dataset_types=["ocean"],
        variables_hint=["dissolved_inorganic_carbon", "alkalinity", "dissolved_oxygen",
                        "nutrients", "pH", "carbon"],
        authority_score=0.95,
        scientific_acceptance=0.97,
        historical_reliability=0.96,
    ),

    ProviderDefinition(
        source_id="cchdo",
        name="CCHDO — GO-SHIP / WOCE / CLIVAR CTD Data",
        base_url="https://cchdo.ucsd.edu",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "High-quality global CTD and hydrographic data from GO-SHIP, WOCE, "
            "and CLIVAR cruises."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "dissolved_oxygen", "nutrients",
                        "chlorophyll", "pressure"],
        api_docs_url="https://cchdo.ucsd.edu/api/v1/docs",
        authority_score=0.95,
        scientific_acceptance=0.97,
        historical_reliability=0.96,
    ),

    ProviderDefinition(
        source_id="ooi",
        name="OOI — Ocean Observatories Initiative",
        base_url="https://oceanobservatories.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Physical, chemical, geological, and biological ocean observations — "
            "open to anyone via OOI Data Explorer."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "dissolved_oxygen", "pH",
                        "chlorophyll", "ocean_currents", "acoustic", "seismic"],
        api_docs_url="https://oceanobservatories.org/data-streams/",
        authority_score=0.92,
        scientific_acceptance=0.93,
        historical_reliability=0.88,
    ),

    ProviderDefinition(
        source_id="ioos_data_catalog",
        name="IOOS Data Catalog",
        base_url="https://www.ioos.us",
        access_type=AccessType.free,
        discoverer_type="erddap",
        description=(
            "Thousands of datasets from 11 US regional ocean observing associations — "
            "coastal oceanography, ecology, fisheries."
        ),
        dataset_types=["ocean"],
        variables_hint=["sea_surface_temperature", "salinity", "currents",
                        "wave_height", "water_level"],
        api_docs_url="https://ioos.us/data/",
        authority_score=0.90,
        scientific_acceptance=0.90,
        historical_reliability=0.88,
    ),

    ProviderDefinition(
        source_id="allen_coral_atlas",
        name="Allen Coral Atlas",
        base_url="https://allencoralatlas.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Global coral reef mapping at 5m resolution — free download. "
            "Benthic cover and geomorphic zonation."
        ),
        dataset_types=["ocean", "biodiversity"],
        variables_hint=["coral_reef_mapping", "benthic_cover", "reef_extent"],
        authority_score=0.93,
        scientific_acceptance=0.95,
        historical_reliability=0.92,
    ),

    ProviderDefinition(
        source_id="arctic_data_center",
        name="Arctic Data Center",
        base_url="https://arcticdata.io",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Arctic research datasets — open access NSF-funded repository. "
            "Sea ice, permafrost, oceanography, atmosphere."
        ),
        dataset_types=["ocean", "climate"],
        variables_hint=["arctic_ocean", "sea_ice", "permafrost", "atmosphere"],
        authority_score=0.90,
        scientific_acceptance=0.93,
        historical_reliability=0.90,
    ),

    ProviderDefinition(
        source_id="scripps",
        name="Scripps Institution of Oceanography",
        base_url="https://scripps.ucsd.edu/research/data-programs",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "CalCOFI, Argo, atmospheric gas, and coastal ocean data from UCSD Scripps. "
            "Publicly accessible."
        ),
        dataset_types=["ocean"],
        variables_hint=["co2_atmospheric", "ocean_temperature", "salinity",
                        "coastal_currents", "calcofi"],
        authority_score=0.94,
        scientific_acceptance=0.96,
        historical_reliability=0.93,
    ),

    ProviderDefinition(
        source_id="whoi_data",
        name="WHOI Data Portal",
        base_url="https://www.whoi.edu/what-we-do/understand/data",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Float, mooring, seafloor, and oceanic data from the Woods Hole "
            "Oceanographic Institution."
        ),
        dataset_types=["ocean"],
        variables_hint=["float_data", "mooring_data", "seafloor_data", "oceanic_data"],
        authority_score=0.94,
        scientific_acceptance=0.96,
        historical_reliability=0.93,
    ),

    ProviderDefinition(
        source_id="ices_data",
        name="ICES Data Centre",
        base_url="https://www.ices.dk/data",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Fisheries, hydrographic, and ecosystem data for the North Atlantic — "
            "most datasets freely downloadable."
        ),
        dataset_types=["ocean", "fisheries"],
        variables_hint=["fisheries_data", "hydrographic", "ecosystem", "fish_stock"],
        api_docs_url="https://www.ices.dk/data/tools/Pages/API.aspx",
        authority_score=0.93,
        scientific_acceptance=0.94,
        historical_reliability=0.92,
    ),

    ProviderDefinition(
        source_id="sea_around_us",
        name="Sea Around Us",
        base_url="https://www.seaaroundus.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Reconstructed fisheries data by EEZ — goes beyond FAO official statistics "
            "to include unreported and discarded catches."
        ),
        dataset_types=["fisheries"],
        variables_hint=["fish_catch", "eez_catch", "reconstructed_fisheries"],
        authority_score=0.90,
        scientific_acceptance=0.93,
        historical_reliability=0.88,
    ),

    ProviderDefinition(
        source_id="openaq",
        name="OpenAQ",
        base_url="https://openaq.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Open air quality data from government-grade sensors worldwide — "
            "PM2.5, PM10, ozone, NO2, CO, SO2. Free API."
        ),
        dataset_types=["atmosphere"],
        variables_hint=["pm25", "pm10", "ozone", "no2", "co", "so2", "air_quality"],
        api_docs_url="https://docs.openaq.org/",
        authority_score=0.88,
        scientific_acceptance=0.85,
        historical_reliability=0.88,
    ),

    ProviderDefinition(
        source_id="openstreetmap",
        name="OpenStreetMap",
        base_url="https://www.openstreetmap.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Open collaborative map — coastlines, ports, infrastructure for GIS overlays. "
            "Used for spatial context in ocean and disaster analysis."
        ),
        dataset_types=["terrain"],
        variables_hint=["coastlines", "roads", "ports", "buildings", "land_use"],
        api_docs_url="https://wiki.openstreetmap.org/wiki/API",
        authority_score=0.82,
        scientific_acceptance=0.75,
        historical_reliability=0.92,
    ),

    ProviderDefinition(
        source_id="wmo_climate_catalogue",
        name="WMO Climate Data Catalogue",
        base_url="https://climatedata-catalogue.wmo.int",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Climate datasets assessed using WMO Stewardship Maturity Matrix — "
            "open browse and download."
        ),
        dataset_types=["climate"],
        variables_hint=["temperature", "precipitation", "wind", "humidity", "pressure"],
        authority_score=0.93,
        scientific_acceptance=0.93,
        historical_reliability=0.90,
    ),

    # ── Repositories & indexes (added from sources.docx gap check) ──────

    ProviderDefinition(
        source_id="figshare",
        name="Figshare",
        base_url="https://figshare.com",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Open research data repository — figures, datasets, and preprints. "
            "Free to browse and download."
        ),
        dataset_types=["multi"],
        variables_hint=["research_data", "figures", "preprints"],
        api_docs_url="https://docs.figshare.com/",
        authority_score=0.80,
        scientific_acceptance=0.80,
        historical_reliability=0.85,
    ),

    ProviderDefinition(
        source_id="dryad",
        name="Dryad",
        base_url="https://datadryad.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Peer-reviewed research data repository, biology-heavy. "
            "Free to browse and download."
        ),
        dataset_types=["biology"],
        variables_hint=["species_data", "ecological_data", "research_data"],
        api_docs_url="https://datadryad.org/api/v2/docs/",
        authority_score=0.85,
        scientific_acceptance=0.87,
        historical_reliability=0.85,
    ),

    ProviderDefinition(
        source_id="aquadocs",
        name="AquaDocs",
        base_url="https://aquadocs.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Joint UNESCO/IOC IODE open-access repository — marine science "
            "publications, technical reports, and working papers."
        ),
        dataset_types=["ocean"],
        variables_hint=["marine_science", "technical_reports", "publications"],
        authority_score=0.83,
        scientific_acceptance=0.83,
        historical_reliability=0.85,
    ),

    ProviderDefinition(
        source_id="openalex",
        name="OpenAlex",
        base_url="https://openalex.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Open scholarly graph covering works, authors, and institutions. "
            "No login required."
        ),
        dataset_types=["multi"],
        variables_hint=["scholarly_works", "citations", "authors"],
        api_docs_url="https://docs.openalex.org/",
        authority_score=0.82,
        scientific_acceptance=0.82,
        historical_reliability=0.85,
    ),

    ProviderDefinition(
        source_id="crossref_api",
        name="Crossref API",
        base_url="https://api.crossref.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "DOI metadata and citation data — fully open, no registration."
        ),
        dataset_types=["multi"],
        variables_hint=["doi_metadata", "citations"],
        api_docs_url="https://api.crossref.org/swagger-ui/index.html",
        authority_score=0.85,
        scientific_acceptance=0.85,
        historical_reliability=0.90,
    ),

    ProviderDefinition(
        source_id="arxiv",
        name="arXiv",
        base_url="https://arxiv.org",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "Preprints in physics, geoscience, and computer science. "
            "Completely open access."
        ),
        dataset_types=["multi"],
        variables_hint=["preprints", "research_papers"],
        api_docs_url="https://info.arxiv.org/help/api/index.html",
        authority_score=0.82,
        scientific_acceptance=0.85,
        historical_reliability=0.90,
    ),

    ProviderDefinition(
        source_id="gcrmn",
        name="GCRMN",
        base_url="https://gcrmn.net",
        access_type=AccessType.free,
        discoverer_type="web",
        description=(
            "Global Coral Reef Monitoring Network — coral reef monitoring "
            "reports and status assessments."
        ),
        dataset_types=["biology", "ocean"],
        variables_hint=["coral_reef", "reef_health", "monitoring_reports"],
        authority_score=0.85,
        scientific_acceptance=0.87,
        historical_reliability=0.85,
    ),

    ProviderDefinition(
        source_id="openaire_graph",
        name="OpenAIRE Research Graph",
        base_url="https://graph.openaire.eu",
        access_type=AccessType.free,
        discoverer_type="rest",
        description=(
            "European open science graph linking publications, datasets, "
            "software, and projects."
        ),
        dataset_types=["multi"],
        variables_hint=["research_outputs", "publications", "datasets"],
        api_docs_url="https://graph.openaire.eu/docs/",
        authority_score=0.83,
        scientific_acceptance=0.83,
        historical_reliability=0.85,
    ),

    ProviderDefinition(
        source_id="marinecadastre",
        name="MarineCadastre.gov",
        base_url="https://marinecadastre.gov",
        access_type=AccessType.free,
        discoverer_type="web",
        description=(
            "US vessel traffic (AIS) and marine spatial planning data. "
            "Free download."
        ),
        dataset_types=["ocean"],
        variables_hint=["vessel_traffic", "ais", "marine_spatial_planning"],
        authority_score=0.85,
        scientific_acceptance=0.85,
        historical_reliability=0.87,
    ),

    ProviderDefinition(
        source_id="google_dataset_search",
        name="Google Dataset Search",
        base_url="https://datasetsearch.research.google.com",
        access_type=AccessType.free,
        discoverer_type="web",
        description=(
            "Index of datasets across the web — directs users to the "
            "original hosting source to download. Not a direct download portal."
        ),
        dataset_types=["multi"],
        variables_hint=["dataset_discovery"],
        browse_only=True,
        authority_score=0.78,
        scientific_acceptance=0.75,
        historical_reliability=0.85,
    ),

    ProviderDefinition(
        source_id="geo_portal",
        name="GEO (Group on Earth Observations)",
        base_url="https://earthobservations.org",
        access_type=AccessType.free,
        discoverer_type="web",
        description=(
            "Discovery and coordination layer for Earth observation data "
            "across member agencies. Not a direct download portal."
        ),
        dataset_types=["multi"],
        variables_hint=["earth_observation", "coordination"],
        browse_only=True,
        authority_score=0.80,
        scientific_acceptance=0.80,
        historical_reliability=0.82,
    ),

    ProviderDefinition(
        source_id="waqi_viewer",
        name="World Air Quality Index (viewer)",
        base_url="https://aqicn.org",
        access_type=AccessType.api_key,
        discoverer_type="rest",
        description=(
            "Global air quality index viewer. Free to browse; an API token "
            "is needed for programmatic access to individual station data."
        ),
        dataset_types=["atmosphere"],
        variables_hint=["air_quality_index", "pm2_5", "pm10", "ozone", "no2"],
        api_docs_url="https://aqicn.org/api/",
        browse_only=True,
        authority_score=0.78,
        scientific_acceptance=0.75,
        historical_reliability=0.82,
    ),

]


# ======================================================================
# TIER 2 — ADDITIONAL REGISTRATION PROVIDERS
# ======================================================================

TIER2_ADDITIONAL: List[ProviderDefinition] = [

    ProviderDefinition(
        source_id="esgf",
        name="ESGF — Earth System Grid Federation (CMIP)",
        base_url="https://esgf.llnl.gov",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "CMIP5/CMIP6 climate model output — temperature, precipitation, sea level "
            "projections. Free account per ESGF node."
        ),
        dataset_types=["climate"],
        variables_hint=["cmip_temperature", "cmip_precipitation", "cmip_sea_level",
                        "climate_projections", "cmip"],
        login_url="https://esgf-node.llnl.gov/user/add/",
        api_docs_url="https://esgf.github.io/esgf-user-support/",
        requires_login=True,
        authority_score=0.95,
        scientific_acceptance=0.97,
        historical_reliability=0.90,
    ),

    ProviderDefinition(
        source_id="bodc",
        name="BODC — British Oceanographic Data Centre",
        base_url="https://www.bodc.ac.uk",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "UK and global marine data — hydrography, acoustics, biology. "
            "Free, some datasets require registration."
        ),
        dataset_types=["ocean"],
        variables_hint=["temperature", "salinity", "currents", "chemistry", "biological"],
        login_url="https://www.bodc.ac.uk/account/register/",
        api_docs_url="https://www.bodc.ac.uk/data/bodc_database/nodb/data_delivery/",
        requires_login=True,
        authority_score=0.93,
        scientific_acceptance=0.94,
        historical_reliability=0.92,
    ),

    ProviderDefinition(
        source_id="opentopography",
        name="OpenTopography",
        base_url="https://opentopography.org",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "High-resolution terrain and LiDAR datasets — DEMs, point clouds, "
            "coastal bathymetry. Free account required."
        ),
        dataset_types=["terrain"],
        variables_hint=["lidar_dem", "point_cloud", "topography", "bathymetry_coastal"],
        login_url="https://portal.opentopography.org/login",
        api_docs_url="https://opentopography.org/developers",
        requires_login=True,
        authority_score=0.92,
        scientific_acceptance=0.93,
        historical_reliability=0.92,
    ),

    ProviderDefinition(
        source_id="iucn_red_list",
        name="IUCN Red List",
        base_url="https://www.iucnredlist.org",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "Global species assessments for 172,600+ species with spatial distribution maps — "
            "free account to download."
        ),
        dataset_types=["biodiversity"],
        variables_hint=["species_status", "species_range_maps", "population_trend",
                        "extinction_risk"],
        login_url="https://www.iucnredlist.org/join",
        api_docs_url="https://apiv3.iucnredlist.org/api/v3/docs",
        requires_login=True,
        authority_score=0.96,
        scientific_acceptance=0.97,
        historical_reliability=0.95,
    ),

    ProviderDefinition(
        source_id="noaa_nodd",
        name="NOAA NODD — Cloud Datasets (AWS/GCP/Azure)",
        base_url="https://www.noaa.gov/nodd",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "NOAA datasets on AWS, Google Cloud, and Azure — Global Drifter Program, "
            "GFS, NCEP/NCAR Reanalysis, Climate Data Records. Some need NOAA account."
        ),
        dataset_types=["ocean", "atmosphere", "climate"],
        variables_hint=["gfs_forecast", "ncep_reanalysis", "global_drifters",
                        "climate_data_records"],
        login_url="https://www.noaa.gov/information-technology/open-data-dissemination",
        requires_login=True,
        authority_score=0.95,
        scientific_acceptance=0.95,
        historical_reliability=0.93,
    ),

    ProviderDefinition(
        source_id="dataverse",
        name="Dataverse (Harvard Dataverse)",
        base_url="https://dataverse.org",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "Research data repositories including Harvard Dataverse — "
            "free accounts for full access."
        ),
        dataset_types=["repository"],
        variables_hint=["research_data", "marine_data", "environmental_data"],
        login_url="https://dataverse.harvard.edu/loginpage.xhtml",
        api_docs_url="https://guides.dataverse.org/en/latest/api/",
        requires_login=True,
        authority_score=0.87,
        scientific_acceptance=0.88,
        historical_reliability=0.90,
    ),

    ProviderDefinition(
        source_id="geoss_portal",
        name="GEOSS Portal",
        base_url="https://www.geoportal.org",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "Index of thousands of Earth observation datasets globally — "
            "free login for full discovery and download features."
        ),
        dataset_types=["multi"],
        variables_hint=["earth_observation", "satellite", "environmental"],
        login_url="https://www.geoportal.org/",
        requires_login=True,
        authority_score=0.88,
        scientific_acceptance=0.87,
        historical_reliability=0.87,
    ),

    ProviderDefinition(
        source_id="earthdata_earthexplorer",
        name="USGS EarthExplorer",
        base_url="https://earthexplorer.usgs.gov",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "USGS satellite imagery and terrain data — Landsat archive, ASTER, DEMs, "
            "aerial photography. Uses same URS login as NASA Earthdata."
        ),
        dataset_types=["satellite", "terrain"],
        variables_hint=["landsat", "aster", "dem", "aerial_photography", "sar"],
        login_url="https://ers.cr.usgs.gov/register",
        api_docs_url="https://m2m.cr.usgs.gov/api/docs/json/",
        requires_login=True,
        authority_score=0.96,
        scientific_acceptance=0.95,
        historical_reliability=0.94,
    ),

    # ── Added from sources.docx gap check ────────────────────────────────

    ProviderDefinition(
        source_id="whoi_open_access",
        name="WHOI Open Access Server (WHOAS)",
        base_url="https://whoas.whoi.edu",
        access_type=AccessType.registration,
        discoverer_type="web",
        description=(
            "Publications, grey literature, and data from the Woods Hole "
            "scientific community. Free; some items may need account access."
        ),
        dataset_types=["ocean"],
        variables_hint=["publications", "grey_literature", "research_data"],
        authority_score=0.85,
        scientific_acceptance=0.87,
        historical_reliability=0.87,
    ),

    ProviderDefinition(
        source_id="scar_antarctic",
        name="SCAR Antarctic Data Centre",
        base_url="https://www.scar.org",
        access_type=AccessType.registration,
        discoverer_type="web",
        description=(
            "Antarctic science data coordinated by the Scientific Committee "
            "on Antarctic Research. Free, some datasets need account access."
        ),
        dataset_types=["climate", "biology", "cryosphere"],
        variables_hint=["antarctic_research", "ice", "biodiversity"],
        authority_score=0.88,
        scientific_acceptance=0.90,
        historical_reliability=0.88,
    ),

    ProviderDefinition(
        source_id="oads_ocean_acidification",
        name="OADS (Ocean Acidification Data Stewardship)",
        base_url="https://oceanacidification.noaa.gov",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "Ocean acidification data — free; a NOAA account may be needed "
            "for some datasets."
        ),
        dataset_types=["ocean"],
        variables_hint=["ocean_acidification", "ph", "dissolved_inorganic_carbon"],
        authority_score=0.90,
        scientific_acceptance=0.91,
        historical_reliability=0.88,
    ),

    ProviderDefinition(
        source_id="pubmed",
        name="PubMed",
        base_url="https://pubmed.ncbi.nlm.nih.gov",
        access_type=AccessType.registration,
        discoverer_type="rest",
        description=(
            "Biomedical and life sciences literature. Free; account is optional."
        ),
        dataset_types=["biology"],
        variables_hint=["biomedical_literature", "life_sciences"],
        api_docs_url="https://www.ncbi.nlm.nih.gov/home/develop/api/",
        authority_score=0.90,
        scientific_acceptance=0.92,
        historical_reliability=0.93,
    ),

]


# ======================================================================
# TIER 4 — ADDITIONAL PAID PROVIDER (Descartes Labs was missing)
# ======================================================================

TIER4_ADDITIONAL: List[ProviderDefinition] = [

    ProviderDefinition(
        source_id="descartes_labs",
        name="Descartes Labs",
        base_url="https://www.descarteslabs.com",
        access_type=AccessType.paid,
        discoverer_type="rest",
        description=(
            "Geospatial analytics platform — satellite data, models, and "
            "AI-ready datasets. Commercial platform."
        ),
        dataset_types=["satellite", "land", "agriculture"],
        variables_hint=["vegetation", "land_use", "crop_monitoring", "change_detection"],
        login_url="https://www.descarteslabs.com/contact/",
        pricing_url="https://www.descarteslabs.com/",
        requires_login=True,
        requires_payment=True,
        price_estimate="Contact Descartes Labs for pricing.",
        authority_score=0.83,
        scientific_acceptance=0.80,
        historical_reliability=0.80,
    ),

]


# ======================================================================
# MASTER REGISTRY — combine all tiers
# ======================================================================

ALL_PROVIDERS: List[ProviderDefinition] = (
    TIER1_FREE +
    TIER1_ADDITIONAL +
    TIER2_REGISTRATION +
    TIER2_ADDITIONAL +
    TIER4_PAID +
    TIER4_ADDITIONAL
)

PROVIDERS_BY_ID: dict = {p.source_id: p for p in ALL_PROVIDERS}


def _normalize_phrase(s: str) -> str:
    """Lowercases and collapses underscores/whitespace to single spaces,
    so 'Satellite_Observations' and 'satellite observations' compare equal."""
    return re.sub(r"[_\s]+", " ", s.lower().strip())


def _phrase_matches(needle: str, haystack: str) -> bool:
    """
    Word/phrase-boundary match in either direction.

    Replaces exact string equality / exact set-membership, which
    silently failed to match free-text Agent-1 output like
    "Satellite observations" (two words) against registry tags like
    "satellite" (one word) -- meaning a request for satellite imagery
    matched zero providers even though plenty exist. This is the most
    likely real cause behind a "Provider registry: 0 provider(s)"
    symptom: the registry itself is populated (ALL_PROVIDERS is built
    from TIER1_FREE + ... + TIER4_ADDITIONAL, all hardcoded), but exact
    matching against free-text request fields can legitimately return
    zero matches even when the registry has hundreds of entries.

    Word-boundary regex (not naive substring) avoids false positives
    like "ocean" matching inside "oceanographic" -- only whole-word/
    phrase containment counts as a match.
    """
    needle = _normalize_phrase(needle)
    haystack = _normalize_phrase(haystack)
    if not needle or not haystack:
        return False
    if re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack):
        return True
    if re.search(r"(?<!\w)" + re.escape(haystack) + r"(?!\w)", needle):
        return True
    return False


def get_provider(source_id: str) -> ProviderDefinition | None:
    """Look up a provider by source_id. Returns None if not found."""
    return PROVIDERS_BY_ID.get(source_id)


def get_providers_by_access(access_type: AccessType) -> List[ProviderDefinition]:
    """Return all providers of a given access tier."""
    return [p for p in ALL_PROVIDERS if p.access_type == access_type]


def get_providers_by_dataset_type(dataset_type: str) -> List[ProviderDefinition]:
    """
    Return all providers whose dataset_types includes the given type.

    CHANGED: was exact equality, which required dataset_type to be
    character-for-character one of the registry's single-word tags.
    Now uses word/phrase-boundary matching via _phrase_matches.
    """
    return [
        p for p in ALL_PROVIDERS
        if any(_phrase_matches(dataset_type, t) for t in p.dataset_types)
    ]


def get_providers_for_variables(variables: List[str]) -> List[ProviderDefinition]:
    """
    Return providers that mention at least one of the requested variables
    in their variables_hint list. Used for fast pre-filtering before
    the discoverer makes live network calls.

    CHANGED: was exact set-intersection on whole variable strings.
    Now uses word/phrase-boundary matching, consistent with
    get_providers_by_dataset_type above.
    """
    result = []
    for provider in ALL_PROVIDERS:
        if any(
            _phrase_matches(var, hint)
            for var in variables
            for hint in provider.variables_hint
        ):
            result.append(provider)
    return result


def get_free_providers() -> list:
    return get_providers_by_access(AccessType.free)


def get_registration_providers() -> list:
    return get_providers_by_access(AccessType.registration)


def get_paid_providers() -> list:
    return get_providers_by_access(AccessType.paid)


# ======================================================================
# Flood-domain routing
#
# ADDED: agent3_discovery.py imports and calls is_flood_related_query()
# and get_providers_for_flood_domain(), but neither was defined anywhere
# in this file (or the rest of the project) -- a hard ImportError that
# blocked Agent 3 from running at all. These compose from EXISTING
# provider entries already in ALL_PROVIDERS rather than inventing new
# records, since no Sentinel-1/Copernicus-EMS/CWC/GRDC-specific provider
# entries exist yet -- flood-extent mapping in this registry is served
# via copernicus_data_space (the real Sentinel-1/2/3 SAR access point,
# see its description/variables_hint above) plus the existing
# disaster-tagged and GIS-tagged providers.
# ======================================================================

_FLOOD_QUERY_KEYWORDS = [
    "flood", "inundation", "waterlog", "river overflow", "river discharge",
    "water level", "storm surge", "sar flood",
]

# Source IDs known to be genuinely useful for flood-domain queries,
# in scientific priority order (SAR/optical flood-extent mapping first,
# then hydrology/water-level, then general disaster alerting). Any ID
# not present in ALL_PROVIDERS is simply skipped -- this list is a
# priority hint, not a hard requirement that every entry exist.
_FLOOD_PRIORITY_SOURCE_IDS = [
    "copernicus_data_space",   # Sentinel-1 SAR / Sentinel-2 optical
    "bhuvan_nrsc",             # India-specific GIS, land cover/terrain for inundation context
    "deltares",                # storm surge / coastal flood hazard modeling
    "gdacs",                   # multi-hazard alerts including flood_extent
    "reliefweb",               # humanitarian flood disaster reporting
]


def is_flood_related_query(dataset_types: List[str], variables: List[str]) -> bool:
    """
    Returns True if the request's dataset types or variables indicate a
    flood/inundation/hydrology-related query, so phase1a_catalog_query
    can route to flood-domain-prioritized providers instead of generic
    type+variable matching.
    """
    combined = " ".join(
        [t.lower() for t in dataset_types] + [v.lower() for v in variables]
    )
    return any(kw in combined for kw in _FLOOD_QUERY_KEYWORDS)


def get_providers_for_flood_domain() -> List[ProviderDefinition]:
    """
    Returns flood-relevant providers in scientific priority order
    (see _FLOOD_PRIORITY_SOURCE_IDS). Falls back to any provider whose
    dataset_types includes "disaster" if the curated list is empty for
    some reason, so this never silently returns nothing for a query
    already identified as flood-related.
    """
    ordered = []
    for source_id in _FLOOD_PRIORITY_SOURCE_IDS:
        provider = PROVIDERS_BY_ID.get(source_id)
        if provider is not None:
            ordered.append(provider)

    if not ordered:
        ordered = [p for p in ALL_PROVIDERS if "disaster" in [t.lower() for t in p.dataset_types]]

    return ordered
