"""
Scientific Knowledge Base for Agent 3 Earth Observation discovery.

Implements the reusable mapping requested for Agent 3:

    Scientific Variable -> Measurement Concept Expansion
                         -> Sensors / Instruments
                         -> Preferred Providers (real source_ids in
                            sources/providers.py, where they exist)

This is a DATA layer, not discovery logic. Adding a new variable means
adding one entry to SCIENTIFIC_VARIABLE_KB below -- nothing in
agent3_discovery.py, the discoverers, or the prompt needs to change.

Design notes / honesty about scope:
  - sensor_instruments lists the scientifically correct sensors/missions
    for each variable (e.g. GPM, Sentinel-1, MODIS) regardless of
    whether sources/providers.py has a dedicated entry for that exact
    mission yet -- this keeps the science correct and the mapping
    genuinely reusable as the provider registry grows.
  - preferred_source_ids lists REAL source_ids that already exist in
    sources/providers.py / ALL_PROVIDERS, in scientific priority order,
    for the providers that are actually closest matches for this
    variable today. This list intentionally does NOT invent provider
    entries that don't exist (e.g. there is no dedicated "GPM" or
    "INSAT" entry in the current registry) -- get_preferred_providers()
    below silently skips any ID that isn't currently registered, so
    this stays correct as new providers are added without needing
    code changes here.
  - expansion_terms drives query expansion (used by discoverers and the
    LLM prompt) and provider-aware search term generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VariableKnowledge:
    """One scientific variable's full knowledge-base entry."""
    canonical_name: str
    domain: str                              # e.g. "meteorology", "oceanography"
    expansion_terms: List[str] = field(default_factory=list)
    sensor_instruments: List[str] = field(default_factory=list)
    preferred_source_ids: List[str] = field(default_factory=list)


# ====================================================================== #
# The knowledge base itself.                                             #
# Keys are canonical lowercase variable names. Lookups in this module    #
# match user-supplied variable strings against these keys AND against    #
# each entry's own expansion_terms (so "precipitation" or "IMERG" both   #
# resolve to the "rainfall" entry).                                      #
# ====================================================================== #

SCIENTIFIC_VARIABLE_KB: Dict[str, VariableKnowledge] = {

    # ---------------------------------------------------------------- #
    # Meteorology                                                        #
    # ---------------------------------------------------------------- #
    "rainfall": VariableKnowledge(
        canonical_name="rainfall",
        domain="meteorology",
        expansion_terms=["precipitation", "imerg", "gpm", "rain gauge", "monsoon rainfall"],
        sensor_instruments=["GPM (IMERG)", "INSAT", "rain gauge networks"],
        preferred_source_ids=["imd", "open_meteo", "noaa_nomads", "copernicus_cds_era5"],
    ),
    "temperature": VariableKnowledge(
        canonical_name="temperature",
        domain="meteorology",
        expansion_terms=["air temperature", "surface temperature", "2m temperature"],
        sensor_instruments=["ERA5 reanalysis", "ECMWF model", "NOAA surface stations"],
        preferred_source_ids=["copernicus_cds_era5", "ecmwf_open_data", "open_meteo", "noaa_nomads"],
    ),
    "wind": VariableKnowledge(
        canonical_name="wind",
        domain="meteorology",
        expansion_terms=["wind speed", "wind direction", "ascat", "scatterometer wind"],
        sensor_instruments=["ASCAT scatterometer", "INSAT", "ERA5 reanalysis"],
        preferred_source_ids=["copernicus_cds_era5", "noaa_nomads", "open_meteo", "imd"],
    ),
    "humidity": VariableKnowledge(
        canonical_name="humidity",
        domain="meteorology",
        expansion_terms=["relative humidity", "specific humidity", "moisture content"],
        sensor_instruments=["ERA5 reanalysis", "ECMWF model"],
        preferred_source_ids=["copernicus_cds_era5", "ecmwf_open_data", "open_meteo"],
    ),

    # ---------------------------------------------------------------- #
    # Floods / Hydrology                                                 #
    # ---------------------------------------------------------------- #
    "flood extent": VariableKnowledge(
        canonical_name="flood extent",
        domain="floods",
        expansion_terms=[
            "flood", "inundation", "sar flood", "waterlogging",
            "river overflow", "flood mapping",
        ],
        sensor_instruments=["Sentinel-1 SAR", "Sentinel-2", "Landsat", "Copernicus EMS"],
        preferred_source_ids=["copernicus_data_space", "bhuvan_nrsc", "gdacs", "reliefweb"],
    ),
    "water level": VariableKnowledge(
        canonical_name="water level",
        domain="floods",
        expansion_terms=["river stage", "gauge height", "river discharge"],
        sensor_instruments=["CWC gauge network", "SWOT altimetry", "Jason altimetry"],
        preferred_source_ids=["incois_portal", "noaa_gloss", "noaa_dart"],
    ),
    "river discharge": VariableKnowledge(
        canonical_name="river discharge",
        domain="floods",
        expansion_terms=["streamflow", "discharge rate", "flow rate"],
        sensor_instruments=["CWC gauge network", "GRDC"],
        preferred_source_ids=["deltares", "noaa_nomads"],
    ),

    # ---------------------------------------------------------------- #
    # Oceanography                                                       #
    # ---------------------------------------------------------------- #
    "sea surface temperature": VariableKnowledge(
        canonical_name="sea surface temperature",
        domain="oceanography",
        expansion_terms=["sst", "ghrsst", "skin temperature", "ocean skin temperature"],
        sensor_instruments=["GHRSST multi-sensor blend", "AVHRR", "MODIS", "VIIRS"],
        preferred_source_ids=["noaa_ghrsst", "incois_portal", "copernicus_marine", "noaa_erddap_coastwatch"],
    ),
    "chlorophyll-a": VariableKnowledge(
        canonical_name="chlorophyll-a",
        domain="oceanography",
        expansion_terms=[
            "chlorophyll", "harmful algal bloom", "hab", "phytoplankton",
            "cyanobacteria", "ocean colour", "ocean color", "fluorescence",
            "red tide",
        ],
        sensor_instruments=["Sentinel-3 OLCI", "MODIS Aqua", "VIIRS"],
        preferred_source_ids=[
            "noaa_erddap_coastwatch", "copernicus_marine", "incois_portal", "nasa_gibs",
        ],
    ),
    "ocean color": VariableKnowledge(
        canonical_name="ocean color",
        domain="oceanography",
        expansion_terms=["ocean colour", "water color", "remote sensing reflectance"],
        sensor_instruments=["Sentinel-3 OLCI", "MODIS Aqua"],
        preferred_source_ids=["copernicus_data_space", "noaa_erddap_coastwatch", "copernicus_marine"],
    ),
    "wave height": VariableKnowledge(
        canonical_name="wave height",
        domain="oceanography",
        expansion_terms=["significant wave height", "swell height", "wave period"],
        sensor_instruments=["altimetry", "wave buoys", "WAVEWATCH III model"],
        preferred_source_ids=["incois_portal", "copernicus_marine", "noaa_wavewatch3"],
    ),
    "ocean currents": VariableKnowledge(
        canonical_name="ocean currents",
        domain="oceanography",
        expansion_terms=["surface currents", "current velocity"],
        sensor_instruments=["altimetry-derived currents", "HYCOM model"],
        preferred_source_ids=["incois_portal", "copernicus_marine", "hycom"],
    ),
    "salinity": VariableKnowledge(
        canonical_name="salinity",
        domain="oceanography",
        expansion_terms=["sea surface salinity", "sss"],
        sensor_instruments=["SMOS", "SMAP", "Argo floats"],
        preferred_source_ids=["copernicus_marine", "argo_gdac_global", "noaa_erddap_coastwatch"],
    ),

    # ---------------------------------------------------------------- #
    # Agriculture / Land                                                 #
    # ---------------------------------------------------------------- #
    "ndvi": VariableKnowledge(
        canonical_name="ndvi",
        domain="agriculture",
        expansion_terms=["vegetation index", "vegetation health", "greenness index"],
        sensor_instruments=["Sentinel-2", "Landsat", "MODIS"],
        preferred_source_ids=["copernicus_land", "planetary_computer", "google_earth_engine"],
    ),
    "soil moisture": VariableKnowledge(
        canonical_name="soil moisture",
        domain="agriculture",
        expansion_terms=["root zone moisture", "surface soil moisture"],
        sensor_instruments=["SMAP", "SMOS"],
        preferred_source_ids=["copernicus_land", "copernicus_cds_era5"],
    ),
    "land cover": VariableKnowledge(
        canonical_name="land cover",
        domain="agriculture",
        expansion_terms=["land use", "land classification", "world cover"],
        sensor_instruments=["ESA WorldCover", "Dynamic World", "Sentinel-2"],
        preferred_source_ids=["copernicus_land", "bhuvan_nrsc", "google_earth_engine"],
    ),

    # ---------------------------------------------------------------- #
    # Air Quality                                                        #
    # ---------------------------------------------------------------- #
    "no2": VariableKnowledge(
        canonical_name="no2",
        domain="air_quality",
        expansion_terms=["nitrogen dioxide", "tropospheric no2"],
        sensor_instruments=["Sentinel-5P TROPOMI"],
        preferred_source_ids=["copernicus_atmosphere", "openaq", "waqi_bulk_api"],
    ),
    "aerosols": VariableKnowledge(
        canonical_name="aerosols",
        domain="air_quality",
        expansion_terms=["aerosol optical depth", "aod", "particulate aerosol"],
        sensor_instruments=["MODIS", "VIIRS"],
        preferred_source_ids=["copernicus_atmosphere", "nasa_gibs"],
    ),
    "pm estimation": VariableKnowledge(
        canonical_name="pm estimation",
        domain="air_quality",
        expansion_terms=["pm2.5", "pm10", "particulate matter"],
        sensor_instruments=["CAMS reanalysis", "Sentinel-5P"],
        preferred_source_ids=["copernicus_atmosphere", "waqi_bulk_api", "openaq", "waqi_viewer"],
    ),
}


# Maps every expansion term and canonical name back to its KB key, built
# once at import time, so lookups work regardless of which synonym the
# request happens to use (e.g. "SST", "skin temperature", and "sea
# surface temperature" all resolve to the same entry).
_TERM_TO_KEY: Dict[str, str] = {}
for _key, _entry in SCIENTIFIC_VARIABLE_KB.items():
    _TERM_TO_KEY[_key.lower()] = _key
    for _term in _entry.expansion_terms:
        _TERM_TO_KEY[_term.lower()] = _key


def lookup_variable(text: str) -> Optional[VariableKnowledge]:
    """
    Resolves a free-text variable/measurement string to its
    VariableKnowledge entry, matching against canonical names and all
    known expansion terms/synonyms. Returns None if nothing matches --
    callers must treat that as "no knowledge-base entry yet", not an
    error, since the KB is intentionally incomplete and growable.
    """
    text_lower = text.lower().strip()
    if not text_lower:
        return None

    if text_lower in _TERM_TO_KEY:
        return SCIENTIFIC_VARIABLE_KB[_TERM_TO_KEY[text_lower]]

    # Fall back to substring containment in either direction, so
    # "chlorophyll concentration" still matches "chlorophyll-a" via its
    # "chlorophyll" expansion term, and "rainfall over Odisha" matches
    # "rainfall".
    for term, key in _TERM_TO_KEY.items():
        if term in text_lower or text_lower in term:
            return SCIENTIFIC_VARIABLE_KB[key]

    return None


def expand_variables(variables: List[str]) -> List[str]:
    """
    Expands a list of requested variables into the union of their
    canonical names + all expansion terms, deduplicated. Variables with
    no KB match are passed through unchanged (never dropped) --
    expansion is a bonus signal, not a filter.

    Ordering is BREADTH-FIRST across variables, not depth-first: every
    requested variable's own term is added first (in request order),
    THEN each variable's canonical name and expansion terms are
    interleaved one "round" at a time. This matters because every
    discoverer caps this list to a small N (3-4) before building a
    network query string, for query-length reasons. A depth-first
    ordering would let variable 1's full synonym list crowd out
    variable 2 entirely whenever there are 2+ requested variables (e.g.
    a request for ["chlorophyll-a", "sea surface temperature"] capped
    at 3 would produce 3 chlorophyll synonyms and ZERO representation
    of sea surface temperature). Breadth-first guarantees every
    requested variable appears at or near the front of the list.
    """
    seen = set()
    rounds: List[List[str]] = []   # rounds[0] = each var's own term,
                                    # rounds[1] = each var's canonical name,
                                    # rounds[2+] = each var's expansion terms, one per round

    def _round_for(idx: int) -> List[str]:
        while len(rounds) <= idx:
            rounds.append([])
        return rounds[idx]

    def _add(term: str, round_idx: int):
        key = term.lower().strip()
        if key and key not in seen:
            seen.add(key)
            _round_for(round_idx).append(term)

    for var in variables:
        _add(var, 0)
        kb_entry = lookup_variable(var)
        if kb_entry:
            _add(kb_entry.canonical_name, 1)
            for i, term in enumerate(kb_entry.expansion_terms):
                _add(term, 2 + i)

    expanded: List[str] = []
    for r in rounds:
        expanded.extend(r)
    return expanded


def get_preferred_providers(variables: List[str]) -> List[str]:
    """
    Returns the union of preferred_source_ids for every variable that
    has a KB match, in scientific priority order (first variable's
    providers first, ties broken by first-seen order). Only returns
    source_ids -- callers should resolve against
    sources.providers.PROVIDERS_BY_ID and silently skip any ID that
    isn't currently registered, since this list is allowed to name
    providers slightly ahead of the registry's current coverage.
    """
    ordered: List[str] = []
    seen = set()
    for var in variables:
        kb_entry = lookup_variable(var)
        if not kb_entry:
            continue
        for source_id in kb_entry.preferred_source_ids:
            if source_id not in seen:
                seen.add(source_id)
                ordered.append(source_id)
    return ordered


def get_sensor_instruments(variables: List[str]) -> List[str]:
    """Returns the union of sensor/instrument names relevant to the
    given variables, for use in provider-aware search term generation
    and LLM prompt context. Deduplicated, order-preserved."""
    ordered: List[str] = []
    seen = set()
    for var in variables:
        kb_entry = lookup_variable(var)
        if not kb_entry:
            continue
        for sensor in kb_entry.sensor_instruments:
            if sensor not in seen:
                seen.add(sensor)
                ordered.append(sensor)
    return ordered


def is_flood_related(variables: List[str], dataset_types: Optional[List[str]] = None) -> bool:
    """
    Knowledge-base-driven flood detection, used as an additional signal
    alongside (not a replacement for) the keyword check in
    sources/providers.py:is_flood_related_query. Returns True if any
    variable resolves to a "floods" domain KB entry.
    """
    for var in variables:
        kb_entry = lookup_variable(var)
        if kb_entry and kb_entry.domain == "floods":
            return True
    if dataset_types:
        for dt in dataset_types:
            if "flood" in dt.lower() or "hydrology" in dt.lower():
                return True
    return False
