"""
security/provenance.py — Dataset Provenance Tracking.

A standalone, agent-independent module for recording and querying the
complete lineage of every dataset that will later be downloaded by Agent 4.

Every provenance record answers:
  • Where did the dataset come from?       → provider, provider_url, download_url
  • Which provider supplied it?            → provider, documentation_url
  • Why was it selected?                   → selection_reason, Agent 3 scores
  • Which agent selected it?               → selected_by_agent
  • Which scientific query requested it?   → original_user_query, scientific_goal
  • Which version was downloaded?          → dataset_version
  • Which checksum belongs to it?          → sha256_checksum (from integrity.py)
  • When was it downloaded?                → download_timestamp
  • How was it downloaded?                 → retrieval_method, authentication_required

Database storage:
    All provenance records are appended to a single JSON file at
    security/provenance_db.json, stored as a JSON array of record objects.
    Records are never deleted or overwritten — save_provenance_record only
    appends, and attempting to save a duplicate dataset_id raises an error.

Intended for later integration with Agent 4 (dataset download). Has no
import-time dependency on any agent — every function takes plain Python
values so it can be imported and used in isolation, including from this
file's own __main__ demo block.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

SECURITY_DIR  = Path(__file__).resolve().parent
PROVENANCE_DB = SECURITY_DIR / "provenance_db.json"


# --------------------------------------------------------------------------- #
# Required fields — validated before every save                               #
# --------------------------------------------------------------------------- #

# Fields that MUST be present and non-empty for a record to be saved.
# Grouped here so it is easy to extend without touching validation logic.
_REQUIRED_FIELDS: list[str] = [
    # Dataset identity
    "dataset_id",
    "dataset_name",
    "provider",
    "provider_url",
    "download_url",
    # Scientific context
    "original_user_query",
    "scientific_goal",
    # Agent 3 selection
    "selected_by_agent",
    "selection_reason",
]


# --------------------------------------------------------------------------- #
# Record factory                                                               #
# --------------------------------------------------------------------------- #

def create_provenance_record(
    # ── Dataset identity ──────────────────────────────────────────────────── #
    dataset_id:              str,
    dataset_name:            str,
    provider:                str,
    provider_url:            str,
    download_url:            str,
    documentation_url:       str                   = "",
    dataset_version:         str                   = "Unknown",
    license:                 str                   = "Unknown",

    # ── Dataset content ───────────────────────────────────────────────────── #
    variables:               list[str]             = None,
    spatial_coverage:        str                   = "Unknown",
    temporal_coverage:       str                   = "Unknown",
    spatial_resolution:      str                   = "Unknown",
    temporal_resolution:     str                   = "Unknown",
    file_format:             str                   = "Unknown",
    retrieval_method:        str                   = "Unknown",
    authentication_required: bool                  = False,

    # ── Agent 3 evaluation ────────────────────────────────────────────────── #
    rank:                         int              = 0,
    overall_score:                float            = 0.0,
    confidence_score:             float            = 0.0,
    authority_score:              float            = 0.0,
    freshness_score:              float            = 0.0,
    relevance_score:              float            = 0.0,
    resolution_score:             float            = 0.0,
    completeness_score:           float            = 0.0,
    consistency_score:            float            = 0.0,
    metadata_quality_score:       float            = 0.0,
    historical_reliability_score: float            = 0.0,
    scientific_acceptance_score:  float            = 0.0,
    realtime_availability_score:  float            = 0.0,
    selected_by_agent:            str              = "Agent 3",
    selection_reason:             str              = "",

    # ── Scientific context ────────────────────────────────────────────────── #
    original_user_query:  str        = "",
    scientific_goal:      str        = "",
    study_area:           str        = "Unknown",
    time_period:          str        = "Unknown",
    requested_variables:  list[str]  = None,

    # ── Download placeholders (populated later by Agent 4) ────────────────── #
    download_timestamp:           Optional[str]  = None,
    download_status:              str            = "pending",
    downloaded_by:                str            = "",
    local_file_path:              str            = "",
    file_size:                    Optional[int]  = None,
    mime_type:                    str            = "",

    # ── Integrity (from security/integrity.py — do NOT recompute here) ────── #
    sha256_checksum:                  str            = "",
    integrity_verified:               bool           = False,
    integrity_verification_timestamp: Optional[str]  = None,
) -> dict[str, Any]:
    """
    Build and return a provenance record dict.

    Does NOT save anything to disk — call save_provenance_record() to
    persist the returned record.

    All scores should be in the 0.0–1.0 range, consistent with Agent 3's
    SourceScoreCard.  dataset_id should be the same identifier used in
    integrity.py's store_integrity_record() so the two records can be
    cross-referenced.

    Args:
        dataset_id:              Stable, unique identifier for the dataset.
                                 Must match the id used in integrity.py.
        dataset_name:            Human-readable dataset name.
        provider:                Organisation or catalog that supplies the data.
        provider_url:            Root URL of the data provider.
        download_url:            Direct URL that Agent 4 will retrieve.
        documentation_url:       URL to dataset documentation / metadata page.
        dataset_version:         Version string or date tag for the dataset.
        license:                 Data license (e.g. "CC BY 4.0", "Open Government").
        variables:               List of scientific variable names in the dataset.
        spatial_coverage:        Geographic extent description or bounding box.
        temporal_coverage:       Date range covered by the dataset.
        spatial_resolution:      Spatial resolution (e.g. "0.25°", "30 m").
        temporal_resolution:     Temporal resolution (e.g. "daily", "hourly").
        file_format:             File format (e.g. "NetCDF", "GeoTIFF", "CSV").
        retrieval_method:        How Agent 4 will retrieve the file
                                 (e.g. "ERDDAP REST", "OPeNDAP", "STAC").
        authentication_required: True if Agent 4 will need credentials.
        rank:                    Agent 3 rank (1 = best).
        overall_score:           Agent 3 final weighted score (0.0–1.0).
        confidence_score:        Fraction of expected metadata fields populated.
        authority_score:         Authority criterion score (0.0–1.0).
        freshness_score:         Freshness criterion score (0.0–1.0).
        relevance_score:         Relevance criterion score (0.0–1.0).
        resolution_score:        Resolution criterion score (0.0–1.0).
        completeness_score:      Completeness criterion score (0.0–1.0).
        consistency_score:       Consistency criterion score (0.0–1.0).
        metadata_quality_score:  Metadata quality criterion score (0.0–1.0).
        historical_reliability_score: Historical reliability score (0.0–1.0).
        scientific_acceptance_score:  Scientific acceptance score (0.0–1.0).
        realtime_availability_score:  Real-time availability score (0.0–1.0).
        selected_by_agent:       Name of the agent that selected this dataset.
        selection_reason:        Human-readable explanation of why it was selected.
        original_user_query:     The raw query the user submitted.
        scientific_goal:         Structured scientific goal parsed from the query.
        study_area:              Geographic study area name.
        time_period:             Requested time period (human-readable).
        requested_variables:     Variables the user asked for.
        download_timestamp:      ISO-8601 timestamp; None until Agent 4 fills this.
        download_status:         "pending" | "success" | "failed".
        downloaded_by:           Agent that performed the download.
        local_file_path:         Absolute path where the file was saved.
        file_size:               File size in bytes; None until downloaded.
        mime_type:               MIME type of the downloaded file.
        sha256_checksum:         SHA-256 hex digest from security/integrity.py.
                                 Do NOT recompute here — pass the value through.
        integrity_verified:      True if integrity.py confirmed the checksum.
        integrity_verification_timestamp: ISO-8601 timestamp of last verification.

    Returns:
        A dict containing all provenance fields plus a
        provenance_record_created_at timestamp (UTC ISO-8601).
    """
    return {
        # ── Dataset identity ─────────────────────────────────────────────── #
        "dataset_id":        dataset_id,
        "dataset_name":      dataset_name,
        "provider":          provider,
        "provider_url":      provider_url,
        "download_url":      download_url,
        "documentation_url": documentation_url,
        "dataset_version":   dataset_version,
        "license":           license,

        # ── Dataset content ──────────────────────────────────────────────── #
        "variables":               variables or [],
        "spatial_coverage":        spatial_coverage,
        "temporal_coverage":       temporal_coverage,
        "spatial_resolution":      spatial_resolution,
        "temporal_resolution":     temporal_resolution,
        "file_format":             file_format,
        "retrieval_method":        retrieval_method,
        "authentication_required": authentication_required,

        # ── Agent 3 evaluation ───────────────────────────────────────────── #
        "rank":                         rank,
        "overall_score":                overall_score,
        "confidence_score":             confidence_score,
        "authority_score":              authority_score,
        "freshness_score":              freshness_score,
        "relevance_score":              relevance_score,
        "resolution_score":             resolution_score,
        "completeness_score":           completeness_score,
        "consistency_score":            consistency_score,
        "metadata_quality_score":       metadata_quality_score,
        "historical_reliability_score": historical_reliability_score,
        "scientific_acceptance_score":  scientific_acceptance_score,
        "realtime_availability_score":  realtime_availability_score,
        "selected_by_agent":            selected_by_agent,
        "selection_reason":             selection_reason,

        # ── Scientific context ───────────────────────────────────────────── #
        "original_user_query": original_user_query,
        "scientific_goal":     scientific_goal,
        "study_area":          study_area,
        "time_period":         time_period,
        "requested_variables": requested_variables or [],

        # ── Download placeholders (Agent 4 fills these later) ────────────── #
        "download_timestamp": download_timestamp,
        "download_status":    download_status,
        "downloaded_by":      downloaded_by,
        "local_file_path":    local_file_path,
        "file_size":          file_size,
        "mime_type":          mime_type,

        # ── Integrity (sourced from security/integrity.py only) ──────────── #
        "sha256_checksum":                  sha256_checksum,
        "integrity_verified":               integrity_verified,
        "integrity_verification_timestamp": integrity_verification_timestamp,

        # ── Internal audit field ─────────────────────────────────────────── #
        "provenance_record_created_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Database read / write helpers                                                #
# --------------------------------------------------------------------------- #

def _load_db() -> list[dict]:
    """
    Load all provenance records from disk as a list of dicts.

    Returns an empty list if the database file does not exist yet or is
    empty. On corrupt JSON, backs up the file with a .corrupt suffix,
    prints a warning, and returns an empty list rather than crashing.
    """
    if not PROVENANCE_DB.exists():
        return []

    try:
        raw = PROVENANCE_DB.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        backup = PROVENANCE_DB.with_suffix(".json.corrupt")
        try:
            PROVENANCE_DB.replace(backup)
            print(
                f"[provenance] Warning: database at {PROVENANCE_DB} was invalid "
                f"({exc}). Backed up to {backup} and starting a fresh database."
            )
        except OSError:
            print(
                f"[provenance] Warning: database at {PROVENANCE_DB} was invalid "
                f"({exc}) and could not be backed up. Starting a fresh in-memory database."
            )
        return []

    if not isinstance(data, list):
        print(
            f"[provenance] Warning: database at {PROVENANCE_DB} did not contain "
            "a JSON array. Starting a fresh database."
        )
        return []

    return data


def _write_db(records: list[dict]) -> None:
    """
    Write the full list of records to disk atomically.

    Writes to a .tmp file first, then replaces the real database,
    so a crash mid-write cannot corrupt the existing records.
    Creates the parent security/ directory if it does not exist.
    """
    PROVENANCE_DB.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROVENANCE_DB.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    tmp.replace(PROVENANCE_DB)


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #

def _validate_record(record: dict[str, Any]) -> None:
    """
    Validate that all required fields are present and non-empty.

    Raises:
        ValueError: with a descriptive message listing every missing or
            empty required field found in the record.
    """
    missing = [
        field for field in _REQUIRED_FIELDS
        if not record.get(field)
    ]
    if missing:
        raise ValueError(
            f"Provenance record is missing required field(s): {', '.join(missing)}. "
            "All of these must be non-empty before the record can be saved."
        )


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def save_provenance_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and append a provenance record to the database.

    The database file is created automatically if it does not exist.
    Existing records are never modified or deleted — this function only
    appends. Attempting to save a record whose dataset_id already exists
    in the database raises a ValueError rather than silently overwriting.

    Args:
        record: a dict produced by create_provenance_record().

    Returns:
        The record that was saved (unchanged).

    Raises:
        ValueError: if a required field is missing/empty, or if a record
            with this dataset_id already exists in the database.
    """
    _validate_record(record)

    records = _load_db()

    existing_ids = {r.get("dataset_id") for r in records}
    if record["dataset_id"] in existing_ids:
        raise ValueError(
            f"A provenance record for dataset_id '{record['dataset_id']}' already "
            "exists. Records are immutable — create a new record with a distinct "
            "dataset_id if a new version or re-download needs to be tracked."
        )

    records.append(record)
    _write_db(records)

    print(f"[provenance] Saved record for dataset_id='{record['dataset_id']}'.")
    return record


def get_record_by_dataset_id(dataset_id: str) -> Optional[dict[str, Any]]:
    """
    Retrieve the provenance record for a specific dataset_id.

    Args:
        dataset_id: the unique dataset identifier to look up.

    Returns:
        The matching record dict, or None if no record exists for this id.
    """
    records = _load_db()
    for record in records:
        if record.get("dataset_id") == dataset_id:
            return record
    return None


def get_records_by_provider(provider: str) -> list[dict[str, Any]]:
    """
    Retrieve all provenance records from a specific provider.

    Matching is case-insensitive so "NASA EarthData" and "nasa earthdata"
    both return the same results.

    Args:
        provider: the provider name to filter by.

    Returns:
        A list of matching records (empty list if none found).
    """
    provider_lower = provider.lower()
    return [
        r for r in _load_db()
        if r.get("provider", "").lower() == provider_lower
    ]


def get_records_by_query(query: str) -> list[dict[str, Any]]:
    """
    Retrieve all provenance records associated with a specific user query.

    Matching is against the original_user_query field and is
    case-insensitive substring match, so partial queries (e.g. "Bay of
    Bengal") will match records whose full query contains that phrase.

    Args:
        query: the user query string (or substring of it) to search for.

    Returns:
        A list of matching records (empty list if none found).
    """
    query_lower = query.lower()
    return [
        r for r in _load_db()
        if query_lower in r.get("original_user_query", "").lower()
    ]


def list_all_records() -> list[dict[str, Any]]:
    """
    Return all provenance records currently stored in the database.

    Returns:
        A list of all record dicts (empty list if the database is empty
        or does not exist yet).
    """
    return _load_db()


# --------------------------------------------------------------------------- #
# Demo / self-test — runnable independently of any agent                      #
# --------------------------------------------------------------------------- #

def run_demo() -> None:
    """
    Demonstrates, end to end and independently of any agent:
      1. Creating a realistic provenance record.
      2. Saving it to the database.
      3. Retrieving it by dataset_id.
      4. Retrieving records by provider.
      5. Retrieving records by scientific query.
      6. Listing all records.
      7. Verifying that duplicate dataset_id is rejected.

    Uses realistic field values representative of an actual Agent 3
    result for a Bay of Bengal flood monitoring query.

    Idempotent: each run generates a unique dataset_id suffix via uuid4,
    so repeated executions never collide with existing records in
    provenance_db.json.  Duplicate protection for real (non-demo)
    dataset_ids is unchanged.
    """
    print("=" * 65)
    print("security.provenance — demo")
    print("=" * 65)

    # Unique suffix so repeated runs never collide with prior demo records.
    # Real production records use stable, meaningful IDs (e.g. from Agent 3).
    run_id = uuid.uuid4().hex[:8]

    # ── Step 1: Create a realistic provenance record ─────────────────────── #
    print("\n1. Creating provenance record ...")

    # BUG 1 FIX: dataset_id is now unique per run (uuid suffix) so
    # save_provenance_record() never raises "already exists" on repeated
    # executions.  The core dataset_id format is preserved; only a short
    # hex suffix distinguishes demo records from each other.
    demo_id_1 = f"incois_sst_bay_bengal_2024_v1_{run_id}"
    demo_id_2 = f"cmems_sst_bay_bengal_2024_v1_{run_id}"

    record = create_provenance_record(
        # Dataset identity
        dataset_id        = demo_id_1,
        dataset_name      = "INCOIS Sea Surface Temperature — Bay of Bengal 2024",
        provider          = "INCOIS",
        provider_url      = "https://incois.gov.in",
        download_url      = (
            "https://erddap.incois.gov.in/erddap/griddap/"
            "INCOIS_SST_BAY_BENGAL.nc?sst[(2024-06-01T00:00:00Z)]"
        ),
        documentation_url = "https://incois.gov.in/portal/datainfo/sst.jsp",
        dataset_version   = "2024-06-01",
        license           = "Government of India Open Data License",

        # Dataset content
        variables            = ["sea_surface_temperature", "sst_anomaly"],
        spatial_coverage     = "Bay of Bengal (5°N–23°N, 80°E–100°E)",
        temporal_coverage    = "2000-01-01 – present",
        spatial_resolution   = "0.25°",
        temporal_resolution  = "daily",
        file_format          = "NetCDF",
        retrieval_method     = "ERDDAP REST",
        authentication_required = True,

        # Agent 3 evaluation
        rank                         = 1,
        overall_score                = 0.91,
        confidence_score             = 0.88,
        authority_score              = 0.95,
        freshness_score              = 0.90,
        relevance_score              = 0.94,
        resolution_score             = 0.85,
        completeness_score           = 0.88,
        consistency_score            = 0.87,
        metadata_quality_score       = 0.82,
        historical_reliability_score = 0.93,
        scientific_acceptance_score  = 0.96,
        realtime_availability_score  = 0.89,
        selected_by_agent            = "Agent 3",
        selection_reason             = (
            "Highest-ranked source for Bay of Bengal SST. INCOIS is the "
            "domain-preferred provider for Indian Ocean datasets. Complete "
            "metadata, daily temporal resolution, and strong historical "
            "reliability from Qdrant cache (used 12 times, 12 successes). "
            "Authentication required — delegated to Agent 4."
        ),

        # Scientific context
        original_user_query = (
            "Monitor sea surface temperature anomalies in the Bay of Bengal "
            "during the 2024 pre-monsoon season to assess cyclone formation risk."
        ),
        scientific_goal     = (
            "Quantify SST anomalies in the Bay of Bengal (May–June 2024) "
            "and identify regions exceeding +1°C above the 30-year climatological mean."
        ),
        study_area          = "Bay of Bengal",
        time_period         = "May 2024 – June 2024",
        requested_variables = ["sea_surface_temperature", "sst_anomaly"],

        # Download placeholders — Agent 4 fills these
        download_timestamp  = None,
        download_status     = "pending",
        downloaded_by       = "",
        local_file_path     = "",
        file_size           = None,
        mime_type           = "",

        # Integrity — sourced from security/integrity.py, not recomputed here
        sha256_checksum                  = "",
        integrity_verified               = False,
        integrity_verification_timestamp = None,
    )

    print(f"   dataset_id      : {record['dataset_id']}")
    print(f"   dataset_name    : {record['dataset_name']}")
    print(f"   provider        : {record['provider']}")
    print(f"   overall_score   : {record['overall_score']}")
    print(f"   rank            : {record['rank']}")
    print(f"   download_status : {record['download_status']}")
    print(f"   created_at      : {record['provenance_record_created_at']}")

    # ── Step 2: Save it ───────────────────────────────────────────────────── #
    print("\n2. Saving record to database ...")
    save_provenance_record(record)
    print(f"   Saved to: {PROVENANCE_DB}")

    # Save a second record (different provider) to make the list queries useful
    record2 = create_provenance_record(
        dataset_id        = demo_id_2,
        dataset_name      = "CMEMS Global SST Analysis — Bay of Bengal 2024",
        provider          = "CMEMS",
        provider_url      = "https://marine.copernicus.eu",
        download_url      = (
            "https://nrt.cmems-du.eu/thredds/dodsC/"
            "cmems_obs-sst_glo_phy_nrt_l4_P1D-m"
        ),
        documentation_url = "https://marine.copernicus.eu/product/SST_GLO",
        dataset_version   = "2024-06-01",
        license           = "Copernicus Marine Service License",
        variables            = ["analysed_sst", "analysis_error"],
        spatial_coverage     = "Global (includes Bay of Bengal)",
        temporal_coverage    = "2007-01-01 – present",
        spatial_resolution   = "0.05°",
        temporal_resolution  = "daily",
        file_format          = "NetCDF",
        retrieval_method     = "OPeNDAP",
        authentication_required = True,
        rank                         = 2,
        overall_score                = 0.87,
        confidence_score             = 0.84,
        authority_score              = 0.93,
        freshness_score              = 0.88,
        relevance_score              = 0.91,
        resolution_score             = 0.95,
        completeness_score           = 0.85,
        consistency_score            = 0.83,
        metadata_quality_score       = 0.80,
        historical_reliability_score = 0.90,
        scientific_acceptance_score  = 0.94,
        realtime_availability_score  = 0.86,
        selected_by_agent            = "Agent 3",
        selection_reason             = (
            "High-resolution global SST analysis (0.05°) with excellent "
            "scientific acceptance. Ranked 2nd behind INCOIS for this "
            "query due to INCOIS's domain-preferred provider status for "
            "Indian Ocean datasets."
        ),
        original_user_query = (
            "Monitor sea surface temperature anomalies in the Bay of Bengal "
            "during the 2024 pre-monsoon season to assess cyclone formation risk."
        ),
        scientific_goal     = (
            "Quantify SST anomalies in the Bay of Bengal (May–June 2024) "
            "and identify regions exceeding +1°C above the 30-year climatological mean."
        ),
        study_area          = "Bay of Bengal",
        time_period         = "May 2024 – June 2024",
        requested_variables = ["sea_surface_temperature", "sst_anomaly"],
        download_status     = "pending",
        sha256_checksum     = "",
        integrity_verified  = False,
    )
    save_provenance_record(record2)
    print(f"   Second record saved: {record2['dataset_id']}")

    # ── Step 3: Retrieve by dataset_id ───────────────────────────────────── #
    print("\n3. Retrieving record by dataset_id ...")
    # BUG 2 FIX: look up by the actual saved ID (demo_id_1), not a
    # hardcoded string — otherwise this step would always find the stale
    # record from the first-ever run rather than the one just saved.
    retrieved = get_record_by_dataset_id(demo_id_1)
    assert retrieved is not None, f"Expected to find the INCOIS record (id={demo_id_1})."
    print(f"   Found: {retrieved['dataset_name']}")
    print(f"   Provider: {retrieved['provider']}")
    print(f"   Rank: {retrieved['rank']}  |  Overall score: {retrieved['overall_score']}")
    print(f"   Selection reason: {retrieved['selection_reason'][:80]}...")

    missing = get_record_by_dataset_id("does_not_exist_999")
    assert missing is None, "Expected None for unknown dataset_id."
    print("   Non-existent id → None (correct).")

    # ── Step 4: Retrieve by provider ─────────────────────────────────────── #
    print("\n4. Retrieving records by provider ...")
    incois_records = get_records_by_provider("INCOIS")
    print(f"   Provider 'INCOIS' → {len(incois_records)} record(s):")
    for r in incois_records:
        print(f"     • {r['dataset_id']}  (score={r['overall_score']})")

    cmems_records = get_records_by_provider("CMEMS")
    print(f"   Provider 'CMEMS' → {len(cmems_records)} record(s):")
    for r in cmems_records:
        print(f"     • {r['dataset_id']}  (score={r['overall_score']})")

    none_records = get_records_by_provider("NonExistentProvider")
    assert none_records == [], "Expected empty list for unknown provider."
    print("   Provider 'NonExistentProvider' → [] (correct).")

    # ── Step 5: Retrieve by scientific query ──────────────────────────────── #
    print("\n5. Retrieving records by scientific query ...")
    query_results = get_records_by_query("Bay of Bengal")
    print(f"   Query 'Bay of Bengal' → {len(query_results)} record(s):")
    for r in query_results:
        print(f"     • {r['dataset_id']}")
    # BUG 3 FIX: assert >= 2, not == 2.  Each run adds 2 more matching
    # records to the database, so after multiple runs the count grows.
    # The meaningful invariant is "at least the 2 we just saved match",
    # not "exactly 2 records exist in the database total".
    assert len(query_results) >= 2, (
        f"Expected at least 2 records matching 'Bay of Bengal', "
        f"got {len(query_results)}."
    )

    no_results = get_records_by_query("Arctic Sea Ice Extent")
    assert no_results == [], "Expected empty list for unrelated query."
    print("   Query 'Arctic Sea Ice Extent' → [] (correct).")

    # ── Step 6: List all records ──────────────────────────────────────────── #
    print("\n6. Listing all records in database ...")
    all_records = list_all_records()
    print(f"   Total records: {len(all_records)}")
    for i, r in enumerate(all_records, start=1):
        print(
            f"   [{i}] {r['dataset_id']}"
            f"  |  provider={r['provider']}"
            f"  |  rank={r['rank']}"
            f"  |  score={r['overall_score']}"
            f"  |  status={r['download_status']}"
        )

    # ── Step 7: Duplicate rejection ───────────────────────────────────────── #
    print("\n7. Verifying duplicate dataset_id is rejected ...")
    # BUG 4 FIX: use demo_id_1 (the ID we actually saved in this run)
    # rather than a hardcoded string.  Without this fix, the duplicate
    # check would try to re-insert the original "incois_sst_bay_bengal_2024_v1"
    # (from provenance_db.json) which is not the same as the uuid-suffixed
    # ID we saved in Step 2 — so the ValueError would never be raised and
    # the AssertionError below would fire instead.
    duplicate = create_provenance_record(
        dataset_id          = demo_id_1,          # same ID as Step 1/2
        dataset_name        = "Duplicate INCOIS Record",
        provider            = "INCOIS",
        provider_url        = "https://incois.gov.in",
        download_url        = "https://erddap.incois.gov.in/erddap/griddap/duplicate",
        selected_by_agent   = "Agent 3",
        selection_reason    = "Duplicate — should be rejected.",
        original_user_query = "Test duplicate rejection.",
        scientific_goal     = "Test.",
    )
    try:
        save_provenance_record(duplicate)
        raise AssertionError("Expected ValueError for duplicate dataset_id — none was raised.")
    except ValueError as exc:
        print(f"   Correctly rejected: {exc}")

    print("\n" + "=" * 65)
    print("Demo completed successfully — all expected outcomes matched.")
    print(f"Database location: {PROVENANCE_DB}")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
