"""
Agent 4 — Size Estimator.

Thin wrapper around the connector registry's probe_size. Kept separate
so the access-resolution and size-approval steps don't need to know
about connectors directly.
"""

from connectors.base_connector import FetchRequest
from connectors.connector_factory import get_connector
from models.agent4_schemas import DatasetDescriptor, DatasetMetadata, SizeEstimate, format_bytes
from models.website_analysis_schemas import SourceSnapshot


def estimate_size(snapshot: SourceSnapshot, variables: list, bounding_box=None, time_range=None) -> SizeEstimate:
    # 1. Immediate override for WMS endpoints
    if "wms" in snapshot.url.lower():
        estimated_size = 2048 * 2048 * 4
        return SizeEstimate(
            source_id=snapshot.source_id,
            estimated_bytes=float(estimated_size),
            is_exact=False,
            method="wms_canvas_estimate",
            human_readable=format_bytes(estimated_size)
        )

    connector = get_connector(snapshot)
    fetch_request = FetchRequest(variables=variables, bounding_box=bounding_box, time_range=time_range)
    
    try:
        estimate = connector.probe_size(snapshot, fetch_request)
        # 2. Intercept 'unavailable' responses and replace with a safe 50MB fallback
        if estimate.method == "unavailable":
            fallback_size = 50.0 * 1024 * 1024
            return SizeEstimate(
                source_id=snapshot.source_id,
                estimated_bytes=fallback_size,
                is_exact=False,
                method="orchestrator_fallback",
                human_readable=format_bytes(fallback_size)
            )
        return estimate
    except Exception as exc:
        print(f"[Size Estimator] Probe failed for {snapshot.source_id} (non-fatal): {exc}")
        # 3. Safe fallback in case of connector probe exception
        fallback_size = 50.0 * 1024 * 1024
        return SizeEstimate(
            source_id=snapshot.source_id, 
            estimated_bytes=fallback_size,
            is_exact=False,
            method="error_fallback",
            human_readable=format_bytes(fallback_size)
        )


def discover_source_datasets(
    snapshot: SourceSnapshot,
    variables: list,
    bounding_box=None,
    time_range=None,
) -> list[DatasetDescriptor]:
    connector = get_connector(snapshot)
    context = {
        "variables": variables,
        "bounding_box": bounding_box,
        "time_range": time_range,
    }
    try:
        return list(connector.discover_datasets(snapshot, context) or [])
    except NotImplementedError:
        return []
    except Exception as exc:
        print(f"[Dataset Discovery] Discovery failed for {snapshot.source_id} (non-fatal): {exc}")
        return []


def probe_dataset_metadata(snapshot: SourceSnapshot, variables: list, bounding_box=None, time_range=None) -> DatasetMetadata:
    """Best-effort dataset metadata probe before download.

    Agent 3 discovers and ranks sources; Agent 4 only asks the chosen
    connector for retrieval metadata needed to execute and validate the
    download.
    """
    connector = get_connector(snapshot)
    fetch_request = FetchRequest(variables=variables, bounding_box=bounding_box, time_range=time_range)
    try:
        return connector.probe_metadata(snapshot, fetch_request)
    except Exception as exc:
        print(f"[Metadata Probe] Probe failed for {snapshot.source_id} (non-fatal): {exc}")
        return DatasetMetadata(
            source_id=snapshot.source_id,
            dataset_id=snapshot.source_id,
            download_endpoint=snapshot.url,
            api_endpoint=snapshot.url,
            metadata_endpoint=snapshot.url,
            variables=list(getattr(snapshot, "variables_available", []) or variables or []),
            retrieval_method="unavailable",
            unavailable_reason=f"Metadata probe failed: {exc}",
        )
