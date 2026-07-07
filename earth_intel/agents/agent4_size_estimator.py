"""
Agent 4 — Size Estimator.

Thin wrapper around the connector registry's probe_size. Kept separate
so the access-resolution and size-approval steps don't need to know
about connectors directly.
"""

from agents.agent4_connectors.base import FetchRequest
from agents.agent4_connectors.registry import get_connector
from models.agent4_schemas import SizeEstimate
from models.website_analysis_schemas import SourceSnapshot


def estimate_size(snapshot: SourceSnapshot, variables: list, bounding_box=None, time_range=None) -> SizeEstimate:
    connector = get_connector(snapshot)
    fetch_request = FetchRequest(variables=variables, bounding_box=bounding_box, time_range=time_range)
    try:
        return connector.probe_size(snapshot, fetch_request)
    except Exception as exc:
        print(f"[Size Estimator] Probe failed for {snapshot.source_id} (non-fatal): {exc}")
        return SizeEstimate(source_id=snapshot.source_id, method="unavailable")
