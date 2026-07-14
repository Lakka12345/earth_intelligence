"""
Backward-compatible shim for the Phase 1 connector architecture.

New code should import from connectors.base_connector.
"""

from connectors.base_connector import BaseConnector, ConnectorDescriptor, ConnectorMatch, Credentials, FetchRequest

__all__ = [
    "BaseConnector",
    "ConnectorDescriptor",
    "ConnectorMatch",
    "Credentials",
    "FetchRequest",
]
