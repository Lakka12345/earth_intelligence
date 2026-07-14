"""
Backward-compatible shim for connector selection.

New code should use connectors.connector_factory.get_connector.
"""

from connectors.connector_factory import get_connector
from connectors.connector_registry import list_connectors, register_connector, supports_provider

__all__ = [
    "get_connector",
    "list_connectors",
    "register_connector",
    "supports_provider",
]
