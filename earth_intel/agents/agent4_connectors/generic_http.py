"""
Backward-compatible shim for GenericHTTPConnector.

New code should import from connectors.generic_http_connector.
"""

from connectors.generic_http_connector import GenericHTTPConnector

__all__ = ["GenericHTTPConnector"]
