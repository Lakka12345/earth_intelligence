"""
Connector Registry.

Add new provider-specific connectors here as they're built. Order
matters: registry tries each in order and uses the first that
can_handle() the source; GenericHTTPConnector is always last since it
always returns True from can_handle().
"""

from typing import List

from agents.agent4_connectors.base import BaseConnector
from agents.agent4_connectors.erddap import ERDDAPConnector
from agents.agent4_connectors.generic_http import GenericHTTPConnector
from models.website_analysis_schemas import SourceSnapshot

_CONNECTORS: List[BaseConnector] = [
    ERDDAPConnector(),
    # Add provider-specific connectors here as they're built, e.g.:
    # STACConnector(), NASAEarthdataConnector(), CKANConnector(), ...
    GenericHTTPConnector(),  # must stay last -- universal fallback
]


def get_connector(snapshot: SourceSnapshot) -> BaseConnector:
    for connector in _CONNECTORS:
        if connector.can_handle(snapshot):
            return connector
    return _CONNECTORS[-1]  # unreachable in practice, GenericHTTPConnector always matches
