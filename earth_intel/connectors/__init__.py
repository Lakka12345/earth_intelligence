"""
Agent 4 connector architecture.

Importing this package registers the built-in generic HTTP fallback.
Future connector modules can register themselves with
connectors.connector_registry.register_connector.
"""

from connectors.base_connector import BaseConnector, ConnectorDescriptor, Credentials, FetchRequest
from connectors.connector_factory import ConnectorFactory, ConnectorSelectionContext, get_connector
from connectors.connector_registry import (
    ConnectorRegistry,
    get_connector_by_id,
    list_connectors,
    register_connector,
    supports_provider,
)
from connectors.connector_types import (
    AccessType,
    AuthenticationType,
    CapabilityFlags,
    ConnectorType,
    DatasetType,
    RetryStrategy,
    ValidationStrategy,
)
from connectors.copernicus_cds_connector import CopernicusCDSConnector
from connectors.copernicus_marine_connector import CopernicusMarineConnector
from connectors.dataone_connector import DataONEConnector
from connectors.earth_engine_connector import EarthEngineConnector
from connectors.generic_http_connector import GenericHTTPConnector
from connectors.ghrsst_connector import GHRSSTConnector
from connectors.incois_connector import INCOISConnector
from connectors.nasa_earthdata_connector import NASAEarthDataConnector
from connectors.noaa_connector import NOAAConnector
from connectors.open_meteo_connector import OpenMeteoConnector
from connectors.planetary_computer_connector import PlanetaryComputerConnector
from connectors.protocol_connectors import (
    CKANConnector,
    ERDDAPConnector,
    FTPConnector,
    GenericRESTConnector,
    OGCAPIConnector,
    OPeNDAPConnector,
    S3Connector,
    STACConnector,
    THREDDSConnector,
)

__all__ = [
    "AccessType",
    "AuthenticationType",
    "BaseConnector",
    "CapabilityFlags",
    "ConnectorDescriptor",
    "ConnectorFactory",
    "ConnectorRegistry",
    "ConnectorSelectionContext",
    "ConnectorType",
    "CopernicusCDSConnector",
    "CopernicusMarineConnector",
    "Credentials",
    "DataONEConnector",
    "DatasetType",
    "EarthEngineConnector",
    "CKANConnector",
    "ERDDAPConnector",
    "FetchRequest",
    "FTPConnector",
    "GHRSSTConnector",
    "GenericHTTPConnector",
    "GenericRESTConnector",
    "INCOISConnector",
    "NASAEarthDataConnector",
    "NOAAConnector",
    "OGCAPIConnector",
    "OPeNDAPConnector",
    "OpenMeteoConnector",
    "PlanetaryComputerConnector",
    "RetryStrategy",
    "S3Connector",
    "STACConnector",
    "THREDDSConnector",
    "ValidationStrategy",
    "get_connector",
    "get_connector_by_id",
    "list_connectors",
    "register_connector",
    "supports_provider",
]
