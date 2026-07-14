"""
Connector registry for Agent 4.

Future connector files should register themselves by importing
register_connector and calling it once at module import time.
"""

from collections import OrderedDict
from typing import Iterable, List, Optional, Type, Union

from connectors.base_connector import BaseConnector

ConnectorInput = Union[BaseConnector, Type[BaseConnector]]


class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: "OrderedDict[str, BaseConnector]" = OrderedDict()

    def register(self, connector: ConnectorInput) -> BaseConnector:
        instance = connector() if isinstance(connector, type) else connector
        connector_id = instance.descriptor.connector_id
        if connector_id in self._connectors:
            return self._connectors[connector_id]
        self._connectors[connector_id] = instance
        return instance

    def unregister(self, connector_id: str) -> None:
        self._connectors.pop(connector_id, None)

    def list_connectors(self) -> List[BaseConnector]:
        return list(self._connectors.values())

    def connector_ids(self) -> List[str]:
        return list(self._connectors.keys())

    def get(self, connector_id: str) -> Optional[BaseConnector]:
        return self._connectors.get(connector_id)

    def supports_provider(self, provider_name: str) -> bool:
        normalized = (provider_name or "").strip().lower()
        if not normalized:
            return False
        return any(
            connector.descriptor.provider_name.lower() == normalized
            for connector in self._connectors.values()
        )

    def extend(self, connectors: Iterable[ConnectorInput]) -> None:
        for connector in connectors:
            self.register(connector)


registry = ConnectorRegistry()


def register_connector(connector: ConnectorInput) -> BaseConnector:
    return registry.register(connector)


def list_connectors() -> List[BaseConnector]:
    return registry.list_connectors()


def get_connector_by_id(connector_id: str) -> Optional[BaseConnector]:
    return registry.get(connector_id)


def supports_provider(provider_name: str) -> bool:
    return registry.supports_provider(provider_name)
