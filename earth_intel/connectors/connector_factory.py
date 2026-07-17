"""
Connector selection for Agent 4.

The factory is the only place that chooses a connector from Agent 3's
handoff context. Agent 4 should ask the factory instead of hardcoding
provider checks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from connectors.base_connector import BaseConnector
from connectors.connector_registry import registry
from models.website_analysis_schemas import SourceSnapshot

_BUILTINS_LOADED = False


@dataclass(frozen=True)
class ConnectorSelectionContext:
    provider_name: str = ""
    api_type: str = "unknown"
    dataset_type: str = "unknown"
    access_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SourceSnapshot,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ConnectorSelectionContext":
        return cls(
            provider_name=snapshot.name,
            api_type=snapshot.api_type,
            dataset_type=snapshot.dataset_type,
            metadata=metadata or {},
        )


class ConnectorFactory:
    def __init__(self, connector_registry=registry) -> None:
        self.registry = connector_registry

    def _ensure_builtin_connectors(self) -> None:
        global _BUILTINS_LOADED
        if _BUILTINS_LOADED:
            return
        import connectors.copernicus_cds_connector  # noqa: F401
        import connectors.copernicus_marine_connector  # noqa: F401
        import connectors.dataone_connector  # noqa: F401
        import connectors.earth_engine_connector  # noqa: F401
        import connectors.gdacs_connector  # noqa: F401
        import connectors.ghrsst_connector  # noqa: F401
        import connectors.incois_connector  # noqa: F401
        import connectors.nasa_earthdata_connector  # noqa: F401
        import connectors.noaa_connector  # noqa: F401
        import connectors.open_meteo_connector  # noqa: F401
        import connectors.planetary_computer_connector  # noqa: F401
        import connectors.protocol_connectors  # noqa: F401
        import connectors.generic_http_connector  # noqa: F401
        _BUILTINS_LOADED = True

    def select_connector(
        self,
        snapshot: SourceSnapshot,
        context: Optional[ConnectorSelectionContext] = None,
    ) -> BaseConnector:
        self._ensure_builtin_connectors()
        context = context or ConnectorSelectionContext.from_snapshot(snapshot)
        context_dict = {
            "provider_name": context.provider_name,
            "api_type": context.api_type,
            "dataset_type": context.dataset_type,
            "access_type": context.access_type,
            "metadata": context.metadata,
        }

        # CHANGED: previously started best_score = -1, so when NO connector
        # genuinely matched (every can_handle() correctly returned False,
        # every score was 0), the first-registered connector still "won"
        # the tie purely because 0 > -1 the first time through the loop --
        # subsequent 0-scores never beat it (strict >). copernicus_cds_connector
        # is first in the import list above, so any source nothing else
        # recognized silently defaulted to the CDS connector instead of the
        # intended generic_rest fallback. Starting at 0 means only a real
        # match (score > 0) can ever become best_connector; if nothing
        # matches, we fall through to an explicit, visible fallback below
        # instead of an accidental one based on import order.
        best_connector = None
        best_score = 0
        for connector in self.registry.list_connectors():
            match = connector.match_score(snapshot, context_dict)
            if match.score > best_score:
                best_connector = connector
                best_score = match.score

        if best_connector is None:
            # Nothing genuinely matched -- explicitly use the generic
            # fallback connector rather than letting registration order
            # decide. Look it up by connector_id first (fast path); if the
            # id ever changes, fall back to a provider_name search so this
            # doesn't silently break.
            best_connector = self.registry.get("generic_rest") or self.registry.get("generic_http")
            if best_connector is None:
                for connector in self.registry.list_connectors():
                    if "generic" in connector.descriptor.provider_name.lower():
                        best_connector = connector
                        break

        if best_connector is None:
            raise RuntimeError("No connectors are registered. GenericHTTPConnector should be registered as fallback.")
        return best_connector


factory = ConnectorFactory()


def get_connector(
    snapshot: SourceSnapshot,
    context: Optional[ConnectorSelectionContext] = None,
) -> BaseConnector:
    return factory.select_connector(snapshot, context)
