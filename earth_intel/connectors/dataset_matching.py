"""
Reusable dataset matching helpers for provider connectors.

These helpers keep Phase 2 provider connectors focused on official
catalog metadata instead of duplicating matching boilerplate.
"""

from typing import Iterable, List, Optional

from connectors.base_connector import BaseConnector, ConnectorDescriptor, ConnectorMatch, FetchRequest
from models.agent4_schemas import DatasetDescriptor, DatasetMetadata, SizeEstimate, format_bytes
from models.website_analysis_schemas import SourceSnapshot


def normalize(value: str) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").replace("-", " ").split())


def any_token_matches(needles: Iterable[str], haystack: str) -> bool:
    normalized_haystack = normalize(haystack)
    return any(normalize(needle) in normalized_haystack for needle in needles if normalize(needle))


def variable_score(requested: Iterable[str], supported: Iterable[str]) -> int:
    score = 0
    supported_text = " ".join(supported)
    for variable in requested:
        normalized = normalize(variable)
        if not normalized:
            continue
        if any(normalized in normalize(candidate) or normalize(candidate) in normalized for candidate in supported):
            score += 3
        elif normalized in normalize(supported_text):
            score += 1
    return score


class StaticDatasetConnector(BaseConnector):
    datasets: List[DatasetDescriptor] = []
    provider_keywords: tuple = ()
    api_keywords: tuple = ()

    def can_handle(self, snapshot: SourceSnapshot) -> bool:
        provider_text = f"{snapshot.name} {snapshot.url} {snapshot.api_type} {snapshot.dataset_type}"
        return any_token_matches(self.provider_keywords, provider_text)

    def match_score(self, snapshot: SourceSnapshot, context=None) -> ConnectorMatch:
        if not self.can_handle(snapshot):
            return ConnectorMatch(score=0, reason="Provider keywords did not match.")
        score = max(10, 1000 - self.descriptor.priority)
        provider_text = f"{snapshot.name} {snapshot.url} {snapshot.api_type} {snapshot.dataset_type}"
        if any_token_matches(
            (self.descriptor.provider_name, self.descriptor.connector_id.replace("_", " ")),
            provider_text,
        ):
            score += 100
        api_text = f"{snapshot.api_type} {(context or {}).get('api_type', '')}"
        if any_token_matches(self.api_keywords, api_text):
            score += 25
        return ConnectorMatch(score=score, reason=f"{self.descriptor.provider_name} provider match.")

    def discover_datasets(self, snapshot: SourceSnapshot, context=None) -> List[DatasetDescriptor]:
        requested = list((context or {}).get("variables") or getattr(snapshot, "variables_available", []) or [])
        ranked = sorted(
            self.datasets,
            key=lambda dataset: (
                variable_score(requested, dataset.supported_variables),
                any_token_matches([snapshot.name], dataset.dataset_name),
            ),
            reverse=True,
        )
        return ranked or []

    def _best_dataset(self, snapshot: SourceSnapshot, fetch_request: FetchRequest) -> Optional[DatasetDescriptor]:
        datasets = self.discover_datasets(snapshot, {"variables": fetch_request.variables})
        return datasets[0] if datasets else None

    def probe_metadata(self, snapshot: SourceSnapshot, fetch_request: FetchRequest) -> DatasetMetadata:
        dataset = self._best_dataset(snapshot, fetch_request)
        if dataset is None:
            return DatasetMetadata(
                source_id=snapshot.source_id,
                dataset_id=snapshot.source_id,
                variables=list(snapshot.variables_available or fetch_request.variables or []),
                retrieval_method=f"{self.name} dataset catalog",
                unavailable_reason="No matching dataset descriptor was found in this connector catalog.",
            )
        return dataset.to_metadata(snapshot.source_id)

    def probe_size(self, snapshot: SourceSnapshot, fetch_request: FetchRequest) -> SizeEstimate:
        dataset = self._best_dataset(snapshot, fetch_request)
        if dataset and dataset.estimated_size_bytes is not None:
            return SizeEstimate(
                source_id=snapshot.source_id,
                estimated_bytes=dataset.estimated_size_bytes,
                is_exact=False,
                method=f"{self.name} dataset metadata estimate",
                human_readable=format_bytes(dataset.estimated_size_bytes),
            )
        return SizeEstimate(
            source_id=snapshot.source_id,
            method=f"{self.name} metadata unavailable",
            human_readable="Unknown",
        )
