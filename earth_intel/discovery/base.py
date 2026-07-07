"""
Base class for all discovery plugins.

Every discoverer must implement search(request) and return a list of
CandidateSource objects.  Failures inside a discoverer are caught by
the DiscoveryEngine — they never crash the pipeline.
"""

from abc import ABC, abstractmethod
from typing import List

from models.discovery_schemas import CandidateSource
from models.retrieval_request import RetrievalRequest


class BaseDiscoverer(ABC):

    @abstractmethod
    def search(self, request: RetrievalRequest) -> List[CandidateSource]:
        """
        Search for datasets matching the request.
        Must return a list of CandidateSource objects.
        Must NOT raise exceptions — catch internally and return [].
        """
        ...
