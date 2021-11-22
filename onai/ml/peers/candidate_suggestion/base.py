import abc
from typing import List, Tuple

from onai.ml.peers.types import CompanyDetail, PeerSuggestion


class CandidateSuggestion(abc.ABC):
    @abc.abstractmethod
    def suggest_candidates(
        self, base_company: CompanyDetail, start_year: int, end_year: int, size: int
    ) -> List[PeerSuggestion]:
        pass

    @abc.abstractmethod
    def suggest_candidates_by_name(
        self, base_company_name: str, stat_year: int, end_year: int, size: int
    ) -> Tuple[CompanyDetail, List[PeerSuggestion]]:
        pass

    @abc.abstractmethod
    def score_candidates(
        self,
        base_company: CompanyDetail,
        peer_candidates: List[CompanyDetail],
        end_year: int,
    ) -> List[float]:
        pass
