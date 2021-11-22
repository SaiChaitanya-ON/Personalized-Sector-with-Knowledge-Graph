import datetime
import random
import string
from dataclasses import dataclass
from math import ceil
from typing import List, Tuple

from onai.ml.config import BaseConfig
from onai.ml.peers.candidate_suggestion.base import CandidateSuggestion
from onai.ml.peers.dp_requests import DEFAULT_FINANCIALS
from onai.ml.peers.types import CompanyDetail, Financial, PeerSuggestion


@dataclass
class Config(BaseConfig):
    @staticmethod
    def populate_argparser(parser):
        parser.add_argument(
            "--start_year", type=int, help="starting date of the fetched financials"
        )
        parser.add_argument(
            "--end_year", type=int, help="End date of the fetched financials"
        )
        parser.add_argument("--max_peers", type=int)

    start_year: int
    end_year: int
    max_peers: int = 20


class RandomCandidateSuggestion(CandidateSuggestion):

    _char_list = string.ascii_lowercase + " "

    def suggest_candidates(
        self,
        base_company: CompanyDetail,
        start_year: int = 2008,
        end_year: int = 2019,
        size: int = 20,
    ) -> List[PeerSuggestion]:
        return self.suggest_candidates_by_name(base_company.name, size)[1]

    def suggest_candidates_by_name(
        self,
        base_company_name: str,
        start_year: int = 2008,
        end_year: int = 2019,
        size: int = 20,
    ) -> Tuple[CompanyDetail, List[PeerSuggestion]]:
        n_peers = int(
            ceil(max(1, min(random.gauss(self.cfg.max_peers, 1), self.cfg.max_peers)))
        )
        base_company_detail = self._generate_company_detail()
        base_company_detail.name = base_company_name
        return (
            base_company_detail,
            [
                PeerSuggestion(rank, self._generate_company_detail())
                for rank in range(1, n_peers + 1)
            ],
        )

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _random_date(start, end):
        """Generate a random datetime between `start` and `end`"""
        return start + datetime.timedelta(
            # Get a random amount of seconds between `start` and `end`
            seconds=random.randint(0, int((end - start).total_seconds()))
        )

    @staticmethod
    def _generate_random_str(length: int = 20):
        return "".join(
            random.choice(RandomCandidateSuggestion._char_list) for _ in range(length)
        )

    def _generate_currency(self):
        return random.choice(["EUR", "USD", None, "CNY"])

    def _generate_company_detail(self):
        return CompanyDetail(
            name=RandomCandidateSuggestion._generate_random_str(random.randint(5, 30)),
            description=RandomCandidateSuggestion._generate_random_str(
                random.randint(100, 400)
            ),
            region=RandomCandidateSuggestion._generate_random_str(
                random.randint(5, 10)
            ),
            fye=RandomCandidateSuggestion._random_date(
                datetime.date(2000, 1, 1), datetime.date(2020, 12, 31)
            ),
            financials={
                k: self._generate_financial(self._generate_currency())
                for k in DEFAULT_FINANCIALS
            },
        )

    def _generate_financial(self, currency: None):
        ret = []
        for yr in range(self.cfg.start_year, self.cfg.end_year + 1):
            if random.random() < 0.1:
                ret.append(Financial(None, yr, None))
            else:
                magnitude = random.choice(["m", "k", None])
                ret.append(Financial(random.uniform(100, 500), yr, currency, magnitude))
        return ret
