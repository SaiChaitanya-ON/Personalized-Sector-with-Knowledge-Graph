import datetime
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pycountry

logger = logging.getLogger(__name__)


@dataclass
class Financial:
    val: Optional[float]
    year: int
    currency: Optional[str] = None
    magnitude: Optional[str] = None


Financials = Dict[str, List[Financial]]
ISO3_CODE = str


@dataclass
class CompanyDetail:
    name: str
    description: str
    region: str
    fye: Optional[datetime.date] = None
    financials: Financials = field(default_factory=dict)
    sector_description: str = ""
    country: Optional[ISO3_CODE] = None
    predicted_industries: List[str] = field(default_factory=list)
    entity_id: Optional[str] = None

    def __post_init__(self):
        for fs in self.financials.values():
            fs.sort(key=lambda x: x.year)

        if self.country is not None:
            self.country = self.country.upper()
            if not pycountry.countries.get(alpha_3=self.country.upper()):
                logger.warning("Ignoring unrecognised ISO3 code: %s", self.country)
                self.country = None


@dataclass
class PeerSuggestion:
    rank: int
    detail: CompanyDetail
    score: Optional[float] = None
