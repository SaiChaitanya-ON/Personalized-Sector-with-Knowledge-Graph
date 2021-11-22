import datetime
import logging
import math
import ssl
from collections import Counter, defaultdict
from dataclasses import dataclass
from json import JSONDecodeError
from typing import List, Optional, Tuple

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.connection import create_ssl_context

from onai.ml.config import BaseConfig
from onai.ml.peers.candidate_suggestion.base import CandidateSuggestion
from onai.ml.peers.dp import (
    RateConversionRequest,
    fetch_conversion_rates_v2,
    get_retriable_session,
    similar_name_ed,
)
from onai.ml.peers.dp_requests import (
    GET_FINANCIAL_FIELDS_BULK_QUERY,
    HEADERS,
    get_financial_fields_bulk_query,
)
from onai.ml.peers.types import CompanyDetail, Financial, Financials, PeerSuggestion
from onai.ml.tools.argparse import add_bool_argument
from onai.ml.utils import deep_get

logger = logging.getLogger(__name__)

DP_REQ_SIZE = 100


@dataclass
class Config(BaseConfig):
    @staticmethod
    def populate_argparser(parser):
        parser.add_argument("--tgt_currency")
        parser.add_argument("--internal_ds")
        parser.add_argument("--es_host")
        parser.add_argument("--es_port", type=int)
        parser.add_argument("--es_index", type=str)
        add_bool_argument(parser, "use_ssl", default=False)
        parser.add_argument("--dp_financials", default=[], nargs="*")

    dp_financials: List[str]
    tgt_currency: str = "EUR"
    internal_ds: Optional[str] = None
    es_host: str = "host.docker.internal"
    es_port: int = 9200
    es_index: str = "company"
    use_ssl: bool = False


class ESCandidateSuggestion(CandidateSuggestion):

    _ES_BATCH_SIZE = 10

    @classmethod
    def from_cfg(cls, cfg: Config):
        return cls(
            cfg.es_host,
            cfg.es_port,
            cfg.internal_ds,
            es_index=cfg.es_index,
            tgt_currency=cfg.tgt_currency,
            use_ssl=cfg.use_ssl,
        )

    def __init__(
        self,
        es_host,
        es_port,
        internal_ds: str = None,
        use_ssl=False,
        es_index="company",
        min_doc_freq=1,
        min_term_freq=0,
        max_query_terms=10,
        tgt_currency="EUR",
        dp_financials=None,
        retry_on_timeout=False,
        max_request_timeout=10,
    ):
        if use_ssl:
            ssl_context = create_ssl_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        else:
            ssl_context = None

        self.client = Elasticsearch(
            hosts=[{"host": es_host, "port": es_port}],
            indices=[es_index],
            scheme="https" if use_ssl else "http",
            ssl_context=ssl_context,
            retry_on_timeout=retry_on_timeout,
        )

        self.index = es_index
        self.total_docs = self.client.count()["count"]
        self.min_doc_freq = min_doc_freq
        self.min_term_freq = min_term_freq
        self.max_query_terms = max_query_terms
        self.tgt_currency = tgt_currency
        self.dp_financials = dp_financials
        if internal_ds:
            self.internal_ds = pd.read_excel(internal_ds, ["Info", "Financials"])
        else:
            self.internal_ds = None
        self.max_request_timeout = max_request_timeout

    def get_companies_detail_by_id(
        self, entity_ids: List[str], start_year: int = 2008, end_year: int = 2019
    ) -> List[CompanyDetail]:
        es_res_by_id = {
            es_res["entity_id"]: es_res
            for es_res in self.search_by_entity_ids(entity_ids)
        }
        fs = fetch_financials_from_dp(
            entity_ids, start_year, end_year, self.tgt_currency, self.dp_financials
        )
        assert len(fs) == len(entity_ids)
        ret = []
        for eid, f in zip(entity_ids, fs):
            if eid not in es_res_by_id:
                logger.warning("Cannot get information from ES on entity %s", eid)
                ret.append(None)
                continue
            es_res = es_res_by_id[eid]
            ret.append(
                CompanyDetail(
                    name=es_res["name"],
                    description=es_res["business_description"],
                    region=es_res["region"],
                    fye=f[1],
                    financials=f[0],
                    sector_description=es_res.get("primary_sic_node_desc"),
                    country=es_res.get("country_of_incorporation_iso"),
                    predicted_industries=es_res.get("predicted_industries", []),
                    entity_id=es_res["entity_id"],
                )
            )
        assert len(ret) == len(entity_ids)
        return ret

    # TODO: figure out a way to batch query es
    # for now, since this is just applicable in annotation stage
    # this would be fine
    def get_companies_detail_by_name(
        self, company_names: List[str], start_year: int = 2008, end_year: int = 2019
    ) -> List[CompanyDetail]:
        tgt_currency = self.tgt_currency

        ret = []
        unresolved_entity_ids = set()
        entity_id_to_name = {}
        logger.info("Fetching info from ES.")
        info = self.internal_ds["Info"] if self.internal_ds else None
        for name in company_names:
            name = name.strip()
            # first check if that exists in internal_ds and it is marked as INTERNAL
            info_res = (
                info.loc[info["Base Borrower Name"] == name]
                if info is not None
                else None
            )

            # this base company is in internal data, can be found from an excel sheet
            if info_res is not None and info_res.size and info_res.iloc[0]["INTERNAL"]:
                financials = self.internal_ds["Financials"]
                s = info_res.iloc[0]
                industries = []
                internal_id = s["Case"]
                financial_res = financials.loc[financials["Case"] == internal_id]
                cur = s["currency"].strip().upper()
                mag = s["magnitude"]
                base_company_financials = {
                    k: [Financial(None, yr) for yr in range(start_year, end_year + 1)]
                    for k in self.dp_financials
                }
                # TODO: make sure the currency/magnitude here is converted into the same magnitude/currency fetched from
                # external data platform.

                # pre-populate data structure
                for _, r in financial_res.iterrows():
                    outcome_ls = base_company_financials[r["Data Item"]]
                    for yr in range(start_year, end_year + 1):
                        if yr in r and not pd.isna(r[yr]):
                            if r["Data Item"] in ("EBITDA_MARG", "TOTAL_DEBT_EQUITY"):
                                outcome_ls[yr - start_year] = Financial(r[yr], yr)
                            else:
                                val = (
                                    fetch_conversion_rates_v2(
                                        [
                                            RateConversionRequest(
                                                cur,
                                                tgt_currency,
                                                datetime.date(end_year, 1, 1)
                                                if pd.isna(s["FYE"])
                                                else datetime.date(
                                                    yr, int(s["FYE"]), 1
                                                ),
                                            )
                                        ]
                                    )[0]
                                    * r[yr]
                                )
                                outcome_ls[yr - start_year] = Financial(
                                    val, yr, tgt_currency, mag
                                )

                ret.append(
                    CompanyDetail(
                        name=name,
                        description=s["Base Borrower Description"],
                        region=s["Base Borrower Region"],
                        fye=datetime.date(end_year, int(s["FYE"]), 1)
                        if not pd.isna(s["FYE"])
                        else None,
                        financials=base_company_financials,
                        sector_description=s["Base Borrower Sector"],
                        predicted_industries=industries,
                    )
                )
            # information of this base company can be found in DP
            else:
                es_res = self.search_query(name, size=5)
                if not similar_name_ed(es_res[0]["name"], name):
                    logger.warning(
                        "Cannot get information for %s from ES. Closest result is %s. Skipping...",
                        name,
                        es_res[0]["name"],
                    )
                    logger.warning(str([e["name"] for e in es_res]))
                    ret.append(None)
                    continue
                # if there are multiple _exact_ matches, remove this item
                matches = sum(r["name"] == name for r in es_res)
                if matches > 1:
                    logger.warning(
                        "Cannot get information for %s from ES, as there are multiple exact matches.",
                        name,
                    )

                es_res = es_res[0]
                ret.append(
                    CompanyDetail(
                        name=name,
                        description=es_res["business_description"],
                        region=es_res["region"],
                        fye=None,  # will be populated by batch dp request
                        financials={},
                        sector_description=es_res.get("primary_sic_node_desc"),
                        country=es_res.get("country_of_incorporation_iso"),
                        predicted_industries=es_res.get("predicted_industries", []),
                        entity_id=es_res["entity_id"],
                    )
                )
                entity_id_to_name[es_res["entity_id"]] = name
                unresolved_entity_ids.add(es_res["entity_id"])

        # ensure the ordering
        logger.info("Fetch info from DP")
        unresolved_entity_ids = list(unresolved_entity_ids)
        fs = fetch_financials_from_dp(
            unresolved_entity_ids,
            start_year,
            end_year,
            tgt_currency,
            self.dp_financials,
        )
        assert len(fs) == len(unresolved_entity_ids)
        name_to_financial = {
            entity_id_to_name[entity_id]: f
            for entity_id, f in zip(unresolved_entity_ids, fs)
        }

        for r in ret:
            if r and len(r.financials) == 0:
                # unresolved_entity. Fill financials in
                r.financials, r.fye = name_to_financial[r.name]
        return ret

    def get_company_detail(
        self, base_company_name: str, start_year: int = 2008, end_year: int = 2019
    ) -> CompanyDetail:
        return self.get_companies_detail_by_name(
            [base_company_name], start_year, end_year
        )[0]

    def search_by_entity_ids(self, entity_ids: List[str]):
        ret = []
        for i in range(0, len(entity_ids), self._ES_BATCH_SIZE):
            query = {
                "_source": True,
                "query": {"ids": {"values": entity_ids[i : i + self._ES_BATCH_SIZE]}},
            }
            json_res = self.client.search(
                body=query,
                index=self.index,
                timeout=f"{self.max_request_timeout}s",
                request_timeout=self.max_request_timeout,
            )
            ret.extend(hit["_source"] for hit in json_res["hits"]["hits"])
        return ret

    def search_query(self, base_name, size=10):
        fuzzy_query = {
            "_source": True,
            "from": 0,
            "size": size,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": base_name,
                                "fuzziness": "2",
                                "prefix_length": 1,
                                "fields": ["name", "name.cleaned"],
                                "minimum_should_match": "1",
                                "type": "most_fields",
                            }
                        },
                        {
                            "multi_match": {
                                "query": base_name,
                                "fuzziness": "1",
                                "prefix_length": 1,
                                "fields": ["name", "name.cleaned"],
                                "minimum_should_match": "1",
                                "type": "most_fields",
                                "boost": 2,
                            }
                        },
                        {
                            "multi_match": {
                                "query": base_name,
                                "fields": ["name", "name.cleaned"],
                                "minimum_should_match": "1",
                                "type": "most_fields",
                                "boost": 4,
                            }
                        },
                    ]
                }
            },
        }

        json_result = self.client.search(index=self.index, body=fuzzy_query)
        return [hit["_source"] for hit in json_result["hits"]["hits"]]

    def extract_negative_words(
        self,
        company: CompanyDetail,
        min_doc_freq: int = 1,
        min_term_freq: int = 0,
        max_query_terms: int = 10,
        size: int = 20,
        topn_front: int = 15,
        topn_tail: int = 30,
    ):
        """
        Given a base company, extract a set of negative (irelevant) words by suggesting companies that are
        NOT in the same sector as it. This should hopefully get rid of high idf words that are actually not
        indicative of business (e.g. name of country, suffixes like Ltd., Inc. etc).
        """

        synthetic_doc = {}
        synthetic_doc["business_description"] = company.description

        query = {"query": {"bool": {"must": []}}}
        company_region = company.region

        fields_based_filter: dict = defaultdict(list)

        if company_region is not None:
            fields_based_filter["must"].append(
                {"term": {"region.keyword": {"value": company_region}}}
            )

        query["query"]["bool"] = dict(fields_based_filter)
        mlt = {
            "more_like_this": {
                "like": [{"doc": synthetic_doc}],
                "fields": ["business_description"],
                "min_doc_freq": min_doc_freq,
                "min_term_freq": min_term_freq,
                "max_query_terms": max_query_terms,
                "include": True,
            }
        }

        query_boolean_should = query["query"]["bool"].get("should", [])
        query_boolean_should.append(mlt)
        query["query"]["bool"]["should"] = query_boolean_should
        query["size"] = size

        json_result = self.client.search(index=self.index, body=query)

        results = json_result["hits"]["hits"]

        negative_words_idf = {}
        for result in results:
            negative_words_idf.update(
                self.get_document_idf(result["_source"]["business_description"])
            )

        base_company_idf = self.get_document_idf(company.description)

        negative_words = Counter(
            {
                word: idf
                for word, idf in negative_words_idf.items()
                if word in base_company_idf
            }
        )

        # Sort words by their idf and take the top few and bottom few to form our list of negatives.
        sorted_words = negative_words.most_common()

        return [w for w, _ in sorted_words[:topn_front] + sorted_words[-topn_tail:]]

    def get_documents_idf(
        self, business_descriptions: List[str], batch_size=128
    ) -> List[dict]:
        """
        Given a list of business descriptions, use the mtermvectors api to get their token idfs
        """
        n = len(business_descriptions)
        terms_idf = []

        for i in range(0, n, batch_size):
            description_batch = business_descriptions[i : i + batch_size]
            query = {
                "parameters": {
                    "fields": ["business_description"],
                    "term_statistics": True,
                },
                "docs": [
                    {"doc": {"business_description": business_description}}
                    for business_description in description_batch
                ],
            }

            json_result = self.client.mtermvectors(index=self.index, body=query)
            assert len(json_result["docs"]) == len(
                business_descriptions
            ), "Failed to retrieve term statistics for all business descriptions"
            for el in json_result["docs"]:
                term_idf = {}
                terms = (
                    deep_get(el, "term_vectors", "business_description", "terms") or {}
                )
                for term, term_stats in terms.items():
                    term_idf[term] = math.log(
                        self.total_docs / term_stats.get("doc_freq", 1)
                    )
                terms_idf.append(term_idf)

        return terms_idf

    def get_document_idf(self, business_description: str) -> dict:
        """
        Given a business description, return it in terms of its tokens with their idf
        """

        return self.get_documents_idf([business_description])[0]

    def suggest_candidates(
        self,
        company: CompanyDetail,
        start_year: int = 2008,
        end_year: int = 2019,
        size: int = 20,
        get_peer_financial: bool = True,
        must_filters: Optional[List[dict]] = None,
        sort_by: Optional[dict] = None,
    ) -> List[PeerSuggestion]:

        synthetic_doc = {}
        synthetic_doc["business_description"] = company.description

        # TODO: remove this
        synthetic_doc["predicted_industries"] = company.predicted_industries

        query = {"query": {"bool": {}}, "sort": {}}

        fields_based_filter: dict = defaultdict(list)
        fields_based_filter["must"].append(
            {"term": {"region.keyword": {"value": company.region}}}
        )
        if must_filters:
            fields_based_filter["must"].extend(must_filters)

        # TODO: remove this
        for el in synthetic_doc["predicted_industries"]:
            fields_based_filter["should"].append(
                {"match": {"predicted_industries": el}}
            )

        fields_based_filter["must"].append(
            {
                "range": {
                    "last_filing_date": {
                        # feature extractor will render the peers of which last
                        # reported financials are over than 4 years old to be no financial
                        # we relax the hard filter in elastic search to be after the start_year
                        # because for some analysts peers w/o financials are still useful
                        # based on the annotations
                        "gte": (datetime.datetime(start_year, 1, 1))
                    }
                }
            }
        )

        query["query"]["bool"] = dict(fields_based_filter)

        mlt = {
            "more_like_this": {
                "like": [{"doc": synthetic_doc}],
                "fields": ["business_description", "predicted_industries"],
                "min_doc_freq": self.min_doc_freq,
                "min_term_freq": self.min_term_freq,
                "max_query_terms": self.max_query_terms,
                "include": True,
            }
        }

        query_boolean_should = query["query"]["bool"].get("should", [])
        query_boolean_should.append(mlt)
        query["query"]["bool"]["should"] = query_boolean_should
        query["size"] = int(math.ceil(size * 1.2))

        # all documents should have this field (otherwise it is useless company for our analysts anyway)
        query_boolean_filter = query["query"]["bool"]["filter"] = []
        query_boolean_filter.append({"exists": {"field": "business_description"}})

        # control sort order of peers as per callers need
        if sort_by is not None:
            query["sort"]["_script"] = sort_by

        json_result = self.client.search(
            index=self.index,
            body=query,
            timeout=f"{self.max_request_timeout}s",
            request_timeout=self.max_request_timeout,
        )
        results = json_result["hits"]["hits"]

        ret = []
        entity_ids = list()

        for result in results:
            if len(ret) >= size:
                break
            es_res = result["_source"]
            # remove company with exact name as it _must be_ bad peer
            if es_res["name"] == company.name:
                continue

            cd = CompanyDetail(
                name=es_res["name"],
                description=es_res["business_description"],
                region=es_res["region"],
                fye=None,  # will be populated by batch dp request
                financials={},
                sector_description=es_res.get("primary_sic_node_desc"),
                country=es_res.get("country_of_incorporation_iso"),
                predicted_industries=es_res.get("predicted_industries", []),
                entity_id=es_res["entity_id"],
            )
            entity_ids.append(es_res["entity_id"])
            ret.append(
                PeerSuggestion(
                    rank=len(ret) + 1,
                    detail=cd,
                    score=result["_score"] if sort_by is None else result["sort"][0],
                )
            )
        if get_peer_financial:
            fs = fetch_financials_from_dp(
                entity_ids, start_year, end_year, self.tgt_currency, self.dp_financials
            )

            for i, (f, fye) in enumerate(fs):
                ret[i].detail.financials = f
                ret[i].detail.fye = fye
        return ret

    def suggest_candidates_by_name(
        self,
        base_company_name: str,
        start_year: int = 2008,
        end_year: int = 2019,
        size: int = 20,
        must_filters: Optional[List[dict]] = None,
    ) -> Tuple[CompanyDetail, List[PeerSuggestion]]:
        base_company_detail = self.get_company_detail(
            base_company_name, start_year, end_year
        )
        coarse_peer_res = self.suggest_candidates(
            base_company_detail, start_year, end_year, size, must_filters=must_filters
        )
        return base_company_detail, coarse_peer_res

    def score_candidates(
        self,
        base_company: CompanyDetail,
        peer_candidates: List[CompanyDetail],
        end_year: int,
    ) -> List[float]:
        raise NotImplementedError


def fetch_financials_from_dp(
    entity_ids: List[str],
    start_year: int,
    end_year: int,
    tgt_currency: str,
    financials: List[str] = (),
) -> List[Tuple[Financials, datetime.date]]:
    """

    :param entity_ids: A list of UUIDs
    :param start_year: fetch the financial after the start_year (financial.year >= start_year)
    :param end_year: fetch the financial before the end_year (financial.year <= end_year)
    :param tgt_currency: target currency in ISO3 format
    :param financials: financials to fetch in DIME format
    """
    time_series_params = {
        "period": {
            # end_year + 1 to account for end_year-12-31
            "range": {"start": f"{start_year}-01-01", "end": f"{end_year + 1}-01-01"}
        },
        "periodType": "ANNUAL",
        "currency": {"targetCurrency": tgt_currency},
    }
    resp = []
    query = (
        get_financial_fields_bulk_query(financials)
        if financials
        else GET_FINANCIAL_FIELDS_BULK_QUERY
    )

    for i in range(0, len(entity_ids), DP_REQ_SIZE):
        raw_resp = None
        try:
            with get_retriable_session(api_gw_proxy=True) as s:
                raw_resp = s.post(
                    "http://x.data-services.onai.cloud/api/",
                    json={
                        "query": query,
                        "variables": {
                            "companyIds": entity_ids[i : i + DP_REQ_SIZE],
                            "params": time_series_params,
                        },
                    },
                    headers=HEADERS,
                )
                resp.extend(raw_resp.json()["data"]["entities"])

        except JSONDecodeError:
            logger.warning(
                "Cannot decode responce %s with request %s",
                raw_resp.content,
                entity_ids,
            )
            raise
        except KeyError as e:
            logger.warning("Missing Key %s in response %s", e.args[0], raw_resp.content)
            raise

    assert len(resp) == len(entity_ids), (len(resp), len(entity_ids), resp, entity_ids)
    ret = []
    for entity_id, entity in zip(entity_ids, resp):
        fields = entity["fields"]
        financials = {}
        fye = None
        for ts in fields:
            outcome_ls = financials[ts["dataItem"]["mnemonic"]] = [
                Financial(None, yr) for yr in range(start_year, end_year + 1)
            ]
            tp = ts["__typename"]
            if "dataPoints" not in ts:
                continue
            for dp in ts["dataPoints"]:
                event_dt = datetime.datetime.strptime(dp["eventDate"], "%Y-%m-%d")
                fye = event_dt.date()
                assert start_year <= event_dt.year <= end_year, dp
                if tp == "MonetaryAmountTimeSeries":
                    if dp["monetaryAmount"]["currency"]["code"] == tgt_currency:
                        financial = Financial(
                            dp["monetaryAmount"]["value"] / 1e6,
                            event_dt.year,
                            dp["monetaryAmount"]["currency"]["code"],
                            "m",
                        )
                    else:
                        logger.warning(
                            "Expect %s. Get %s for %s on entity %s",
                            tgt_currency,
                            dp["monetaryAmount"]["currency"]["code"],
                            ts["dataItem"]["mnemonic"],
                            entity_id,
                        )
                        financial = Financial(None, event_dt.year)

                elif tp == "FloatTimeSeries":
                    financial = Financial(dp["floatValue"], event_dt.year)
                elif tp == "IntegerTimeSeries":
                    financial = Financial(dp["intValue"], event_dt.year)
                else:
                    assert False, "Unrecognised type of time series"
                outcome_ls[event_dt.year - start_year] = financial

        if fye is None:
            logger.debug("%s has no fye", entity_id)
        ret.append((financials, fye))
    return ret
