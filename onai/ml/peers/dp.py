import datetime
import logging
import string
from dataclasses import dataclass
from typing import List

import editdistance
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from .dp_requests import GET_CONVERSION_RATES_QUERY, HEADERS

punctuation_translation = str.maketrans("", "", string.punctuation)

logger = logging.getLogger(__name__)


def similar_name_ed(base_name, result_name, max_ed=2):
    base_name = base_name.lower().translate(punctuation_translation)
    result_name = result_name.lower().translate(punctuation_translation)
    return editdistance.eval(base_name, result_name) < max_ed


def similar_name(base_name, result_name, match_tresh=0.5):
    base_name = base_name.lower().translate(punctuation_translation)
    result_name = result_name.lower().translate(punctuation_translation)
    n = min(len(base_name), len(result_name))

    m = 0
    for i in range(n):
        if base_name[i] == result_name[i]:
            m += 1

    return m / n >= match_tresh


def render_latest_monetary_values(field):
    ret = {}
    if "dataPoints" not in field:
        return {}
    for data_point in field["dataPoints"]:
        date = data_point["eventDate"].split("-")[0]
        value = data_point["monetaryAmount"]["value"]
        currency = data_point["monetaryAmount"]["currency"]["code"]

        ret[date] = f"{currency} {value/1e6}m"

    return ret


def render_latest_float_values(field):
    ret = {}
    if "dataPoints" not in field:
        return {}
    for data_point in field["dataPoints"]:
        date = data_point["eventDate"].split("-")[0]
        value = data_point["floatValue"]

        ret[date] = str(value)
    return ret


def render_field(field):
    company_dict = {}
    if field["__typename"] == "MonetaryAmountTimeSeries":
        company_dict[field["dataItem"]["mnemonic"]] = render_latest_monetary_values(
            field
        )
        return company_dict

    if field["__typename"] == "FloatTimeSeries":
        company_dict[field["dataItem"]["mnemonic"]] = render_latest_float_values(field)
        return company_dict

    return {}


@dataclass
class RateConversionRequest:
    source_currency: str
    target_currency: str
    event_date: datetime.date


_API_GW_PROXY_HOST = "http://shared-service-proxy.default:8080"


def get_retriable_session(api_gw_proxy: bool = False):
    s = requests.Session()
    retries = Retry(
        total=10,
        backoff_factor=0.1,
        method_whitelist=("POST", "GET"),
        status_forcelist=[502, 503, 504],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))

    if api_gw_proxy:
        s.proxies = {"http": _API_GW_PROXY_HOST}

    return s


def fetch_conversion_rates_v2(
    rate_conversions: List[RateConversionRequest]
) -> List[float]:
    monetary_conversion_keys = [
        {
            "sourceCurrency": r.source_currency,
            "targetCurrency": r.target_currency,
            "spotRate": r.event_date.isoformat(),
        }
        for r in rate_conversions
    ]
    raw_resp = None
    try:
        with get_retriable_session(api_gw_proxy=True) as s:
            raw_resp = s.post(
                "http://x.data-services.onai.cloud/api/",
                json={
                    "query": GET_CONVERSION_RATES_QUERY,
                    "variables": {"keys": monetary_conversion_keys},
                },
                headers=HEADERS,
            )
            res = raw_resp.json()["data"]["currencyConversionRates"]
            return [el["rate"] for el in res]
    except Exception as e:
        logger.exception(
            "Fetch Conversion Rates: response: (%s, %s), param: (%s)",
            raw_resp,
            raw_resp.content if raw_resp else None,
            rate_conversions,
        )
        raise e


def fetch_conversion_rates(source_currencies, event_date, target_currency="EUR"):
    raise DeprecationWarning(
        "This function is no longer maintained. "
        "Use fetch_conversion_rates_v2 instead"
    )
