import datetime

from onai.ml.peers.dp import RateConversionRequest, fetch_conversion_rates_v2


def test_fetch_financial():
    resp = fetch_conversion_rates_v2(
        [RateConversionRequest("USD", "EUR", datetime.date(2017, 1, 1))]
    )
    assert resp[0] is not None

    resp = fetch_conversion_rates_v2(
        [
            RateConversionRequest("USD", "EUR", datetime.date(yr, 1, 1))
            for yr in range(2008, 2019)
        ]
    )
    assert len(resp) == 11
    assert all(r is not None for r in resp)
    resp = fetch_conversion_rates_v2([])
    assert len(resp) == 0
