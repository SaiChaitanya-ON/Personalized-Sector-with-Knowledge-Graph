from onai.ml.peers.types import CompanyDetail, Financial


def test_financial_sorted():
    f = {
        "TEST_FINANCIAL": [
            Financial(None, 2015),
            Financial(None, 2014),
            Financial(None, 2013),
        ]
    }

    cd = CompanyDetail("", "", "", financials=f)

    for v in cd.financials.values():
        assert all(v[i].year <= v[i + 1].year for i in range(len(v) - 1))


def test_country_iso3():
    cd = CompanyDetail("", "", "", financials={}, country="")
    assert cd.country is None

    cd = CompanyDetail("", "", "", financials={}, country="AAA")
    assert cd.country is None

    cd = CompanyDetail("", "", "", financials={}, country="deu")
    assert cd.country == "DEU"
