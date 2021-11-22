import os

from onai.ml.peers.candidate_suggestion.ranknet import RankNetCandidateSuggestion
from onai.ml.peers.types import CompanyDetail


def test_sort_by():
    ranker_path = "s3://oaknorth-staging-non-confidential-ml-artefacts/peers/v1.0.3"
    ret = RankNetCandidateSuggestion(
        es_host="berry-es-test.ml.onai.cloud",
        es_port=80,
        es_index="company",
        dp_financials=["TOTAL_REVENUE", "EBIT", "EBITDA"],
        use_ssl=False,
        scorer_config_path=os.path.join(ranker_path, "scorer_cfg.json"),
        analyser_path=os.path.join(ranker_path, "idf_model.pkl"),
    )
    base = CompanyDetail(
        name="abc",
        description="yachts",
        country="UK",
        region="Europe",
        sector_description="Yacht Builders",
        fye={"month": 10, "day": 1},
    )
    sort_by = {
        "type": "number",
        "script": {
            "source": "Math.abs((doc['total_revenue'].size() == 0 ? Double.POSITIVE_INFINITY : doc['total_revenue'].value) - params.origin)",
            "params": {"origin": 10000},
        },
        "order": "asc",
    }

    output = ret.suggest_candidates(base, sort_by=sort_by)

    output_score = []
    for value in output:
        output_score.append(value.score)
    expected_score = output_score.copy()
    expected_score.sort()

    assert output_score == expected_score, "Sort Failed !"
