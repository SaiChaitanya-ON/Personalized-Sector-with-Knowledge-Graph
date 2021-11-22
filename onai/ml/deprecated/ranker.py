class Ranker:
    @staticmethod
    def find_closest_revenue(candidate, revenue, ebitda):
        if candidate.revenue == 0 or candidate.ebitda == 0:
            return candidate.score / 100

        candidate_ebitda_marg = candidate.ebitda / candidate.revenue
        ebitda_marg = ebitda / revenue

        if candidate.revenue < revenue:
            score_revenue = 1 - candidate.revenue / revenue
        else:
            score_revenue = 1 - revenue / candidate.revenue

        if candidate_ebitda_marg < ebitda_marg:
            score_ebitda_marg = 1 - candidate_ebitda_marg / ebitda_marg
        else:
            score_ebitda_marg = 1 - ebitda_marg / candidate_ebitda_marg

        return (score_ebitda_marg + score_revenue) / 2

    def rank_by_revenue(self, query, candidates, threshold=0):
        if threshold > 0:
            candidates = [x for x in candidates if x.score > threshold]

        sort_by_revenue_result = sorted(
            candidates[0:21],
            key=lambda x: self.find_closest_revenue(
                x, query["_source"]["total_revenue"], query["_source"]["ebitda"]
            ),
        )

        return sort_by_revenue_result
