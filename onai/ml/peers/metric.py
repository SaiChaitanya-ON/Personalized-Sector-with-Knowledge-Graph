import numpy as np


# reverse: True
# Ascending order
# reverse: False
# descending order
def precision_at_k(df, score_column, reverse=False):
    tst = [(el["relevance_score"], el[score_column]) for _, el in df.iterrows()]
    sorted_tst = [el for el in sorted(tst, key=lambda el: el[1] if reverse else -el[1])]

    rel = 0
    precision_at_ks = np.zeros(len(sorted_tst))
    for k, (relevance, _) in enumerate(sorted_tst):
        if relevance > 0:
            rel += 1
        precision_at_ks[k] = rel / (k + 1)
    return precision_at_ks


def average_precision(df, score_column, reverse=False):
    tst = [(el["relevance_score"], el[score_column]) for _, el in df.iterrows()]
    sorted_tst = [el for el in sorted(tst, key=lambda el: el[1] if reverse else -el[1])]

    rel = 0
    total = 0
    m = 0
    for k, (relevance, _) in enumerate(sorted_tst):
        if relevance > 0:
            rel += 1
            total += rel / (k + 1)
            m += 1

    if m == 0:
        return 0
    return total / m


def ndcg_df(df, score_column, reverse=False):
    tst = [(el["relevance_score"], el[score_column]) for _, el in df.iterrows()]
    sorted_tst = [el for el in sorted(tst, key=lambda el: el[1] if reverse else -el[1])]
    return ndcg([relevance for relevance, _ in sorted_tst])


def ndcg(labels):
    return dcg(labels) / perfect_dcg(labels)


def perfect_dcg(labels):
    return dcg(sorted(labels, reverse=True))


def dcg(labels):
    labels = np.array(labels, dtype=np.float)
    labels = 2 ** labels - 1
    denoms = np.log2(np.arange(len(labels)) + 2)
    return (labels / denoms).sum()
