import pandas as pd
from gensim.utils import tokenize

from onai.ml.peers.document_analyzer import Preprocess
from onai.ml.peers.experiment.modeling.bert import BertScorer, BertScorerConfig


def test_empty_input():
    scorer = BertScorer(BertScorerConfig(["1", "2", "3"]))
    scores, feats = scorer.predict_for_basecompany(pd.DataFrame(), True)
    assert len(feats) == len(scores) == 0
    scores = scorer.predict_for_basecompany(pd.DataFrame(), False)
    assert len(scores) == 0


def test_analyser():
    preprocess = Preprocess()
    res = preprocess.document_processor(
        "this company sells pizzas on the 5th avenue in new york"
    )
    assert type(res) == list


def test_ngrams():
    preprocess = Preprocess()
    tokens = tokenize("this company sells pizzas in new york")
    expected_res = [
        "this company",
        "company sells",
        "sells pizzas",
        "pizzas in",
        "in new",
        "new york",
    ]

    assert preprocess.ngrams(tokens, n=2) == expected_res, "ngrams are not consistent"
