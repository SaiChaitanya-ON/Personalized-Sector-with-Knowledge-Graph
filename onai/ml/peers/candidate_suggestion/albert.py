import json
from dataclasses import dataclass
from itertools import islice
from typing import List, Optional, Tuple

import pandas as pd
import smart_open
import torch

from onai.ml.peers.candidate_suggestion.es import Config as ESCfg
from onai.ml.peers.candidate_suggestion.es import ESCandidateSuggestion
from onai.ml.peers.experiment.modeling.albert import AlbertScorer, AlbertScorerConfig
from onai.ml.peers.feature_extractor import PeerFeatureExtractor
from onai.ml.peers.types import CompanyDetail, PeerSuggestion


@dataclass
class Config(ESCfg):
    @staticmethod
    def populate_argparser(parser):
        ESCfg.populate_argparser(parser)
        parser.add_argument("--initial_es_candidates", type=int)
        parser.add_argument("--scorer_config_path", type=str, required=True)
        parser.add_argument("--scorer_state_dict_path", type=str, required=True)

    scorer_config_path: str = None
    scorer_state_dict_path: str = None
    initial_es_candidates: int = 100


class AlbertRankNetCandidateSuggestion(ESCandidateSuggestion):
    @classmethod
    def from_cfg(cls, cfg: Config):
        return cls(**(cfg.to_dict()))

    def __init__(
        self,
        scorer_config_path: str,
        scorer_state_dict_path: str,
        analyser_path: Optional[str] = None,
        initial_es_candidates=100,
        **kwargs,
    ):
        super(AlbertRankNetCandidateSuggestion, self).__init__(**kwargs)
        with smart_open.open(scorer_config_path, "r") as f:
            scorer_cfg_dict = json.load(f)
        scorer = AlbertScorer(AlbertScorerConfig.from_dict(scorer_cfg_dict))
        with smart_open.open(scorer_state_dict_path, "rb") as f:
            scorer.load_state_dict(torch.load(f))

        self.albert_scorer = scorer
        self.feature_extractor = PeerFeatureExtractor(
            self, analyser_path=analyser_path, albert_tokenizer=scorer.tokeniser
        )
        self.initial_es_candidates = initial_es_candidates

    def _extract_peers_with_features(
        self,
        base_company: CompanyDetail,
        start_year: int,
        end_year: int,
        size: int = 100,
    ) -> Tuple[List[PeerSuggestion], pd.DataFrame]:
        peer_suggestions = super(
            AlbertRankNetCandidateSuggestion, self
        ).suggest_candidates(base_company, start_year, end_year, size=size)

        candidates_with_features = self.feature_extractor.extract_features_for_suggestions(
            base_company, peer_suggestions, pretrained_emb=False, bert_tokens="albert"
        )
        return peer_suggestions, pd.DataFrame(candidates_with_features)

    def suggest_candidates(
        self,
        base_company: CompanyDetail,
        start_year: int = 2008,
        end_year: int = 2019,
        size: int = 100,
    ) -> List[PeerSuggestion]:
        generated_candidates, features_for_candidates = self._extract_peers_with_features(
            base_company, start_year, end_year, self.initial_es_candidates
        )
        scores, feats = self.albert_scorer.predict_for_basecompany(
            features_for_candidates, return_feats=True
        )

        assert len(scores) == len(generated_candidates)

        ret = []
        for rank, (score, peer) in islice(
            enumerate(sorted(zip(scores, generated_candidates), key=lambda x: -x[0])),
            size,
        ):
            cd = peer.detail
            ret.append(PeerSuggestion(rank=rank + 1, detail=cd, score=score))
        return ret
