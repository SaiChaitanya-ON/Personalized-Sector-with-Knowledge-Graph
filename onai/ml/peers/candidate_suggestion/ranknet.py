import os
from dataclasses import dataclass
from itertools import islice
from typing import List, Optional, Tuple

import fsspec
import pandas as pd
import torch

from onai.ml.peers.candidate_suggestion.es import Config as ESCfg
from onai.ml.peers.candidate_suggestion.es import ESCandidateSuggestion
from onai.ml.peers.experiment.modeling.base import Scorer, ScorerConfig
from onai.ml.peers.feature_extractor import PeerFeatureExtractor
from onai.ml.peers.types import CompanyDetail, PeerSuggestion


@dataclass
class Config(ESCfg):
    @staticmethod
    def populate_argparser(parser):
        ESCfg.populate_argparser(parser)
        parser.add_argument("--scorer_config_path", type=str, required=True)
        parser.add_argument("--scorer_state_dict_path", type=str, required=True)
        parser.add_argument("--initial_es_candidates", type=int)

    scorer_config_path: str = None
    scorer_state_dict_path: str = None
    initial_es_candidates: int = 100


class RankNetCandidateSuggestion(ESCandidateSuggestion):
    @classmethod
    def from_cfg(cls, cfg: Config):
        return cls(**(cfg.to_dict()))

    def __init__(
        self,
        scorer_config_path: str,
        analyser_path: str,
        scorer_state_dict_path: Optional[str] = None,
        initial_es_candidates=100,
        **kwargs
    ):
        super(RankNetCandidateSuggestion, self).__init__(**kwargs)
        ranker_dir = os.path.dirname(scorer_config_path)
        scorer_state_dict_path = scorer_state_dict_path or os.path.join(
            ranker_dir, "best_model.pth"
        )

        self.feature_extractor = PeerFeatureExtractor(self, analyser_path=analyser_path)
        self.scorer = Scorer(ScorerConfig.from_json_file(scorer_config_path))
        self.scorer.eval()
        with fsspec.open(scorer_state_dict_path, "rb") as fin:
            self.scorer.load_state_dict(torch.load(fin))
        self.initial_es_candidates = initial_es_candidates

    def _extract_peers_with_features(
        self,
        base_company: CompanyDetail,
        start_year: int,
        end_year: int,
        size: int = 100,
        must_filters: Optional[List[dict]] = None,
        sort_by: Optional[dict] = None,
    ) -> Tuple[List[PeerSuggestion], pd.DataFrame]:
        peer_suggestions = super(RankNetCandidateSuggestion, self).suggest_candidates(
            base_company,
            start_year,
            end_year,
            size=size,
            must_filters=must_filters,
            sort_by=sort_by,
        )
        candidates_with_features = None
        if sort_by is None:
            candidates_with_features = self.feature_extractor.extract_features_for_suggestions(
                base_company,
                peer_suggestions,
                pretrained_emb=False,
                infer_subsidiary=True,
            )
        return peer_suggestions, pd.DataFrame(candidates_with_features)

    def suggest_candidates(
        self,
        base_company: CompanyDetail,
        start_year: int = 2008,
        end_year: int = 2019,
        size=100,
        must_filters: Optional[List[dict]] = None,
        sort_by: Optional[dict] = None,
    ) -> List[PeerSuggestion]:
        (
            generated_candidates,
            features_for_candidates,
        ) = self._extract_peers_with_features(
            base_company,
            start_year,
            end_year,
            self.initial_es_candidates,
            must_filters,
            sort_by,
        )
        scores = (
            [score[0] for score in self.scorer.predict_from_df(features_for_candidates)]
            if sort_by is None
            else [candidate.score for candidate in generated_candidates]
        )

        assert len(scores) == len(generated_candidates)

        ret = []
        for rank, (score, peer) in islice(
            enumerate(
                zip(scores, generated_candidates)
                if sort_by is not None
                else sorted(zip(scores, generated_candidates), key=lambda x: -x[0])
            ),
            size,
        ):
            cd = peer.detail
            ret.append(PeerSuggestion(rank=rank + 1, detail=cd, score=score))
        return ret
