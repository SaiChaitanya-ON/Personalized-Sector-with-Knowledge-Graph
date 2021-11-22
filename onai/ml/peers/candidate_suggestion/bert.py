import os
from dataclasses import dataclass
from itertools import islice
from typing import List, Optional, Tuple

import fsspec
import pandas as pd
import torch

from onai.ml.peers.candidate_suggestion.es import Config as ESCfg
from onai.ml.peers.candidate_suggestion.es import ESCandidateSuggestion
from onai.ml.peers.experiment.modeling.bert import BertScorer, BertScorerConfig
from onai.ml.peers.feature_extractor import PeerFeatureExtractor
from onai.ml.peers.types import CompanyDetail, PeerSuggestion
from onai.ml.tools.argparse import add_bool_argument
from onai.ml.tools.file import cached_fs_open


@dataclass
class Config(ESCfg):
    @staticmethod
    def populate_argparser(parser):
        ESCfg.populate_argparser(parser)
        parser.add_argument("--initial_es_candidates", type=int)
        parser.add_argument("--scorer_config_path", type=str, required=True)
        parser.add_argument("--scorer_state_dict_path", type=str, required=True)
        add_bool_argument(parser, "cuda")

    scorer_config_path: str = None
    scorer_state_dict_path: str = None
    cuda: bool = False
    initial_es_candidates: int = 100


class BertRankNetCandidateSuggestion(ESCandidateSuggestion):
    @classmethod
    def from_cfg(cls, cfg: Config):
        return cls(**(cfg.to_dict()))

    def __init__(
        self,
        scorer_config_path: str,
        scorer_state_dict_path: Optional[str] = None,
        analyser_path: Optional[str] = None,
        initial_es_candidates=100,
        cuda: bool = False,
        **kwargs,
    ):
        super(BertRankNetCandidateSuggestion, self).__init__(**kwargs)
        cfg = BertScorerConfig.from_json_file(scorer_config_path)
        # we are loading from state_dict, so no need to load from pretrained_last_layer_path
        cfg.pretrained_last_layer_path = None
        ranker_dir = os.path.dirname(scorer_config_path)
        cfg.pretrained_bert_cfg_path = ranker_dir
        scorer = BertScorer(cfg)

        analyser_path = (
            os.path.join(ranker_dir, "idf_model.pkl")
            if analyser_path is None
            and fsspec.filesystem("s3").exists(
                os.path.join(ranker_dir, "idf_model.pkl")
            )
            else analyser_path
        )

        scorer_state_dict_path = scorer_state_dict_path or os.path.join(
            ranker_dir, "best_model.pth"
        )

        if cuda:
            device = "cuda"
        else:
            device = "cpu"
        with cached_fs_open(scorer_state_dict_path, "rb") as f:
            scorer.load_state_dict(torch.load(f, map_location=device))

        self.bert_scorer = scorer
        self.feature_extractor = PeerFeatureExtractor(
            self, analyser_path=analyser_path, bert_tokenizer=scorer.tokeniser
        )
        self.initial_es_candidates = initial_es_candidates
        if cuda:
            self.bert_scorer = self.bert_scorer.cuda()

    def _extract_peers_with_features(
        self,
        base_company: CompanyDetail,
        start_year: int,
        end_year: int,
        size: int = 100,
        must_filters: Optional[List[dict]] = None,
    ) -> Tuple[List[PeerSuggestion], pd.DataFrame]:
        peer_suggestions = super(
            BertRankNetCandidateSuggestion, self
        ).suggest_candidates(
            base_company, start_year, end_year, size=size, must_filters=must_filters
        )

        candidates_with_features = self.feature_extractor.extract_features_for_suggestions(
            base_company,
            peer_suggestions,
            pretrained_emb=False,
            bert_tokens="bert",
            infer_subsidiary=True,
            end_year=end_year,
        )
        return peer_suggestions, pd.DataFrame(candidates_with_features)

    def suggest_candidates(
        self,
        base_company: CompanyDetail,
        start_year: int = 2008,
        end_year: int = 2019,
        size=100,
        must_filters: Optional[List[dict]] = None,
        **kwargs,
    ) -> List[PeerSuggestion]:
        generated_candidates, features_for_candidates = self._extract_peers_with_features(
            base_company, start_year, end_year, self.initial_es_candidates, must_filters
        )
        scores, feats = self.bert_scorer.predict_for_basecompany(
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

    def score_candidates(
        self,
        base_company: CompanyDetail,
        peer_candidates: List[CompanyDetail],
        end_year: int,
    ) -> List[float]:
        features_for_candidates = self.feature_extractor.extract_features_for_suggestions(
            base_company,
            [PeerSuggestion(1, c, 1.0) for c in peer_candidates],
            pretrained_emb=False,
            bert_tokens="bert",
            infer_subsidiary=True,
            end_year=end_year,
        )

        return self.bert_scorer.predict_for_basecompany(
            pd.DataFrame(features_for_candidates)
        )
