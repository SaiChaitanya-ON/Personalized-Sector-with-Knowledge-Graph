import argparse
import copy
import json
import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from smart_open import open
from torch import nn as nn
from transformers import AlbertModel, AlbertTokenizer

from onai.ml.config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class AlbertScorerConfig(BaseConfig):
    cols: List[str]
    max_seq_length: int = 64
    pretrained_albert_name: str = "albert-base-v2"
    pretrained_last_layer_path: str = None
    pred_batch_size: int = 64
    hidden_dropout_prob: Optional[float] = None
    attention_probs_dropout_prob: Optional[float] = None

    @staticmethod
    def populate_argparser(parser):
        parser.add_argument("--max_seq_length", type=int)
        parser.add_argument("--pretrained_albert_name")
        parser.add_argument("--pred_batch_size", type=int)
        parser.add_argument("--hidden_dropout_prob", type=float)
        parser.add_argument("--attention_probs_dropout_prob", type=float)
        parser.add_argument("--pretrained_last_layer_path")


class AlbertScorer(nn.Module):
    def __init__(self, config: AlbertScorerConfig):
        super().__init__()
        self.cols = config.cols
        self.config = copy.copy(config)
        logger.info(f"Initialising Albert Scorer with config: \n{config}")
        override_albert_cfg = {}
        if config.hidden_dropout_prob is not None:
            override_albert_cfg["hidden_dropout_prob"] = config.hidden_dropout_prob
        if config.attention_probs_dropout_prob is not None:
            override_albert_cfg[
                "attention_probs_dropout_prob"
            ] = config.attention_probs_dropout_prob

        self.albert = AlbertModel.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_albert_name,
            **override_albert_cfg,
        )

        # Number of columns plus one for embedding distance
        self.fc = nn.Linear(len(config.cols) + 1, 1, bias=False)
        self.tokeniser = AlbertTokenizer.from_pretrained(config.pretrained_albert_name)
        logger.info(config.pretrained_last_layer_path)
        try:
            if config.pretrained_last_layer_path:
                with open(config.pretrained_last_layer_path, "r") as fin:
                    init_weights_by_name = json.load(fin)
                # the final 0.0 is initial weight of albert
                init_weights = [init_weights_by_name.get(k, 0.0) for k in self.cols] + [
                    0.0
                ]
                with torch.no_grad():
                    self.fc.weight.copy_(torch.tensor(init_weights))
                logger.info(f"Initialised weight: {self.fc.weight}")
        except:
            logger.exception(
                "Failed to initialise weight. Using random initialisation."
            )

    def _get_pooled_emb(self, **kwargs):
        outputs = self.albert(**kwargs)[0]
        attention_mask = kwargs["attention_mask"].to(dtype=torch.float)
        masked_outputs = outputs * attention_mask.unsqueeze(2)
        # B x hidden_size
        return torch.sum(masked_outputs, 1) / attention_mask.sum(1, keepdim=True)

    def _df_to_input(self, df):
        x = torch.tensor(df[self.cols].values).float()
        base_desc = df["base_token_ids"]
        peer_desc = df["peer_token_ids"]
        return x, base_desc, peer_desc

    def predict_from_df(self, df):
        x_scalar, base_token_ids, peer_token_ids = self._df_to_input(df)
        return self.predict(x_scalar, base_token_ids, peer_token_ids)

    def predict_for_basecompany(self, df, return_feats=False):
        x_scalar, base_input_ids, peer_input_ids = self._df_to_input(df)

        assert (
            base_input_ids.astype(str).nunique() == 1
        ), "Base company IDs are not the same"

        base_input_ids = base_input_ids[:1]

        original_mode = self.training
        self.eval()

        with torch.no_grad():
            assert x_scalar.shape[0] == len(peer_input_ids) and len(base_input_ids) == 1
            scores = []
            feats = []
            batch_sz = self.config.pred_batch_size
            for i in range(0, x_scalar.shape[0], self.config.pred_batch_size):
                score, feat = self.forward(
                    x_scalar[i : i + batch_sz],
                    base_input_ids,
                    peer_input_ids[i : i + batch_sz],
                    return_feat=True,
                )
                scores.extend(score.detach().cpu().numpy())
                feats.extend(feat.detach().cpu().numpy())
        self.train(original_mode)
        if return_feats:
            return scores, feats

        return scores

    def predict(self, x_scalar, base_input_ids, peer_input_ids):
        # mini-batching
        original_mode = self.training
        self.eval()
        with torch.no_grad():
            assert x_scalar.shape[0] == len(base_input_ids) == len(peer_input_ids)
            scores = []
            batch_sz = self.config.pred_batch_size
            for i in range(0, x_scalar.shape[0], self.config.pred_batch_size):
                scores.extend(
                    self.forward(
                        x_scalar[i : i + batch_sz],
                        base_input_ids[i : i + batch_sz],
                        peer_input_ids[i : i + batch_sz],
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
        self.train(original_mode)
        return scores

    def forward(
        self,
        x_scalar,
        base_input_ids: List[List[int]],
        peer_input_ids: List[List[int]],
        return_feat=False,
    ):
        # B x hidden_size
        x_scalar = x_scalar.to(device=self.fc.weight.device)
        base_business_descs = self._token_ids_to_albert_inputs(base_input_ids)
        peer_business_descs = self._token_ids_to_albert_inputs(peer_input_ids)
        base_emb = self._get_pooled_emb(**base_business_descs)
        # B x hidden_size
        peer_emb = self._get_pooled_emb(**peer_business_descs)

        # B x 1
        dot_prod_desc = torch.sum(base_emb * peer_emb, dim=1, keepdim=True)

        # feat: B x (scalar_feat_dim + 1)
        feat = torch.cat([x_scalar, dot_prod_desc], dim=1)
        if return_feat:
            return self.fc(feat), feat
        else:
            return self.fc(feat)

    def _token_ids_to_albert_inputs(self, token_ids: List[List[int]]):

        max_seq_len = min(max(len(i) for i in token_ids), self.config.max_seq_length)

        encoded_details = [
            self.tokeniser.encode_plus(
                token_id, max_length=max_seq_len, pad_to_max_length=True
            )
            for token_id in token_ids
        ]

        device = self.fc.weight.device

        input_ids = torch.cat(
            [
                torch.as_tensor(detail["input_ids"]).reshape([1, -1])
                for detail in encoded_details
            ],
            dim=0,
        ).to(device=device)
        attention_mask = torch.cat(
            [
                torch.as_tensor(detail["attention_mask"]).reshape([1, -1])
                for detail in encoded_details
            ],
            dim=0,
        ).to(device=device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@dataclass
class AlbertRankNetConfig(BaseConfig):
    l1_last_layer: float = 0.0
    l2_last_layer: float = 0.0
    repr_reg: float = 0.0

    @staticmethod
    def populate_argparser(parser):
        parser.add_argument("--l1_last_layer", type=float)
        parser.add_argument("--l2_last_layer", type=float)
        parser.add_argument("--repr_reg", type=float)


class AlbertRankNet(nn.Module):
    def __init__(self, cfg: AlbertRankNetConfig, scorer: AlbertScorer):
        super().__init__()
        self.scorer = scorer
        self.cfg = copy.copy(cfg)
        logger.info(f"Initialising RankNet with config: \n {cfg}")

    # TODO: this should output loss (including regularisation, etc.) directly as well.
    def forward(
        self,
        peer1_scalar_feats,
        peer2_scalar_feats,
        base_desc_token_ids: List[List[int]],
        peer1_desc_token_ids: List[List[int]],
        peer2_desc_token_ids: List[List[int]],
    ):
        # TODO: this madness of device moving should not exist
        # Use Pytorch lightning to remove all these nonsense
        device = self.scorer.fc.weight.device
        scalar_feat_dim = len(self.scorer.config.cols)

        assert peer1_scalar_feats.shape[1] == scalar_feat_dim
        assert peer2_scalar_feats.shape[1] == scalar_feat_dim
        # B x hidden_size
        base_business_descs = self.scorer._token_ids_to_albert_inputs(
            base_desc_token_ids
        )
        peer1_business_descs = self.scorer._token_ids_to_albert_inputs(
            peer1_desc_token_ids
        )
        peer2_business_descs = self.scorer._token_ids_to_albert_inputs(
            peer2_desc_token_ids
        )

        base_emb = self.scorer._get_pooled_emb(**base_business_descs)

        # B x hidden_size
        peer1_emb = self.scorer._get_pooled_emb(**peer1_business_descs)
        peer2_emb = self.scorer._get_pooled_emb(**peer2_business_descs)

        peer1_feat = torch.cat(
            [peer1_scalar_feats, torch.sum(base_emb * peer1_emb, dim=1, keepdim=True)],
            dim=1,
        )
        peer2_feat = torch.cat(
            [peer2_scalar_feats, torch.sum(base_emb * peer2_emb, dim=1, keepdim=True)],
            dim=1,
        )

        return {
            "loss": self.scorer.fc(peer1_feat) - self.scorer.fc(peer2_feat),
            "base_emb": base_emb,
            "peer1_emb": peer1_emb,
            "peer2_emb": peer2_emb,
            "peer1_feat": peer1_feat,
            "peer2_feat": peer2_feat,
        }
