import copy
import json
import logging
import os
import tarfile
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.core.decorators import auto_move_data
from smart_open import open
from torch import nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

from onai.ml.config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class BertScorerConfig(BaseConfig):
    cols: List[str]
    max_seq_length: int = 64
    pretrained_bert_name: Optional[str] = "bert-base-uncased"
    pretrained_last_layer_path: str = None
    pred_batch_size: int = 64
    last_k_layer_trainable: Optional[int] = None
    pretrained_bert_cfg_path: Optional[str] = None

    @staticmethod
    def populate_argparser(parser):
        parser.add_argument("--max_seq_length", type=int)
        parser.add_argument("--pretrained_bert_name")
        parser.add_argument("--pred_batch_size", type=int)
        parser.add_argument("--pretrained_last_layer_path")
        parser.add_argument("--last_k_layer_trainable", type=int)


class BertScorer(pl.LightningModule):
    def __init__(self, config: BertScorerConfig):
        super().__init__()
        self.cols = config.cols
        self.config = copy.copy(config)
        logger.info("Initialising Bert Scorer with config: \n %r", config)

        assert (
            config.pretrained_bert_name or config.pretrained_bert_cfg_path
        ), "Neither a pretrained BERT model or pretrained BERT config is specified"

        if config.pretrained_bert_name:
            logger.info("Loading from pre-trained model")
            self.bert = BertModel.from_pretrained(
                pretrained_model_name_or_path=config.pretrained_bert_name
            )
        elif config.pretrained_bert_cfg_path:
            with open(
                os.path.join(config.pretrained_bert_cfg_path, "bert_cfg.json"), "r"
            ) as fin:
                self.bert = BertModel(BertConfig.from_dict(json.load(fin)))
        else:
            assert False
        self.bert.train()

        # Number of columns plus one for embedding distance
        self.fc = nn.Linear(len(config.cols) + 1, 1, bias=False)

        if config.pretrained_bert_name:
            self.tokeniser = BertTokenizer.from_pretrained(config.pretrained_bert_name)
        elif config.pretrained_bert_cfg_path:
            with tempfile.TemporaryDirectory() as tmpdir, open(
                os.path.join(config.pretrained_bert_cfg_path, "berttokeniser.tar.gz"),
                "rb",
            ) as fin, tarfile.open(fileobj=fin, mode="r:gz") as tar_fout:
                tar_fout.extractall(tmpdir)
                self.tokeniser = BertTokenizer.from_pretrained(tmpdir)
        else:
            assert False

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
                logger.info("Initialised weight: \n %r", self.fc.weight)
        except Exception:
            logger.exception(
                "Failed to initialise weight. Using random initialisation."
            )

        # all bert layers are by default not optimisable
        # this might not be ideal as it seems there is a strong positive effect to allow fine-tuning on the embedding
        # layers.
        for parm in self.bert.parameters():
            parm.requires_grad = False

        if config.last_k_layer_trainable is not None:
            bert_layers = self.bert.encoder.layer
            assert 0 <= config.last_k_layer_trainable <= len(bert_layers), (
                f"There are only {len(bert_layers)} and you want last {config.last_k_layer_trainable} "
                f"layers to be trainable!"
            )
            for layer in self.bert.encoder.layer[
                : -(config.last_k_layer_trainable + 1) : -1
            ]:
                for parm in layer.parameters():
                    parm.requires_grad = True

    @auto_move_data
    def _get_pooled_emb(self, **kwargs):
        outputs = self.bert(**kwargs)[0]
        attention_mask = kwargs["attention_mask"].to(dtype=torch.float)
        masked_outputs = outputs * attention_mask.unsqueeze(2)
        # B x hidden_size
        # TODO: try max pooling instead of average pooling.
        return torch.sum(masked_outputs, 1) / attention_mask.sum(1, keepdim=True)

    def _df_to_input(
        self, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, List[List[int]], List[List[int]]]:
        x = torch.tensor(df[self.cols].values).float()
        base_desc = df["base_token_ids"]
        peer_desc = df["peer_token_ids"]

        return x, base_desc.tolist(), peer_desc.tolist()

    def predict_from_df(self, df):
        x_scalar, base_token_ids, peer_token_ids = self._df_to_input(df)
        return self.predict(x_scalar, base_token_ids, peer_token_ids)

    def predict_for_basecompany(self, df, return_feats=False):
        scores = []
        feats = []
        if len(df.index) > 0:
            x_scalar, base_input_ids, peer_input_ids = self._df_to_input(df)
            assert (
                len(set(tuple(x) for x in base_input_ids)) == 1
            ), "Base company IDs are not the same"

            # a dirty hack to fix companies that has empty descriptions.
            # this will ensure tokeniser won't report any errors
            masked_idxes = []
            for idx, pid in enumerate(peer_input_ids):
                if len(pid) == 0:
                    pid.append(1)
                    masked_idxes.append(idx)

            base_input_ids = base_input_ids[:1]

            original_mode = self.training
            self.eval()

            with torch.no_grad():
                assert (
                    x_scalar.shape[0] == len(peer_input_ids)
                    and len(base_input_ids) == 1
                )
                batch_sz = self.config.pred_batch_size
                for i in range(0, x_scalar.shape[0], self.config.pred_batch_size):
                    score, feat = self.forward(
                        x_scalar[i : i + batch_sz],
                        base_input_ids,
                        peer_input_ids[i : i + batch_sz],
                        return_feat=True,
                    )
                    scores.extend(score.detach().cpu().numpy().squeeze(1).tolist())
                    feats.extend(feat.detach().cpu().numpy())
            self.train(original_mode)

            for idx in masked_idxes:
                scores[idx] = float("-inf")

        if return_feats:
            return scores, feats

        return scores

    def predict(
        self, x_scalar, base_input_ids: List[List[int]], peer_input_ids: List[List[int]]
    ):
        # mini-batching
        original_mode = self.training
        self.eval()
        with torch.no_grad():
            assert x_scalar.shape[0] == len(base_input_ids) == len(peer_input_ids)
            scores = []
            batch_sz = self.config.pred_batch_size
            for i in range(0, x_scalar.shape[0], batch_sz):
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

        device = self.fc.weight.device
        if device.type == "cpu":
            # device memory is enough, we will pack everything to be executed in a single go
            descs = self._token_ids_to_bert_inputs(base_input_ids + peer_input_ids)
            embs = self._get_pooled_emb(**descs)
            base_emb = embs[: len(base_input_ids)]
            peer_emb = embs[len(base_input_ids) :]
        else:
            base_business_descs = self._token_ids_to_bert_inputs(base_input_ids)
            peer_business_descs = self._token_ids_to_bert_inputs(peer_input_ids)

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

    @auto_move_data
    def _token_ids_to_bert_inputs(self, token_ids: List[List[int]]):
        max_seq_len = min(
            max(len(i) for i in token_ids) + self.tokeniser.num_special_tokens_to_add(),
            self.config.max_seq_length,
        )

        # TODO: we can utilise batch_encode_plus instead of hand-crafting our own
        # batching script
        encoded_details = [
            self.tokeniser.encode_plus(
                token_id,
                max_length=max_seq_len,
                pad_to_max_length=True,
                truncation=True,
            )
            for token_id in token_ids
        ]

        input_ids = torch.cat(
            [
                torch.as_tensor(detail["input_ids"]).reshape([1, -1])
                for detail in encoded_details
            ],
            dim=0,
        )
        attention_mask = torch.cat(
            [
                torch.as_tensor(detail["attention_mask"]).reshape([1, -1])
                for detail in encoded_details
            ],
            dim=0,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@dataclass
class BertRankNetConfig(BaseConfig):
    l1_last_layer: float = 0.0
    l2_last_layer: float = 0.0
    repr_reg: float = 0.0

    @staticmethod
    def populate_argparser(parser):
        parser.add_argument("--l1_last_layer", type=float)
        parser.add_argument("--l2_last_layer", type=float)
        parser.add_argument("--repr_reg", type=float)


class BertRankNet(pl.LightningModule):
    def __init__(self, cfg: BertRankNetConfig, scorer: BertScorer):
        super().__init__()
        self.scorer = scorer
        self.criterion = nn.BCEWithLogitsLoss()
        self.cfg = copy.copy(cfg)
        logger.info("Initialising RankNet with config: \n %r", cfg)

    # TODO: this should output loss directly (including regularisation) as well.
    def forward(
        self,
        peer1_scalar_feats,
        peer2_scalar_feats,
        y,
        base_desc_token_ids: List[List[int]],
        peer1_desc_token_ids: List[List[int]],
        peer2_desc_token_ids: List[List[int]],
    ):
        scalar_feat_dim = len(self.scorer.config.cols)

        assert peer1_scalar_feats.shape[1] == scalar_feat_dim
        assert peer2_scalar_feats.shape[1] == scalar_feat_dim
        # B x hidden_size
        self.scorer.to(self.device)
        base_business_descs = self.scorer._token_ids_to_bert_inputs(base_desc_token_ids)
        peer1_business_descs = self.scorer._token_ids_to_bert_inputs(
            peer1_desc_token_ids
        )
        peer2_business_descs = self.scorer._token_ids_to_bert_inputs(
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

        loss = self.scorer.fc(peer1_feat) - self.scorer.fc(peer2_feat)
        loss = self.criterion(loss, y)

        reg_loss_l1 = 0
        reg_loss_l2 = 0
        for param in self.scorer.fc.parameters():
            reg_loss_l1 += param.norm(1)
            reg_loss_l2 += param.norm(2)

        repr_reg_loss = (
            base_emb.pow(2).sum(dim=1).mean()
            + peer1_emb.pow(2).sum(dim=1).mean()
            + peer2_emb.pow(2).sum(dim=1).mean()
        ) / 3

        loss += (
            self.cfg.l1_last_layer * reg_loss_l1
            + self.cfg.l2_last_layer * reg_loss_l2
            + self.cfg.repr_reg * repr_reg_loss
        )
        return {
            "loss": loss,
            "base_emb": base_emb,
            "peer1_emb": peer1_emb,
            "peer2_emb": peer2_emb,
            "peer1_feat": peer1_feat,
            "peer2_feat": peer2_feat,
        }
