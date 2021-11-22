import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import pycountry
import torch
from omegaconf import MISSING, OmegaConf
from smart_open import open
from torch import nn
from transformers import DistilBertConfig, DistilBertTokenizer
from transformers.modeling_distilbert import DistilBertModel

from onai.ml.peers.experiment.financial_quantiser import FinancialQuantiser

logger = logging.getLogger(__name__)


@dataclass
class SurrogateReprConfig:
    quantiser_cfg_path: str = MISSING
    max_seq_length: int = 128
    text_distill_bert_tokeniser: str = "distilbert-base-uncased"
    text_encoding_layers: int = 3
    text_attention_heads: int = 8
    hidden_dim: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    financials: Tuple[str, ...] = ("TOTAL_REVENUE", "EBIT", "EBITDA")
    financial_encoding_layer: int = 1
    financial_attention_heads: int = 2
    financial_hidden_dim: int = 32
    max_financial_yrs: int = 16
    dist_type: str = "inner"
    countries: Tuple[str, ...] = tuple(c.alpha_3 for c in pycountry.countries)
    region_dim: int = 32


class FinancialEncoder(nn.Module):
    def __init__(
        self, cfg: SurrogateReprConfig, financial: str, quantiser: FinancialQuantiser
    ):
        super().__init__()
        self.financial_encoder = DistilBertModel(
            DistilBertConfig(
                quantiser.n_quantisation_points,
                quantiser.cfg.max_periods,
                n_layers=cfg.financial_encoding_layer,
                n_heads=cfg.financial_attention_heads,
                dim=cfg.financial_hidden_dim,
                hidden_dim=cfg.financial_hidden_dim * 4,
                dropout=cfg.dropout,
                attention_dropout=cfg.attention_dropout,
                pad_token_id=quantiser.pad_token_id,
            )
        )
        self.financial = financial

    def forward(self, financial_ts, mask=None):
        # B x T x dim
        last_hidden_states: torch.Tensor = self.financial_encoder(
            input_ids=financial_ts, attention_mask=mask
        )[0]
        return last_hidden_states.mean(dim=1)


class SurrogateRepr(nn.Module):
    def __init__(self, cfg: SurrogateReprConfig):
        super().__init__()
        text_tokeniser = DistilBertTokenizer.from_pretrained(
            cfg.text_distill_bert_tokeniser
        )
        self.text_encoder = DistilBertModel(
            DistilBertConfig(
                vocab_size=text_tokeniser.vocab_size,
                max_position_embeddings=cfg.max_seq_length,
                n_layers=cfg.text_encoding_layers,
                n_heads=cfg.text_attention_heads,
                dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim * 4,
                dropout=cfg.dropout,
                attention_dropout=cfg.attention_dropout,
                pad_token_id=text_tokeniser.pad_token_id,
            )
        )
        with open(cfg.quantiser_cfg_path, "rb") as fin:
            quantiser = FinancialQuantiser(OmegaConf.load(fin))

        self.financial_encoders = nn.ModuleDict(
            {k: FinancialEncoder(cfg, k, quantiser) for k in cfg.financials}
        )

        self.country_emb = nn.Embedding(len(cfg.countries), cfg.region_dim)
        self.cfg = cfg

    def forward(self, text_ids, text_mask, financial_inputs: dict, stacked_countries):
        # B x t_dim
        text_reprs = self.text_encoder(input_ids=text_ids, attention_mask=text_mask)[
            0
        ].mean(1)
        # (F) x B x f_dim
        financial_reprs = {
            k: self.financial_encoders[k](financial_inputs[k])
            for k in self.cfg.financials
        }
        # B x r_dim
        country_reprs = self.country_emb(stacked_countries)
        # B x (t_dim + f_dim * F + r_dim)
        concat_reprs = torch.cat(
            [text_reprs]
            + [financial_reprs[k] for k in self.cfg.financials]
            + [country_reprs],
            dim=1,
        )
        return {
            "concat": concat_reprs,
            "text": text_reprs,
            "financials": financial_reprs,
            "region": country_reprs,
        }
