from typing import List

import torch
from omegaconf import OmegaConf
from smart_open import open
from transformers import DistilBertTokenizer

from onai.ml.peers.candidate_suggestion.es import ESCandidateSuggestion
from onai.ml.peers.experiment.financial_quantiser import FinancialQuantiser
from onai.ml.peers.experiment.modeling.surrogate import (
    SurrogateRepr,
    SurrogateReprConfig,
)
from onai.ml.peers.types import CompanyDetail


class SurrogateCandidateSuggestion(ESCandidateSuggestion):
    def __init__(
        self,
        scorer_cfg_path: str,
        scorer_state_dict_path: str,
        cuda: bool = False,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        with open(scorer_cfg_path) as fin:
            cfg: SurrogateReprConfig = OmegaConf.load(fin)
        self.cfg = cfg
        net = self.repr = SurrogateRepr(cfg)
        with open(scorer_state_dict_path, "rb") as fin:
            blob = torch.load(fin, map_location=torch.device("cpu"))
        net.load_state_dict(blob)
        net.eval()
        with open(cfg.quantiser_cfg_path, "r") as fin:
            self.quantiser = FinancialQuantiser(OmegaConf.load(fin))
        self.id_by_country = {v: idx for idx, v in enumerate(cfg.countries)}
        # TODO: this needs to be redone
        # this has an external dependency
        self.tokeniser = DistilBertTokenizer.from_pretrained(
            cfg.text_distill_bert_tokeniser
        )
        self.device = "cpu"
        if cuda:
            self.repr = self.repr.cuda()
            self.device = "cuda"

    def get_repr(self, companies: List[CompanyDetail]):
        device = torch.device(self.device)
        # TODO: tokeniser is overpadding. Fix it
        text_outputs = self.tokeniser.batch_encode_plus(
            [c.description for c in companies],
            max_length=self.cfg.max_seq_length,
            pad_to_max_length=True,
            return_tensors="pt",
            return_attention_masks=True,
            add_special_tokens=True,
        )

        financial_outputs = {
            k: v.to(device=device)
            for k, v in self.quantiser.align_pad_quantise_financials(
                [c.financials for c in companies],
                self.cfg.financials,
                self.cfg.max_financial_yrs,
            ).items()
        }

        country_outputs = torch.tensor(
            [self.id_by_country[c.country] for c in companies], device=device
        )

        assert not self.repr.training
        with torch.no_grad():
            repr = self.repr(
                text_outputs["input_ids"].to(device=device),
                text_outputs["attention_mask"].to(device=device),
                financial_outputs,
                country_outputs,
            )["concat"]
        return repr

    def score_candidates(
        self,
        base_company: CompanyDetail,
        peer_candidates: List[CompanyDetail],
        end_year: int,
    ) -> List[float]:

        # (B + 1) x h
        repr = self.get_repr([base_company] + peer_candidates)

        # assert not self.repr.training
        with torch.no_grad():
            # 1 x h
            base_repr = repr[0:1, :]
            # B x h
            peer_reprs = repr[1:, :]
            # 1 x B
            if self.repr.cfg.dist_type == "inner":
                sims = torch.matmul(base_repr, peer_reprs.transpose(0, 1))
            elif self.repr.cfg.dist_type == "l2":
                sims = -torch.norm(base_repr - peer_reprs, p=2, dim=1)
        return sims.squeeze().tolist()
