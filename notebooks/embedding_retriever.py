import logging
from typing import List

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

logger = logging.getLogger(__name__)


class EmbeddingRetriever:
    BATCH_SIZE = 16

    def __init__(self, cuda=False):
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.device = "cuda:0" if cuda else "cpu:0"

        if cuda:
            self.bert_model = self.bert_model.cuda()

    def get_bert_embedding(self, text: str, topn: int = 512):
        return self.get_bert_embeddings([text], topn)

    def get_bert_embeddings(self, texts: List[str], topn: int = 512):
        with torch.no_grad():
            ret = []
            for i in range(0, len(texts), self.BATCH_SIZE):
                texts_batch = texts[i : i + self.BATCH_SIZE]
                encoded_inputs = self.bert_tokenizer.batch_encode_plus(
                    texts_batch,
                    max_length = topn,
                    truncation = True,
                    return_tensors="pt",
                    pad_to_max_length=True,
                )
                # B x T x d
                outputs = self.bert_model(
                    encoded_inputs["input_ids"], encoded_inputs["attention_mask"]
                )[0]
                ret.append(outputs.max(axis=1).values.detach().cpu().numpy())
            return np.concatenate(ret)
