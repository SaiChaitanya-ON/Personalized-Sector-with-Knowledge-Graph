import pickle
import re
from collections import Counter
from itertools import tee
from typing import Dict, List

import smart_open
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import tokenize
from scipy.sparse import lil_matrix


class Preprocess:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def stemming(self, data):
        ret = [
            self.stemmer.stem(word)
            for word in data
            if (word not in STOPWORDS and len(word) > 1)
        ]
        return ret

    def clean_text(self, data):
        data = data.replace(r"\n", " ").strip()
        data = re.sub(r"[^a-zA-Z]+", " ", data)
        data = re.sub(r" +", " ", data)
        return data

    def ngrams(self, tokens: List, n=2):
        iter_grams = tee(tokens, n)
        for ix, i in enumerate(iter_grams):
            for _ in range(ix):
                next(i, None)
        return [" ".join(w) for w in zip(*iter_grams)]

    def document_processor(self, text) -> List:
        """normalize the raw text."""
        tokens = tokenize(self.clean_text(text.lower()))
        tokens = self.stemming(tokens)
        tokens.extend(self.ngrams(tokens))
        return tokens


class Analyser:
    def __init__(self, analyser_path: str):
        with smart_open.open(analyser_path, "rb") as fin:
            self.idf_model = pickle.load(fin)
        assert isinstance(self.idf_model, dict)
        self.vocab_len = len(self.idf_model)
        self.token_to_idx = {i: ix for ix, i in enumerate(self.idf_model)}
        self.preprocess = Preprocess()

    def get_document_idf(self, query: str) -> Dict[str, int]:
        return {
            token: self.idf_model[token]
            for token in self.preprocess.document_processor(query)
            if token in self.token_to_idx
        }

    def transform(self, query: str):
        """ convert a document query into a vector return sparse matrix of
        size (1,vocab), crs_matrix and lil_matrix used for better performance"""
        tokens = [
            token
            for token in self.preprocess.document_processor(query)
            if token in self.token_to_idx
        ]
        vector = lil_matrix((1, self.vocab_len))
        for token, count in Counter(tokens).items():
            tf = 1
            vector[0, self.token_to_idx[token]] = tf * self.idf_model[token]
        return vector.tocsr()
