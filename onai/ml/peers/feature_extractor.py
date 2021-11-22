import datetime
import logging
from typing import List, Optional, Set

import numpy as np
import spacy
from scipy.sparse.linalg import norm as sci_norm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AlbertTokenizer, BertTokenizer
from werkzeug.utils import cached_property

from onai.ml.peers.candidate_suggestion.es import ESCandidateSuggestion
from onai.ml.peers.document_analyzer import Analyser
from onai.ml.peers.embedding_retriever import EmbeddingRetriever
from onai.ml.peers.types import CompanyDetail, Financial, Financials, PeerSuggestion

logger = logging.getLogger(__name__)

_LAST_REVENUE_WINDOWS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


# Given financials, and a query, find out the financial in financials that are closest in time
def extract_feat_last_yr_financial(
    base_company: CompanyDetail,
    peer_company: CompanyDetail,
    financial: str,
    windows: List[float],
    feat_name: str,
    end_year: int,
):
    base_company_financial = last_reported_financial(
        base_company.financials, financial, end_year - 4
    )
    peer_financial = (
        None
        if base_company_financial is None
        else closest_financial(
            peer_company.financials, base_company_financial, financial
        )
    )

    ret = {}

    for window_size in windows:
        ret[f"{feat_name}_{window_size:.2f}"] = 0.0

    if (
        base_company_financial is not None
        and base_company_financial.val is not None
        and peer_financial is not None
        and peer_financial.val is not None
        # if the closest reported financial is within 2 years.
        and abs(base_company_financial.year - peer_financial.year) <= 2
    ):
        assert base_company_financial.currency == peer_financial.currency
        base_company_val = convert_monetary_financial_to_value(base_company_financial)
        peer_val = convert_monetary_financial_to_value(peer_financial)

        for window_size in windows:
            if ((peer_val < 0) == (base_company_val < 0)) and (
                abs(base_company_val) * (10 ** -window_size)
                < abs(peer_val)
                < abs(base_company_val) * (10 ** window_size)
            ):
                ret[f"{feat_name}_{window_size:.2f}"] = 1.0
        # ignore outliers, ie we don't have any feature activated when the difference between two financials are
        # too large
        ret.update(
            {
                f"no_{feat_name}": 0.0,
                f"last_peer_{financial}": peer_val,
                f"base_company_{financial}": base_company_val,
            }
        )
    else:
        ret.update(
            {
                f"no_{feat_name}": 1.0,
                f"last_peer_{financial}": None,
                f"base_company_{financial}": None,
            }
        )

    return ret


class PeerFeatureExtractor:
    def __init__(
        self,
        es_client: ESCandidateSuggestion,
        analyser_path: Optional[str] = None,
        embedding_retriever: Optional[EmbeddingRetriever] = None,
        albert_tokenizer: Optional[AlbertTokenizer] = None,
        bert_tokenizer: Optional[BertTokenizer] = None,
    ):
        self.albert_tokenizer = albert_tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.embedding_retriever = embedding_retriever
        self.es_client = es_client
        self.analyser = Analyser(analyser_path) if analyser_path else None

    @cached_property
    def _sentenciser(self):
        nlp = spacy.blank("en")
        nlp.add_pipe(nlp.create_pipe("sentencizer"))
        return nlp

    @cached_property
    def _full_en_nlp(self):
        return spacy.load("en_core_web_lg")

    def _subsidiary_entity(self, company_description: str) -> Set[str]:
        # get the last three sentences of the company description
        doc = self._sentenciser(company_description)
        sents = list(doc.sents)[:-4:-1]
        for sent in sents:
            sent_str = sent.string
            if "subsidiary" in sent_str.lower():
                # get out all entities in that sentence
                s_doc = self._full_en_nlp(sent_str)
                for tok in s_doc:
                    if "subsidiary" in tok.string:
                        for child in tok.children:
                            if child.string.strip().lower() == "of":
                                return {
                                    " ".join(
                                        c.string.strip()
                                        for c in child.subtree
                                        if c != child
                                    )
                                }
                # if there is nothing this might be a wrong sentence, try the next sentence!
        return set()

    def compute_item_features(
        self,
        base_company: CompanyDetail,
        result: PeerSuggestion,
        borrower_description_words: dict,
        peer_description_words: dict,
        negative_words: Set[str],
        base_embedding: np.array,
        peer_embedding: np.array,
        base_subsidiary: Set[str],
        max_score: float,
        pretrained_emb: bool = True,
        infer_subsidiary: bool = False,
        end_year: int = datetime.datetime.today().year,
    ) -> dict:
        """
        Given a base company and a peer result (from ES), retrieve its features as related to the
        base company.
        """

        peer_details = result.detail
        country_overlap = 0
        if base_company.country == peer_details.country:
            country_overlap = 1

        symmetric_diff = (
            peer_description_words.keys() ^ borrower_description_words.keys()
        )

        # Average of idfs in description symmetric diff.
        weighted_symmetric_diff = 0
        for word in symmetric_diff:
            if word in peer_description_words:
                weighted_symmetric_diff += peer_description_words[word]
            else:
                weighted_symmetric_diff += borrower_description_words[word]
        if len(symmetric_diff) > 0:
            weighted_symmetric_diff /= len(symmetric_diff)

        # Average idf of words in peer but not in base company.
        peer_diff_words = list(
            sorted(
                [
                    (idf, word)
                    for word, idf in peer_description_words.items()
                    if word not in borrower_description_words
                ],
                reverse=True,
            )
        )
        peer_diff = 0

        for idf, word in peer_diff_words[:10]:
            peer_diff += idf

        if len(peer_diff_words) > 0:
            peer_diff /= min(len(peer_diff_words), peer_diff_words[0][0])

        # IDF-Jaccard similarity between base company description and peer description.
        intersection = peer_description_words.keys() & borrower_description_words.keys()
        all_words = peer_description_words.keys() | borrower_description_words.keys()
        weighted_intersection = 0
        if all_words:
            weighted_intersection = sum(
                peer_description_words[word] for word in intersection
            )

            weighted_intersection /= sum(
                peer_description_words[word]
                if word in peer_description_words
                else borrower_description_words[word]
                for word in all_words
            )

        ret = {}

        if self.analyser:
            """then calculate vector_similarity using the analyser """
            # Get similarity  between tfidf vectors for the base borrower and peers
            vector_peer = self.analyser.transform(result.detail.description)
            vector_base = self.analyser.transform(base_company.description)
            assert vector_base.shape == vector_peer.shape

            vector_similarity = vector_base.dot(vector_peer.transpose())[0, 0] / (
                sci_norm(vector_base) * sci_norm(vector_peer)
            )
            ret.update(idf_vector_similarity=vector_similarity)

        else:
            """else calculate the negatives features"""
            # IDF-Jaccard similarity between base company description and peer description, ignoring
            # negative words sample. We compute this with both the complete set of negative words,
            # and only the negative words of lowest idf.
            all_words_negatives = {
                word for word in all_words if word not in negative_words
            }
            intersection_negatives = {
                word for word in intersection if word not in negative_words
            }
            intersection_negatives_tail_end = {
                word for word in intersection if word not in negative_words[-10:]
            }

            weighted_intersection_negatives = sum(
                peer_description_words[word] for word in intersection_negatives
            )
            weighted_intersection_negatives_tail_end = sum(
                peer_description_words[word] for word in intersection_negatives_tail_end
            )
            if len(all_words_negatives) > 0:
                norm = sum(
                    peer_description_words[word]
                    if word in peer_description_words
                    else borrower_description_words[word]
                    for word in all_words_negatives
                )
                weighted_intersection_negatives /= norm
                weighted_intersection_negatives_tail_end /= norm

            ret.update(
                dict(
                    weighted_intersection_negative_sample=weighted_intersection_negatives,
                    weighted_intersection_negative_sample_tail_end=weighted_intersection_negatives_tail_end,
                )
            )

        # Get bert embedding of peer's description, so we can compute cosine similarity with base
        # company.
        if pretrained_emb:
            ret.update(
                dict(
                    bert_embedding_sim=cosine_similarity(
                        base_embedding, peer_embedding.reshape(1, -1)
                    )[0, 0]
                )
            )

        ret.update(
            extract_feat_last_yr_financial(
                base_company,
                peer_details,
                "TOTAL_REVENUE",
                _LAST_REVENUE_WINDOWS,
                "last_revenue_diff",
                end_year,
            )
        )
        ret.update(
            extract_feat_last_yr_financial(
                base_company,
                peer_details,
                "EBITDA",
                _LAST_REVENUE_WINDOWS,
                "last_ebitda_diff",
                end_year,
            )
        )
        ret.update(
            extract_feat_last_yr_financial(
                base_company,
                peer_details,
                "EBIT",
                _LAST_REVENUE_WINDOWS,
                "last_ebit_diff",
                end_year,
            )
        )

        if infer_subsidiary:
            peer_subsidary = self._subsidiary_entity(peer_details.description)

            ret["is_subsidiary"] = float(
                base_company.name in peer_subsidary
                or peer_details.name in base_subsidiary
                or len(base_subsidiary & peer_subsidary) > 0
            )

        ret.update(
            dict(
                base_name=base_company.name,
                peer_name=peer_details.name,
                country_overlap=country_overlap,
                weighted_symmetric_diff=weighted_symmetric_diff,
                weighted_intersection=weighted_intersection,
                peer_diff=peer_diff,
                es_score=result.score,
                business_description=peer_details.description,
            )
        )
        return ret

    def extract_features_for_suggestions(
        self,
        company: CompanyDetail,
        suggested_peers: List[PeerSuggestion],
        pretrained_emb: bool = True,
        bert_tokens: Optional[str] = None,
        infer_subsidiary: bool = False,
        end_year: int = datetime.datetime.today().year,
    ) -> List[dict]:
        """
        For a given base company and a list of suggested peers, return a list of
        extracted features.
        """
        ret = []

        if self.analyser:
            base_company_terms = self.analyser.get_document_idf(company.description)
            peer_terms = [
                self.analyser.get_document_idf(peer.detail.description)
                for peer in suggested_peers
            ]
            negative_words = None
        else:
            base_company_terms = self.es_client.get_document_idf(company.description)
            peer_terms = self.es_client.get_documents_idf(
                [peer.detail.description for peer in suggested_peers]
            )
            negative_words = self.es_client.extract_negative_words(company)

        if pretrained_emb:
            embs = self.embedding_retriever.get_bert_embeddings(
                [company.description]
                + [peer.detail.description for peer in suggested_peers]
            )
            base_embedding = embs[0:1, :]
            peer_embeddings = embs[1:, :]

            assert peer_embeddings.shape[0] == len(suggested_peers)
        else:
            base_embedding = None
            peer_embeddings = [None] * len(suggested_peers)
        if infer_subsidiary:
            base_subdiary = self._subsidiary_entity(company.description)
        else:
            base_subdiary = set()

        def tokenise(s):
            if bert_tokens == "albert":
                return self.albert_tokenizer.encode(s, add_special_tokens=False)
            elif bert_tokens == "bert":
                return self.bert_tokenizer.encode(
                    s,
                    add_special_tokens=False,
                    max_length=self.bert_tokenizer.max_len,
                    truncation=True,
                )
            elif bert_tokens is None:
                return None
            assert False, "Unsupported type of bert tokeniser %s" % bert_tokens

        base_token_ids = tokenise(company.description)

        max_score = None

        for peer_term, peer_embedding, result in zip(
            peer_terms, peer_embeddings, suggested_peers
        ):
            if max_score is None:
                max_score = result.score
            peer_features = self.compute_item_features(
                company,
                result,
                base_company_terms,
                peer_term,
                negative_words,
                base_embedding,
                peer_embedding,
                base_subdiary,
                max_score,
                pretrained_emb,
                infer_subsidiary,
                end_year,
            )
            if base_token_ids is not None:
                peer_features.update(
                    {
                        "base_token_ids": base_token_ids,
                        "peer_token_ids": tokenise(result.detail.description),
                    }
                )

            ret.append(peer_features)

        return ret


def convert_monetary_financial_to_value(f: Financial):
    if f.magnitude is None:
        multiplier = 1
    elif f.magnitude == "b":
        multiplier = 1e9
    elif f.magnitude == "m" or f.magnitude == "mn":
        multiplier = 1e6
    elif f.magnitude == "k":
        multiplier = 1e3
    elif f is None:
        multiplier = 1
    else:
        assert False, f"Unknown magnitude {f.magnitude}"
    return f.val * multiplier


def closest_financial(
    financials: Financials,
    query: Financial,
    mnenomic: str,
    # we accept closest financial that is at most 4 years old
    start_year_threshold: int = datetime.datetime.today().year - 4,
) -> Optional[Financial]:
    if mnenomic not in financials:
        return None
    financials = financials[mnenomic]
    target_year = query.year
    ret = financials[0]
    for f in financials:
        if (
            f.val
            and abs(f.year - target_year) < abs(ret.year - target_year)
            and f.year >= start_year_threshold
        ):
            ret = f
    return ret


def last_reported_financial(
    financials: Financials,
    mnenomic: str,
    # by default, we will accept anything that is 4 years old.
    start_year_threshold: int = datetime.datetime.today().year - 4,
) -> Optional[Financial]:
    if mnenomic not in financials:
        return None
    financials = financials[mnenomic]
    # Assuming revenues are sorted
    for f in financials[::-1]:
        if f.val and f.year >= start_year_threshold:
            return f
