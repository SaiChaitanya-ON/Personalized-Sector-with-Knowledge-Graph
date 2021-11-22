import unittest
from collections import deque
from unittest import mock
from unittest.mock import MagicMock

from onai.ml.search.qparser.query_parser import (
    parse_free_text_query,
    preprocess_sentence,
    replace_tags,
)


class TestStringMethods(unittest.TestCase):
    def test_replace_tags(self):
        test_query = (
            "drug stores in Europe with more than 55 employees and revenue less than 5"
        )
        mock_tagged_dict = {
            "entities": [
                {"start_pos": 15, "end_pos": 21, "type": "LOC"},
                {"start_pos": 37, "end_pos": 39, "type": "CD"},
                {"start_pos": 72, "end_pos": 73, "type": "CD"},
            ]
        }

        expected_tagged_query = "drug stores in <loc> with more than <cd> employees and revenue less than <cd>"
        expected_tags = {"<loc>": deque(["Europe"]), "<cd>": deque(["55", "5"])}

        def mock_to_dict_function(tagger_type):
            if tagger_type == "pos":
                return mock_tagged_dict
            return {}

        mock_tagged_sentence = MagicMock()
        mock_tagged_sentence.to_dict.side_effect = mock_to_dict_function

        tagged_query, tags = replace_tags(test_query, mock_tagged_sentence, "pos")

        self.assertEqual(tags, expected_tags)
        self.assertEqual(tagged_query, expected_tagged_query)

    def test_preprocess_sentence(self):
        test_query = "real estate companies in Germany with at least 5 employees and between 7 and 90 revenue"

        mock_cardinal_sentence = "real estate companies in Germany with at least <cd> employees and between <cd> and <cd> revenue"
        mock_cardinal_tags = {"<cd>": deque(["5", "7", "90"])}

        mock_comparator_sentence = "real estate companies <in> Germany <in> gte <cardinal> employees <cc> <range_start> <cardinal> <range_end> <cardinal> revenue"

        mock_ner_sentence = "real estate companies  <gpe>  gte <cardinal> employees  <range_start> <cardinal> <range_end> <cardinal> revenue"
        mock_ner_tags = {"<gpe>": deque(["Germany"])}

        def mock_replace_tags(sentence, tagged_sentence, tagger_type, tags_to_replace):
            if tagger_type == "pos":
                if tags_to_replace == ["<cd>"]:
                    return mock_cardinal_sentence, mock_cardinal_tags
                if tags_to_replace == ["<in>", "<cc>", "<dt>"]:
                    return mock_comparator_sentence, {}
            if tagger_type == "ner":
                return mock_ner_sentence, mock_ner_tags

        expected_preprocessed_sentence = "real estate companies  <gpe>  gte <cardinal> employees  <range_start> <cardinal> <range_end> <cardinal> revenue"
        expected_tags = {
            "<gpe>": deque(["Germany"]),
            "<cardinal>": deque(["5", "7", "90"]),
        }

        with mock.patch("onai.ml.search.qparser.query_parser.ner_tagger"), mock.patch(
            "onai.ml.search.qparser.query_parser.pos_tagger"
        ), mock.patch(
            "onai.ml.search.qparser.query_parser.replace_tags", mock_replace_tags
        ):
            sentence, tags = preprocess_sentence(test_query)

        self.assertEqual(sentence, expected_preprocessed_sentence)
        self.assertEqual(tags, expected_tags)

    def test_query(self):
        test_query = "real estate companies in Germany with at least 5 employees and between 7 and 90 revenue"

        mock_cardinal_sentence = "real estate companies in Germany with at least <cd> employees and between <cd> and <cd> revenue"
        mock_cardinal_tags = {"<cd>": deque(["5", "7", "90"])}

        mock_comparator_sentence = "real estate companies <in> Germany <in> gte <cardinal> employees <cc> <range_start> <cardinal> <range_end> <cardinal> revenue"

        mock_ner_sentence = "real estate companies  <gpe>  gte <cardinal> employees  <range_start> <cardinal> <range_end> <cardinal> revenue"
        mock_ner_tags = {"<gpe>": deque(["Germany"])}

        def mock_replace_tags(sentence, tagged_sentence, tagger_type, tags_to_replace):
            if tagger_type == "pos":
                if tags_to_replace == ["<cd>"]:
                    return mock_cardinal_sentence, mock_cardinal_tags
                if tags_to_replace == ["<in>", "<cc>", "<dt>"]:
                    return mock_comparator_sentence, {}
            if tagger_type == "ner":
                return mock_ner_sentence, mock_ner_tags

        with mock.patch("onai.ml.search.qparser.query_parser.ner_tagger"), mock.patch(
            "onai.ml.search.qparser.query_parser.pos_tagger"
        ), mock.patch(
            "onai.ml.search.qparser.query_parser.replace_tags", mock_replace_tags
        ):
            parse_dict = parse_free_text_query(test_query)

        expected_parse_dict = {
            "parseSuccess": True,
            "numEmployees": [("gte", 5)],
            "revenue": [("gte", 7), ("lte", 90)],
            "country": "Germany",
            "freeText": "real estate companies",
        }

        self.assertEqual(parse_dict, expected_parse_dict)
