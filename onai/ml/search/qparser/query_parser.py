import os
from collections import deque

import smart_open
from semantic.numbers import NumberService

from flair.data import Sentence
from flair.models import SequenceTagger
from nltk import ChartParser
from nltk.grammar import CFG
from nltk.tree import Tree

number_parser = NumberService()


ner_tagger = SequenceTagger.load("ner-ontonotes-fast")
pos_tagger = SequenceTagger.load("pos-fast")

comparator_expressions = {
    "gte": [
        "greater than or equal to",
        "minimum",
        "min",
        "at least",
        "not less than",
        "not fewer than",
        ">=",
        "no less than",
        "no fewer than",
    ],
    "gt": ["greater than", "more than", ">", "above", "over"],
    "lte": [
        "less than or equal to",
        "maximum",
        "max",
        "at most",
        "not greater than",
        "not more than",
        "<=",
        "no more than",
    ],
    "lt": ["less than", "fewer than", "<", "below", "under"],
}

reverse_comparator_expressions = {
    v: k for k, vs in comparator_expressions.items() for v in vs
}

DEFAULT_KW_FILE = (
    "s3://oaknorth-staging-non-confidential-ml-artefacts/query_parsing/industry_kw.csv"
)
keywords_file = os.environ.get("GRAMMAR_KEYWORDS_FILE", DEFAULT_KW_FILE)
industry_keywords = {}
with smart_open.open(keywords_file) as f:
    for line in f:
        word, freq = line.strip().split(",")
        industry_keywords[word] = int(freq)

company_grammar = CFG.fromstring(
    f"""
    CP -> CT FP | FP CT | CT | FP | FP CT FP
    FP -> FPC FP | FPC
    FPC -> REV | EMP | LOC | Link REV | Link EMP | Link LOC
    REV -> CO 'revenue' | 'revenue' CO | 'revenue' Range | Range 'revenue'
    EMP -> CO 'employees' | 'employees' CO | 'employees' Range | Range 'employees'
    LOC -> L COUNTRY | L REGION | COUNTRY | REGION
    COUNTRY -> '<gpe>'
    REGION -> '<loc>'
    L -> 'in'
    CO -> Ineq '<cardinal>'
    Range -> '<range_start>' '<cardinal>' '<range_end>' '<cardinal>'
    Ineq -> 'gt' | 'gte' | 'lt' | 'lte'
    Link -> 'having' | 'with' | 'and'
    CT -> {"|".join(["{kw} CT | {kw}".format(kw=repr(kw)) for kw in industry_keywords])}"""
)


def replace_tags(original_sentence, tagged_sentence, tagger_type, tags_to_replace=None):
    """ Replace natural language sequence with its found tag
    :param original_sentence: str
    :param tagged_sentence: flair Sentence object that as been tagged by a Tagger
    :param tagger_type: str which tagger has been used 'ner' or 'pos'
    :param tags_to_replace: List[str] which tags found by Tagger to replace. If None, replace all tags

    :return : Tuple[str, Dict[str, deque[str]]] returns the sentence with tags replaced and the original values that
              the tags had in the text. This latter return is used when parsing the syntax tree to re-populate the
              initial textual values. Processing them in a FIFO manner ensures consistency when doing this (so initial
              values get mapped to the correct syntax tree leaves)
    """

    tags = {}

    tagged_string = ""
    prev_end = 0

    for entity in tagged_sentence.to_dict(tagger_type)["entities"]:
        start_pos = entity["start_pos"]
        end_pos = entity["end_pos"]
        tag_type = f"<{entity['type'].lower()}>"
        if tags_to_replace is not None and tag_type not in tags_to_replace:
            continue
        if tag_type not in tags:
            tags[tag_type] = deque()
        tags[tag_type].append(original_sentence[start_pos:end_pos])

        tagged_string += original_sentence[prev_end:start_pos]
        tagged_string += tag_type
        prev_end = end_pos
    tagged_string += original_sentence[prev_end:]

    return tagged_string, tags


def preprocess_sentence(original_sentence):
    """ Preprocess a free text query to something our grammar can handle. This works as follows:
        * do POS tagging on the initial sequence, replacing all found cardinals (this is because NER seems to have
          some strange assumptions around what a cardinal is)
        * from the above result, replace comparators with a canonical representation (e.g. 'fewer than' -> '<')
        * remove uninformative words such as interjections and conjunctions
        * do NER tagging to identify countries and regions

    :param : str the original sentence

    :return : Tuple[str, Dict[str, deque[str]]], return the tag-replaced sentence together with the actual initial tag values
    """

    sentence = Sentence(original_sentence)
    pos_tagger.predict(sentence)
    pos_sentence, ca_tags = replace_tags(
        original_sentence, sentence, tagger_type="pos", tags_to_replace=["<cd>"]
    )

    if "<cd>" in ca_tags:
        ca_tags["<cardinal>"] = ca_tags["<cd>"]
        del ca_tags["<cd>"]

    pos_sentence = pos_sentence.replace("<cd>", "<cardinal>")

    for matcher, comparator in sorted(
        reverse_comparator_expressions.items(), key=lambda a: -len(a[0])
    ):
        pos_sentence = pos_sentence.replace(
            matcher + " <cardinal>", comparator + " <cardinal>"
        )

    pos_sentence = pos_sentence.replace(
        "between <cardinal> and <cardinal>",
        "<range_start> <cardinal> <range_end> <cardinal>",
    )

    sentence = Sentence(pos_sentence)
    pos_tagger.predict(sentence)
    pos_sentence, _ = replace_tags(
        pos_sentence,
        sentence,
        tagger_type="pos",
        tags_to_replace=["<in>", "<cc>", "<dt>"],
    )

    pos_sentence = (
        pos_sentence.replace("<in>", "").replace("<cc>", "").replace("<dt>", "")
    )

    sentence = Sentence(pos_sentence)
    ner_tagger.predict(sentence)

    ner_sentence, ner_tags = replace_tags(
        pos_sentence, sentence, tagger_type="ner", tags_to_replace=["<gpe>", "<loc>"]
    )

    ner_tags.update(ca_tags)

    return ner_sentence, ner_tags


def _process_parse_tree(tree, tags):
    """ Given a syntax tree and a mapping from tags to initial value, process the syntax tree in a DFS fashion to
        populate the with the original comparison values.

        Because all syntax trees have the same leaf order regardless of their actual structure, and because DFS
        traversal retrieves this order, we can safely populate the initial values in the order in which they are
        encountered in this traversal by popping from the appropriate tags queue. E.g. when we encounter a 'cardinal'
        leaf it is always safe to pop the first value in the tags['cardinal'] queue and label the respective
        leaf with that value.
    """

    tag_terminals = {"<cardinal>", "<gpe>", "<loc>", "<money>"}
    redundant_terminals = {"employees", "revenue"}

    def dfs_label(root=tree):
        if isinstance(root, str):
            if root in tag_terminals:
                return tags[root].popleft()
            if root not in redundant_terminals:
                return root
            else:
                return None

        new_children = []
        for el in root:
            labelled_child = dfs_label(el)
            if labelled_child is not None:
                new_children.append(labelled_child)
        return Tree(root.label(), new_children)

    tree = dfs_label()
    return tree


def _populate_query(tree):
    """ Given a labelled parse tree (i.e. with tags replaces with their original values), attempt to
        generate a mapping of the form field -> condition, which can be used to create an ES querystring
    """
    ret = {}

    def process_range_subtree(root):
        return [
            ("gte", number_parser.parse(root[1])),
            ("lte", number_parser.parse(root[3])),
        ]

    def process_comparator_subtree(root):
        co, ca = None, None
        for child in root:
            if isinstance(child, Tree) and child.label() == "Ineq":
                co = child[0]
            else:
                ca = child
        return [(co, number_parser.parse(ca))]

    def process_numerical_field_subtree(root):
        for child in root:
            if isinstance(child, Tree) and child.label() == "CO":
                return process_comparator_subtree(child)
            if isinstance(child, Tree) and child.label() == "Range":
                return process_range_subtree(child)

    def process_companytype_subtree(root):
        if isinstance(root, Tree):
            for child in root:
                for kw in process_companytype_subtree(child):
                    yield kw
        else:
            yield root

    def dfs_query_parser(root=tree):
        for child in root:
            if child.label() == "Link":
                continue
            if child.label() == "L":
                # TODO: parse L so we distinguish between various location-type queries
                continue
            if child.label() == "CT":
                company_freetext = " ".join(
                    [el for el in process_companytype_subtree(child)]
                )
                ret["freeText"] = company_freetext
            elif child.label() == "REGION":
                if "region" in ret:
                    raise ValueError("Can only have one region identifier")
                ret["region"] = list(child)[0]
            elif child.label() == "COUNTRY":
                if "country" in ret:
                    raise ValueError("Can only have one country identifier")
                ret["country"] = list(child)[0]
            elif child.label() == "EMP":
                if "numEmployees" not in ret:
                    ret["numEmployees"] = []
                ret["numEmployees"].extend(process_numerical_field_subtree(child))
            elif child.label() == "REV":
                if "revenue" not in ret:
                    ret["revenue"] = []
                ret["revenue"].extend(process_numerical_field_subtree(child))
            else:
                dfs_query_parser(child)

    dfs_query_parser()
    return ret


def parse_free_text_query(input_query):
    # N.B flair taggers are the execution bottleneck at the moment
    tagged_sentence, tags = preprocess_sentence(input_query)
    tagged_sentence = tagged_sentence.lower()

    parser = ChartParser(company_grammar)
    grammar_tree = None
    try:
        for tree in parser.parse(tagged_sentence.split()):
            grammar_tree = tree
    except Exception as e:
        return {
            "parseSuccess": False,
            "parseErrorMessage": f"Failed to parse grammar {e}",
        }

    if grammar_tree is None:
        return {
            "parseSuccess": False,
            "parseErrorMessage": "Failed to construct syntax tree.",
        }

    processed_parse_tree = _process_parse_tree(grammar_tree, tags)
    parse_dict = _populate_query(processed_parse_tree)
    parse_dict["parseSuccess"] = True

    return parse_dict
