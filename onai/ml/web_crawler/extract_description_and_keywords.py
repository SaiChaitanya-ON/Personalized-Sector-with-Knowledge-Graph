import argparse
import csv
import string
from time import time

import gensim
import pandas as pd
from gensim.utils import tokenize
from langdetect import detect
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

weird_char = "\xfe"


def process_text(text):
    words = tokenize(text, lower=True)
    return " ".join(
        list(
            filter(
                lambda word: word not in string.punctuation and word.isalpha(), words
            )
        )
    )


def check_language(text):
    try:
        return detect(text)
    except:
        return ""


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def process_too_long_text(text):
    sent = list(gensim.summarization.textcleaner.get_sentences(text))
    idx = min(len(sent), 3)
    return " ".join(sent[0:idx])


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def extract_description_pattern_is(text, company_name):
    # Extract description following pattern: <Company name> is ...
    description = ""
    if company_name in text:
        idx = text.find(company_name)
        potential_text = text[idx : idx + len(company_name) + 30]
        if " is " in potential_text:
            end_idx = len(text)
            for sym in ["*", "#", "|", ">"]:
                sym_idx = text.find(sym, idx)
                if sym_idx != -1:
                    end_idx = min(end_idx, sym_idx)

            extracted = text[idx:end_idx]
            sent = list(gensim.summarization.textcleaner.get_sentences(extracted))
            extracted_desc = sent[0]
            if extracted_desc.split(" ")[-1] in {"Inc.", "U.S."} and len(sent) > 1:
                extracted_desc += " " + sent[1]
            if len(extracted_desc.split()) > 5:
                description = extracted_desc.replace("\n", " ")
    return description


def extract_description_pattern_who_we_are(text):
    # Extract description following pattern: Who we are, ...
    description = ""
    text = text.split("\n")
    for idx, line in enumerate(text):
        for marker in [
            "# who we are",
            "# overview",
            "# company overview",
            "# about",
            "# mission",
            "# our mission",
        ]:
            if marker in line.lower():
                start_idx = idx + 1
                while start_idx < len(text):
                    if text[start_idx] == "":
                        start_idx += 1
                    else:
                        break

                end_idx = start_idx + 1
                while end_idx < len(text):
                    if text[end_idx] != "":
                        end_idx += 1
                    else:
                        break

                if start_idx < len(text):
                    extracted = " ".join(text[start_idx:end_idx])
                    extracted_desc = process_too_long_text(extracted)

                    extracted_desc_splits = extracted_desc.split()
                    if (
                        len(extracted_desc_splits) > 5
                        and extracted_desc_splits[0][0] != "*"
                    ):
                        if extracted_desc_splits[0][0] == "#":
                            extracted_desc = " ".join(extracted_desc_splits[1:])
                        description = extracted_desc
                        break
    return description


def preprocess_df(df):
    df["text"] = df["text"].str.replace(weird_char, "\n")
    df["about_us_text"] = df["about_us_text"].str.replace(weird_char, "\n")
    df["description"] = df["description"].str.replace(weird_char, "\n")

    df["text_lang"] = df["text"].apply(lambda x: check_language(x))
    df["about_us_text_lang"] = df["text"].apply(lambda x: check_language(x))

    return df


def get_tfidf_transform(df):
    docs = []
    for index, row in df.iterrows():
        for field, field_lang in {
            "text": "text_lang",
            "about_us_text": "about_us_text_lang",
        }.items():
            if row[field] and row[field_lang] == "en":
                docs.append(process_text(row[field]))

    # remove words appearing in more than 95% / less than 1% of all documents
    count_vectorizer = CountVectorizer(
        max_df=0.95, min_df=0.01, stop_words=gensim.parsing.preprocessing.STOPWORDS
    )
    word_count_vector = count_vectorizer.fit_transform(docs)

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    feature_names = count_vectorizer.get_feature_names()

    return tfidf_transformer, feature_names, count_vectorizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from URLs")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    start_time = time()

    df = pd.read_csv(args.input_file, sep="\t", quoting=csv.QUOTE_NONE).fillna("")
    df = preprocess_df(df)

    tfidf_transformer, feature_names, count_vectorizer = get_tfidf_transform(df)

    count_valid_description = 0
    for index, row in df.iterrows():

        # check if the description from meta field is in English and has more than 3 tokens
        if (
            check_language(row["description"]) == "en"
            and len(row["description"].split()) > 3
        ):
            description = row["description"]
            description = process_too_long_text(description)
            count_valid_description += 1
        else:
            description = ""

        keywords_text = []
        for field, field_lang in {
            "text": "text_lang",
            "about_us_text": "about_us_text_lang",
        }.items():
            if row[field]:
                if row[field_lang] != "en":
                    continue

                keywords_text.append(row[field])

                if not description:
                    description = extract_description_pattern_is(
                        row[field], row["name"]
                    )
                    if not description:
                        description = extract_description_pattern_who_we_are(row[field])

        df.loc[index, "description"] = description

        keywords_text = " ".join(keywords_text)

        # Check if the website text + about us page text has more than 50 tokens
        if len(keywords_text.split()) > 50:
            keywords_text += " " + description
            tf_idf_vector = tfidf_transformer.transform(
                count_vectorizer.transform([keywords_text])
            )
            sorted_items = sort_coo(tf_idf_vector.tocoo())
            keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
        else:
            keywords = {}

        df.loc[index, "keywords"] = ", ".join(keywords)

    df[["name", "url", "description", "keywords"]].to_csv(args.output_file, sep="\t")

    print("Execution time: %fs" % (time() - start_time))
