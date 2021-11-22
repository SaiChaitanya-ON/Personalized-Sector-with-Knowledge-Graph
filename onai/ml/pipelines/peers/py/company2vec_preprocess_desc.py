#!/usr/bin/env python
# coding: utf-8

import argparse
import math
import string
from collections import Counter

import pyspark.sql.functions as F
import pyspark.sql.types as T
import smart_open
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import tokenize
from langdetect import detect
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession

parser = argparse.ArgumentParser()
parser.add_argument("--word2id-path")
parser.add_argument("--data-path")

args = parser.parse_args()
word2id_path = args.word2id_path
data_path = args.data_path


spark = (
    SparkSession.builder.appName("CompanyDescriptionProcess")
    .config("spark.kryoserializer.buffer.max", "2047m")
    .config("spark.driver.maxResultSize", "10g")
    .config("spark.executor.memory", "47696M")
    .config("spark.default.parallelism", 1000)
    .config("spark.sql.shuffle.partitions", 1000)
    .config("spark.task.cpus", 1)
    .getOrCreate()
)

companies_raw = spark.read.load(
    "s3://ai-data-lake-dev-eu-west-1/business/company_data_denormalized"
)


def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False


is_english_udf = F.udf(is_english, T.BooleanType())


def tokenize_text(text, remove_sws=False):
    if not text:
        return []
    if remove_sws:
        text = remove_stopwords(text)
    text = tokenize(text, lowercase=True)

    return text


tokenize_text_udf = F.udf(tokenize_text, T.ArrayType(T.StringType()))

p = PorterStemmer()


def lemmatize_text(text):
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text]
    words = [p.stem(word) for word in words]
    return list(
        filter(
            lambda word: word not in string.punctuation
            and word.isalpha()
            and len(word) > 1,
            words,
        )
    )


lemmatize_text_udf = F.udf(lemmatize_text, T.ArrayType(T.StringType()))


def sparse_bow(*args):
    ret = []
    for el in args:
        if not el:
            continue
        ret.extend(el)
    return dict(Counter(ret))


sparse_bow_udf = F.udf(sparse_bow, T.MapType(T.StringType(), T.IntegerType()))

join_text_udf = F.udf(lambda words: " ".join(words), T.StringType())


duplicate_ids = [
    row.entity_id
    for row in companies_raw.groupBy("entity_id")
    .agg(F.count("*").alias("count"))
    .filter(F.col("count") > 1)
    .collect()
]

companies = (
    companies_raw.filter(~F.col("entity_id").isin(duplicate_ids))
    .filter(
        (
            F.col("business_description").isNotNull()
            & (F.length("business_description") > 0)
        )
    )
    .withColumn(
        "bow_description_lemmatized",
        sparse_bow_udf(
            lemmatize_text_udf(tokenize_text_udf("business_description", F.lit(True)))
        ),
    )
    .withColumn(
        "bow_description_not_lemmatized",
        sparse_bow_udf(tokenize_text_udf("business_description")),
    )
    .withColumn("ebitda_marg_calc", F.col("ebitda") / F.col("total_revenue"))
    .filter(F.size("bow_description_lemmatized") > 0)
    .repartition(1000)
).cache()

(companies.write.parquet(f"{data_path}/enhanced_companies_es", mode="overwrite"))

num_docs = companies.count()


def save_words(description_column, min_df=0.00001):
    min_df = num_docs * 0.00001

    words_df = {
        row.key: row.n_docs
        for row in companies.select(F.explode(description_column))
        .groupBy("key")
        .agg(F.count("*").alias("n_docs"))
        .filter(F.col("n_docs") > min_df)
        .collect()
    }
    num_words = len(words_df)
    words_idf = {k: math.log((num_docs + 1) / (v + 1)) for k, v in words_df.items()}
    idx = 0
    id2word = {}
    word2id = {}
    for word in sorted(words_idf):
        id2word[idx] = word
        word2id[word] = idx
        idx += 1

    print(f"Saving words to {word2id_path}/{description_column}")
    with smart_open.open(
        f"{word2id_path}/{description_column}/words_idf.csv", "w"
    ) as f:
        for word, idf in sorted(words_idf.items()):
            f.write(f"{word},{idf}\n")
    with smart_open.open(f"{word2id_path}/{description_column}/word2id.csv", "w") as f:
        for word, idd in sorted(word2id.items()):
            f.write(f"{word},{idd}\n")


save_words("bow_description_lemmatized")
save_words("bow_description_not_lemmatized")


def extract_features(feature_column):
    words_idf = {}
    with smart_open.open(f"{word2id_path}/{feature_column}/words_idf.csv", "r") as f:
        for line in f:
            word, idf = line.strip().split(",")
            words_idf[word] = float(idf)

    word2id = {}
    with smart_open.open(f"{word2id_path}/{feature_column}/word2id.csv", "r") as f:
        for line in f:
            word, idd = line.strip().split(",")
            word2id[word] = int(idd)
    num_words = len(word2id)

    def words_tfidf(bow):
        dct = {
            word2id[k]: math.log(v + 1) * words_idf[k]
            for k, v in bow.items()
            if k in word2id
        }
        return Vectors.sparse(num_words, dct)

    words_tfidf_udf = F.udf(words_tfidf, VectorUDT())

    companies_tfidf = companies.withColumn(
        "bow_tfidf", words_tfidf_udf(feature_column)
    ).cache()

    vectorizer = VectorAssembler(
        inputCols=["bow_tfidf"], outputCol="features", handleInvalid="skip"
    )

    pipeline = Pipeline(stages=[vectorizer])

    pipeline_fit = pipeline.fit(companies_tfidf)
    processed_companies = pipeline_fit.transform(companies_tfidf).repartition(1000)

    get_size = F.udf(lambda vec: vec.size, T.IntegerType())
    get_indices = F.udf(
        lambda vec: [int(el) for el in vec.indices], T.ArrayType(T.IntegerType())
    )
    get_values = F.udf(
        lambda vec: [float(el) for el in vec.values], T.ArrayType(T.DoubleType())
    )

    print(f"Saving data to {data_path}/raw_company_features_{feature_column}")

    (
        processed_companies.select(
            "entity_id",
            get_size("features").alias("size"),
            get_indices("features").alias("feature_indices"),
            get_values("features").alias("feature_values"),
            feature_column,
        )
        .repartition(32)
        .write.parquet(
            f"{data_path}/raw_company_features_{feature_column}", mode="overwrite"
        )
    )


extract_features("bow_description_lemmatized")
extract_features("bow_description_not_lemmatized")
