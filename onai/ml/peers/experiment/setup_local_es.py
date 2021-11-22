import argparse
import logging
import os

import boto3
import requests
from onai_datalake.utils.es_helper import ElasticSearchHelper
from pyspark.sql import SparkSession
from requests import HTTPError

from onai.ml.tools.logging import setup_logger

payload_settings = {
    "settings": {
        "analysis": {
            "filter": {
                "filter_stemmer": {"type": "stemmer", "language": "english"},
                "filter_shingle": {
                    "type": "shingle",
                    "max_shingle_size": 4,
                    "min_shingle_size": 2,
                    "output_unigrams": "true",
                },
            },
            "analyzer": {
                "tags_analyzer": {
                    "type": "custom",
                    "filter": ["lowercase", "filter_stemmer"],
                    "tokenizer": "standard",
                },
                "ngram_analyzer": {
                    "type": "custom",
                    "filter": ["lowercase", "filter_stemmer", "filter_shingle"],
                    "tokenizer": "standard",
                },
            },
        }
    }
}

payload_mappings = {
    "properties": {
        "business_description": {"analyzer": "tags_analyzer", "type": "text"},
        "capex": {"type": "float"},
        "cash_and_equiv": {"type": "float"},
        "cff": {"type": "float"},
        "cfi": {"type": "float"},
        "cfo": {"type": "float"},
        "company_type_name": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "country_of_incorporation": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "country_of_incorporation_iso": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "earn_cont_ops": {"type": "float"},
        "ebit": {"type": "float"},
        "ebitda": {"type": "float"},
        "ebitda_marg": {"type": "float"},
        "ebt_incl_xo": {"type": "float"},
        "entity_id": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "fcf": {"type": "float"},
        "gross_profit": {"type": "float"},
        "icr_ebit": {"type": "float"},
        "id_bvd": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "id_capiq": {"type": "long"},
        "long_term_debt_issued": {"type": "float"},
        "long_term_debt_repaid": {"type": "float"},
        "name": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "net_inc": {"type": "float"},
        "net_inc_incl_xo": {"type": "float"},
        "number_employees": {"type": "long"},
        "oper_inc": {"type": "float"},
        "primary_naics_node": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "primary_naics_node_desc": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "primary_sic_node": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "primary_sic_node_desc": {"analyzer": "tags_analyzer", "type": "text"},
        "region": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "revenue": {"type": "float"},
        "short_description": {"analyzer": "tags_analyzer", "type": "text"},
        "clean_description": {"analyzer": "tags_analyzer", "type": "text"},
        "enhanced_sic_desc": {"analyzer": "tags_analyzer", "type": "text"},
        "st_debt_issued": {"type": "float"},
        "st_debt_repaid": {"type": "float"},
        "status": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "total_assets": {"type": "float"},
        "total_debt_issued": {"type": "float"},
        "total_debt_repaid": {"type": "float"},
        "total_equity": {"type": "float"},
        "total_liab": {"type": "float"},
        "total_revenue": {"type": "float"},
        "vendor": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
    }
}

logger = logging.getLogger(__name__)


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--es_host", default="127.0.0.1")
    parser.add_argument("--es_port", default="9200")
    parser.add_argument("--index", default="company")
    parser.add_argument(
        "--input",
        default="s3://oaknorth-ml-dev-eu-west-1/company2vec/l2r/company_data_denormalized_ind_pred",
    )
    args = parser.parse_args()
    es = ElasticSearchHelper(hosts=args.es_host, port=args.es_port)
    try:
        es.delete_index(args.index)
        logger.info(f"Deleted existed index {args.index}")
    except HTTPError:
        logger.info(f"Index {args.index} does not exist. Won't delete it.")
    host_url = f"http://{args.es_host}:{args.es_port}/{args.index}/"
    r = requests.put(host_url, json=payload_settings)
    logger.info(r.status_code)
    logger.info(r.content)
    r = requests.put(
        host_url + "_mapping/_doc?include_type_name=true", json=payload_mappings
    )
    logger.info(r.status_code)
    logger.info(r.content)

    sts_client = boto3.client("sts")

    # Call the assume_role method of the STSConnection object and pass the role
    # ARN and a role session name.
    assumed_role_object = sts_client.assume_role(
        RoleArn="arn:aws:iam::823139504911:role/Admin",
        RoleSessionName="SparkMLDevAdmin",
    )
    credentials = assumed_role_object["Credentials"]

    aws_access_key_id = credentials["AccessKeyId"]
    aws_secret_access_key = credentials["SecretAccessKey"]
    aws_session_token = credentials["SessionToken"]

    spark = (
        SparkSession.builder.master(os.environ.get("SPARK_MASTER", "local[*]"))
        .appName("ES")
        .config("spark.sql.catalogImplementation", "hive")
        .config(
            "spark.hadoop.fs.AbstractFileSystem.s3.impl", "org.apache.hadoop.fs.s3a.S3A"
        )
        .config(
            "spark.hadoop.fs.AbstractFileSystem.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3A",
        )
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider",
        )
        .config(
            "spark.hadoop.fs.s3.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider",
        )
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.executor.extraLibraryPath", "/usr/lib64/libsnappy.so")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "2047m")
        .config("spark.driver.memory", "20g")
        .config("spark.worker.cleanup.enabled", "true")
        .config("spark.worker.cleanup.interval", "60")
        .config("spark.hadoop.fs.s3.access.key", aws_access_key_id)
        .config("spark.hadoop.fs.s3.secret.key", aws_secret_access_key)
        .config("spark.hadoop.fs.s3.session.token", aws_session_token)
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key_id)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_access_key)
        .config("spark.hadoop.fs.s3a.session.token", aws_session_token)
        .enableHiveSupport()
        .getOrCreate()
    )

    pred_ind_path = args.input
    es.create_index_from_parquet(
        spark=spark,
        source=pred_ind_path,
        es_index=f"{args.index}/_doc",
        id_field="entity_id",
    )


if __name__ == "__main__":
    main()
