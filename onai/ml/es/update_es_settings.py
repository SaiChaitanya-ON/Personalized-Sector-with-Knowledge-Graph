import requests

payload_settings = {
    "settings": {
        "analysis": {
            "filter": {"filter_stemmer": {"type": "stemmer", "language": "english"}},
            "analyzer": {
                "tags_analyzer": {
                    "type": "custom",
                    "filter": ["lowercase", "filter_stemmer"],
                    "tokenizer": "standard",
                }
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

r = requests.put("http://localhost:9200/company/", json=payload_settings)
print(r.status_code)
print(r.content)

r = requests.put(
    "http://localhost:9200/company/_mapping/_doc?include_type_name=true",
    json=payload_mappings,
)
print(r.status_code)
print(r.content)
