import ssl

import urllib3
from elasticsearch import Elasticsearch
from elasticsearch.connection import create_ssl_context
from elasticsearch_dsl import Q, Search

urllib3.disable_warnings()

private_labels = {
    "AT9110366855": ["DE5230002160", "IT02341060289", "SE5564461043", "DE5030404497"],
    "AT9150130980": ["AT9070146571", "AT9070013095", "AT9030105402", "DE7330013290"],
    "AT9070292371": ["US943156479", "GB03128724", "DE8170815005", "CN31893PC"],
    "DE6210018060": [
        "AT9050048223",
        "AT9050000773",
        "AT9130034177",
        "AT9090048034",
        "DE8250456168",
    ],
    "AT9030129604": ["GB01797397", "US262750628L", "US128612417L"],
    "AT9110100823": ["AT9090057018", "AT9110389976", "AT9110473749", "AT9070111785"],
    "AT9110248277": ["AT9110191444", "AT9030010233", "AT9110005665", "AT9110130890"],
    "AT9110004036": [
        "AT9030068392",
        "GB03466427",
        "SE5564874534",
        "AT9070049425",
        "AT9030005573",
    ],
    "AT9070414232": ["AT9150066778", "AT9110465986"],
    "AT9110660590": ["AT9110043437", "AT9070124525", "DE5030402838"],
    "AT9110254456": [],
    "AT9110622968": ["AT9110487541", "AT9070088609", "AT9090024429"],
    "AT9050174224": ["DE8190503933", "AT9150027751", "DE8250149230"],
    "AT9110351503": ["DE3410045617", "AT9050184608", "AT9150174279"],
    "AT9110072476": ["AT9110090301", "FR851192963", "AT9090044860", "AT9010001555"],
    "SK50340336": [
        "DE4150153965",
        "AT9050022806",
        "AT9070002717",
        "AT9110100823",
        "ESA80183916",
        "AT9110826855",
        "AT9110694976",
    ],
    "AT9110125056": [
        "GB05921584",
        "IE048247",
        "PT980477468",
        "GB02171434",
        "GB06497614",
    ],
    "AT9110060440": ["FR352256424", "DK19986292", "FR420189409"],
    "AT9110836851": ["AT9150063166", "DE2190021063", "AT9070040428"],
    "AT9110283885": ["AT9090024294", "AT9010036275", "AT9050066950"],
}

HOST = "127.0.0.1"
PORT = "4443"
SSL = True
INDEX = "company"

if SSL:
    ssl_context = create_ssl_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
else:
    ssl_context = None

client = Elasticsearch(
    hosts=[{"host": HOST, "port": PORT}],
    indices=[INDEX],
    scheme="https" if SSL else "http",
    ssl_context=ssl_context,
)

labels = {}

missed = 0
for public_label, peers in private_labels.items():
    private_label_entity_id = ""

    q = Q("match", id_bvd=str(public_label))

    s = (
        Search(index="company")
        .using(client)
        .query(q)
        .params(search_type="dfs_query_then_fetch")
    )

    for hit in s[0:1]:
        private_label_entity_id = hit.entity_id

    if not private_label_entity_id:
        missed += 1
        continue

    peers_entity_id = []
    for peer in peers:
        q = Q("match", id_bvd=str(peer))

        s = (
            Search(index="company")
            .using(client)
            .query(q)
            .params(search_type="dfs_query_then_fetch")
        )

        for hit in s[0:1]:
            peers_entity_id.append(hit.entity_id)

    labels[private_label_entity_id] = peers_entity_id

print(labels)
print(len(labels))

print(missed)
