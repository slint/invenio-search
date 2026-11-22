# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""ParadeDB search backend tests.

The backend tests need Postgres with the ParadeDB ``pg_search`` extension; set
``PARADEDB_DSN`` (e.g. ``postgresql://postgres@localhost/test``) to run them.
"""

import os
import uuid

import pytest

from invenio_search.engine import SEARCH_DISTRIBUTION

pytestmark = pytest.mark.skipif(
    SEARCH_DISTRIBUTION != "OpenSearch", reason="needs opensearch-py"
)


@pytest.mark.parametrize(
    "query,tree",
    [
        ("a AND b", ("bool", [("term", None, "a", False), ("term", None, "b", False)], [], [])),
        # Lucene: a prohibited clause next to an optional one leaves it optional.
        ("a -b", ("bool", [], [("term", None, "a", False)], [("term", None, "b", False)])),
        ("+a b", ("bool", [("term", None, "a", False)], [("term", None, "b", False)], [])),
        # Plain words stay together, so the field's analyzer sees them at once.
        ("title:(a b)", ("text", "title", "a b")),
        ("lovelac~1", ("fuzzy", None, "lovelac", "1")),
        ("title:x^3", ("boost", ("term", "title", "x", False), 3.0)),
        ('title:"glacier melt"', ("phrase", "title", "glacier melt")),
        ("glac*", ("term", None, "glac\0", True)),
        ("year:[2000 TO 2010}", ("range", "year", "2000", "2010", True, False)),
        ("_exists_:doi", ("exists", "doi")),
        ("*", ("all",)),
    ],
)  # fmt: skip
def test_query_string_parse(query, tree):
    from invenio_search_paradedb.querystring import parse

    assert parse(query) == tree


def test_terms_partition():
    """Partitions as OpenSearch assigned them to these keyword terms."""
    from invenio_search_paradedb.query import term_partition

    assert [term_partition(str(i), 1) for i in range(3)] == [0, 0, 0]
    assert {term_partition(str(i), 3) for i in range(30)} == {0, 1, 2}


@pytest.fixture()
def client():
    dsn = os.environ.get("PARADEDB_DSN")
    if not dsn:
        pytest.skip("PARADEDB_DSN not set")
    import psycopg2
    from opensearchpy import OpenSearch

    from invenio_search_paradedb.connection import ParadeDBConnection

    schema = "test_" + uuid.uuid4().hex[:8]
    yield OpenSearch(
        hosts=["paradedb"], connection_class=ParadeDBConnection, dsn=dsn, schema=schema
    )
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")


MAPPING = {
    "settings": {"index.query.default_field": ["title", "creators.name"]},
    "mappings": {
        "properties": {
            "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "notes": {"type": "text"},
            "access": {"type": "keyword"},
            "owners": {"type": "integer"},
            "date": {"type": "date"},
            "creators": {
                "type": "nested",
                "properties": {
                    "name": {"type": "text"},
                    "role": {"type": "keyword"},
                },
            },
        }
    },
}
DOCS = {
    "1": {
        "title": "Glacier melt measurements",
        "notes": "ocean",
        "access": "public",
        "owners": [1, 2],
        "date": "2020-05-01",
        "creators": [{"name": "Ada Lovelace", "role": "editor"}],
    },
    "2": {
        "title": "Ocean temperature",
        "notes": "glacier",
        "access": "restricted",
        "owners": [2],
        "date": "2021-01-01",
        "creators": [
            {"name": "Ada Byron", "role": "author"},
            {"name": "Mary Lovelace", "role": "editor"},
        ],
    },
    "3": {
        "title": "Protein folding",
        "access": "public",
        "owners": [3],
        "date": "2021-12-31",
        "creators": [{"name": "Grace Hopper", "role": "author"}],
    },
}


def ids(result):
    return sorted(h["_id"] for h in result["hits"]["hits"])


def test_search(client):
    client.indices.create("recs-v1", body=MAPPING)
    client.indices.put_alias(index="recs-v1", name="recs")
    for doc_id, doc in DOCS.items():
        client.index("recs", doc, id=doc_id, version=2, version_type="external_gte")

    def search(query, **body):
        return ids(client.search(index="recs", body={"query": query, **body}))

    # Unfielded text only searches the default fields ("glacier" is also in notes).
    assert search({"query_string": {"query": "glacier"}}) == ["1"]
    assert search({"query_string": {"query": "glac*"}}) == ["1"]
    assert search({"query_string": {"query": "lovelace -ocean"}}) == ["1"]
    assert search({"query_string": {"query": 'title:"temperature ocean"'}}) == []
    assert search(
        {"query_string": {"query": "notes:glacier OR date:[2021-06-01 TO *]"}}
    ) == ["2", "3"]
    # Permission-style filters match exact values, also inside arrays.
    assert search(
        {
            "bool": {
                "filter": [{"terms": {"owners": [2]}}],
                "must_not": [{"term": {"access": "restricted"}}],
            }
        }
    ) == ["1"]
    # Nested conditions must hold on the same object: doc 2 has an author and
    # a creator named Lovelace, but they are different creators.
    nested = {
        "nested": {
            "path": "creators",
            "query": {
                "bool": {
                    "must": [
                        {"term": {"creators.role": "author"}},
                        {"match": {"creators.name": "lovelace"}},
                    ]
                }
            },
        }
    }
    assert search(nested) == []
    assert search({"range": {"date": {"gte": "2021", "lt": "2021-12-31"}}}) == ["2"]
    result = client.search(
        index="recs",
        body={
            "size": 0,
            "aggs": {
                "access": {"terms": {"field": "access"}},
                "years": {
                    "date_histogram": {
                        "field": "date",
                        "calendar_interval": "year",
                        "format": "yyyy",
                    }
                },
            },
        },
    )
    assert result["hits"]["total"]["value"] == 3
    assert [
        (b["key"], b["doc_count"]) for b in result["aggregations"]["access"]["buckets"]
    ] == [
        ("public", 2),
        ("restricted", 1),
    ]
    assert [
        (b["key_as_string"], b["doc_count"])
        for b in result["aggregations"]["years"]["buckets"]
    ] == [("2020", 1), ("2021", 2)]
    sorted_hits = client.search(index="recs", body={"sort": [{"date": "desc"}]})
    assert [h["_id"] for h in sorted_hits["hits"]["hits"]] == ["3", "2", "1"]

    from opensearchpy import ConflictError, RequestError

    with pytest.raises(ConflictError):
        client.index("recs", DOCS["1"], id="1", version=1, version_type="external_gte")
    with pytest.raises(RequestError):
        client.search(index="recs", body={"query": {"geo_shape": {"x": {}}}})
    client.delete("recs", "3", version=3, version_type="external_gte")
    assert search({"match_all": {}}) == ["1", "2"]


EDGE = {
    "settings": {
        "analysis": {
            "analyzer": {
                "edge": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", "grams"],
                },
                "plain": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                },
            },
            "filter": {"grams": {"type": "edge_ngram", "min_gram": 2, "max_gram": 20}},
        }
    },
    "mappings": {
        "properties": {
            "name": {"type": "text", "analyzer": "edge", "search_analyzer": "plain"},
            "place": {"type": "text", "analyzer": "edge", "search_analyzer": "plain"},
            "note": {"type": "text"},
        }
    },
}


def test_accents_and_word_beginnings(client):
    """Expected hits are what OpenSearch returns for these analyzers."""
    client.indices.create("v2", body={**EDGE, "aliases": {"names": {}}})
    # An older index version behind the same alias maps ``place`` as keyword.
    client.indices.create(
        "v1",
        body={
            "mappings": {"properties": {"place": {"type": "keyword"}}},
            "aliases": {"names": {}},
        },
    )
    client.index(
        "v2",
        id="1",
        body={"name": "Bonnat, Catherine", "place": "Université de Zürich"},
    )
    client.index(
        "v2", id="2", body={"name": "Hopper, Grace", "place": "CERN", "note": "Café"}
    )
    client.indices.refresh("names")

    def find(query, fields, kind="best_fields"):
        body = {
            "multi_match": {
                "query": query,
                "fields": fields,
                "type": kind,
                "operator": "and",
            }
        }
        return ids(client.search(index="names", body={"query": body}))

    assert find("zur", ["place"]) == ["1"]  # word beginning, accent folded
    assert find("ZÜRICH", ["place"]) == ["1"]
    assert find("univ", ["place"]) == ["1"]
    assert find("cafe", ["note"]) == ["2"]  # folded without an edge analyzer
    assert find("rich", ["place"]) == []  # n-grams start at word beginnings
    assert find("de bon", ["name", "place"]) == []  # no single field has both
    assert find("de bon", ["name", "place"], "cross_fields") == ["1"]


def test_percolate(client):
    client.indices.create("recs-v1", body=MAPPING)
    client.indices.create(
        "sets", body={"mappings": {**MAPPING["mappings"], "properties": {
            **MAPPING["mappings"]["properties"], "query": {"type": "percolator"}}}}
    )  # fmt: skip
    client.index(
        "sets", {"query": {"query_string": {"query": "title:glacier"}}}, id="glaciers"
    )
    client.index("sets", {"query": {"term": {"access": "public"}}}, id="public")
    result = client.search(
        index="sets",
        body={
            "query": {"percolate": {"field": "query", "documents": list(DOCS.values())}}
        },
    )
    slots = {
        h["_id"]: h["fields"]["_percolator_document_slot"]
        for h in result["hits"]["hits"]
    }
    assert slots == {"glaciers": [0], "public": [0, 2]}
    # The percolated documents are not stored.
    assert client.count(index="sets")["count"] == 2
