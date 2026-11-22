# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""opensearch-py connection class that stores and searches indices in Postgres.

Instead of sending HTTP requests to a cluster, :class:`ParadeDBConnection`
answers the OpenSearch REST API calls Invenio makes from a Postgres database
with the ParadeDB ``pg_search`` extension. All indices share one documents
table with a BM25 index; index metadata, aliases and templates live in small
side tables. Requests it cannot answer raise a 400 error rather than being
ignored.

It is meant for small instances (local development, demos, test
environments), not as a replacement for an OpenSearch cluster.
"""

import datetime as dt
import fnmatch
import functools
import json
import re
import threading
import time
import uuid
from copy import deepcopy

from opensearchpy.connection.base import Connection

from .hashing import f32
from .query import (
    DATE_TYPES,
    EPOCH,
    NUMERIC_TYPES,
    ROOT,
    Aggregations,
    Fields,
    Translator,
    Unsupported,
    _listify,
    filter_source,
    fold,
    leaves,
    parse_date,
    values_at,
)
from .querystring import ParseError

try:
    import psycopg2 as psycopg
except ImportError:  # pragma: no cover
    import psycopg

SEARCH_KEYS = {
    "query",
    "post_filter",
    "from",
    "size",
    "sort",
    "aggs",
    "aggregations",
    "_source",
    "track_total_hits",
    "track_scores",
    "version",
    "timeout",
    "explain",
    "search_after",
}
LONG_MAX = 2**63 - 1
LONG_MIN = -(2**63)


class Error(Exception):
    """An error answered with an OpenSearch-style error response."""

    def __init__(self, status, kind, reason):
        """Constructor."""
        super().__init__(reason)
        self.status, self.kind, self.reason = status, kind, reason


def _not_found(name):
    return Error(404, "index_not_found_exception", f"no such index [{name}]")


def _merge(base, extra):
    """Deep merge ``extra`` into a copy of ``base``."""
    out = deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _flat(settings, prefix=""):
    """Flatten index settings to dotted keys without the ``index.`` prefix."""
    out = {}
    for key, value in (settings or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, path + "."))
        else:
            out[path[6:] if path.startswith("index.") else path] = value
    return out


def _true(value):
    return value in (True, "true", "True", 1, "1")


class ParadeDBConnection(Connection):
    """Answer OpenSearch API calls from Postgres with ParadeDB ``pg_search``.

    Usually enabled by :class:`~invenio_search_paradedb.ext.InvenioSearchParadeDB`
    from a ``paradedb://`` URL in ``SEARCH_HOSTS``. ``dsn`` is the Postgres
    connection string; without it the connection uses the application's
    ``SQLALCHEMY_DATABASE_URI``.
    """

    _ready = set()
    _ready_lock = threading.Lock()
    _scrolls = {}

    def __init__(self, dsn=None, schema="invenio_search", **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.dsn = dsn
        self.schema = schema
        self.docs = f"{schema}.docs"
        self._local = threading.local()

    # Connection management

    def _dsn(self):
        if self.dsn:
            return self.dsn
        from flask import current_app

        dsn = current_app.config["SQLALCHEMY_DATABASE_URI"]
        scheme, rest = dsn.split("://", 1)
        return scheme.split("+")[0] + "://" + rest

    def _connect(self):
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            conn = psycopg.connect(self._dsn())
            self._local.conn = conn
        key = (self._dsn(), self.schema)
        if key not in self._ready:
            with self._ready_lock:
                if key not in self._ready:
                    self._setup(conn)
                    self._ready.add(key)
        return conn

    def _setup(self, conn):
        s = self.schema
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search CASCADE")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {s}.indices ("
                "name text PRIMARY KEY, mappings jsonb NOT NULL, "
                "settings jsonb NOT NULL, created bigint NOT NULL)"
            )
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {s}.aliases ("
                f"alias text, index text REFERENCES {s}.indices ON DELETE CASCADE, "
                "PRIMARY KEY (alias, index))"
            )
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {s}.templates ("
                "kind text, name text, body jsonb NOT NULL, PRIMARY KEY (kind, name))"
            )
            # Like Lucene doc ids, ``seq`` grows on every write; it breaks ties.
            cur.execute(f"CREATE SEQUENCE IF NOT EXISTS {s}.docs_seq")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {s}.docs ("
                "pk bigserial PRIMARY KEY, "
                f"idx text NOT NULL REFERENCES {s}.indices ON DELETE CASCADE, "
                "id text NOT NULL, version bigint NOT NULL, "
                f"seq bigint NOT NULL DEFAULT nextval('{s}.docs_seq'), text text, "
                "doc jsonb NOT NULL, idoc jsonb, UNIQUE (idx, id))"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS docs_bm25 ON {s}.docs "
                "USING bm25 (pk, text, idoc) "
                "WITH (key_field = 'pk', json_fields = '{\"idoc\": {\"fast\": true}}')"
            )
        conn.commit()

    def close(self):
        """Close this thread's database connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # Request dispatch

    def perform_request(
        self,
        method,
        url,
        params=None,
        body=None,
        timeout=None,
        ignore=(),
        headers=None,
    ):
        """Answer one API request. Returns ``(status, headers, body)``."""
        start = time.time()
        path, _, query = url.partition("?")
        # The client passes some values as bytes, such as ``version_type``.
        params = {
            k: v.decode() if isinstance(v, bytes) else v
            for k, v in (params or {}).items()
        }
        for pair in filter(None, query.split("&")):
            key, _, value = pair.partition("=")
            params.setdefault(key, value)
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        parts = [p for p in path.split("/") if p]
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                status, data = self._route(cur, method, parts, params, body)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            if isinstance(exc, Error):
                status, data = exc.status, self._error(exc)
            elif isinstance(exc, Unsupported):
                status = 400
                data = self._error(
                    Error(400, "illegal_argument_exception", f"unsupported: {exc}")
                )
            elif isinstance(exc, (ParseError, ValueError, KeyError, TypeError)):
                status = 400
                data = self._error(
                    Error(400, "search_phase_execution_exception", str(exc))
                )
            elif isinstance(exc, psycopg.Error):
                status = 500
                data = self._error(Error(500, "exception", str(exc).strip()))
            else:
                raise
        raw = "" if method == "HEAD" or data is None else json.dumps(data)
        duration = time.time() - start
        if isinstance(ignore, int):
            ignore = (ignore,)
        if not (200 <= status < 300) and status not in ignore:
            self.log_request_fail(method, url, path, body, duration, status, raw)
            self._raise_error(status, raw, "application/json")
        self.log_request_success(method, url, path, body, status, raw, duration)
        return status, {"content-type": "application/json"}, raw

    @staticmethod
    def _error(err):
        cause = {"type": err.kind, "reason": err.reason}
        return {
            "error": {**cause, "root_cause": [cause]},
            "status": err.status,
        }

    def _route(self, cur, method, parts, params, body):
        data = json.loads(body) if body and parts[-1:] != ["_bulk"] else {}
        head = parts[0] if parts else ""
        if not parts:
            return 200, self._info()
        if head == "_cluster" and parts[1:2] == ["health"]:
            return 200, {"status": "green", "timed_out": False, "number_of_nodes": 1}
        if head in ("_template", "_index_template", "_component_template"):
            return self._template(cur, method, head, parts[1:], data)
        if head == "_aliases":
            return self._update_aliases(cur, method, data)
        if head == "_alias":
            return self._aliases(cur, method, None, parts[1:])
        if head == "_bulk":
            return self._bulk(cur, None, body, params)
        if head == "_search" and parts[1:2] == ["scroll"]:
            return self._scroll(method, data, params)
        if head in ("_search", "_count"):
            parts = ["*"] + parts
        if head in ("_refresh", "_flush", "_forcemerge"):
            return 200, {"_shards": {"total": 1, "successful": 1, "failed": 0}}
        if head.startswith("_") and parts[0] != "*":
            raise Unsupported(f"endpoint [/{'/'.join(parts)}]")

        name, rest = parts[0], parts[1:]
        action = rest[0] if rest else None
        if action is None:
            return self._index(cur, method, name, data, params)
        if action in ("_doc", "_create", "_update"):
            method = "PUT" if method == "POST" else method
            return self._document(cur, method, name, action, rest[1:], data, params)
        if action == "_count":
            return 200, self._search(cur, name, data, params, count=True)
        if action == "_search" and params.get("scroll"):
            size = int(data.get("size", params.get("size", 10)))
            result = self._search(cur, name, data, params, limit=False)
            return 200, self._start_scroll(result, size)
        if action == "_search":
            return 200, self._search(cur, name, data, params)
        if action == "_bulk":
            return self._bulk(cur, name, body, params)
        if action in ("_refresh", "_flush", "_forcemerge"):
            self._resolve(cur, name, params)
            return 200, {"_shards": {"total": 1, "successful": 1, "failed": 0}}
        if action in ("_alias", "_aliases"):
            return self._aliases(cur, method, name, rest[1:])
        if action == "_mapping":
            return self._mapping(cur, method, name, rest[1:], data, params)
        if action == "_settings" and method == "GET":
            names = self._resolve(cur, name, params)
            meta = self._meta(cur, names)
            return 200, {n: {"settings": {"index": meta[n][1]}} for n in names}
        if action == "_delete_by_query":
            return 200, self._delete_by_query(cur, name, data, params)
        raise Unsupported(f"endpoint [{method} /{'/'.join(parts)}]")

    def _info(self):
        from opensearchpy import VERSION

        return {
            "name": "postgres",
            "cluster_name": "postgres",
            "version": {
                "distribution": "opensearch",
                "number": ".".join(map(str, VERSION)),
            },
            "tagline": "Invenio search on Postgres",
        }

    # Index metadata

    def _names(self, cur):
        cur.execute(f"SELECT name FROM {self.schema}.indices")
        names = {r[0] for r in cur.fetchall()}
        cur.execute(f"SELECT alias, index FROM {self.schema}.aliases")
        aliases = {}
        for alias, index in cur.fetchall():
            aliases.setdefault(alias, []).append(index)
        return names, aliases

    def _resolve(self, cur, expr, params=None, aliases_ok=True):
        """Resolve an index expression to concrete index names."""
        params = params or {}
        names, aliases = self._names(cur)
        lenient = _true(params.get("ignore_unavailable"))
        out = []
        for part in expr.split(","):
            if part.startswith("-"):
                excluded = set(fnmatch.filter(names, part[1:]))
                out = [n for n in out if n not in excluded]
                continue
            if part in ("_all", "*"):
                found = sorted(names)
            elif "*" in part or "?" in part:
                found = sorted(fnmatch.filter(names, part))
                if aliases_ok:
                    for alias in fnmatch.filter(aliases, part):
                        found += aliases[alias]
            elif part in names:
                found = [part]
            elif aliases_ok and part in aliases:
                found = sorted(aliases[part])
            elif lenient:
                found = []
            else:
                raise _not_found(part)
            out += [n for n in found if n not in out]
        return out

    def _meta(self, cur, names):
        cur.execute(
            f"SELECT name, mappings, settings FROM {self.schema}.indices "
            "WHERE name = ANY(%s::text[])",
            [list(names)],
        )
        return {name: (m, s) for name, m, s in cur.fetchall()}

    def _templates_for(self, cur, name):
        """Mappings, settings and aliases from templates matching ``name``."""
        cur.execute(f"SELECT kind, name, body FROM {self.schema}.templates")
        rows = cur.fetchall()
        components = {n: b for k, n, b in rows if k == "_component_template"}
        legacy = sorted(
            (b for k, n, b in rows if k == "_template" and self._matches(b, name)),
            key=lambda b: b.get("order", 0),
        )
        composable = sorted(
            (
                b
                for k, n, b in rows
                if k == "_index_template" and self._matches(b, name)
            ),
            key=lambda b: b.get("priority", 0),
        )
        result = {"mappings": {}, "settings": {}, "aliases": {}}
        if composable:
            best = composable[-1]
            for component in best.get("composed_of") or []:
                result = _merge(
                    result, (components.get(component) or {}).get("template")
                )
            return _merge(result, best.get("template"))
        for template in legacy:
            result = _merge(
                result, {k: template.get(k) or {} for k in result if k in template}
            )
        return result

    @staticmethod
    def _matches(template, name):
        patterns = template.get("index_patterns") or []
        if isinstance(patterns, str):
            patterns = [patterns]
        return any(fnmatch.fnmatchcase(name, p) for p in patterns)

    def _create(self, cur, name, data):
        names, aliases = self._names(cur)
        if name in names or name in aliases:
            raise Error(
                400,
                "resource_already_exists_exception",
                f"index [{name}] already exists",
            )
        spec = _merge(self._templates_for(cur, name), data)
        cur.execute(
            f"INSERT INTO {self.schema}.indices VALUES (%s, %s, %s, %s)",
            [
                name,
                json.dumps(spec.get("mappings") or {}),
                json.dumps(_flat(spec.get("settings"))),
                int(time.time() * 1000),
            ],
        )
        for alias in spec.get("aliases") or {}:
            cur.execute(
                f"INSERT INTO {self.schema}.aliases VALUES (%s, %s)", [alias, name]
            )

    def _index(self, cur, method, name, data, params):
        if method == "PUT":
            self._create(cur, name, data)
            return 200, {
                "acknowledged": True,
                "shards_acknowledged": True,
                "index": name,
            }
        if method == "HEAD":
            names, aliases = self._names(cur)
            if "*" in name or "," in name:
                return (200 if self._resolve(cur, name, params) else 404), None
            return (200 if name in names or name in aliases else 404), None
        if method == "DELETE":
            for index in self._resolve(cur, name, params, aliases_ok=False):
                cur.execute(
                    f"DELETE FROM {self.schema}.indices WHERE name = %s", [index]
                )
            return 200, {"acknowledged": True}
        if method == "GET":
            names = self._resolve(cur, name, params)
            meta = self._meta(cur, names)
            _, aliases = self._names(cur)
            return 200, {
                n: {
                    "aliases": {a: {} for a, idx in aliases.items() if n in idx},
                    "mappings": meta[n][0],
                    "settings": {"index": meta[n][1]},
                }
                for n in names
            }
        raise Unsupported(f"[{method}] on an index")

    def _template(self, cur, method, kind, rest, data):
        if not rest:
            raise Unsupported(f"[{method} /{kind}] without a name")
        name = rest[0]
        table = f"{self.schema}.templates"
        if method in ("PUT", "POST"):
            cur.execute(
                f"INSERT INTO {table} VALUES (%s, %s, %s) ON CONFLICT (kind, name) "
                "DO UPDATE SET body = EXCLUDED.body",
                [kind, name, json.dumps(data)],
            )
            return 200, {"acknowledged": True}
        cur.execute(f"SELECT name, body FROM {table} WHERE kind = %s", [kind])
        found = [(n, b) for n, b in cur.fetchall() if fnmatch.fnmatchcase(n, name)]
        if method == "HEAD":
            return (200 if found else 404), None
        if not found:
            raise Error(404, "resource_not_found_exception", f"[{name}] missing")
        if method == "DELETE":
            for n, _ in found:
                cur.execute(
                    f"DELETE FROM {table} WHERE kind = %s AND name = %s", [kind, n]
                )
            return 200, {"acknowledged": True}
        if kind == "_template":
            return 200, dict(found)
        key = kind[1:] + "s"
        return 200, {
            key: [{"name": n, kind[1:]: b} for n, b in found],
        }

    def _aliases(self, cur, method, index, rest):
        alias = rest[0] if rest else None
        table = f"{self.schema}.aliases"
        if method in ("PUT", "POST"):
            for name in self._resolve(cur, index, aliases_ok=False):
                for a in alias.split(","):
                    cur.execute(
                        f"INSERT INTO {table} VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        [a, name],
                    )
            return 200, {"acknowledged": True}
        _, aliases = self._names(cur)
        indices = set(self._resolve(cur, index)) if index else None
        patterns = alias.split(",") if alias and alias not in ("_all", "*") else None
        pairs = [
            (a, i)
            for a, idx in aliases.items()
            for i in idx
            if (indices is None or i in indices)
            and (patterns is None or any(fnmatch.fnmatchcase(a, p) for p in patterns))
        ]
        if method == "DELETE":
            if not pairs:
                raise Error(404, "aliases_not_found_exception", f"[{alias}] missing")
            for a, i in pairs:
                cur.execute(
                    f"DELETE FROM {table} WHERE alias = %s AND index = %s", [a, i]
                )
            return 200, {"acknowledged": True}
        if method == "HEAD":
            return (200 if pairs else 404), None
        if patterns and not pairs:
            raise Error(404, "aliases_not_found_exception", f"alias [{alias}] missing")
        out = {i: {"aliases": {}} for i in (indices or [])}
        for a, i in pairs:
            out.setdefault(i, {"aliases": {}})["aliases"][a] = {}
        return 200, out

    def _update_aliases(self, cur, method, data):
        if method != "POST":
            return self._aliases(cur, method, None, [])
        for action in data.get("actions") or []:
            ((op, spec),) = action.items()
            indices = spec.get("indices") or [spec.get("index")]
            aliases = spec.get("aliases") or [spec.get("alias")]
            for index in indices:
                if op == "remove_index":
                    self._index(cur, "DELETE", index, {}, {})
                    continue
                for alias in aliases:
                    self._aliases(
                        cur, "PUT" if op == "add" else "DELETE", index, [alias]
                    )
        return 200, {"acknowledged": True}

    def _mapping(self, cur, method, name, rest, data, params):
        names = self._resolve(cur, name, params)
        meta = self._meta(cur, names)
        if method == "PUT":
            for n in names:
                cur.execute(
                    f"UPDATE {self.schema}.indices SET mappings = %s WHERE name = %s",
                    [json.dumps(_merge(meta[n][0], data)), n],
                )
            return 200, {"acknowledged": True}
        if rest[:1] == ["field"]:
            wanted = rest[1].split(",")
            out = {}
            for n in names:
                fields = Fields([meta[n][0]])
                found = {}
                for field in wanted:
                    for full in fnmatch.filter(fields.map, field):
                        spec = self._field_spec(meta[n][0], full)
                        if spec is not None:
                            leaf = full.rsplit(".", 1)[-1]
                            found[full] = {"full_name": full, "mapping": {leaf: spec}}
                out[n] = {"mappings": found}
            return 200, out
        return 200, {n: {"mappings": meta[n][0]} for n in names}

    @staticmethod
    def _field_spec(mapping, path):
        node = {"properties": mapping.get("properties") or {}}
        for segment in path.split("."):
            props = node.get("properties") or {}
            if segment in props:
                node = props[segment]
            elif segment in (node.get("fields") or {}):
                node = node["fields"][segment]
            else:
                return None
        return node

    # Documents

    def _write_index(self, cur, name, cache):
        """Resolve the index a write to ``name`` goes to: ``{index, text}``."""
        if name in cache:
            return cache[name]
        names, aliases = self._names(cur)
        if name in names:
            index = name
        elif name in aliases:
            if len(aliases[name]) != 1:
                raise Error(
                    400,
                    "illegal_argument_exception",
                    f"alias [{name}] has more than one index",
                )
            index = aliases[name][0]
        else:
            self._create(cur, name, {})
            index = name
        mappings, settings = self._meta(cur, [index])[index]
        cache[name] = {"index": index, **_analysis(mappings, settings)}
        return cache[name]

    def _store(self, cur, entry, doc_id, doc, version):
        """Write a document. Returns its pk."""
        cur.execute(
            f"INSERT INTO {self.docs} (idx, id, version, text, doc, idoc) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (idx, id) DO UPDATE SET "
            "version = EXCLUDED.version, text = EXCLUDED.text, doc = EXCLUDED.doc, "
            "idoc = EXCLUDED.idoc, "
            f"seq = nextval('{self.schema}.docs_seq') RETURNING pk",
            [
                entry["index"],
                doc_id,
                version,
                _text(doc, entry["text"], entry["edge"]),
                json.dumps(doc),
                json.dumps(_analysed(doc, entry["edge"])),
            ],
        )
        return cur.fetchone()[0]

    def _put(self, cur, entry, doc_id, doc, version, version_type, op):
        """Store a document. Returns ``(status, result, version)``."""
        index = entry["index"]
        cur.execute(
            f"SELECT version FROM {self.docs} WHERE idx = %s AND id = %s FOR UPDATE",
            [index, doc_id],
        )
        row = cur.fetchone()
        current = row[0] if row else None
        if op == "create" and current is not None:
            raise Error(
                409,
                "version_conflict_engine_exception",
                f"[{doc_id}]: document already exists",
            )
        if version_type in ("external", "external_gte"):
            version = int(version)
            if current is not None and (
                version < current or (version_type == "external" and version == current)
            ):
                raise Error(
                    409,
                    "version_conflict_engine_exception",
                    f"[{doc_id}]: version conflict, current version [{current}] "
                    f"is higher or equal to the one provided [{version}]",
                )
        else:
            version = (current or 0) + 1
        self._store(cur, entry, doc_id, doc, version)
        if current is None:
            return 201, "created", version
        return 200, "updated", version

    def _delete(self, cur, index, doc_id, version, version_type):
        cur.execute(
            f"SELECT version FROM {self.docs} WHERE idx = %s AND id = %s FOR UPDATE",
            [index, doc_id],
        )
        row = cur.fetchone()
        if row is None:
            return 404, "not_found", int(version or 1)
        if version_type in ("external", "external_gte") and int(version) < row[0]:
            raise Error(
                409,
                "version_conflict_engine_exception",
                f"[{doc_id}]: version conflict, current version [{row[0]}] "
                f"is higher than the one provided [{version}]",
            )
        cur.execute(
            f"DELETE FROM {self.docs} WHERE idx = %s AND id = %s", [index, doc_id]
        )
        new = (
            int(version)
            if version_type in ("external", "external_gte")
            else (row[0] + 1)
        )
        return 200, "deleted", new

    @staticmethod
    def _doc_result(index, doc_id, status, result, version):
        return {
            "_index": index,
            "_id": doc_id,
            "_version": version,
            "result": result,
            "_shards": {"total": 1, "successful": 1, "failed": 0},
            "_seq_no": 0,
            "_primary_term": 1,
        }

    def _document(self, cur, method, name, action, rest, data, params):
        doc_id = rest[0] if rest else uuid.uuid4().hex
        version, version_type = params.get("version"), params.get("version_type")
        if method in ("GET", "HEAD"):
            indices = self._resolve(cur, name, params)
            cur.execute(
                f"SELECT idx, version, doc FROM {self.docs} "
                "WHERE idx = ANY(%s::text[]) AND id = %s",
                [indices, doc_id],
            )
            row = cur.fetchone()
            if method == "HEAD":
                return (200 if row else 404), None
            if row is None:
                return 404, {"_index": name, "_id": doc_id, "found": False}
            return 200, {
                "_index": row[0],
                "_id": doc_id,
                "_version": row[1],
                "_seq_no": 0,
                "_primary_term": 1,
                "found": True,
                "_source": filter_source(row[2], self._source_param(params)),
            }
        if method == "DELETE":
            index = self._resolve(cur, name, params)
            if len(index) != 1:
                raise _not_found(name)
            status, result, v = self._delete(
                cur, index[0], doc_id, version, version_type
            )
            return status, self._doc_result(index[0], doc_id, status, result, v)
        entry = self._write_index(cur, name, {})
        index = entry["index"]
        if action == "_update":
            doc = self._updated(cur, index, doc_id, data)
            status, result, v = self._put(cur, entry, doc_id, doc, None, None, "index")
        else:
            op = (
                "create"
                if action == "_create" or params.get("op_type") == "create"
                else "index"
            )
            status, result, v = self._put(
                cur, entry, doc_id, data, version, version_type, op
            )
        return status, self._doc_result(index, doc_id, status, result, v)

    def _updated(self, cur, index, doc_id, data):
        if "script" in data:
            raise Unsupported("scripted updates")
        cur.execute(
            f"SELECT doc FROM {self.docs} WHERE idx = %s AND id = %s", [index, doc_id]
        )
        row = cur.fetchone()
        if row is None:
            if data.get("doc_as_upsert"):
                return data.get("doc") or {}
            if "upsert" in data:
                return data["upsert"]
            raise Error(404, "document_missing_exception", f"[{doc_id}]: missing")
        return _merge(row[0], data.get("doc"))

    def _bulk(self, cur, default_index, body, params):
        lines = [json.loads(line) for line in (body or "").splitlines() if line.strip()]
        items, errors, cache, i = [], False, {}, 0
        while i < len(lines):
            ((op, meta),) = lines[i].items()
            source = None
            if op != "delete":
                source = lines[i + 1]
                i += 1
            i += 1
            name = meta.get("_index") or default_index
            doc_id = meta.get("_id") or uuid.uuid4().hex
            version = meta.get("version", meta.get("_version"))
            version_type = meta.get("version_type", meta.get("_version_type"))
            cur.execute("SAVEPOINT bulk_item")
            try:
                if op == "delete":
                    indices = self._resolve(cur, name)
                    index = indices[0]
                    status, result, v = self._delete(
                        cur, index, doc_id, version, version_type
                    )
                else:
                    entry = self._write_index(cur, name, cache)
                    index = entry["index"]
                    if op == "update":
                        source = self._updated(cur, index, doc_id, source)
                        version, version_type = None, None
                    status, result, v = self._put(
                        cur, entry, doc_id, source, version, version_type, op
                    )
                item = self._doc_result(index, doc_id, status, result, v)
                item["status"] = status
                cur.execute("RELEASE SAVEPOINT bulk_item")
            except Error as err:
                cur.execute("ROLLBACK TO SAVEPOINT bulk_item")
                errors = True
                status = err.status
                item = {
                    "_index": name,
                    "_id": doc_id,
                    "status": err.status,
                    "error": {"type": err.kind, "reason": err.reason},
                }
            items.append({op: item})
        return 200, {"took": 0, "errors": errors, "items": items}

    # Search

    @staticmethod
    def _source_param(params, body=None):
        if body and "_source" in body:
            return body["_source"]
        includes = params.get("_source_includes") or params.get("_source")
        excludes = params.get("_source_excludes")
        if includes in ("true", "false"):
            return includes == "true"
        if not includes and not excludes:
            return None
        return {
            "includes": includes.split(",") if includes else [],
            "excludes": excludes.split(",") if excludes else [],
        }

    def _search(self, cur, expr, body, params, count=False, limit=True):
        unknown = set(body) - SEARCH_KEYS
        if unknown:
            raise Unsupported(f"search options {sorted(unknown)}")
        if params.get("q"):
            body = {**body, "query": {"query_string": {"query": params["q"]}}}
        if not _true(params.get("ignore_unavailable")) and "*" not in expr:
            self._resolve(cur, expr, params)
        names = self._resolve(cur, expr, {"ignore_unavailable": "true"})
        meta = self._meta(cur, names)
        fields = Fields([m for m, _ in meta.values()])
        slots = {}
        tr = Translator(
            fields,
            self.docs,
            percolate=lambda field, docs: self._percolate(
                cur, names, meta, field, docs, slots
            ),
        )

        def matching(query):
            """Pks of the documents matching a query, without scores."""
            cond, cond_params = Translator(fields, self.docs, tr.percolate).where(
                query, ROOT, False
            )
            cur.execute(
                f"SELECT d.pk FROM {self.docs} d WHERE d.idx = ANY(%s::text[]) "
                f"AND COALESCE({cond}, FALSE)",
                [names] + cond_params,
            )
            return {r[0] for r in cur.fetchall()}

        # Text clauses become pg_search BM25 queries; their scores are added up.
        cond, cond_params = tr.where(body.get("query"))
        ctes, joins, score_params = [], [], []
        for n, (expr_sql, p) in enumerate(tr.scoring):
            ctes.append(
                f"s{n} AS (SELECT pk, paradedb.score(pk) AS sc FROM {self.docs} "
                f"WHERE pk @@@ {expr_sql})"
            )
            joins.append(f"LEFT JOIN s{n} ON s{n}.pk = d.pk")
            score_params += p
        score = " + ".join(f"COALESCE(s{n}.sc, 0)" for n in range(len(ctes)))
        cur.execute(
            ("WITH " + ", ".join(ctes) + " " if ctes else "")
            + f"SELECT d.pk, ({score or '1.0'})::real FROM {self.docs} d "
            f"{' '.join(joins)} WHERE d.idx = ANY(%s::text[]) "
            f"AND COALESCE({cond}, FALSE)",
            score_params + [names] + cond_params,
        )
        scores = dict(cur.fetchall())
        shards = {"total": 1, "successful": 1, "skipped": 0, "failed": 0}
        if count:
            return {"count": len(scores), "_shards": shards}

        aggs_spec = body.get("aggs") or body.get("aggregations")
        aggregations = None
        if aggs_spec:
            aggs = Aggregations(fields, aggs_spec, matching)
            select, select_params, keys = aggs.select()
            cur.execute(
                f"SELECT {select} FROM {self.docs} d WHERE d.pk = ANY(%s::bigint[])",
                select_params + [list(scores)],
            )
            rows = [dict(zip(keys, row)) for row in cur.fetchall()]
            aggregations = aggs.compute(rows)

        if body.get("post_filter"):
            keep = matching(body["post_filter"])
            scores = {pk: v for pk, v in scores.items() if pk in keep}

        specs = _sort_specs(body.get("sort"))
        with_score = any(s[0] == "_score" for s in specs) or _true(
            body.get("track_scores")
        )
        need_doc = any(s[0] not in ("_score", "_doc", "_id") for s in specs)
        cur.execute(
            f"SELECT d.pk, d.idx, d.id, d.seq"
            f"{', d.doc' if need_doc else ''} FROM {self.docs} d "
            "WHERE d.pk = ANY(%s::bigint[])",
            [list(scores)],
        )
        entries = []
        for row in cur.fetchall():
            pk, idx, doc_id, seq = row[:4]
            doc = row[4] if need_doc else None
            values = [_sort_value(fields, spec, doc, scores[pk], row) for spec in specs]
            entries.append((values, seq, pk, idx, doc_id))
        entries.sort(
            key=functools.cmp_to_key(
                lambda a, b: (
                    _compare(a[0], b[0], specs) or (a[1] > b[1]) - (a[1] < b[1])
                )
            )
        )
        if body.get("search_after") is not None:
            after = body["search_after"]
            entries = [e for e in entries if _compare(e[0], after, specs) > 0]
        total = len(scores)

        size = int(body.get("size", params.get("size", 10)))
        frm = int(body.get("from", params.get("from", 0)))
        page = entries[frm : frm + size] if limit else entries
        cur.execute(
            f"SELECT pk, version, doc FROM {self.docs} WHERE pk = ANY(%s::bigint[])",
            [[e[2] for e in page]],
        )
        stored = {pk: (v, doc) for pk, v, doc in cur.fetchall()}
        version = _true(params.get("version")) or body.get("version") is True
        source = self._source_param(params, body)
        hits = []
        for values, _, pk, idx, doc_id in page:
            hit = {
                "_index": idx,
                "_id": doc_id,
                "_score": _java_float(scores[pk]) if with_score else None,
            }
            if version:
                hit["_version"] = stored[pk][0]
            src = filter_source(stored[pk][1], source)
            if src is not None:
                hit["_source"] = src
            if pk in slots:
                hit["fields"] = {"_percolator_document_slot": slots[pk]}
            if any(spec[0] != "_score" for spec in specs):
                hit["sort"] = [
                    _java_float(v) if spec[0] == "_score" else v
                    for v, spec in zip(values, specs)
                ]
            hits.append(hit)
        scored = list(scores.values()) if with_score else []
        track = body.get("track_total_hits", 10000)
        total_hits = None
        if track not in (False, "false"):
            relation = "eq"
            limit_total = None if track in (True, "true") else int(track)
            if limit_total is not None and total > limit_total:
                total, relation = limit_total, "gte"
            total_hits = {"value": total, "relation": relation}
        result = {
            "took": 0,
            "timed_out": False,
            "_shards": shards,
            "hits": {
                "max_score": _java_float(max(scored)) if scored else None,
                "hits": hits,
            },
        }
        if total_hits is not None:
            result["hits"] = {"total": total_hits, **result["hits"]}
        if aggregations is not None:
            result["aggregations"] = aggregations
        return result

    def _percolate(self, cur, names, meta, field, documents, slots):
        """Run stored queries against ``documents``; returns matching query rows."""
        matched = []
        cur.execute("SAVEPOINT percolate")
        try:
            for name in names:
                mappings, settings = meta[name]
                fields = Fields([mappings])
                if fields.resolve(field)[1] != "percolator":
                    continue
                entry = {"index": name, **_analysis(mappings, settings)}
                temp = [
                    self._store(cur, entry, f"_percolate-{slot}", doc, 1)
                    for slot, doc in enumerate(documents)
                ]
                cur.execute(
                    f"SELECT pk, doc -> %s FROM {self.docs} "
                    "WHERE idx = %s AND doc ? %s AND NOT (pk = ANY(%s::bigint[]))",
                    [field, name, field, temp],
                )
                for pk, stored in cur.fetchall():
                    cond = Translator(fields, self.docs).where(stored)
                    cur.execute(
                        f"SELECT d.pk FROM {self.docs} d "
                        f"WHERE d.pk = ANY(%s::bigint[]) AND COALESCE({cond[0]}, FALSE)",
                        [temp] + cond[1],
                    )
                    hit_slots = sorted(temp.index(r[0]) for r in cur.fetchall())
                    if hit_slots:
                        matched.append(pk)
                        slots[pk] = hit_slots
        finally:
            cur.execute("ROLLBACK TO SAVEPOINT percolate")
        return matched

    def _scroll(self, method, data, params):
        ids = data.get("scroll_id") or params.get("scroll_id")
        if method == "DELETE":
            ids = [ids] if isinstance(ids, str) else ids or []
            freed = sum(self._scrolls.pop(i, None) is not None for i in ids)
            return 200, {"succeeded": True, "num_freed": freed}
        state = self._scrolls.get(ids)
        if state is None:
            raise Error(
                404, "search_context_missing_exception", f"No search context [{ids}]"
            )
        result, pos, size = state
        page = deepcopy({k: v for k, v in result.items() if k != "aggregations"})
        page["hits"]["hits"] = result["hits"]["hits"][pos : pos + size]
        page["_scroll_id"] = ids
        self._scrolls[ids] = (result, pos + size, size)
        return 200, page

    def _start_scroll(self, result, size):
        scroll_id = uuid.uuid4().hex
        self._scrolls[scroll_id] = (result, size, size)
        page = deepcopy(result)
        page["hits"]["hits"] = result["hits"]["hits"][:size]
        page["_scroll_id"] = scroll_id
        return page

    def _delete_by_query(self, cur, name, data, params):
        result = self._search(
            cur, name, {"query": data.get("query")}, params, limit=False
        )
        for hit in result["hits"]["hits"]:
            cur.execute(
                f"DELETE FROM {self.docs} WHERE idx = %s AND id = %s",
                [hit["_index"], hit["_id"]],
            )
        n = len(result["hits"]["hits"])
        return {"took": 0, "timed_out": False, "total": n, "deleted": n, "failures": []}


# Sorting
# =======


def _sort_specs(sort):
    """``[(field, descending, missing, mode)]``; by default by score."""
    if not sort:
        return [("_score", True, "_last", None)]
    out = []
    for item in _listify(sort):
        if isinstance(item, str):
            field, opts = item, {}
        else:
            ((field, opts),) = item.items()
        if isinstance(opts, str):
            opts = {"order": opts}
        order = str(opts.get("order") or ("desc" if field == "_score" else "asc"))
        if order.lower() not in ("asc", "desc"):
            raise Unsupported(f"sort order [{order}]")
        out.append(
            (
                field,
                order.lower() == "desc",
                opts.get("missing", "_last"),
                opts.get("mode"),
            )
        )
    return out


def _sort_value(fields, spec, doc, score, row):
    """The value OpenSearch sorts a hit by, as it reports it in ``sort``."""
    field, desc, missing, mode = spec
    if field == "_score":
        return score
    if field == "_doc":
        return row[3]
    if field == "_id":
        return row[2]
    path, kind = fields.resolve(field)
    values = []
    for value in values_at(doc, path):
        if value is None or isinstance(value, (dict, list)):
            continue
        if kind in DATE_TYPES:
            d = parse_date(value)
            value = (d - EPOCH) // dt.timedelta(milliseconds=1) if d else None
        elif isinstance(value, bool):
            value = "true" if value else "false" if kind == "keyword" else int(value)
        elif kind in NUMERIC_TYPES and isinstance(value, str):
            value = float(value) if "." in value else int(value)
        elif kind == "keyword":
            value = str(value)
        if value is not None:
            values.append(value)
    if values:
        if mode == "sum":
            return sum(values)
        if mode == "avg":
            return sum(values) / len(values)
        if mode == "max" or (mode is None and desc):
            return max(values)
        return min(values)
    if kind not in NUMERIC_TYPES | DATE_TYPES | {"boolean"}:
        return None
    return LONG_MAX if desc == (missing == "_first") else LONG_MIN


def _compare(a, b, specs):
    for x, y, (_, desc, missing, _) in zip(a, b, specs):
        if x == y:
            continue
        if x is None or y is None:
            res = 1 if x is None else -1
            return -res if missing == "_first" else res
        res = -1 if x < y else 1
        return -res if desc else res
    return 0


def _java_float(value):
    """A 32-bit float score as OpenSearch prints it: the shortest exact digits."""
    for digits in range(1, 10):
        short = float(f"{value:.{digits}g}")
        if f32(short) == value:
            return short
    return value


def _analysis(mappings, settings):
    """What indexing needs from an index's mapping and settings.

    ``text``: document paths for the catch-all column (``None`` for all).
    ``edge``: paths whose analyzer emits edge n-grams, with ``(min, max)``.
    """
    default = settings.get("query.default_field")
    fields = Fields([mappings])
    text = None
    if default and default != "*" and default != ["*"]:
        text = {fields.resolve(f)[0] for f in _listify(default)}
    grams = _edge_analyzers(settings)
    edge = {}

    def walk(props, prefix):
        for name, spec in (props or {}).items():
            path = prefix + name
            if spec.get("type") == "search_as_you_type":
                continue  # its prefixes come from bool_prefix/phrase_prefix
            # A multi-field resolves to its parent path, so its n-grams go there.
            for field in [spec, *(spec.get("fields") or {}).values()]:
                if field.get("analyzer") in grams:
                    edge[path] = grams[field["analyzer"]]
            walk(spec.get("properties"), path + ".")

    walk((mappings or {}).get("properties"), "")
    return {"text": text, "edge": edge}


def _edge_analyzers(settings):
    """``{analyzer: (min_gram, max_gram)}`` for analyzers that edge n-gram."""
    out = {}
    for key, value in settings.items():
        m = re.fullmatch(r"analysis\.analyzer\.([^.]+)\.(filter|tokenizer)", key)
        if not m:
            continue
        for part in _listify(value):
            base = f"analysis.{m.group(2)}.{part}."
            if settings.get(base + "type", part) in ("edge_ngram", "edgeNGram"):
                low = int(settings.get(base + "min_gram", 1))
                out[m.group(1)] = (low, int(settings.get(base + "max_gram", 2)))
    return out


def _grams(value, span):
    """Folded ``value`` followed by the edge n-grams of each of its words.

    pg_search tokenises the result into words, so a query word matches a
    word beginning, as on an OpenSearch edge n-gram field.
    """
    low, high = span
    words = re.findall(r"\w+", value.lower())
    grams = {w[:n] for w in words for n in range(low, min(high, len(w)) + 1)}
    return " ".join([value, *sorted(grams - set(words))])


def _analysed(doc, edge, prefix=""):
    """``doc`` as pg_search indexes it: strings folded, n-grams added."""
    if isinstance(doc, dict):
        return {
            k: _analysed(v, edge, f"{prefix}.{k}" if prefix else k)
            for k, v in doc.items()
        }
    if isinstance(doc, list):
        return [_analysed(v, edge, prefix) for v in doc]
    if isinstance(doc, str):
        value = fold(doc)
        return _grams(value, edge[prefix]) if prefix in edge else value
    return doc


def _text(doc, spec, edge):
    """The catch-all text that unfielded queries search, folded."""
    values = []
    for path, value in leaves(doc):
        if isinstance(value, bool) or value is None:
            continue
        if (
            spec is None
            or path in spec
            or any(fnmatch.fnmatchcase(path, p) for p in spec if "*" in p)
        ):
            value = fold(str(value))
            values.append(_grams(value, edge[path]) if path in edge else value)
    return "\n".join(values)
