# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""SQL conditions for filter clauses, and aggregations over matching documents.

:class:`Translator` turns filter clauses (``term`` on numbers and dates,
``range``, ``exists``, ``ids``, ``nested``, ...) into exact SQL/JSON path
conditions on the stored documents. :class:`Aggregations` computes OpenSearch
aggregations from the field values of the matched documents. Anything not
handled raises :class:`Unsupported` instead of being ignored.

SQL fragments are ``(sql, params)`` tuples with psycopg2 ``%s`` placeholders;
the documents table is aliased ``d``.
"""

import calendar
import datetime as dt
import json
import re
from fnmatch import fnmatchcase

from . import querystring
from .hashing import murmur3_32

TEXT_TYPES = {"text", "search_as_you_type", "match_only_text", "completion"}
NUMERIC_TYPES = {
    "long",
    "integer",
    "short",
    "byte",
    "double",
    "float",
    "half_float",
    "scaled_float",
    "unsigned_long",
}
DATE_TYPES = {"date", "date_nanos"}

TRUE = ("TRUE", [])
FALSE = ("FALSE", [])
ROOT = ("d.doc", "", 0)
EPOCH = dt.datetime(1970, 1, 1)


class Unsupported(Exception):
    """The request uses a feature the Postgres backend does not implement."""


def jsonpath(path):
    """Build a lax JSON path for a dotted field path."""
    if not path:
        return "$"
    segments = (s.replace("\\", "\\\\").replace('"', '\\"') for s in path.split("."))
    return "$" + "".join(f'."{s}"' for s in segments)


def each_value(path):
    """JSON path to every value at a dotted path, one per array element."""
    return jsonpath(path) + "[*]"


def _literal(text):
    """Quote text as a JSON path string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _join(parts, op):
    parts = list(parts)
    if not parts:
        return TRUE if op == "AND" else FALSE
    if len(parts) == 1:
        return parts[0]
    return (
        "(" + f" {op} ".join(p[0] for p in parts) + ")",
        [x for p in parts for x in p[1]],
    )


def _not(part):
    return (f"NOT COALESCE({part[0]}, FALSE)", part[1])


def _listify(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _tokens(text):
    return re.findall(r"\w+", str(text).lower())


def _wild(value):
    """Turn a query string wildcard value into a regular expression."""
    out = []
    for ch in value:
        if ch == "\0":
            out.append(".*")
        elif ch == "\1":
            out.append(".")
        elif ch.isalnum() or ch == "_" or ch.isspace():
            out.append(ch)
        else:
            out.append("\\" + ch)
    return "".join(out)


def _escape_re(value):
    return "".join(c if c.isalnum() or c in "_ " else "\\" + c for c in value)


def _variants(value):
    """Values that OpenSearch treats as equal to ``value`` in a term query."""
    out = [value]
    if isinstance(value, bool):
        out.append("true" if value else "false")
    elif isinstance(value, (int, float)):
        out.append(str(value))
        if isinstance(value, float) and value.is_integer():
            out += [int(value), str(int(value))]
    elif isinstance(value, str):
        if value in ("true", "false"):
            out.append(value == "true")
        else:
            try:
                out.append(int(value))
            except ValueError:
                try:
                    out.append(float(value))
                except ValueError:
                    pass
    return out


def _distance(fuzziness, text):
    """Edit distance for a fuzziness setting (``AUTO`` uses the longest word)."""
    if fuzziness in (None, 0, "0"):
        return 0
    spec = str(fuzziness).upper()
    if spec.startswith("AUTO"):
        low, high = 3, 6
        if ":" in spec:
            low, high = (int(x) for x in spec[5:].split(","))
        n = max((len(w) for w in _tokens(text)), default=0)
        return 0 if n < low else 1 if n < high else 2
    return min(int(float(spec)), 2)


def _msm(value, count, default):
    """Resolve ``minimum_should_match`` for ``count`` optional clauses."""
    if value is None:
        return default
    spec = str(value).strip()
    if "<" in spec:
        result = count
        for cond in spec.split():
            limit, inner = cond.split("<", 1)
            if count > int(limit):
                result = _msm(inner, count, default)
        return result
    if spec.endswith("%"):
        pct = float(spec[:-1])
        n = int(count * abs(pct) / 100)
        return count - n if pct < 0 else n
    n = int(spec)
    return count + n if n < 0 else n


# Dates
# =====

_DATE_SQL = (
    "(CASE WHEN length({s}) >= 19 THEN replace(left({s}, 19), ' ', 'T') "
    "ELSE {s} || substr('0000-01-01T00:00:00', length({s}) + 1) END)"
)


def _pad(text):
    """Normalise a date string to ``YYYY-MM-DDTHH:MM:SS`` for comparison."""
    text = text.strip()
    if len(text) >= 19:
        return text[:19].replace(" ", "T")
    return text + "0000-01-01T00:00:00"[len(text) :]


def parse_date(value):
    """Parse a date value from a document, or return ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return EPOCH + dt.timedelta(milliseconds=value)
    text = str(value).strip()
    try:
        d = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        if d.tzinfo is not None:
            d = d.astimezone(dt.timezone.utc).replace(tzinfo=None)
        # OpenSearch keeps dates as whole milliseconds.
        return d.replace(microsecond=d.microsecond // 1000 * 1000)
    except ValueError:
        pass
    try:
        return dt.datetime.fromisoformat(_pad(text))
    except ValueError:
        return None


def _add_months(d, months):
    year, month = divmod(d.month - 1 + months, 12)
    year += d.year
    day = min(d.day, calendar.monthrange(year, month + 1)[1])
    return d.replace(year=year, month=month + 1, day=day)


def floor_date(d, unit):
    """Round a datetime down to the start of a calendar unit."""
    d = d.replace(microsecond=0)
    if unit == "y":
        return d.replace(month=1, day=1, hour=0, minute=0, second=0)
    if unit == "q":
        return d.replace(
            month=(d.month - 1) // 3 * 3 + 1, day=1, hour=0, minute=0, second=0
        )
    if unit == "M":
        return d.replace(day=1, hour=0, minute=0, second=0)
    if unit == "w":
        d -= dt.timedelta(days=d.weekday())
        return d.replace(hour=0, minute=0, second=0)
    if unit == "d":
        return d.replace(hour=0, minute=0, second=0)
    if unit == "h":
        return d.replace(minute=0, second=0)
    if unit == "m":
        return d.replace(second=0)
    return d


def step_date(d, unit, n=1):
    """Move a datetime by ``n`` calendar units."""
    if unit == "y":
        return _add_months(d, 12 * n)
    if unit == "q":
        return _add_months(d, 3 * n)
    if unit == "M":
        return _add_months(d, n)
    seconds = {"w": 604800, "d": 86400, "h": 3600, "H": 3600, "m": 60, "s": 1}[unit]
    return d + dt.timedelta(seconds=seconds * n)


def date_bound(value, round_up):
    """Resolve a range bound, including date math such as ``now-1d/d``."""
    text = str(value)
    if text.startswith("now") or "||" in text:
        if "||" in text:
            base, math = text.split("||", 1)
            d = parse_date(base) or dt.datetime.utcnow()
        else:
            d, math = dt.datetime.utcnow(), text[3:]
        for m in re.finditer(r"([+-])(\d+)([yMwdhHms])|/([yMwdhHms])", math):
            if m.group(4):
                unit = "h" if m.group(4) == "H" else m.group(4)
                d = floor_date(d, unit)
                if round_up:
                    d = step_date(d, unit) - dt.timedelta(seconds=1)
            else:
                n = int(m.group(2)) * (1 if m.group(1) == "+" else -1)
                d = step_date(d, m.group(3), n)
        return d.strftime("%Y-%m-%dT%H:%M:%S")
    return _pad(text)


# Fields
# ======


class Fields:
    """Field types from index mappings, with multi-fields resolved to their parent."""

    def __init__(self, mappings):
        """Constructor."""
        self.map = {}
        for mapping in mappings:
            self._walk((mapping or {}).get("properties") or {}, "")

    def _walk(self, props, prefix):
        for name, spec in props.items():
            path = prefix + name
            kind = spec.get("type") or ("object" if "properties" in spec else None)
            self.map.setdefault(path, (path, kind))
            for sub, sub_spec in (spec.get("fields") or {}).items():
                self.map.setdefault(f"{path}.{sub}", (path, sub_spec.get("type")))
            if "properties" in spec:
                self._walk(spec["properties"], path + ".")

    def resolve(self, field):
        """Return ``(document path, mapping type)`` for a field name."""
        field = field.split("^")[0]
        return self.map.get(field, (field, None))


# Queries
# =======


class Translator:
    """Translate one search request's queries into SQL conditions."""

    def __init__(self, fields, table, percolate=None):
        """Constructor.

        :param fields: :class:`Fields` of the searched indices.
        :param table: name of the documents table.
        :param percolate: callable ``(field, documents) -> [pk]`` that runs
            stored percolator queries, or ``None`` when not supported.
        """
        self.fields = fields
        self.table = table
        self.percolate = percolate
        self.scoring = []

    def where(self, query, ctx=ROOT, scoring=True):
        """Translate a query clause into an SQL condition."""
        if not query:
            return TRUE
        if not isinstance(query, dict) or len(query) != 1:
            raise Unsupported(f"malformed query {json.dumps(query)[:80]}")
        ((kind, body),) = query.items()
        method = getattr(self, "_q_" + kind, None)
        if method is None:
            raise Unsupported(f"query [{kind}]")
        return method(body, ctx, scoring)

    @staticmethod
    def _rel(ctx, path):
        prefix = ctx[1]
        if prefix and path.startswith(prefix + "."):
            return path[len(prefix) + 1 :]
        return path

    @staticmethod
    def _single(body, name):
        items = [(k, v) for k, v in body.items() if k not in ("boost", "_name")]
        if len(items) != 1:
            raise Unsupported(f"[{name}] needs exactly one field")
        return items[0]

    def _bm25(self, expr, params, ctx, scoring):
        if ctx[2]:
            raise Unsupported("full-text query inside a nested query")
        if scoring:
            self.scoring.append((expr, params))
        return (f"d.pk IN (SELECT pk FROM {self.table} WHERE pk @@@ {expr})", params)

    def _exact(self, ctx, path, kind, values):
        if path in ("_id", "_index"):
            if ctx[2]:
                raise Unsupported(f"[{path}] inside a nested query")
            column = "d.id" if path == "_id" else "d.idx"
            return (f"{column} = ANY(%s::text[])", [[str(v) for v in values]])
        if kind in TEXT_TYPES and not ctx[2]:
            return _join(
                (
                    self._bm25(
                        "paradedb.term(%s, %s)", ["doc." + path, str(v)], ctx, False
                    )
                    for v in values
                ),
                "OR",
            )
        variants, seen = [], set()
        for value in values:
            for v in _variants(value):
                key = (type(v).__name__, v)
                if key not in seen:
                    seen.add(key)
                    variants.append(v)
        if not variants:
            return FALSE
        cond = " || ".join(f"@ == $v{i}" for i in range(len(variants)))
        return (
            f"jsonb_path_exists({ctx[0]}, %s::jsonpath, %s::jsonb)",
            [
                f"{jsonpath(self._rel(ctx, path))} ? ({cond})",
                json.dumps({f"v{i}": v for i, v in enumerate(variants)}),
            ],
        )

    def _like(self, ctx, path, pattern, ci):
        flag = ' flag "i"' if ci else ""
        return (
            f"jsonb_path_exists({ctx[0]}, %s::jsonpath)",
            [
                f"{jsonpath(self._rel(ctx, path))} ? "
                f"(@ like_regex {_literal(pattern)}{flag})"
            ],
        )

    def _words(self, ctx, path, words, conjunction):
        parts = [self._like(ctx, path, f"\\y{_escape_re(w)}\\y", True) for w in words]
        return _join(parts, "AND" if conjunction else "OR")

    def _text_match(self, ctx, column, path, text, conjunction, scoring, fuzzy=None):
        """Analysed match on the catch-all column (``path=None``) or a field."""
        if not ctx[2]:
            distance = _distance(fuzzy, text)
            return self._bm25(
                "paradedb.match(%s, %s, conjunction_mode => %s, distance => %s)",
                [column, str(text), conjunction, distance],
                ctx,
                scoring,
            )
        return self._words(ctx, path, _tokens(text), conjunction)

    def _prefix(self, ctx, column, path, kind, prefix, scoring):
        if kind in TEXT_TYPES or path is None:
            if not ctx[2]:
                return self._bm25(
                    "paradedb.phrase_prefix(%s, %s)",
                    [column, [prefix.lower()]],
                    ctx,
                    scoring,
                )
            return self._like(ctx, path, f"\\y{_escape_re(prefix)}", True)
        return self._like(ctx, path, "^" + _escape_re(prefix), False)

    def _phrase(self, ctx, column, path, text, scoring):
        if not ctx[2]:
            words = " ".join(_tokens(text))
            return self._bm25(
                "paradedb.parse(%s, lenient => true)",
                [f'{column}:"{words}"'],
                ctx,
                scoring,
            )
        return self._like(
            ctx, path, "\\y" + "\\W+".join(map(_escape_re, _tokens(text))) + "\\y", True
        )

    def _target(self, field):
        """Return ``(paradedb column, document path, type)`` for a text target."""
        if field in (None, "*", "_all"):
            return "text", None, "text"
        path, kind = self.fields.resolve(field)
        return "doc." + path, path, kind

    # Leaf and compound clauses

    def _q_match_all(self, body, ctx, scoring):
        return TRUE

    def _q_match_none(self, body, ctx, scoring):
        return FALSE

    def _q_bool(self, body, ctx, scoring):
        must = _listify(body.get("must"))
        filters = _listify(body.get("filter"))
        should = _listify(body.get("should"))
        parts = [self.where(q, ctx, scoring) for q in must]
        parts += [self.where(q, ctx, False) for q in filters]
        parts += [
            _not(self.where(q, ctx, False)) for q in _listify(body.get("must_not"))
        ]
        if should:
            clauses = [self.where(q, ctx, scoring) for q in should]
            default = 0 if (must or filters) else 1
            n = _msm(body.get("minimum_should_match"), len(clauses), default)
            if n > len(clauses):
                return FALSE
            if n == 1:
                parts.append(_join(clauses, "OR"))
            elif n > 1:
                parts.append(
                    (
                        "("
                        + " + ".join(f"({c[0]})::int" for c in clauses)
                        + f") >= {n}",
                        [x for c in clauses for x in c[1]],
                    )
                )
        return _join(parts, "AND")

    def _q_constant_score(self, body, ctx, scoring):
        return self.where(body.get("filter"), ctx, False)

    def _q_dis_max(self, body, ctx, scoring):
        return _join((self.where(q, ctx, scoring) for q in body["queries"]), "OR")

    def _q_term(self, body, ctx, scoring):
        field, value = self._single(body, "term")
        if isinstance(value, dict):
            value = value["value"]
        return self._q_terms({field: [value]}, ctx, scoring)

    def _q_terms(self, body, ctx, scoring):
        field, values = self._single(body, "terms")
        if not isinstance(values, list):
            raise Unsupported("[terms] lookup")
        path, kind = self.fields.resolve(field)
        if kind in DATE_TYPES:
            # A date term matches the whole span it denotes, e.g. ``...||/d``.
            return _join(
                (
                    self._q_range({field: {"gte": v, "lte": v}}, ctx, scoring)
                    for v in values
                ),
                "OR",
            )
        return self._exact(ctx, path, kind, values)

    def _q_ids(self, body, ctx, scoring):
        return self._exact(ctx, "_id", None, _listify(body.get("values")))

    def _q_exists(self, body, ctx, scoring):
        path, _ = self.fields.resolve(body["field"])
        return (
            f"jsonb_path_exists({ctx[0]}, %s::jsonpath)",
            [f"{jsonpath(self._rel(ctx, path))} ? (@ != null)"],
        )

    def _q_range(self, body, ctx, scoring):
        field, spec = self._single(body, "range")
        path, kind = self.fields.resolve(field)
        lower = [
            (spec[k], op)
            for k, op in (("gte", ">="), ("gt", ">"))
            if spec.get(k) is not None
        ]
        upper = [
            (spec[k], op)
            for k, op in (("lte", "<="), ("lt", "<"))
            if spec.get(k) is not None
        ]
        if spec.get("from") is not None:
            lower.append(
                (spec["from"], ">=" if spec.get("include_lower", True) else ">")
            )
        if spec.get("to") is not None:
            upper.append((spec["to"], "<=" if spec.get("include_upper", True) else "<"))
        is_range = bool(kind and kind.endswith("_range"))
        base = kind[: -len("_range")] if is_range else kind
        bounds = [b for b, _ in lower + upper]
        if base in NUMERIC_TYPES or base in ("integer", "float", "double", "long"):
            mode = "num"
        elif base in DATE_TYPES:
            mode = "date"
        elif base in ("keyword", "text", "ip"):
            mode = "str"
        elif bounds and all(
            isinstance(b, (int, float)) and not isinstance(b, bool) for b in bounds
        ):
            mode = "num"
        else:
            mode = "date"

        def value(sql):
            if mode == "num":
                return f"({sql})::numeric"
            if mode == "date":
                return _DATE_SQL.format(s=sql)
            return sql

        def bound(b, op):
            if mode == "num":
                return float(b)
            if mode == "date":
                return date_bound(b, round_up=op in ("<=", ">"))
            return str(b)

        conds, params = [], []
        if is_range:
            low = value("COALESCE(v->>'gte', v->>'gt')")
            high = value("COALESCE(v->>'lte', v->>'lt')")
            guard = "jsonb_typeof(v) = 'object'"
            for b, op in lower:
                conds.append(f"({high} IS NULL OR {high} {op} %s)")
                params.append(bound(b, op))
            for b, op in upper:
                conds.append(f"({low} IS NULL OR {low} {op} %s)")
                params.append(bound(b, op))
        else:
            guard = (
                "jsonb_typeof(v) = 'number'"
                if mode == "num"
                else ("jsonb_typeof(v) = 'string'")
            )
            scalar = "v" if mode == "num" else "(v #>> '{}')"
            for b, op in lower + upper:
                conds.append(f"{value(scalar)} {op} %s")
                params.append(bound(b, op))
        cond = " AND ".join([guard] + conds)
        return (
            f"EXISTS (SELECT 1 FROM jsonb_path_query({ctx[0]}, %s::jsonpath) v "
            f"WHERE {cond})",
            [each_value(self._rel(ctx, path))] + params,
        )

    def _q_nested(self, body, ctx, scoring):
        path, _ = self.fields.resolve(body["path"])
        alias = f"e{ctx[2] + 1}"
        inner = self.where(body["query"], (alias, path, ctx[2] + 1), False)
        return (
            f"EXISTS (SELECT 1 FROM jsonb_path_query({ctx[0]}, %s::jsonpath) {alias} "
            f"WHERE {inner[0]})",
            [each_value(self._rel(ctx, path))] + inner[1],
        )

    # Full-text clauses

    def _q_match(self, body, ctx, scoring):
        field, opts = self._single(body, "match")
        opts = opts if isinstance(opts, dict) else {"query": opts}
        column, path, kind = self._target(field)
        if path is not None and kind not in TEXT_TYPES and kind is not None:
            return self._exact(ctx, path, kind, [opts["query"]])
        conjunction = str(opts.get("operator", "or")).lower() == "and"
        return self._text_match(
            ctx,
            column,
            path,
            opts["query"],
            conjunction,
            scoring,
            opts.get("fuzziness"),
        )

    def _q_match_phrase(self, body, ctx, scoring):
        field, opts = self._single(body, "match_phrase")
        opts = opts if isinstance(opts, dict) else {"query": opts}
        column, path, kind = self._target(field)
        if path is not None and kind not in TEXT_TYPES and kind is not None:
            return self._exact(ctx, path, kind, [opts["query"]])
        return self._phrase(ctx, column, path, opts["query"], scoring)

    def _q_match_phrase_prefix(self, body, ctx, scoring):
        field, opts = self._single(body, "match_phrase_prefix")
        opts = opts if isinstance(opts, dict) else {"query": opts}
        column, path, kind = self._target(field)
        words = _tokens(opts["query"])
        if not ctx[2] and words:
            return self._bm25(
                "paradedb.phrase_prefix(%s, %s)", [column, words], ctx, scoring
            )
        return self._prefix(ctx, column, path, kind, " ".join(words), scoring)

    def _q_prefix(self, body, ctx, scoring):
        field, value = self._single(body, "prefix")
        if isinstance(value, dict):
            value = value["value"]
        column, path, kind = self._target(field)
        return self._prefix(ctx, column, path, kind, str(value), scoring)

    def _q_wildcard(self, body, ctx, scoring):
        field, value = self._single(body, "wildcard")
        if isinstance(value, dict):
            value = value.get("value", value.get("wildcard"))
        value = str(value).replace("*", "\0").replace("?", "\1")
        return self._qs_term(field, value, True, ctx, None, scoring)

    def _q_multi_match(self, body, ctx, scoring):
        fields = body.get("fields") or [None]
        query = body["query"]
        kind = body.get("type", "best_fields")
        conjunction = str(body.get("operator", "or")).lower() == "and"
        fuzzy = body.get("fuzziness")
        targets = [self._target(f) for f in fields]
        if kind in ("phrase", "phrase_prefix"):
            parts = []
            for column, path, _ in targets:
                if kind == "phrase":
                    parts.append(self._phrase(ctx, column, path, query, scoring))
                elif not ctx[2]:
                    parts.append(
                        self._bm25(
                            "paradedb.phrase_prefix(%s, %s)",
                            [column, _tokens(query)],
                            ctx,
                            scoring,
                        )
                    )
            return _join(parts, "OR")
        if kind == "bool_prefix":
            words = _tokens(query)
            if not words:
                return FALSE
            per_word = []
            for i, word in enumerate(words):
                last = i == len(words) - 1
                alternatives = []
                for column, path, ftype in targets:
                    if last:
                        alternatives.append(
                            self._prefix(
                                ctx, column, path, ftype or "text", word, scoring
                            )
                        )
                    else:
                        alternatives.append(
                            self._text_match(ctx, column, path, word, False, scoring)
                        )
                per_word.append(_join(alternatives, "OR"))
            return _join(per_word, "AND" if conjunction else "OR")
        parts = []
        for column, path, ftype in targets:
            if path is not None and ftype not in TEXT_TYPES and ftype is not None:
                parts.append(self._exact(ctx, path, ftype, [query]))
            else:
                parts.append(
                    self._text_match(
                        ctx, column, path, query, conjunction, scoring, fuzzy
                    )
                )
        return _join(parts, "OR")

    def _q_query_string(self, body, ctx, scoring):
        fields = body.get("fields") or body.get("default_field")
        if isinstance(fields, str):
            fields = [fields]
        tree = querystring.parse(body["query"], body.get("default_operator", "OR"))
        return self._qs(tree, ctx, fields, scoring)

    _q_simple_query_string = _q_query_string

    def _qs(self, node, ctx, fields, scoring):
        kind = node[0]
        if kind == "all":
            return TRUE
        if kind == "bool":
            _, must, should, must_not = node
            parts = [self._qs(n, ctx, fields, scoring) for n in must]
            parts += [_not(self._qs(n, ctx, fields, False)) for n in must_not]
            if should:
                clauses = [self._qs(n, ctx, fields, scoring) for n in should]
                if not must:
                    parts.append(_join(clauses, "OR"))
            return _join(parts, "AND") if (parts or not should) else FALSE
        if kind == "term":
            _, field, value, wildcard = node
            targets = [field] if field else (fields or [None])
            return _join(
                (
                    self._qs_term(t, value, wildcard, ctx, fields, scoring)
                    for t in targets
                ),
                "OR",
            )
        if kind == "phrase":
            _, field, value = node
            parts = []
            for target in [field] if field else (fields or [None]):
                column, path, ftype = self._target(target)
                if path is not None and ftype not in TEXT_TYPES and ftype is not None:
                    parts.append(self._exact(ctx, path, ftype, [value]))
                else:
                    parts.append(self._phrase(ctx, column, path, value, scoring))
            return _join(parts, "OR")
        if kind == "range":
            _, field, low, high, include_low, include_high = node
            if not field:
                return FALSE
            spec = {}
            if low is not None:
                spec["gte" if include_low else "gt"] = low
            if high is not None:
                spec["lte" if include_high else "lt"] = high
            return self._q_range({field: spec}, ctx, scoring)
        if kind == "exists":
            return self._q_exists({"field": node[1]}, ctx, scoring)
        if kind == "boost":
            # Boosts only change scores, which are approximate on this backend.
            return self._qs(node[1], ctx, fields, scoring)
        if kind in ("text", "fuzzy", "regexp"):
            field = node[1]
            targets = [field] if field else (fields or [None])
            return _join(
                (self._qs_other(kind, t, node, ctx, scoring) for t in targets), "OR"
            )
        raise Unsupported(f"query string node [{kind}]")

    def _qs_other(self, kind, target, node, ctx, scoring):
        column, path, ftype = self._target(target)
        text_like = path is None or ftype in TEXT_TYPES or ftype is None
        if kind == "text":
            if text_like:
                return self._text_match(ctx, column, path, node[2], False, scoring)
            return self._exact(ctx, path, ftype, [node[2]])
        if kind == "fuzzy":
            if text_like and not ctx[2]:
                return self._bm25(
                    "paradedb.fuzzy_term(%s, %s, distance => %s)",
                    [column, str(node[2]).lower(), _distance(node[3], node[2])],
                    ctx,
                    scoring,
                )
            return self._exact(ctx, path, ftype, [node[2]])
        if text_like and not ctx[2]:
            return self._bm25(
                "paradedb.regex(%s, %s)", [column, str(node[2]).lower()], ctx, scoring
            )
        return self._like(ctx, path, f"^({node[2]})$", False)

    def _qs_term(self, target, value, wildcard, ctx, fields, scoring):
        column, path, kind = self._target(target)
        text_like = path is None or kind in TEXT_TYPES or kind is None
        if wildcard:
            prefix_only = (
                value.endswith("\0") and "\0" not in value[:-1] and "\1" not in value
            )
            if text_like:
                if prefix_only and value[:-1]:
                    return self._prefix(ctx, column, path, "text", value[:-1], scoring)
                if not ctx[2] and path is None:
                    return self._bm25(
                        "paradedb.regex(%s, %s)",
                        [column, _wild(value).lower()],
                        ctx,
                        scoring,
                    )
                return self._like(ctx, path, f"\\y{_wild(value)}\\y", True)
            return self._like(ctx, path, f"^{_wild(value)}$", False)
        if text_like:
            return self._text_match(ctx, column, path, value, False, scoring)
        return self._exact(ctx, path, kind, [value])

    def _q_percolate(self, body, ctx, scoring):
        if self.percolate is None or ctx[2]:
            raise Unsupported("[percolate] in this context")
        documents = body.get("documents")
        if documents is None and "document" in body:
            documents = [body["document"]]
        if documents is None:
            raise Unsupported("[percolate] of indexed documents")
        pks = self.percolate(body.get("field", "query"), documents)
        return ("d.pk = ANY(%s::bigint[])", [pks])


# Documents
# =========


def leaves(node, prefix=""):
    """Yield ``(dotted path, value)`` for every scalar in a document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for value in node:
            yield from leaves(value, prefix)
    else:
        yield prefix, node


def values_at(doc, path):
    """Return all values at a dotted path, flattening arrays."""
    current = [doc]
    for segment in path.split("."):
        nxt = []
        for node in current:
            value = node.get(segment) if isinstance(node, dict) else None
            if isinstance(value, list):
                nxt.extend(value)
            elif value is not None:
                nxt.append(value)
        current = nxt
    return current


def _source_match(path, patterns):
    return any(path == p or fnmatchcase(path, p) for p in patterns)


def _source_partial(path, patterns):
    return any(p.startswith(path + ".") or "*" in p for p in patterns)


def _prune(node, prefix, includes, excludes):
    if isinstance(node, list):
        return [_prune(v, prefix, includes, excludes) for v in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if _source_match(path, excludes):
            continue
        if not includes or _source_match(path, includes):
            out[key] = _prune(value, path, [], excludes)
        elif _source_partial(path, includes) and isinstance(value, (dict, list)):
            sub = _prune(value, path, includes, excludes)
            if sub:
                out[key] = sub
    return out


def filter_source(source, spec):
    """Apply ``_source`` filtering (``True``, ``False``, list or includes/excludes)."""
    if spec is None or spec is True:
        return source
    if spec is False:
        return None
    if isinstance(spec, str):
        includes, excludes = [spec], []
    elif isinstance(spec, list):
        includes, excludes = spec, []
    else:
        includes = _listify(spec.get("includes") or spec.get("include"))
        excludes = _listify(spec.get("excludes") or spec.get("exclude"))
    return _prune(source, "", includes, excludes)


# Aggregations
# ============

_UNITS = {
    "year": "y", "1y": "y", "y": "y",
    "quarter": "q", "1q": "q", "q": "q",
    "month": "M", "1M": "M", "M": "M",
    "week": "w", "1w": "w", "w": "w",
    "day": "d", "1d": "d", "d": "d", "24h": "d",
    "hour": "h", "1h": "h", "h": "h",
    "minute": "m", "1m": "m", "m": "m",
}  # fmt: skip
_METRICS = {"sum", "min", "max", "avg", "value_count", "cardinality", "stats"}


def _split(agg):
    sub = agg.get("aggs") or agg.get("aggregations") or {}
    kinds = [k for k in agg if k not in ("aggs", "aggregations", "meta")]
    if len(kinds) != 1:
        raise Unsupported("aggregation needs exactly one type")
    return kinds[0], agg[kinds[0]], sub


def _java_format(d, fmt):
    millis = f"{d.microsecond // 1000:03d}"
    if not fmt:
        return d.strftime("%Y-%m-%dT%H:%M:%S.") + millis + "Z"
    for java, py in (
        ("yyyy", "%Y"),
        ("MM", "%m"),
        ("dd", "%d"),
        ("HH", "%H"),
        ("mm", "%M"),
        ("ss", "%S"),
        ("SSS", millis),
    ):
        fmt = fmt.replace(java, py)
    return d.strftime(fmt.replace("'", ""))


def term_partition(value, partitions):
    """Partition of a term in a ``terms`` aggregation, as OpenSearch hashes it."""
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        mask = (1 << 64) - 1
        z = value & mask
        z = ((z ^ (z >> 32)) * 0x4CD6944C5CC20B6D) & mask
        z = ((z ^ (z >> 29)) * 0xFC12C5B19D3259E9) & mask
        z ^= z >> 32
        return (z - (1 << 64) if z >> 63 else z) % partitions
    return murmur3_32(str(value).encode("utf-8"), 31) % partitions


def _sort_key(value):
    return (type(value).__name__, value)


class Aggregations:
    """Compute aggregations from the field values of the matched documents.

    Field values come from one SQL query over the matching rows; bucketing
    happens in Python, which suits the small instances this backend targets.
    """

    def __init__(self, fields, spec, match):
        """Constructor.

        :param match: callable returning the set of pks matching a query.
        """
        self.fields = fields
        self.match = match
        self.spec = spec
        self.columns = {}
        self.filters = {}
        self.need_doc = False
        self._collect(spec)

    def _field(self, field):
        path, _ = self.fields.resolve(field)
        self.columns.setdefault(
            "f:" + path,
            ("jsonb_path_query_array(d.doc, %s::jsonpath)", [each_value(path)]),
        )
        return "f:" + path

    def _collect(self, spec):
        for agg in (spec or {}).values():
            kind, body, sub = _split(agg)
            if kind in _METRICS or kind in ("terms", "date_histogram"):
                if "field" not in body:
                    raise Unsupported(f"[{kind}] aggregation without a field")
                self._field(body["field"])
            elif kind == "filter":
                self.filters[id(body)] = body
            elif kind == "filters":
                for name, query in body["filters"].items():
                    self.filters[id(query)] = query
            elif kind == "top_hits":
                self.need_doc = True
            else:
                raise Unsupported(f"aggregation [{kind}]")
            self._collect(sub)

    def select(self):
        """SQL select list and parameters for the aggregation rows."""
        sql, params, keys = ["d.pk", "d.idx", "d.id"], [], ["_pk", "_index", "_id"]
        if self.need_doc:
            sql.append("d.doc")
            keys.append("_source")
        for key, (expr, p) in self.columns.items():
            sql.append(expr)
            params += p
            keys.append(key)
        return ", ".join(sql), params, keys

    def compute(self, rows, spec=None):
        """Build the ``aggregations`` response for the given rows."""
        if spec is None:
            for key, query in self.filters.items():
                pks = self.match(query)
                for row in rows:
                    row[key] = row["_pk"] in pks
        out = {}
        for name, agg in (self.spec if spec is None else spec).items():
            kind, body, sub = _split(agg)
            result = getattr(self, "_" + kind)(body, rows, sub)
            if "meta" in agg:
                result["meta"] = agg["meta"]
            out[name] = result
        return out

    def _values(self, row, field):
        return row["f:" + self.fields.resolve(field)[0]] or []

    def _terms(self, body, rows, sub):
        members = {}
        for row in rows:
            seen = set()
            for value in self._values(row, body["field"]):
                if value is None or isinstance(value, (dict, list)):
                    continue
                key = _sort_key(value)
                if key not in seen:
                    seen.add(key)
                    members.setdefault(key, (value, []))[1].append(row)
        include = body.get("include")
        exclude = body.get("exclude")
        partition = None
        if isinstance(include, dict):
            partition = (int(include["partition"]), int(include["num_partitions"]))
            include = None
        items = [
            v
            for v in members.values()
            if (not isinstance(include, list) or v[0] in include)
            and (not isinstance(exclude, list) or v[0] not in exclude)
            and (
                partition is None or term_partition(v[0], partition[1]) == partition[0]
            )
        ]
        if isinstance(include, str) or isinstance(exclude, str):
            raise Unsupported("[terms] include/exclude patterns")
        order = body.get("order") or {"_count": "desc"}
        order = order[0] if isinstance(order, list) else order
        ((order_by, direction),) = order.items()
        if order_by in ("_key", "_term"):
            items.sort(key=lambda v: _sort_key(v[0]), reverse=direction == "desc")
        elif order_by == "_count":
            items.sort(key=lambda v: _sort_key(v[0]))
            items.sort(key=lambda v: len(v[1]), reverse=direction != "asc")
        else:
            raise Unsupported(f"[terms] order by [{order_by}]")
        min_count = body.get("min_doc_count", 1)
        items = [v for v in items if len(v[1]) >= min_count]
        size = body.get("size", 10)
        shown = items[:size]
        buckets = []
        for value, bucket_rows in shown:
            bucket = {"key": value, "doc_count": len(bucket_rows)}
            if isinstance(value, bool):
                bucket = {
                    "key": int(value),
                    "key_as_string": "true" if value else "false",
                    "doc_count": len(bucket_rows),
                }
            bucket.update(self.compute(bucket_rows, sub))
            buckets.append(bucket)
        return {
            "doc_count_error_upper_bound": 0,
            "sum_other_doc_count": sum(len(v[1]) for v in items[size:]),
            "buckets": buckets,
        }

    def _date_histogram(self, body, rows, sub):
        interval = (
            body.get("calendar_interval")
            or body.get("fixed_interval")
            or body.get("interval")
        )
        unit = _UNITS.get(str(interval))
        if unit is None:
            raise Unsupported(f"[date_histogram] interval [{interval}]")
        members = {}
        for row in rows:
            starts = set()
            for value in self._values(row, body["field"]):
                if isinstance(value, dict):
                    low = parse_date(value.get("gte") or value.get("gt"))
                    high = parse_date(value.get("lte") or value.get("lt")) or low
                    if low is None:
                        continue
                    start = floor_date(low, unit)
                    while start <= high:
                        starts.add(start)
                        start = step_date(start, unit)
                else:
                    d = parse_date(value)
                    if d is not None:
                        starts.add(floor_date(d, unit))
            for start in starts:
                members.setdefault(start, []).append(row)
        hard = body.get("hard_bounds") or {}
        low = (
            floor_date(parse_date(date_bound(hard["min"], False)), unit)
            if hard.get("min") is not None
            else None
        )
        high = (
            parse_date(date_bound(hard["max"], True))
            if hard.get("max") is not None
            else None
        )
        keys = sorted(
            k
            for k in members
            if (low is None or k >= low) and (high is None or k <= high)
        )
        min_count = body.get("min_doc_count", 0)
        if min_count == 0 and keys:
            extended = body.get("extended_bounds") or {}
            first, last = keys[0], keys[-1]
            if extended.get("min") is not None:
                first = min(
                    first,
                    floor_date(parse_date(date_bound(extended["min"], False)), unit),
                )
            if extended.get("max") is not None:
                last = max(
                    last,
                    floor_date(parse_date(date_bound(extended["max"], True)), unit),
                )
            if low is not None:
                first = max(first, low)
            if high is not None:
                last = min(last, floor_date(high, unit))
            keys, start = [], first
            while start <= last:
                keys.append(start)
                start = step_date(start, unit)
        buckets = []
        for key in keys:
            bucket_rows = members.get(key, [])
            if len(bucket_rows) < min_count:
                continue
            bucket = {
                "key_as_string": _java_format(key, body.get("format")),
                "key": int((key - EPOCH).total_seconds() * 1000),
                "doc_count": len(bucket_rows),
            }
            bucket.update(self.compute(bucket_rows, sub))
            buckets.append(bucket)
        return {"buckets": buckets}

    def _numbers(self, body, rows):
        """Numeric values of a field; dates count as epoch milliseconds."""
        dates = self.fields.resolve(body["field"])[1] in DATE_TYPES
        out = []
        for row in rows:
            for v in self._values(row, body["field"]):
                if dates and v is not None:
                    d = parse_date(v)
                    if d is not None:
                        out.append((d - EPOCH) // dt.timedelta(milliseconds=1))
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append(v)
        return out

    def _metric(self, body, value):
        out = {"value": None if value is None else float(value)}
        if value is not None and self.fields.resolve(body["field"])[1] in DATE_TYPES:
            d = EPOCH + dt.timedelta(milliseconds=value)
            out["value_as_string"] = _java_format(d, body.get("format"))
        return out

    def _sum(self, body, rows, sub):
        return {"value": float(sum(self._numbers(body, rows)))}

    def _min(self, body, rows, sub):
        values = self._numbers(body, rows)
        return self._metric(body, min(values) if values else None)

    def _max(self, body, rows, sub):
        values = self._numbers(body, rows)
        return self._metric(body, max(values) if values else None)

    def _avg(self, body, rows, sub):
        values = self._numbers(body, rows)
        return {"value": sum(values) / len(values) if values else None}

    def _stats(self, body, rows, sub):
        values = self._numbers(body, rows)
        return {
            "count": len(values),
            "min": float(min(values)) if values else None,
            "max": float(max(values)) if values else None,
            "avg": sum(values) / len(values) if values else None,
            "sum": float(sum(values)),
        }

    def _value_count(self, body, rows, sub):
        return {"value": sum(len(self._values(r, body["field"])) for r in rows)}

    def _cardinality(self, body, rows, sub):
        values = {
            json.dumps(v, sort_keys=True)
            for r in rows
            for v in self._values(r, body["field"])
        }
        return {"value": len(values)}

    def _filter(self, body, rows, sub):
        matched = [r for r in rows if r[id(body)]]
        return {"doc_count": len(matched), **self.compute(matched, sub)}

    def _filters(self, body, rows, sub):
        buckets = {}
        for name, query in body["filters"].items():
            matched = [r for r in rows if r[id(query)]]
            buckets[name] = {"doc_count": len(matched), **self.compute(matched, sub)}
        return {"buckets": buckets}

    def _top_hits(self, body, rows, sub):
        ordered = list(rows)
        for item in reversed(_listify(body.get("sort"))):
            if isinstance(item, str):
                field, opts = item, {}
            else:
                ((field, opts),) = item.items()
            if isinstance(opts, str):
                opts = {"order": opts}
            desc = opts.get("order", "asc") == "desc"
            path = self.fields.resolve(field)[0]
            present = [r for r in ordered if values_at(r["_source"], path)]
            missing = [r for r in ordered if not values_at(r["_source"], path)]
            present.sort(
                key=lambda r: _sort_key(
                    (max if desc else min)(values_at(r["_source"], path), key=_sort_key)
                ),
                reverse=desc,
            )
            ordered = present + missing
        frm = body.get("from", 0)
        hits = [
            {
                "_index": r["_index"],
                "_id": r["_id"],
                "_score": None,
                "_source": filter_source(r["_source"], body.get("_source")),
            }
            for r in ordered[frm : frm + body.get("size", 3)]
        ]
        return {
            "hits": {
                "total": {"value": len(rows), "relation": "eq"},
                "max_score": None,
                "hits": hits,
            }
        }
