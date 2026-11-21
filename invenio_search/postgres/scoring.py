# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""Evaluate OpenSearch queries on one index the way Lucene matches and scores them.

:class:`Searcher` turns a query into ``{pk: score}``. Text queries read the
postings that :mod:`.schema` wrote, and are scored with BM25 using per-shard
statistics, as OpenSearch does. Filters on numbers, dates, booleans, ranges
and nested objects run as exact SQL conditions from :mod:`.query`, with a
constant score like the corresponding OpenSearch queries.
"""

import re

from . import querystring
from .lucene import BM25, f32, idf
from .query import ROOT, Unsupported, _listify, _msm
from .schema import KEYWORD_LIKE, TEXT_LIKE

PREFIX_MAX_CHARS = 20


# Combining clause scores
# =======================


def conj(parts):
    """Documents matching all parts; scores add up."""
    parts = list(parts)
    keys = set(parts[0])
    for part in parts[1:]:
        keys &= part.keys()
    return {k: f32(sum(p[k] for p in parts)) for k in keys}


def disj(parts, minimum=1):
    """Documents matching at least ``minimum`` parts; scores add up."""
    counts, sums = {}, {}
    for part in parts:
        for k, v in part.items():
            counts[k] = counts.get(k, 0) + 1
            sums[k] = sums.get(k, 0.0) + v
    return {k: f32(sums[k]) for k, c in counts.items() if c >= max(minimum, 1)}


def dismax(parts, tie=0.0):
    """Best score among the parts, plus ``tie`` times the others."""
    best, total = {}, {}
    for part in parts:
        for k, v in part.items():
            if k not in best or v > best[k]:
                best[k] = v
            total[k] = total.get(k, 0.0) + v
    if not tie:
        return best
    return {k: f32(best[k] + (total[k] - best[k]) * tie) for k in best}


def const(pks, boost):
    """Constant score for every matching document."""
    score = f32(boost)
    return {pk: score for pk in pks}


def _boost(opts, boost):
    if isinstance(opts, dict) and "boost" in opts:
        return f32(boost * float(opts["boost"]))
    return boost


def edits_for(fuzziness, text):
    """Maximum edit distance for a fuzziness setting and a term."""
    spec = str(fuzziness).upper()
    if spec.startswith("AUTO"):
        low, high = 3, 6
        if ":" in spec:
            low, high = (int(x) for x in spec[5:].split(","))
        n = len(text)
        return 0 if n < low else 1 if n < high else 2
    return min(int(float(spec)), 2)


def osa_distance(a, b, limit):
    """Optimal string alignment distance, or ``limit + 1`` when larger."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if (
                prev2 is not None
                and i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


def wildcard_regex(pattern):
    """Regex for a wildcard pattern where ``\\0`` is ``*`` and ``\\1`` is ``?``."""
    return re.compile(
        "".join(
            ".*" if c == "\0" else "." if c == "\1" else re.escape(c) for c in pattern
        ),
        re.S,
    )


class Searcher:
    """Matches and scores queries against the documents of one index."""

    def __init__(self, cur, schema_name, index, schema, translator, restrict=None):
        """Constructor.

        :param schema_name: the Postgres schema holding the tables.
        :param schema: :class:`~.schema.Schema` of the index.
        :param translator: :class:`~.query.Translator` for SQL conditions.
        :param restrict: only consider these document pks (for percolation).
        """
        self.cur = cur
        self.tables = schema_name
        self.index = index
        self.schema = schema
        self.tr = translator
        self.restrict = restrict
        self.percolate = None
        self._cache = {}

    # Data access

    def _scope(self):
        if self.restrict is None:
            return "d.idx = %s", [self.index]
        return "d.idx = %s AND d.pk = ANY(%s::bigint[])", [self.index, self.restrict]

    def _query(self, key, sql, params):
        if key not in self._cache:
            self.cur.execute(sql, params)
            self._cache[key] = self.cur.fetchall()
        return self._cache[key]

    def _all_rows(self):
        scope, params = self._scope()
        return self._query(
            ("all",),
            f"SELECT d.pk, d.shard FROM {self.tables}.docs d WHERE {scope}",
            params,
        )

    def all(self, boost):
        """Every document, with a constant score."""
        return const((pk for pk, _ in self._all_rows()), boost)

    def _shard_size(self, shard):
        return sum(1 for _, s in self._all_rows() if s == shard)

    def postings(self, field, term):
        """``[(pk, shard, freq, positions, norm)]`` for a term of a field."""
        scope, params = self._scope()
        t = self.tables
        return self._query(
            ("postings", field, term),
            f"SELECT p.pk, d.shard, p.freq, p.positions, n.norm FROM {t}.terms p "
            f"JOIN {t}.docs d ON d.pk = p.pk "
            f"LEFT JOIN {t}.norms n ON n.pk = p.pk AND n.field = p.field "
            f"WHERE {scope} AND p.field = %s AND p.term = %s",
            params + [field, term],
        )

    def field_terms(self, field):
        """``{shard: set of terms}`` indexed for a field."""
        scope, params = self._scope()
        t = self.tables
        rows = self._query(
            ("terms", field),
            f"SELECT DISTINCT d.shard, p.term FROM {t}.terms p "
            f"JOIN {t}.docs d ON d.pk = p.pk WHERE {scope} AND p.field = %s",
            params + [field],
        )
        out = {}
        for shard, term in rows:
            out.setdefault(shard, set()).add(term)
        return out

    def field_stats(self, field):
        """``{shard: (doc count, sum of term frequencies)}`` for a field."""
        scope, params = self._scope()
        t = self.tables
        rows = self._query(
            ("stats", field),
            f"SELECT d.shard, count(*), sum(n.total) FROM {t}.norms n "
            f"JOIN {t}.docs d ON d.pk = n.pk WHERE {scope} AND n.field = %s "
            "GROUP BY d.shard",
            params + [field],
        )
        return {shard: (count, int(total)) for shard, count, total in rows}

    def _with_terms(self, field, terms):
        found = set()
        for term in terms:
            found.update(pk for pk, *_ in self.postings(field, term))
        return found

    def sql(self, clause, boost):
        """Run a clause as an SQL condition; constant score."""
        try:
            cond, cond_params = self.tr.where(clause, ROOT)
        except (ValueError, TypeError):
            return {}  # Lenient: a value that does not parse matches nothing.
        scope, params = self._scope()
        self.cur.execute(
            f"SELECT d.pk FROM {self.tables}.docs d "
            f"WHERE {scope} AND COALESCE({cond}, FALSE)",
            params + cond_params,
        )
        return const((r[0] for r in self.cur.fetchall()), boost)

    # Term-level scoring

    def bm25(self, field, terms, boost, idf_value=None):
        """Score documents containing any of ``terms`` (a synonym group)."""
        stats = self.field_stats(field)
        docs, df = {}, {}
        for term in terms:
            counts = {}
            for pk, shard, freq, _, norm in self.postings(field, term):
                counts[shard] = counts.get(shard, 0) + 1
                entry = docs.setdefault(pk, [shard, 0, norm])
                entry[1] += freq
            for shard, n in counts.items():
                df[shard] = max(df.get(shard, 0), n)
        scorers = {s: BM25(boost, df[s], *stats.get(s, (0, 0))) for s in df}
        return {
            pk: scorers[s].score(freq, norm) for pk, (s, freq, norm) in docs.items()
        }

    def fuzzy(self, field, term, fuzziness, prefix_length, boost, expansions=50):
        """Fuzzy term query, rewritten to the top terms with blended frequencies."""
        edits = edits_for(fuzziness, term)
        if edits == 0:
            return self.bm25(field, [term], boost)
        stats = self.field_stats(field)
        prefix = term[:prefix_length]
        out = {}
        for shard, terms in self.field_terms(field).items():
            found = []
            for candidate in terms:
                if not candidate.startswith(prefix):
                    continue
                d = osa_distance(term, candidate, edits)
                if d <= edits:
                    sim = f32(1 - f32(d) / f32(min(len(candidate), len(term))))
                    found.append((-sim, candidate, sim))
            found.sort()
            found = found[:expansions]
            rows = {
                t: [r for r in self.postings(field, t) if r[1] == shard]
                for _, t, _ in found
            }
            max_df = max((len(r) for r in rows.values()), default=0)
            sums = {}
            for _, t, sim in found:
                scorer = BM25(f32(boost * sim), max_df, *stats.get(shard, (0, 0)))
                for pk, _, freq, _, norm in rows[t]:
                    sums[pk] = sums.get(pk, 0.0) + scorer.score(freq, norm)
            out.update({pk: f32(v) for pk, v in sums.items()})
        return out

    def phrase(self, field, positions, boost):
        """Exact phrase; ``positions`` holds the terms allowed at each position."""
        if len(positions) == 1:
            return self.bm25(field, positions[0], boost)
        stats = self.field_stats(field)
        per_pos, df, shards, norms = [], {}, {}, {}
        for terms in positions:
            docs = {}
            for term in terms:
                counts = {}
                for pk, shard, _, pos, norm in self.postings(field, term):
                    docs.setdefault(pk, set()).update(pos or [])
                    shards[pk], norms[pk] = shard, norm
                    counts[shard] = counts.get(shard, 0) + 1
                for shard, n in counts.items():
                    df.setdefault((term, shard), n)
            per_pos.append(docs)
        candidates = set(per_pos[0])
        for docs in per_pos[1:]:
            candidates &= docs.keys()
        all_terms = [t for terms in positions for t in terms]
        out = {}
        for pk in candidates:
            freq = sum(
                all(p + i in per_pos[i][pk] for i in range(1, len(per_pos)))
                for p in per_pos[0][pk]
            )
            if not freq:
                continue
            shard = shards[pk]
            count, total = stats.get(shard, (0, 0))
            value = f32(sum(idf(df.get((t, shard), 0), count) for t in all_terms))
            scorer = BM25(boost, 0, count, total)
            scorer.weight = f32(f32(boost) * value)
            out[pk] = scorer.score(freq, norms[pk])
        return out

    def prefix(self, field, value, boost):
        """Constant-score prefix query."""
        spec = self.schema.fields.get(field)
        if spec is not None and spec.type == "prefix":
            if len(value) <= PREFIX_MAX_CHARS:
                return const(self._with_terms(field, [value]), boost)
            field = spec.base
        terms = {t for ts in self.field_terms(field).values() for t in ts}
        return const(
            self._with_terms(field, [t for t in terms if t.startswith(value)]), boost
        )

    def pattern(self, field, regex, boost):
        """Constant-score query on the terms matching a compiled regex."""
        terms = {t for ts in self.field_terms(field).values() for t in ts}
        return const(
            self._with_terms(field, [t for t in terms if regex.fullmatch(t)]), boost
        )

    # Analysis

    def _positions(self, field, text):
        out = []
        for term, inc in self.schema.tokens(field, str(text), search=True):
            if inc or not out:
                out.append([term])
            elif term not in out[-1]:
                out[-1].append(term)
        return out

    def _keyword(self, field, value):
        if isinstance(value, bool):
            value = "true" if value else "false"
        return self.schema.tokens(field, str(value))[0][0]

    def _position(self, field, terms, boost, fuzziness, prefix_length, expansions):
        # Like OpenSearch, positions holding several terms (such as folded and
        # original forms) become a synonym query without fuzziness.
        if fuzziness not in (None, 0, "0") and len(terms) == 1:
            return self.fuzzy(
                field.name, terms[0], fuzziness, prefix_length, boost, expansions
            )
        return self.bm25(field.name, terms, boost)

    def _other(self, name, value, boost):
        """Exact value on a numeric, date, boolean or ip field."""
        field = self.schema.field(name)
        if field is not None and field.type in ("date", "date_nanos"):
            return self.sql({"range": {name: {"gte": value, "lte": value}}}, boost)
        return self.sql({"term": {name: value}}, boost)

    def match(
        self,
        name,
        text,
        boost,
        operator="or",
        minimum=None,
        fuzziness=None,
        prefix_length=0,
        expansions=50,
    ):
        """``match`` query on one field."""
        field = self.schema.field(name)
        if field is None:
            return self.sql({"term": {name: text}}, boost) if name == "_id" else {}
        if field.type in KEYWORD_LIKE:
            term = self._keyword(field, text)
            if fuzziness not in (None, 0, "0"):
                return self.fuzzy(
                    field.name, term, fuzziness, prefix_length, boost, expansions
                )
            return const(self._with_terms(field.name, [term]), boost)
        if field.type not in TEXT_LIKE:
            return self._other(field.name, text, boost)
        positions = self._positions(field, text)
        if not positions:
            return {}
        clauses = [
            self._position(field, t, boost, fuzziness, prefix_length, expansions)
            for t in positions
        ]
        if len(clauses) == 1:
            return clauses[0]
        if str(operator).lower() == "and":
            return conj(clauses)
        return disj(clauses, _msm(minimum, len(clauses), 1))

    def match_phrase(self, name, text, boost):
        """``match_phrase`` query on one field."""
        field = self.schema.field(name)
        if field is None:
            return {}
        if field.type in KEYWORD_LIKE:
            return const(
                self._with_terms(field.name, [self._keyword(field, text)]), boost
            )
        if field.type not in TEXT_LIKE:
            return self._other(field.name, text, boost)
        positions = self._positions(field, text)
        return self.phrase(field.name, positions, boost) if positions else {}

    def _prefix_target(self, field):
        if field.type in ("search_as_you_type", "shingle"):
            base = field.base or field.name
            return f"{base}._index_prefix"
        return field.name

    def bool_prefix(
        self, name, text, boost, operator="or", minimum=None, fuzziness=None, pl=0
    ):
        """``match_bool_prefix`` on one field: all terms but the last, which is a prefix."""
        field = self.schema.field(name)
        if field is None:
            return {}
        if field.type in KEYWORD_LIKE:
            return self.prefix(field.name, self._keyword(field, text), boost)
        if field.type not in TEXT_LIKE:
            return {}
        positions = self._positions(field, text)
        if not positions:
            return {}
        clauses = [
            self._position(field, t, boost, fuzziness, pl, 50) for t in positions[:-1]
        ]
        target = self._prefix_target(field)
        clauses.append(disj([self.prefix(target, t, boost) for t in positions[-1]]))
        if len(clauses) == 1:
            return clauses[0]
        if str(operator).lower() == "and":
            return conj(clauses)
        return disj(clauses, _msm(minimum, len(clauses), 1))

    def phrase_prefix(self, name, text, boost, expansions=50):
        """``match_phrase_prefix`` on one field."""
        field = self.schema.field(name)
        if field is None or field.type not in TEXT_LIKE:
            return self.bool_prefix(name, text, boost) if field else {}
        positions = self._positions(field, text)
        if not positions:
            return {}
        last = positions[-1]
        terms = sorted({t for ts in self.field_terms(field.name).values() for t in ts})
        expanded = [t for t in terms if any(t.startswith(p) for p in last)]
        if not expanded:
            return {}
        return self.phrase(field.name, positions[:-1] + [expanded[:expansions]], boost)

    # Query DSL

    def run(self, query, boost=1.0, scoring=True):
        """Evaluate a query: ``{pk: score}`` of the matching documents."""
        if not query:
            return self.all(boost)
        if not isinstance(query, dict) or len(query) != 1:
            raise Unsupported("malformed query")
        ((kind, body),) = query.items()
        method = getattr(self, "_q_" + kind, None)
        if method is None:
            raise Unsupported(f"query [{kind}]")
        return method(body, f32(boost), scoring)

    @staticmethod
    def _single(body, name):
        items = [(k, v) for k, v in body.items() if k not in ("boost", "_name")]
        if len(items) != 1:
            raise Unsupported(f"[{name}] needs exactly one field")
        return items[0]

    def _q_match_all(self, body, boost, scoring):
        return self.all(_boost(body, boost))

    def _q_match_none(self, body, boost, scoring):
        return {}

    def _q_bool(self, body, boost, scoring):
        b = _boost(body, boost)
        must = [self.run(q, b, scoring) for q in _listify(body.get("must"))]
        filters = [self.run(q, 1.0, False) for q in _listify(body.get("filter"))]
        should = [self.run(q, b, scoring) for q in _listify(body.get("should"))]
        must_not = [self.run(q, 1.0, False) for q in _listify(body.get("must_not"))]
        required = must + [dict.fromkeys(f, 0.0) for f in filters]
        if not (required or should):
            out = self.all(b) if not must_not else self.all(0.0)
        else:
            n = _msm(
                body.get("minimum_should_match"), len(should), 0 if required else 1
            )
            if should and n > len(should):
                return {}
            if not required:
                out = disj(should, n)
            else:
                out = conj(required)
                if should:
                    opt = disj(should, max(n, 1))
                    out = {
                        k: f32(v + opt.get(k, 0.0))
                        for k, v in out.items()
                        if not n or k in opt
                    }
        for excluded in must_not:
            for k in excluded:
                out.pop(k, None)
        return out

    def _q_constant_score(self, body, boost, scoring):
        return const(self.run(body["filter"], 1.0, False), _boost(body, boost))

    def _q_dis_max(self, body, boost, scoring):
        b = _boost(body, boost)
        return dismax(
            [self.run(q, b, scoring) for q in body["queries"]],
            float(body.get("tie_breaker", 0)),
        )

    def _q_term(self, body, boost, scoring):
        name, opts = self._single(body, "term")
        value = opts.get("value") if isinstance(opts, dict) else opts
        b = _boost(opts, boost)
        field = self.schema.field(name)
        if field is None:
            if name not in ("_id", "_index"):
                return {}
            return self.sql({"term": {name: value}}, b)
        if field.type not in TEXT_LIKE | KEYWORD_LIKE:
            return self._other(field.name, value, b)
        if field.type in KEYWORD_LIKE:
            return const(self._with_terms(field.name, [self._keyword(field, value)]), b)
        return self.bm25(field.name, [str(value)], b)

    def _q_terms(self, body, boost, scoring):
        name, values = self._single(body, "terms")
        if not isinstance(values, list):
            raise Unsupported("[terms] lookup")
        b = _boost(body, boost)
        field = self.schema.field(name)
        if field is None or field.type not in TEXT_LIKE | KEYWORD_LIKE:
            if field is None and name not in ("_id", "_index"):
                return {}
            return self.sql({"terms": {name: values}}, b)
        if field.type in KEYWORD_LIKE:
            values = [self._keyword(field, v) for v in values]
        return const(self._with_terms(field.name, [str(v) for v in values]), b)

    def _q_ids(self, body, boost, scoring):
        return self.sql({"ids": body}, _boost(body, boost))

    def _q_exists(self, body, boost, scoring):
        return self.sql({"exists": body}, _boost(body, boost))

    def _q_range(self, body, boost, scoring):
        _, spec = self._single(body, "range")
        return self.sql({"range": body}, _boost(spec, boost))

    def _q_nested(self, body, boost, scoring):
        return self.sql({"nested": body}, _boost(body, boost))

    def _term_level(self, body, name):
        field_name, opts = self._single(body, name)
        if not isinstance(opts, dict):
            opts = {"value": opts}
        field = self.schema.field(field_name)
        return field, opts

    def _q_prefix(self, body, boost, scoring):
        field, opts = self._term_level(body, "prefix")
        if field is None or field.type not in TEXT_LIKE | KEYWORD_LIKE:
            return {}
        value = str(opts.get("value", opts.get("prefix")))
        if field.type in KEYWORD_LIKE:
            value = self.schema.normalize(field, value)
        return self.prefix(field.name, value, _boost(opts, boost))

    def _q_wildcard(self, body, boost, scoring):
        field, opts = self._term_level(body, "wildcard")
        if field is None or field.type not in TEXT_LIKE | KEYWORD_LIKE:
            return {}
        value = str(opts.get("value", opts.get("wildcard")))
        value = value.replace("*", "\0").replace("?", "\1")
        if field.type in KEYWORD_LIKE:
            value = self.schema.normalize(field, value)
        return self.pattern(field.name, wildcard_regex(value), _boost(opts, boost))

    def _q_regexp(self, body, boost, scoring):
        field, opts = self._term_level(body, "regexp")
        if field is None or field.type not in TEXT_LIKE | KEYWORD_LIKE:
            return {}
        return self.pattern(
            field.name, re.compile(str(opts["value"]), re.S), _boost(opts, boost)
        )

    def _q_fuzzy(self, body, boost, scoring):
        field, opts = self._term_level(body, "fuzzy")
        if field is None or field.type not in TEXT_LIKE | KEYWORD_LIKE:
            return {}
        value = str(opts["value"])
        if field.type in KEYWORD_LIKE:
            value = self._keyword(field, value)
        return self.fuzzy(
            field.name,
            value,
            opts.get("fuzziness", "AUTO"),
            int(opts.get("prefix_length", 0)),
            _boost(opts, boost),
            int(opts.get("max_expansions", 50)),
        )

    def _match_opts(self, body, name):
        field, opts = self._single(body, name)
        if not isinstance(opts, dict):
            opts = {"query": opts}
        return field, opts

    def _q_match(self, body, boost, scoring):
        name, o = self._match_opts(body, "match")
        return self.match(
            name,
            o["query"],
            _boost(o, boost),
            o.get("operator", "or"),
            o.get("minimum_should_match"),
            o.get("fuzziness"),
            int(o.get("prefix_length", 0)),
            int(o.get("max_expansions", 50)),
        )

    def _q_match_phrase(self, body, boost, scoring):
        name, o = self._match_opts(body, "match_phrase")
        return self.match_phrase(name, o["query"], _boost(o, boost))

    def _q_match_phrase_prefix(self, body, boost, scoring):
        name, o = self._match_opts(body, "match_phrase_prefix")
        return self.phrase_prefix(
            name, o["query"], _boost(o, boost), int(o.get("max_expansions", 50))
        )

    def _q_match_bool_prefix(self, body, boost, scoring):
        name, o = self._match_opts(body, "match_bool_prefix")
        return self.bool_prefix(
            name,
            o["query"],
            _boost(o, boost),
            o.get("operator", "or"),
            o.get("minimum_should_match"),
            o.get("fuzziness"),
            int(o.get("prefix_length", 0)),
        )

    def _fields(self, specs):
        """``[(field, boost)]`` for field patterns such as ``title^2`` or ``*``."""
        if not specs:
            return self.schema.default_fields()
        out = []
        for spec in _listify(specs):
            pattern, _, b = spec.partition("^")
            for name in self.schema.expand(pattern) or [pattern]:
                out.append((name, float(b or 1)))
        return out

    def _q_multi_match(self, body, boost, scoring):
        text = body["query"]
        kind = body.get("type", "best_fields")
        b = _boost(body, boost)
        fields = self._fields(body.get("fields"))
        operator = body.get("operator", "or")
        minimum = body.get("minimum_should_match")
        fuzziness = body.get("fuzziness")
        pl = int(body.get("prefix_length", 0))
        expansions = int(body.get("max_expansions", 50))
        tie = body.get("tie_breaker")
        if tie is None:
            tie = 1.0 if kind in ("most_fields", "bool_prefix") else 0.0
        if kind == "cross_fields":
            return self._cross_fields(text, fields, b, operator, minimum, float(tie))
        parts = []
        for name, fb in fields:
            fboost = f32(b * fb)
            if kind in ("best_fields", "most_fields"):
                parts.append(
                    self.match(
                        name, text, fboost, operator, minimum, fuzziness, pl, expansions
                    )
                )
            elif kind == "phrase":
                parts.append(self.match_phrase(name, text, fboost))
            elif kind == "phrase_prefix":
                parts.append(self.phrase_prefix(name, text, fboost, expansions))
            elif kind == "bool_prefix":
                parts.append(
                    self.bool_prefix(
                        name, text, fboost, operator, minimum, fuzziness, pl
                    )
                )
            else:
                raise Unsupported(f"[multi_match] type [{kind}]")
        return dismax(parts, float(tie))

    def _cross_fields(self, text, fields, boost, operator, minimum, tie):
        groups, parts = {}, []
        for name, fb in fields:
            field = self.schema.field(name)
            if field is None:
                continue
            fboost = f32(boost * fb)
            if field.type in KEYWORD_LIKE:
                term = self._keyword(field, text)
                parts.append(const(self._with_terms(field.name, [term]), fboost))
            elif field.type in TEXT_LIKE:
                key = self.schema.search_analyzer_name(field)
                groups.setdefault(key, []).append((field, fboost))
            else:
                parts.append(self._other(field.name, text, fboost))
        for members in groups.values():
            positions = self._positions(members[0][0], text)
            if not positions:
                continue
            clauses = [self._blended(members, terms, tie) for terms in positions]
            if len(clauses) == 1:
                parts.append(clauses[0])
            elif str(operator).lower() == "and":
                parts.append(conj(clauses))
            else:
                parts.append(disj(clauses, _msm(minimum, len(clauses), 1)))
        return dismax(parts, tie)

    def _blended(self, members, terms, tie):
        """Elasticsearch's ``BlendedTermQuery`` over fields (dis_max blending)."""
        pairs = [(f, fb, t) for t in terms for f, fb in members]
        rows = {(f.name, t): self.postings(f.name, t) for f, _, t in pairs}
        shards = {r[1] for rs in rows.values() for r in rs}
        out = []
        for shard in shards:
            dfs = {key: sum(1 for r in rs if r[1] == shard) for key, rs in rows.items()}
            max_df = max(dfs.values())
            stats = {
                f.name: self.field_stats(f.name).get(shard, (0, 0)) for f, _, _ in pairs
            }
            cap = min(
                [self._shard_size(shard)]
                + [stats[f.name][1] for f, _, t in pairs if dfs[(f.name, t)]]
            )
            actual = min(max_df, cap)
            for f, fb, t in pairs:
                if not dfs[(f.name, t)]:
                    continue
                scorer = BM25(fb, actual, *stats[f.name])
                out.append(
                    {
                        pk: scorer.score(freq, norm)
                        for pk, s, freq, _, norm in rows[(f.name, t)]
                        if s == shard
                    }
                )
        return dismax(out, tie)

    def _q_query_string(self, body, boost, scoring):
        fields = body.get("fields") or body.get("default_field")
        targets = self._fields(fields)
        operator = str(body.get("default_operator", "or")).lower()
        tree = querystring.parse(body["query"], operator.upper())
        options = (targets, operator, float(body.get("tie_breaker", 0)))
        return self._qs(tree, _boost(body, boost), options, top=True)

    _q_simple_query_string = _q_query_string

    def _qs_fields(self, field, options):
        if field is None:
            return options[0]
        return [(name, 1.0) for name in self.schema.expand(field) or [field]]

    def _qs(self, node, boost, options, top=False):
        kind = node[0]
        targets, operator, tie = options
        if kind == "all":
            return self.all(boost)
        if kind == "boost":
            return self._qs(node[1], f32(boost * node[2]), options, top)
        if kind == "bool":
            _, must, should, must_not = node
            required = [self._qs(n, boost, options) for n in must]
            optional = [self._qs(n, boost, options) for n in should]
            if required:
                out = conj(required)
                if optional:
                    opt = disj(optional)
                    out = {k: f32(v + opt.get(k, 0.0)) for k, v in out.items()}
            elif optional:
                out = disj(optional)
            else:
                out = self.all(0.0) if top else {}
            for n in must_not:
                for k in self._qs(n, 1.0, options):
                    out.pop(k, None)
            return out
        if kind == "range":
            _, field, low, high, include_low, include_high = node
            if not field:
                return {}
            spec = {}
            if low is not None:
                spec["gte" if include_low else "gt"] = low
            if high is not None:
                spec["lte" if include_high else "lt"] = high
            return self.sql({"range": {field: spec}}, boost)
        if kind == "exists":
            return self.sql({"exists": {"field": node[1]}}, boost)
        parts = []
        for name, fb in self._qs_fields(node[1], options):
            b = f32(boost * fb)
            field = self.schema.field(name)
            if field is None:
                continue
            if kind in ("text", "term") and not (kind == "term" and node[3]):
                parts.append(self.match(name, node[2], b, operator))
            elif kind == "term":
                parts.append(self._qs_wildcard(field, node[2], b))
            elif kind == "phrase":
                parts.append(self.match_phrase(name, node[2], b))
            elif kind == "fuzzy":
                if field.type in TEXT_LIKE | KEYWORD_LIKE:
                    term = self.schema.normalize(field, node[2])
                    parts.append(self.fuzzy(name, term, node[3], 0, b))
            elif kind == "regexp":
                if field.type in TEXT_LIKE | KEYWORD_LIKE:
                    # Like OpenSearch, the pattern goes through the normalizer.
                    regex = re.compile(self.schema.normalize(field, node[2]), re.S)
                    parts.append(self.pattern(name, regex, b))
            else:
                raise Unsupported(f"query string node [{kind}]")
        return parts[0] if len(parts) == 1 else dismax(parts, tie)

    def _qs_wildcard(self, field, value, boost):
        if field.type not in TEXT_LIKE | KEYWORD_LIKE:
            return {}
        value = self.schema.normalize(field, value)
        if value.endswith("\0") and "\0" not in value[:-1] and "\1" not in value:
            return self.prefix(field.name, value[:-1], boost)
        return self.pattern(field.name, wildcard_regex(value), boost)

    def _q_percolate(self, body, boost, scoring):
        if self.percolate is None:
            raise Unsupported("[percolate] in this context")
        documents = body.get("documents")
        if documents is None and "document" in body:
            documents = [body["document"]]
        if documents is None:
            raise Unsupported("[percolate] of indexed documents")
        return const(self.percolate(body.get("field", "query"), documents), boost)
