# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""Index mappings: field types, dynamic mapping, ``copy_to`` and term indexing.

:class:`Schema` describes the searchable fields of one index the way
OpenSearch sees them, including multi-fields and the ``search_as_you_type``
subfields, and turns a document into the postings and norms that
:mod:`.scoring` searches.
"""

import fnmatch
import re
from copy import deepcopy

from .analysis import Analysis
from .query import Unsupported, leaves

TEXT_LIKE = {"text", "match_only_text", "search_as_you_type", "shingle", "prefix"}
KEYWORD_LIKE = {"keyword", "constant_keyword", "wildcard", "version"}
NUMERIC = {
    "long", "integer", "short", "byte", "double", "float", "half_float",
    "scaled_float", "unsigned_long",
}  # fmt: skip
DATES = {"date", "date_nanos"}
OBJECTS = {"object", "nested"}
# Field types that a query on all fields (``*``) searches.
ALL_FIELD_TYPES = TEXT_LIKE | KEYWORD_LIKE | NUMERIC | DATES | {"ip"}
POSITION_GAP = 100
PREFIX_MAX_CHARS = 20

_DATE = re.compile(
    r"\d{4}(-\d{2}(-\d{2}(T\d{2}(:\d{2}(:\d{2}([.,]\d{1,9})?)?)?"
    r"(Z|[+-]\d{2}(:?\d{2})?)?)?)?)?"
    r"|\d{4}/\d{2}/\d{2}( \d{2}:\d{2}:\d{2})? [+-]\d{4}"
)


class Field:
    """One searchable field."""

    def __init__(self, name, spec, source, kind=None, base=None, shingle=0):
        """Constructor."""
        self.name = name
        self.spec = spec
        self.type = kind or spec.get("type") or "object"
        self.source = source  # document path holding the values
        self.base = base  # for search_as_you_type subfields: the main field
        self.shingle = shingle  # shingle size, or the max size for "prefix"

    def __repr__(self):
        """Representation."""
        return f"Field({self.name}, {self.type})"


def _dynamic_type(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "long"
    if isinstance(value, float):
        return "double"
    if isinstance(value, dict):
        return "object"
    return "string"


def _glob(pattern, value, regex=False):
    if regex:
        return re.fullmatch(pattern, value) is not None
    return any(fnmatch.fnmatchcase(value, p) for p in _listify(pattern))


def _listify(value):
    return value if isinstance(value, list) else [value]


def _fill(node, name, dynamic_type):
    if isinstance(node, str):
        return node.replace("{name}", name).replace("{dynamic_type}", dynamic_type)
    if isinstance(node, dict):
        return {k: _fill(v, name, dynamic_type) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill(v, name, dynamic_type) for v in node]
    return node


class Schema:
    """Fields and analyzers of one index."""

    def __init__(self, mappings, settings):
        """Constructor, from an index's mappings and flattened settings."""
        self.mappings = mappings or {}
        self.settings = settings or {}
        self.analysis = Analysis(self.settings)
        self.fields = {}
        self.copies = {}
        self.objects = {}
        self._walk(self.mappings.get("properties") or {}, "")
        self.shards = int(self.settings.get("number_of_shards", 1))

    # Fields

    def _walk(self, props, prefix):
        for name, spec in props.items():
            path = prefix + name
            kind = spec.get("type") or ("object" if "properties" in spec else None)
            if kind in OBJECTS or kind is None:
                self.objects[path] = spec
                if spec.get("enabled", True) not in (False, "false"):
                    self._walk(spec.get("properties") or {}, path + ".")
                continue
            self._add(path, spec, path)
            for sub, sub_spec in (spec.get("fields") or {}).items():
                self._add(f"{path}.{sub}", sub_spec, path)

    def _add(self, name, spec, source):
        field = Field(name, spec, source)
        self.fields[name] = field
        for target in _listify(spec.get("copy_to") or []):
            self.copies.setdefault(source, []).append(target)
        if field.type == "search_as_you_type":
            size = int(spec.get("max_shingle_size", 3))
            for n in range(2, size + 1):
                sub = f"{name}._{n}gram"
                self.fields[sub] = Field(sub, spec, source, "shingle", name, n)
            sub = f"{name}._index_prefix"
            self.fields[sub] = Field(sub, spec, source, "prefix", name, size)

    def field(self, name):
        """Return the field called ``name``, or ``None``."""
        return self.fields.get(name.split("^")[0])

    def expand(self, pattern):
        """Field names matching a pattern such as ``title.*`` or ``*``."""
        if pattern in ("*", "_all"):
            return [n for n, f in self.fields.items() if f.type in ALL_FIELD_TYPES]
        if "*" in pattern or "?" in pattern:
            return [
                n
                for n, f in self.fields.items()
                if fnmatch.fnmatchcase(n, pattern) and f.type in ALL_FIELD_TYPES
            ]
        return [pattern] if pattern in self.fields else []

    def default_fields(self):
        """Fields searched when a query names none: ``[(name, boost)]``."""
        default = self.settings.get("query.default_field") or ["*"]
        out = []
        for spec in _listify(default):
            pattern, _, boost = spec.partition("^")
            for name in self.expand(pattern):
                out.append((name, float(boost or 1)))
        return out

    # Analysis

    def _analyzer_name(self, field, search):
        spec = self.fields[field.base].spec if field.base else field.spec
        names = self.analysis.tree.get("analyzer") or {}
        if search:
            if spec.get("search_analyzer"):
                return spec["search_analyzer"]
            if spec.get("analyzer"):
                return spec["analyzer"]
            if "default_search" in names:
                return "default_search"
        elif spec.get("analyzer"):
            return spec["analyzer"]
        return "default" if "default" in names else "standard"

    def search_analyzer_name(self, field):
        """Name identifying how a field analyzes queries (for cross_fields)."""
        if field.type in KEYWORD_LIKE:
            return None
        return (self._analyzer_name(field, True), field.type, field.shingle)

    def tokens(self, field, text, search=False):
        """Analyze ``text`` for ``field``: ``[(term, position increment)]``."""
        if field.type in KEYWORD_LIKE:
            normalize = self.analysis.normalizer(field.spec.get("normalizer"))
            value = normalize(text) if normalize else text
            return [(value, 1)]
        analyzer = self.analysis.analyzer(self._analyzer_name(field, search))
        tokens = analyzer.tokens(text)
        if field.type == "shingle":
            return _shingles(tokens, field.shingle)
        if field.type == "prefix" and not search:
            return _prefixes(tokens, field.shingle)
        return tokens

    def normalize(self, field, text):
        """Normalize a wildcard or prefix pattern like the field's analyzer."""
        if field.type in KEYWORD_LIKE:
            normalize = self.analysis.normalizer(field.spec.get("normalizer"))
            return normalize(text) if normalize else text
        return self.analysis.analyzer(self._analyzer_name(field, True)).normalize(text)

    # Documents

    def dynamic(self, doc):
        """Mappings OpenSearch adds for the new fields in ``doc``.

        Returns properties to merge into the mappings, or ``{}``. Raises
        :class:`DynamicError` for a field under ``dynamic: strict``.
        """
        templates = self.mappings.get("dynamic_templates") or []
        date_detection = self.mappings.get("date_detection", True) not in (
            False,
            "false",
        )
        root_dynamic = str(self.mappings.get("dynamic", "true")).lower()
        added = {}

        def visit(node, path, props, dynamic, out):
            for name, value in node.items():
                full = f"{path}.{name}" if path else name
                if value is None or value == []:
                    continue
                spec = props.get(name)
                sample = value[0] if isinstance(value, list) and value else value
                if spec is None:
                    if dynamic == "strict":
                        raise DynamicError(full, path)
                    if dynamic != "true":
                        continue
                    spec = self._new_mapping(
                        full, name, sample, templates, date_detection
                    )
                    if spec is None:
                        continue
                    out[name] = spec
                kind = spec.get("type") or ("object" if "properties" in spec else None)
                if kind not in OBJECTS and kind is not None:
                    continue
                if spec.get("enabled", True) in (False, "false"):
                    continue
                inner_dynamic = str(spec.get("dynamic", dynamic)).lower()
                children = (
                    spec.setdefault("properties", {})
                    if name in out
                    else (spec.get("properties") or {})
                )
                new = {}
                for item in value if isinstance(value, list) else [value]:
                    if isinstance(item, dict):
                        visit(item, full, {**children, **new}, inner_dynamic, new)
                if new:
                    if name in out:
                        out[name].setdefault("properties", {}).update(new)
                    else:
                        out[name] = {"properties": new}

        visit(doc, "", self.mappings.get("properties") or {}, root_dynamic, added)
        return added

    @staticmethod
    def _new_mapping(path, name, value, templates, date_detection):
        kind = _dynamic_type(value)
        if kind == "string" and date_detection and _DATE.fullmatch(str(value)):
            kind = "date"
        for template in templates:
            ((_, t),) = template.items()
            regex = t.get("match_pattern") == "regex"
            mmt = t.get("match_mapping_type")
            if mmt and mmt != "*" and mmt != kind:
                continue
            if "match" in t and not _glob(t["match"], name, regex):
                continue
            if "unmatch" in t and _glob(t["unmatch"], name, regex):
                continue
            if "path_match" in t and not _glob(t["path_match"], path):
                continue
            if "path_unmatch" in t and _glob(t["path_unmatch"], path):
                continue
            dynamic_type = {"string": "text"}.get(kind, kind)
            mapping = _fill(deepcopy(t["mapping"]), name, dynamic_type)
            if "type" not in mapping and kind != "object":
                mapping["type"] = dynamic_type
            return mapping
        if kind == "object":
            return {}
        if kind == "string":
            return {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            }
        if kind == "double":
            return {"type": "float"}
        return {"type": kind}

    def with_copies(self, doc):
        """Return ``doc`` with ``copy_to`` values added to their targets."""
        if not self.copies:
            return doc
        doc = deepcopy(doc)
        for source, targets in self.copies.items():
            values = values_at(doc, source)
            if not values:
                continue
            for target in targets:
                node = doc
                parts = target.split(".")
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                current = node.get(parts[-1])
                current = [] if current is None else _listify(current)
                node[parts[-1]] = current + values
        return doc

    def terms(self, doc):
        """Index a document: ``(postings, norms)``.

        ``postings`` are ``(field, term, freq, positions)`` and ``norms`` are
        ``(field, length, total)``: ``length`` counts tokens that advance the
        position (as Lucene's norms do), ``total`` all tokens, and ``length``
        is ``None`` for fields without norms (keywords).
        """
        postings, norms = [], []
        for name, field in self.fields.items():
            if field.type not in TEXT_LIKE and field.type not in KEYWORD_LIKE:
                continue
            values = [v for v in values_at(doc, field.source) if v is not None]
            values = [v for v in values if not isinstance(v, (dict, list))]
            if not values:
                continue
            if field.type in KEYWORD_LIKE:
                limit = field.spec.get("ignore_above")
                seen = set()
                for value in values:
                    text = _keyword(value)
                    if limit is not None and len(text) > int(limit):
                        continue
                    seen.add(self.tokens(field, text)[0][0])
                if seen:
                    postings += [(name, t, 1, None) for t in seen]
                    norms.append((name, None, len(seen)))
                continue
            positions, length, total, pos = {}, 0, 0, -1
            for i, value in enumerate(values):
                if i:
                    pos += POSITION_GAP
                for term, inc in self.tokens(field, str(value)):
                    pos += inc
                    total += 1
                    if inc:
                        length += 1
                    positions.setdefault(term, []).append(max(pos, 0))
            if total:
                postings += [(name, t, len(p), p) for t, p in positions.items()]
                norms.append((name, length, total))
        return postings, norms


class DynamicError(Unsupported):
    """A document has a field that the mapping does not allow."""

    def __init__(self, field, parent):
        """Constructor."""
        super().__init__(
            f"mapping set to strict, dynamic introduction of [{field.rsplit('.', 1)[-1]}]"
            f" within [{parent or '_doc'}] is not allowed"
        )


def _keyword(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def values_at(doc, path):
    """Return all values at a dotted path, flattening arrays.

    Also finds values stored under keys that contain dots, such as
    ``{"schema.org": ...}``.
    """
    out = []

    def walk(node, parts):
        if isinstance(node, list):
            for item in node:
                walk(item, parts)
            return
        if not parts:
            if node is not None:
                out.append(node)
            return
        if not isinstance(node, dict):
            return
        for i in range(1, len(parts) + 1):
            key = ".".join(parts[:i])
            if key in node:
                walk(node[key], parts[i:])

    walk(doc, path.split("."))
    return out


def _shingles(tokens, size):
    terms = [t for t, inc in tokens if inc]
    out = []
    for i in range(len(terms) - size + 1):
        out.append((" ".join(terms[i : i + size]), 1))
    return out


def _prefixes(tokens, size):
    terms = [t for t, inc in tokens if inc]
    out = []
    for i in range(len(terms)):
        window = terms[i : i + size]
        shingle = " ".join(window + [""] * (size - len(window)))
        inc = 1
        for n in range(1, min(len(shingle), PREFIX_MAX_CHARS) + 1):
            out.append((shingle[:n], inc))
            inc = 0
    return out


__all__ = ("Schema", "Field", "DynamicError", "values_at", "leaves")
