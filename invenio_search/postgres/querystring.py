# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

r"""Parser for the Lucene query string syntax of ``query_string`` queries.

Produces a small tree that the translator turns into SQL:

- ``("all",)``
- ``("bool", must, should, must_not)``
- ``("term", field, value, wildcard)``: in a wildcard value, ``\0`` stands for
  an unescaped ``*`` and ``\1`` for an unescaped ``?``
- ``("text", field, value)``: consecutive plain words without operators,
  analyzed together like Lucene does when it does not split on whitespace
- ``("phrase", field, value)``
- ``("range", field, low, high, include_low, include_high)``: ``None`` is open
- ``("exists", field)``
- ``("fuzzy", field, value, fuzziness)``
- ``("regexp", field, pattern)``
- ``("boost", node, factor)``

The occur rules follow Lucene's ``QueryParserBase.addClause``.
"""

MUST, SHOULD, MUST_NOT = "must", "should", "must_not"
_STOP = set('()[]{}:"^~')


class ParseError(Exception):
    """Query string is not valid Lucene syntax."""


def _tokenize(text):
    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "()":
            tokens.append((c, c))
            i += 1
        elif text.startswith("&&", i) or text.startswith("||", i):
            tokens.append(("AND" if c == "&" else "OR", c * 2))
            i += 2
        elif c in "+-!" and i + 1 < n and not text[i + 1].isspace():
            tokens.append(("NOT" if c == "!" else c, c))
            i += 1
        elif c == ":":
            tokens.append((":", c))
            i += 1
        elif c in "^~":
            i += 1
            start = i
            while i < n and (text[i].isdigit() or text[i] == "."):
                i += 1
            value = text[start:i]
            if c == "^":
                tokens.append(("BOOST", float(value or 1)))
            else:
                tokens.append(("FUZZY", value or None))
        elif c == "/":
            end = i + 1
            while end < n and text[end] != "/":
                end += 2 if text[end] == "\\" else 1
            if end >= n:
                raise ParseError("unterminated regular expression")
            tokens.append(("REGEXP", text[i + 1 : end]))
            i = end + 1
        elif c == '"':
            i += 1
            buf = []
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    i += 1
                buf.append(text[i])
                i += 1
            if i >= n:
                raise ParseError("unterminated phrase")
            i += 1
            tokens.append(("PHRASE", "".join(buf)))
        elif c in "[{":
            end = min(
                (j for j in (text.find("]", i), text.find("}", i)) if j != -1),
                default=-1,
            )
            if end == -1:
                raise ParseError("unterminated range")
            parts = text[i + 1 : end].split(" TO ")
            if len(parts) != 2:
                raise ParseError("invalid range")
            low, high = (p.strip().strip('"') for p in parts)
            tokens.append(
                (
                    "RANGE",
                    (
                        None if low == "*" else low,
                        None if high == "*" else high,
                        c == "[",
                        text[end] == "]",
                    ),
                )
            )
            i = end + 1
        else:
            buf, wildcard = [], False
            while i < n and not text[i].isspace() and text[i] not in _STOP:
                if text[i] == "\\" and i + 1 < n:
                    i += 1
                    buf.append(text[i])
                elif text[i] in "*?":
                    buf.append("\0" if text[i] == "*" else "\1")
                    wildcard = True
                else:
                    buf.append(text[i])
                i += 1
            word = "".join(buf)
            if word in ("AND", "OR", "NOT"):
                tokens.append((word, word))
            else:
                tokens.append(("TERM", (word, wildcard)))
    return tokens


class _Parser:
    def __init__(self, text, default_operator):
        self.tokens = _tokenize(text)
        self.pos = 0
        self.default_and = default_operator.upper() == "AND"

    def peek(self, offset=0):
        i = self.pos + offset
        return self.tokens[i] if i < len(self.tokens) else (None, None)

    def take(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def query(self, field=None, nested=False):
        clauses = []
        while True:
            kind, _ = self.peek()
            if kind is None or (kind == ")" and nested):
                break
            if kind == ")":
                raise ParseError("unbalanced parenthesis")
            conj = None
            if kind in ("AND", "OR"):
                conj = self.take()[0]
                kind, _ = self.peek()
            mods = None
            if kind in ("+", "-", "NOT"):
                mods = self.take()[0]
            node = self.clause(field)
            if node is None:
                continue
            plain = conj is None and mods is None and _plain(node)
            prev = clauses[-1] if clauses else None
            if plain and prev and prev[2] and prev[1][1] == node[1]:
                prev[1] = ("text", node[1], f"{prev[1][2]} {node[2]}")
                continue
            self.add_clause(clauses, conj, mods, node)
            clauses[-1].append(plain)
        return self.build(clauses)

    def add_clause(self, clauses, conj, mods, node):
        if clauses and conj == "AND" and clauses[-1][0] != MUST_NOT:
            clauses[-1][0] = MUST
        if clauses and conj == "OR" and self.default_and and clauses[-1][0] != MUST_NOT:
            clauses[-1][0] = SHOULD
        prohibited = mods in ("-", "NOT")
        if self.default_and:
            required = not prohibited and conj != "OR"
        else:
            required = mods == "+" or (conj == "AND" and not prohibited)
        occur = MUST_NOT if prohibited else MUST if required else SHOULD
        clauses.append([occur, node])

    @staticmethod
    def build(clauses):
        clauses = [c[:2] for c in clauses]
        if len(clauses) == 1 and clauses[0][0] != MUST_NOT:
            return clauses[0][1]
        if not clauses:
            return ("bool", [], [], [])
        return (
            "bool",
            [n for o, n in clauses if o == MUST],
            [n for o, n in clauses if o == SHOULD],
            [n for o, n in clauses if o == MUST_NOT],
        )

    def clause(self, field):
        kind, value = self.peek()
        if kind == "TERM" and self.peek(1)[0] == ":":
            self.pos += 2
            field = None if value[0] in ("*", "\0") else value[0]
            kind, value = self.peek()
        if kind == "(":
            self.take()
            node = self.query(field, nested=True)
            if self.take()[0] != ")":
                raise ParseError("unbalanced parenthesis")
        else:
            node = self.atom(kind, value, field)
        while self.peek()[0] in ("BOOST", "FUZZY"):
            mod, arg = self.take()
            if node is None:
                continue
            if mod == "BOOST":
                node = ("boost", node, arg)
            elif node[0] == "term" and not node[3]:
                node = ("fuzzy", node[1], node[2], arg if arg is not None else "AUTO")
            # A phrase slop (``"a b"~2``) is ignored: the phrase must be exact.
        return node

    def atom(self, kind, value, field):
        self.take()
        if kind == "REGEXP":
            return ("regexp", field, value)
        if kind == "PHRASE":
            return ("phrase", field, value)
        if kind == "RANGE":
            return ("range", field, *value)
        if kind != "TERM":
            return None
        word, wildcard = value
        if field == "_exists_":
            return ("exists", word)
        if word == "\0":
            return ("all",) if field is None else ("exists", field)
        if field and not wildcard:
            for op in (">=", "<=", ">", "<"):
                if word.startswith(op) and len(word) > len(op):
                    bound = word[len(op) :]
                    if op[0] == ">":
                        return ("range", field, bound, None, op == ">=", False)
                    return ("range", field, None, bound, False, op == "<=")
        return ("term", field, word, wildcard)


def _plain(node):
    return node[0] == "text" or (node[0] == "term" and not node[3])


def parse(text, default_operator="OR"):
    """Parse a query string, falling back to plain words on syntax errors."""
    try:
        return _Parser(text, default_operator).query()
    except ParseError:
        words = [
            w
            for w in "".join(c if c.isalnum() else " " for c in text).split()
            if w not in ("AND", "OR", "NOT")
        ]
        if not words:
            return ("bool", [], [], [])
        terms = [("term", None, w, False) for w in words]
        if default_operator.upper() == "AND":
            return ("bool", terms, [], [])
        return ("bool", [], terms, [])
