# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""Text analysis matching the OpenSearch analyzers Invenio's mappings use.

An analyzer turns text into ``(term, position)`` tokens. Tokens produced at the
same position as the previous one (``asciifolding`` with ``preserve_original``,
``edge_ngram``) do not count towards the field length, as in Lucene.
"""

import re
import unicodedata

from .query import Unsupported

ASCII_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
MAX_TOKEN_LENGTH = 255


# Tokenizers
# ==========

# Word break classes (Unicode UAX #29, the parts that matter for word tokens).
LETTER, NUMERIC, EXTEND_NUM_LET, MID_LETTER, MID_NUM, MID_NUM_LET = range(6)
IDEOGRAPH, HIRAGANA, KATAKANA, EXTEND, OTHER, EMOJI = range(6, 12)

# Extended_Pictographic code points, which StandardTokenizer emits as tokens.
_EMOJI_RANGES = (
    (0x00A9, 0x00A9), (0x00AE, 0x00AE), (0x203C, 0x203C), (0x2049, 0x2049),
    (0x2122, 0x2122), (0x2139, 0x2139), (0x2194, 0x2199), (0x21A9, 0x21AA),
    (0x231A, 0x231B), (0x2328, 0x2328), (0x23CF, 0x23CF), (0x23E9, 0x23F3),
    (0x23F8, 0x23FA), (0x24C2, 0x24C2), (0x25AA, 0x25AB), (0x25B6, 0x25B6),
    (0x25C0, 0x25C0), (0x25FB, 0x25FE), (0x2600, 0x27BF), (0x2934, 0x2935),
    (0x2B05, 0x2B07), (0x2B1B, 0x2B1C), (0x2B50, 0x2B50), (0x2B55, 0x2B55),
    (0x3030, 0x3030), (0x303D, 0x303D), (0x3297, 0x3297), (0x3299, 0x3299),
    (0x1F000, 0x1FAFF),
)  # fmt: skip


def _klass(ch):
    o = ord(ch)
    if any(a <= o <= b for a, b in _EMOJI_RANGES):
        return EMOJI
    if ch in "'.\u2018\u2019\u2024\ufe52\uff07\uff0e":
        return MID_NUM_LET
    if ch in ":\u00b7\u0387\u05f4\u2027\ufe13\ufe55\uff1a":
        return MID_LETTER
    if (
        ch
        in ",;\u037e\u0589\u060c\u060d\u066c\u07f8\u2044\ufe10\ufe14\ufe50\ufe54\uff0c\uff1b"
    ):
        return MID_NUM
    if 0x3040 <= o <= 0x309F:
        return HIRAGANA
    if 0x30A0 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return KATAKANA
    if (
        0x4E00 <= o <= 0x9FFF
        or 0x3400 <= o <= 0x4DBF
        or 0x20000 <= o <= 0x2FFFF
        or 0xF900 <= o <= 0xFAFF
        or 0x3005 <= o <= 0x3007
    ):
        return IDEOGRAPH
    cat = unicodedata.category(ch)
    if cat == "Nd":
        return NUMERIC
    if cat == "Pc":
        return EXTEND_NUM_LET
    if cat[0] == "L" or cat == "Nl":
        return LETTER
    if cat[0] == "M" or cat == "Cf":
        return EXTEND
    return OTHER


WORD = {LETTER, NUMERIC, EXTEND_NUM_LET, KATAKANA}


def standard_tokens(text):
    """Split text into word tokens like Lucene's ``StandardTokenizer``."""
    out = []
    classes = [_klass(c) for c in text]
    # Combining marks attach to the preceding character's class.
    for i in range(1, len(classes)):
        if classes[i] == EXTEND and classes[i - 1] in WORD | {IDEOGRAPH, HIRAGANA}:
            classes[i] = classes[i - 1]
    n, i = len(text), 0
    while i < n:
        k = classes[i]
        if k in (IDEOGRAPH, HIRAGANA, EMOJI):
            j = i + 1
            while j < n and classes[j] == EXTEND:
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if k not in WORD:
            i += 1
            continue
        j, last = i + 1, k
        while j < n:
            c = classes[j]
            if c in WORD:
                if c == KATAKANA and last not in (KATAKANA, EXTEND_NUM_LET):
                    break
                if last == KATAKANA and c not in (KATAKANA, EXTEND_NUM_LET):
                    break
                last, j = c, j + 1
                continue
            if c in (MID_LETTER, MID_NUM, MID_NUM_LET) and j + 1 < n:
                nxt = classes[j + 1]
                if last == LETTER and nxt == LETTER and c != MID_NUM:
                    last, j = nxt, j + 2
                    continue
                if last == NUMERIC and nxt == NUMERIC and c != MID_LETTER:
                    last, j = nxt, j + 2
                    continue
            break
        token = text[i:j]
        while len(token) > MAX_TOKEN_LENGTH:
            out.append(token[:MAX_TOKEN_LENGTH])
            token = token[MAX_TOKEN_LENGTH:]
        out.append(token)
        i = j
    return out


_EMAIL = r"[\w!#$%&'*+/=?^`{|}~.-]+@[\w-]+(?:\.[\w-]+)+"
_URL = r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>\"]+[^\s<>\".,;:!?)\]}']"
_URL_EMAIL = re.compile(f"({_EMAIL}|{_URL})")


def url_email_tokens(text):
    """Like ``standard_tokens``, but keep URLs and email addresses whole."""
    out, pos = [], 0
    for m in _URL_EMAIL.finditer(text):
        out += standard_tokens(text[pos : m.start()])
        out.append(m.group(0))
        pos = m.end()
    return out + standard_tokens(text[pos:])


def letter_tokens(text):
    """Split on non-letters (``letter`` tokenizer)."""
    out, buf = [], []
    for ch in text + " ":
        if ch.isalpha():
            buf.append(ch)
        elif buf:
            out.append("".join(buf))
            buf = []
    return out


def whitespace_tokens(text):
    """Split on whitespace."""
    return text.split()


TOKENIZERS = {
    "standard": standard_tokens,
    "uax_url_email": url_email_tokens,
    "letter": letter_tokens,
    "lowercase": lambda t: [lower(x) for x in letter_tokens(t)],
    "whitespace": whitespace_tokens,
    "keyword": lambda t: [t] if t else [],
}


# Character and token filters
# ===========================

_JAVA_CLASSES = {
    r"\p{Punct}": lambda c: c in ASCII_PUNCT,
    r"\p{S}": lambda c: unicodedata.category(c)[0] == "S",
    r"\p{P}": lambda c: unicodedata.category(c)[0] == "P",
}


def _java_pattern(pattern):
    """Translate a Java regex to Python, for the constructs Invenio uses."""
    m = re.fullmatch(r"\[((?:\\p\{\w+\})+)\]", pattern)
    if m:
        tests = [_JAVA_CLASSES[c] for c in re.findall(r"\\p\{\w+\}", m.group(1))]
        chars = lambda c: any(t(c) for t in tests)  # noqa: E731
        return lambda text, repl: "".join(repl if chars(c) else c for c in text)
    if r"\p" in pattern:
        raise Unsupported(f"char filter pattern [{pattern}]")
    regex = re.compile(pattern)
    return lambda text, repl: regex.sub(re.sub(r"\$(\d)", r"\\\1", repl), text)


def char_filter(spec):
    """Build a character filter from its settings."""
    kind = spec.get("type")
    if kind == "pattern_replace":
        replace = _java_pattern(spec["pattern"])
        return lambda text: replace(text, spec.get("replacement", ""))
    if kind == "mapping":
        pairs = [m.split("=>") for m in spec.get("mappings", [])]
        pairs = [(a.strip(), b.strip()) for a, b in pairs]
        pairs.sort(key=lambda p: -len(p[0]))

        def apply(text):
            for a, b in pairs:
                text = text.replace(a, b)
            return text

        return apply
    if kind == "html_strip":
        return lambda text: re.sub(r"<[^>]*>", " ", text)
    raise Unsupported(f"char filter [{kind}]")


_FOLD = {
    "ß": "ss", "æ": "ae", "Æ": "AE", "ø": "o", "Ø": "O", "œ": "oe", "Œ": "OE",
    "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "þ": "th",
    "Þ": "TH", "ı": "i", "ħ": "h", "Ħ": "H", "ŀ": "l", "Ŀ": "L", "ŧ": "t",
    "Ŧ": "T", "ĸ": "q", "ŋ": "n", "Ŋ": "N", "ſ": "s", "‘": "'", "’": "'",
    "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", "ﬁ": "fi", "ﬂ": "fl",
}  # fmt: skip


def fold(text):
    """Fold characters to ASCII like Lucene's ``ASCIIFoldingFilter``."""
    if text.isascii():
        return text
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
        elif ch in _FOLD:
            out.append(_FOLD[ch])
        else:
            base = unicodedata.normalize("NFKD", ch)
            stripped = "".join(c for c in base if not unicodedata.combining(c))
            out.append(stripped if stripped.isascii() and stripped else ch)
    return "".join(out)


def lower(text):
    """Lowercase per code point, like Java's ``Character.toLowerCase``."""
    if text.isascii():
        return text.lower()
    return "".join((c.lower() if len(c.lower()) == 1 else c.lower()[0]) for c in text)


def _lowercase(tokens, spec):
    return [(lower(t), inc) for t, inc in tokens]


def _asciifolding(tokens, spec):
    keep = str(spec.get("preserve_original", False)).lower() == "true"
    out = []
    for token, inc in tokens:
        folded = fold(token)
        if keep and folded != token:
            out += [(folded, inc), (token, 0)]
        else:
            out.append((folded, inc))
    return out


def _edge_ngram(tokens, spec):
    low = int(spec.get("min_gram", 1))
    high = int(spec.get("max_gram", 2))
    keep = str(spec.get("preserve_original", False)).lower() == "true"
    out, pending = [], 0
    for token, inc in tokens:
        pending += inc
        grams = [token[:n] for n in range(low, min(high, len(token)) + 1)]
        if keep and len(token) not in range(low, high + 1):
            grams.append(token)
        for gram in grams:
            out.append((gram, pending))
            pending = 0
    return out


def _unique(tokens, spec):
    out, seen = [], set()
    for token, inc in tokens:
        if token not in seen:
            seen.add(token)
            out.append((token, inc))
    return out


def _stop(tokens, spec):
    words = spec.get("stopwords", "_english_")
    if words == "_english_":
        words = ENGLISH_STOP
    words = set(words)
    out, pending = [], 0
    for token, inc in tokens:
        pending += inc
        if token not in words:
            out.append((token, pending))
            pending = 0
    return out


def _trim(tokens, spec):
    return [(t.strip(), inc) for t, inc in tokens]


TOKEN_FILTERS = {
    "lowercase": _lowercase,
    "asciifolding": _asciifolding,
    "edge_ngram": _edge_ngram,
    "edgeNGram": _edge_ngram,
    "unique": _unique,
    "stop": _stop,
    "trim": _trim,
}
ENGLISH_STOP = (
    "a an and are as at be but by for if in into is it no not of on or such that "
    "the their then there these they this to was will with"
).split()


class Analyzer:
    """A chain of character filters, a tokenizer and token filters."""

    def __init__(self, tokenizer, char_filters=(), token_filters=()):
        """Constructor."""
        self.tokenizer = tokenizer
        self.char_filters = list(char_filters)
        self.token_filters = list(token_filters)

    def tokens(self, text):
        """Return ``[(term, position increment)]`` for a text."""
        for f in self.char_filters:
            text = f(text)
        tokens = [(t, 1) for t in self.tokenizer(text)]
        for f, spec in self.token_filters:
            tokens = f(tokens, spec)
        return tokens

    def normalize(self, text):
        """Apply only the character-level steps (for wildcard and prefix terms)."""
        for f in self.char_filters:
            text = f(text)
        tokens = [(text, 1)]
        for f, spec in self.token_filters:
            if f in (_lowercase, _asciifolding):
                tokens = f(tokens, {})
        return tokens[0][0] if tokens else text


BUILTIN = {
    "standard": lambda: Analyzer(standard_tokens, (), [(_lowercase, {})]),
    "simple": lambda: Analyzer(letter_tokens, (), [(_lowercase, {})]),
    "whitespace": lambda: Analyzer(whitespace_tokens),
    "keyword": lambda: Analyzer(TOKENIZERS["keyword"]),
    "english": lambda: Analyzer(
        standard_tokens, (), [(_lowercase, {}), (_stop, {"stopwords": "_english_"})]
    ),
    "stop": lambda: Analyzer(
        letter_tokens, (), [(_lowercase, {}), (_stop, {"stopwords": "_english_"})]
    ),
}


class Analysis:
    """Analyzers and normalizers defined in one index's settings."""

    def __init__(self, settings):
        """Constructor, from flattened index settings (``analysis.*`` keys)."""
        tree = {}
        for key, value in settings.items():
            if not key.startswith("analysis."):
                continue
            node = tree
            parts = key.split(".")[1:]
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        self.tree = tree
        self._cache = {}

    def _filters(self, names, kind):
        out = []
        for name in names or []:
            spec = (
                (self.tree.get(kind) or {}).get(name) if isinstance(name, str) else name
            )
            if spec is None:
                spec = {"type": name}
            if kind == "char_filter":
                out.append(char_filter(spec))
            else:
                f = TOKEN_FILTERS.get(spec.get("type"))
                if f is None:
                    raise Unsupported(f"token filter [{spec.get('type')}]")
                out.append((f, spec))
        return out

    def analyzer(self, name):
        """Return the analyzer called ``name``."""
        name = name or "standard"
        if name not in self._cache:
            spec = (self.tree.get("analyzer") or {}).get(name)
            if spec is None:
                if name not in BUILTIN:
                    raise Unsupported(f"analyzer [{name}]")
                self._cache[name] = BUILTIN[name]()
            else:
                if spec.get("type", "custom") != "custom" and "tokenizer" not in spec:
                    self._cache[name] = BUILTIN[spec["type"]]()
                    return self._cache[name]
                tokenizer = spec.get("tokenizer", "standard")
                tok_spec = (self.tree.get("tokenizer") or {}).get(tokenizer)
                kind = tok_spec.get("type") if tok_spec else tokenizer
                if kind not in TOKENIZERS:
                    raise Unsupported(f"tokenizer [{kind}]")
                self._cache[name] = Analyzer(
                    TOKENIZERS[kind],
                    self._filters(spec.get("char_filter"), "char_filter"),
                    self._filters(spec.get("filter"), "filter"),
                )
        return self._cache[name]

    def normalizer(self, name):
        """Return a function normalizing keyword values, or ``None``."""
        if not name:
            return None
        key = ("normalizer", name)
        if key not in self._cache:
            if name == "lowercase":
                spec = {"filter": ["lowercase"]}
            else:
                spec = (self.tree.get("normalizer") or {}).get(name)
            if spec is None:
                raise Unsupported(f"normalizer [{name}]")
            analyzer = Analyzer(
                TOKENIZERS["keyword"],
                self._filters(spec.get("char_filter"), "char_filter"),
                self._filters(spec.get("filter"), "filter"),
            )
            self._cache[key] = lambda value: analyzer.normalize(value)
        return self._cache[key]
