# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""Bits of Lucene and OpenSearch needed to reproduce their scores exactly.

OpenSearch scores each shard with that shard's own term statistics, so the
backend assigns documents to shards like OpenSearch does (murmur3 of the id)
and keeps statistics per shard. Field lengths go through Lucene's lossy
one-byte norm encoding, and scores are computed in 32-bit floats.
"""

import math
import struct

K1 = 1.2
B = 0.75


def f32(x):
    """Round to a 32-bit float, like Java ``float`` arithmetic."""
    return struct.unpack("f", struct.pack("f", x))[0]


# Shard routing
# =============


def murmur3_32(data, seed=0):
    """Lucene's ``StringHelper.murmurhash3_x86_32``, as a signed int."""
    c1, c2 = 0xCC9E2D51, 0x1B873593
    h = seed
    n = len(data) // 4 * 4
    for i in range(0, n, 4):
        k = int.from_bytes(data[i : i + 4], "little")
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    k = 0
    tail = data[n:]
    if len(tail) >= 3:
        k ^= tail[2] << 16
    if len(tail) >= 2:
        k ^= tail[1] << 8
    if tail:
        k ^= tail[0]
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h - (1 << 32) if h & 0x80000000 else h


def shard_of(doc_id, shards):
    """Shard OpenSearch routes a document id to."""
    if shards <= 1:
        return 0
    log2 = (shards - 1).bit_length()
    routing_shards = shards << max(1, 10 - min(log2, 10))
    return (murmur3_32(doc_id.encode("utf-16-le")) % routing_shards) // (
        routing_shards // shards
    )


# Norms
# =====


def _long_to_int4(i):
    bits = i.bit_length()
    if bits < 4:
        return i
    shift = bits - 4
    return ((i >> shift) & 0x07) | ((shift + 1) << 3)


def _int4_to_long(i):
    bits, shift = i & 0x07, (i >> 3) - 1
    return bits if shift == -1 else (bits | 0x08) << shift


_FREE = 255 - _long_to_int4(2**31 - 1)


def encode_norm(length):
    """Lucene's ``SmallFloat.intToByte4``."""
    if length < _FREE:
        return length
    return _FREE + _long_to_int4(length - _FREE)


def decode_norm(byte):
    """Lucene's ``SmallFloat.byte4ToInt``."""
    if byte < _FREE:
        return byte
    return _FREE + _int4_to_long(byte - _FREE)


# BM25
# ====


def idf(doc_freq, doc_count):
    """BM25 inverse document frequency."""
    return f32(math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5)))


class BM25:
    """Scores one term (or blended pseudo-term) on one shard of one field."""

    def __init__(self, boost, doc_freq, doc_count, sum_ttf):
        """Constructor."""
        avgdl = f32(sum_ttf / doc_count) if doc_count else f32(1)
        self.weight = f32(f32(boost) * idf(doc_freq, doc_count))
        self.avgdl = avgdl

    def norm_inverse(self, norm):
        """``1 / (k1 * (1 - b + b * dl / avgdl))`` for an encoded norm."""
        dl = f32(decode_norm(norm)) if norm is not None else f32(1)
        return f32(
            1 / f32(f32(K1) * f32(f32(1 - B) + f32(f32(f32(B) * dl) / self.avgdl)))
        )

    def score(self, freq, norm):
        """Score a document with ``freq`` occurrences and an encoded norm."""
        w = self.weight
        return f32(w - f32(w / f32(1 + f32(f32(freq) * self.norm_inverse(norm)))))
