# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""Small pieces of Lucene behaviour that responses depend on.

``murmur3_32`` decides which ``terms`` aggregation partition a term falls in;
``f32`` rounds scores to 32-bit floats, which is how OpenSearch prints them.
"""

import struct


def f32(x):
    """Round to a 32-bit float, like Java ``float`` arithmetic."""
    return struct.unpack("f", struct.pack("f", x))[0]


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
