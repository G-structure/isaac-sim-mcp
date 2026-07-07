"""unitree_sdk2 CRC-32 (MPEG-2 variant) over packed C-struct words.

Mirrors unitree_sdk2's ``crc32_core``: init ``0xFFFFFFFF``, polynomial
``0x04C11DB7``, MSB-first, no input/output reflection, no final XOR. Each 32-bit
word is processed from bit 31 down to bit 0.

``crc32_of_struct`` reads a packed C-struct buffer as little-endian ``uint32``
words and CRCs every word except the trailing ``crc`` field -- i.e. the first
``(len >> 2) - 1`` words -- which is exactly how the robot firmware validates a
LowCmd/LowState frame.

TODO(probe:crc-capture): validate this against a recorded real LowCmd/LowState
frame; the packed layout that feeds these words is our reconstruction of the
unitree_sdk2 memory image, not a captured ground truth.

Run ``python3 crc32.py`` for a dependency-free self-test.
"""
from __future__ import annotations

import struct
from typing import Sequence

_POLY = 0x04C11DB7
_INIT = 0xFFFFFFFF
_MASK = 0xFFFFFFFF

# Precomputed 256-entry CRC-32/MPEG-2 table (MSB-first, no reflection). Feeding a
# word as 4 big-endian bytes through this table is identical to the bit-banged
# polynomial division below, but ~8x fewer GIL-held Python iterations -- material
# because the CRC runs on the packer/producer threads that compete with the
# physics-step callback for the GIL. Equivalence is asserted by the self-test's
# frozen known-answers against the independent bit-wise reference.
_TABLE = []
for _n in range(256):
    _c = _n << 24
    for _ in range(8):
        _c = ((_c << 1) ^ _POLY) & _MASK if _c & 0x80000000 else (_c << 1) & _MASK
    _TABLE.append(_c & _MASK)
_TABLE = tuple(_TABLE)


def crc32_core(words: Sequence[int]) -> int:
    """CRC-32/MPEG-2 over a sequence of 32-bit words, MSB-first per word.

    Table-driven: each word contributes its 4 bytes most-significant-first, which
    is exactly feeding bits 31..0 of the word to the bit-banged polynomial division.
    """
    crc = _INIT
    tbl = _TABLE
    for word in words:
        word &= _MASK
        crc = ((crc << 8) & _MASK) ^ tbl[((crc >> 24) ^ (word >> 24)) & 0xFF]
        crc = ((crc << 8) & _MASK) ^ tbl[((crc >> 24) ^ (word >> 16)) & 0xFF]
        crc = ((crc << 8) & _MASK) ^ tbl[((crc >> 24) ^ (word >> 8)) & 0xFF]
        crc = ((crc << 8) & _MASK) ^ tbl[((crc >> 24) ^ word) & 0xFF]
    return crc & _MASK


def crc32_of_struct(buf: bytes) -> int:
    """CRC a packed struct: little-endian words, excluding the trailing crc word."""
    if len(buf) % 4 != 0:
        raise ValueError(f"struct length {len(buf)} is not a multiple of 4")
    n = (len(buf) >> 2) - 1
    words = struct.unpack_from(f"<{n}I", buf)
    return crc32_core(words)


def _crc32_mpeg2_bytes(data: bytes) -> int:
    """Independent byte-wise CRC-32/MPEG-2 reference for the self-test."""
    crc = _INIT
    for byte in data:
        crc = (crc ^ (byte << 24)) & _MASK
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) & _MASK) ^ _POLY
            else:
                crc = (crc << 1) & _MASK
    return crc & _MASK


def _self_test() -> None:
    # Anchor the polynomial/init/direction to the published CRC-32/MPEG-2 check value.
    assert _crc32_mpeg2_bytes(b"123456789") == 0x0376E6E7, "not CRC-32/MPEG-2"

    # crc32_core (little-endian word reader, MSB-first) must equal the byte-wise
    # reference fed the same words in big-endian order.
    words = (0x00000000, 0xFFFFFFFF, 0x12345678, 0xDEADBEEF, 0x00000001)
    ref = _crc32_mpeg2_bytes(b"".join(struct.pack(">I", w) for w in words))
    got = crc32_core(words)
    assert got == ref, (hex(got), hex(ref))
    assert got == 0xF4BFE41B, hex(got)  # frozen known-answer

    # crc32_of_struct drops the last (crc) word.
    buf = struct.pack("<5I", *words)
    assert crc32_of_struct(buf) == crc32_core(words[:4]) == 0xEC9EEAAB

    print("crc32 self-test OK:", hex(got))


if __name__ == "__main__":
    _self_test()
