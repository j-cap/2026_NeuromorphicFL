"""Byte-equivalent EVS1 acceleration for binary/ternary baseline messages.

Only integer bound one uses the fast path. Other bounds and the rarely forced
dense modes retain the frozen codec. No layout, candidate, tie, padding or
floating-point rule changes. Frozen package files are never modified.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import islice
import math
import re
import struct

import numpy as np
from neuromorphicfl import event_codec as native

IMPLEMENTATION = "EVS1_bound1_packbits_regex_v1"


def encode_pulses(values, *, max_abs=1, force_mode=None):
    if max_abs != 1:
        return native.encode_pulses(values, max_abs=max_abs, force_mode=force_mode)
    vector = native._integers(values, max_abs)
    d = vector.size
    indices = np.flatnonzero(vector)
    count = len(indices)
    gaps = np.diff(np.r_[-1, indices]) - 1
    width = (d - 1).bit_length()
    candidates = []

    def add(mode, b, value_code, bits):
        if force_mode is None or force_mode == native._NAMES[mode]:
            candidates.append(((int(bits) + 7) // 8, mode, b, value_code, int(bits)))

    if not count:
        add(0, 0, 0, 0)
    else:
        for value_code in (0, 1):
            value_bits = count * (1 + value_code)
            for b in range(width + 1):
                add(1, b, value_code, int((gaps >> b).sum()) + count * (b + 1) + value_bits)
            add(2, 0, value_code, count * width + value_bits)
            add(3, 0, value_code, d + value_bits)
            if count == d:
                add(4, 0, value_code, value_bits)
    add(5, 0, 0, d + 2 * count)
    add(6, 0, 0, 2 * d)
    if not candidates:
        raise ValueError("requested codec mode does not represent this support")
    _, mode, b, value_code, predicted = min(candidates)
    if mode in (5, 6):
        return native.encode_pulses(values, max_abs=max_abs, force_mode=force_mode)
    signs = (vector[indices] < 0).astype(np.uint8)
    if mode == 0:
        bits = np.empty(0, dtype=np.uint8)
    elif mode == 1:
        quotient = gaps >> b
        lengths = quotient + b + 2 + value_code
        ends = lengths.cumsum(dtype=np.int64)
        delimiter = ends - (b + 2 + value_code)
        bits = np.ones(predicted, dtype=np.uint8)
        bits[delimiter] = 0
        for offset in range(b):
            bits[delimiter + 1 + offset] = (gaps >> (b - 1 - offset)) & 1
        if value_code:
            bits[ends - 2] = 0  # unary magnitude minus one is zero
        bits[ends - 1] = signs
    elif mode == 2:
        fields = np.zeros((count, width + 1 + value_code), dtype=np.uint8)
        for offset in range(width):
            fields[:, offset] = (indices >> (width - 1 - offset)) & 1
        fields[:, -1] = signs
        bits = fields.ravel()
    else:
        values_bits = np.zeros((count, 1 + value_code), dtype=np.uint8)
        values_bits[:, -1] = signs
        bits = values_bits.ravel() if mode == 4 else np.concatenate(((vector != 0).astype(np.uint8), values_bits.ravel()))
    if bits.size != predicted:
        raise AssertionError("Fast EVS1 length differs from the frozen formula")
    data = native._HEADER.pack(b"EVS1", d, count, max_abs, mode | (value_code << 4), b) + np.packbits(bits, bitorder="big").tobytes()
    return native.PulsePacket(data, f"{native._NAMES[mode]}:{'fixed' if value_code == 0 else 'unary'}:{b}",
                              count, predicted, (-predicted) % 8)


@lru_cache(maxsize=64)
def _rice_pattern(tail_width):
    # The input alphabet is exactly {0,1}. Every symbol starts immediately
    # after its predecessor: unary ones, delimiter, then fixed-width fields.
    return re.compile(b"\x01*\x00" + b"[\x00\x01]" * tail_width)


def _finish(bits, consumed):
    padding = (-consumed) % 8
    if consumed + padding != bits.size:
        raise ValueError("unexpected trailing pulse bytes")
    if np.any(bits[consumed:]):
        raise ValueError("nonzero pulse padding")


def decode_pulses(data, *, expected_dimension=None):
    if len(data) < native._HEADER.size:
        raise ValueError("truncated pulse header")
    magic, d, count, bound, flags, b = native._HEADER.unpack_from(data)
    mode, value_code = flags & 15, flags >> 4
    if bound != 1 or mode in (5, 6):
        return native.decode_pulses(data, expected_dimension=expected_dimension)
    if (magic != b"EVS1" or not 0 < d <= native._MAX_DIMENSION or count > d
            or mode >= len(native._NAMES) or value_code > 1 or b > (d - 1).bit_length()
            or (mode != 1 and b != 0) or (mode == 0 and (count != 0 or value_code != 0))
            or (mode == 4 and count != d) or (expected_dimension is not None and d != expected_dimension)):
        raise ValueError("invalid pulse metadata")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8, offset=native._HEADER.size), bitorder="big")
    result = np.zeros(d, dtype=np.int32)
    value_width = 1 + value_code
    if mode == 0:
        indices, ends = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        consumed = 0
    elif mode == 1:
        matches = islice(_rice_pattern(b + value_width).finditer(bits.tobytes()), count)
        try:
            ends = np.fromiter((match.end() for match in matches), dtype=np.int64, count=count)
        except ValueError as exc:
            raise ValueError("truncated pulse stream") from exc
        starts = np.r_[0, ends[:-1]] if count else np.empty(0, dtype=np.int64)
        quotient = ends - starts - (1 + b + value_width)
        if np.any(quotient < 0) or np.any(quotient > d >> b):
            raise ValueError("pulse code exceeds its declared bound")
        delimiter = starts + quotient
        remainder = np.zeros(count, dtype=np.int64)
        for offset in range(b):
            remainder = (remainder << 1) | bits[delimiter + 1 + offset]
        indices = np.cumsum((quotient << b) + remainder + 1, dtype=np.int64) - 1
        consumed = int(ends[-1]) if count else 0
    elif mode == 2:
        width = (d - 1).bit_length()
        consumed = count * (width + value_width)
        if consumed > bits.size:
            raise ValueError("truncated pulse stream")
        starts = np.arange(count, dtype=np.int64) * (width + value_width)
        ends = starts + width + value_width
        indices = np.zeros(count, dtype=np.int64)
        for offset in range(width):
            indices = (indices << 1) | bits[starts + offset]
    elif mode == 3:
        consumed = d + count * value_width
        if consumed > bits.size:
            raise ValueError("truncated pulse stream")
        indices = np.flatnonzero(bits[:d])
        if len(indices) != count:
            raise ValueError("pulse nonzero count mismatch")
        ends = d + np.arange(1, count + 1, dtype=np.int64) * value_width
    else:  # full support
        consumed = count * value_width
        if consumed > bits.size:
            raise ValueError("truncated pulse stream")
        indices = np.arange(d, dtype=np.int64)
        ends = np.arange(1, count + 1, dtype=np.int64) * value_width
    if (count and (indices[0] < 0 or indices[-1] >= d or np.any(np.diff(indices) <= 0))):
        raise ValueError("unordered or out-of-range pulse position")
    if value_code and np.any(bits[ends - 2]):
        raise ValueError("pulse code exceeds its declared bound")
    result[indices] = np.where(bits[ends - 1], -1, 1)
    if np.count_nonzero(result) != count:
        raise ValueError("pulse nonzero count mismatch")
    _finish(bits, consumed)
    return result


def decode_stc_lossless(data, tensor_sizes):
    if len(data) < 8:
        raise ValueError("truncated lossless STC envelope")
    magic, n = struct.unpack_from(">4sI", data)
    if magic != b"SCL1" or n != len(tensor_sizes) or len(data) < 8 + 4 * n:
        raise ValueError("invalid lossless STC envelope")
    scales = struct.unpack_from(">" + "f" * n, data, 8)
    signs = decode_pulses(data[8 + 4 * n:], expected_dimension=sum(tensor_sizes))
    if np.any(np.abs(signs) > 1):
        raise ValueError("invalid STC signs")
    vector = np.zeros(sum(tensor_sizes), dtype=np.float32)
    start = 0
    for size, scale in zip(tensor_sizes, scales):
        block = signs[start:start + size]
        if not math.isfinite(scale) or scale < 0 or (scale == 0) != (not np.any(block)):
            raise ValueError("invalid STC scale")
        vector[start:start + size] = scale * block.astype(np.float32)
        start += size
    return vector
