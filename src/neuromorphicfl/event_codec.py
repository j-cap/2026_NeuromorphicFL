"""Lossless integer pulse messages and exact Event downlink reconstruction.

The EVS1 frame has a 128-bit header: magic, dimension, nonzero count,
maximum absolute value, mode/value-code flags, and Rice exponent. Model layout,
round ordering and the common jump schedule belong to the surrounding protocol.
Every returned length is an actual byte length, including metadata and padding.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy as np

from .stc import decode_stc

PROTOCOL = "event_integer_portfolio_v1"
_HEADER = struct.Struct(">4sIIHBB")
_ORDERED = struct.Struct(">4sIII")
_PATCH = struct.Struct(">4sII")
_MAX_DIMENSION = 10_000_000
_NAMES = ("empty", "rice", "indices", "bitmap", "full", "dense_unary", "dense_fixed")


class BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.buffer = self.pending = self.bits = 0

    def write(self, value: int, width: int):
        if width < 0 or value < 0 or value >= 1 << width:
            raise ValueError("invalid bit field")
        self.buffer = (self.buffer << width) | int(value)
        self.pending += width
        self.bits += width
        while self.pending >= 8:
            self.pending -= 8
            self.data.append((self.buffer >> self.pending) & 255)
        self.buffer &= (1 << self.pending) - 1

    def unary(self, value: int):
        # Write long runs in chunks; no enormous temporary integer is needed.
        while value >= 32:
            self.write(2**32 - 1, 32)
            value -= 32
        self.write((1 << value) - 1, value)
        self.write(0, 1)

    def finish(self) -> bytes:
        if self.pending:
            self.data.append(self.buffer << (8 - self.pending))
            self.pending = 0
        return bytes(self.data)


class BitReader:
    def __init__(self, data: bytes):
        self.data, self.position = data, 0

    def read(self, width: int) -> int:
        if width < 0 or self.position + width > len(self.data) * 8:
            raise ValueError("truncated pulse stream")
        value = 0
        for _ in range(width):
            value = (value << 1) | ((self.data[self.position // 8] >> (7 - self.position % 8)) & 1)
            self.position += 1
        return value

    def unary(self, maximum: int) -> int:
        value = 0
        while self.read(1):
            value += 1
            if value > maximum:
                raise ValueError("pulse code exceeds its declared bound")
        return value

    def finish(self):
        padding = (-self.position) % 8
        if self.position + padding != len(self.data) * 8:
            raise ValueError("unexpected trailing pulse bytes")
        if self.read(padding):
            raise ValueError("nonzero pulse padding")


@dataclass(frozen=True)
class PulsePacket:
    data: bytes
    mode: str
    nonzeros: int
    payload_bits: int
    padding_bits: int

    @property
    def wire_bits(self) -> int:
        return len(self.data) * 8


def _integers(values, max_abs: int) -> np.ndarray:
    vector = np.asarray(values)
    if (vector.ndim != 1 or not 0 < vector.size <= _MAX_DIMENSION
            or vector.dtype.kind not in "iu" or not 1 <= max_abs <= 32767
            or int(vector.min()) < -max_abs or int(vector.max()) > max_abs):
        raise ValueError("expected a bounded nonempty signed-integer vector")
    return vector.astype(np.int32, copy=False)


def encode_pulses(values, *, max_abs: int = 1, force_mode: str | None = None) -> PulsePacket:
    """Encode an integer vector; choose the smallest fully framed byte length.

    Candidate lengths are exact formulas for the same bit writer used below.
    Mode, value code and Rice exponent all travel in the header. No support,
    sign, magnitude or probability model is a free side channel.
    """
    vector = _integers(values, max_abs)
    d = vector.size
    indices = np.flatnonzero(vector)
    count = int(indices.size)
    magnitudes = np.abs(vector[indices])
    gaps = np.diff(np.r_[-1, indices]) - 1
    index_width = (d - 1).bit_length()
    magnitude_width = (max_abs - 1).bit_length()
    candidates = []

    def add(mode, b, value_code, bits):
        if force_mode is None or force_mode == _NAMES[mode]:
            candidates.append(((int(bits) + 7) // 8, mode, b, value_code, int(bits)))

    if not count:
        add(0, 0, 0, 0)
    else:
        for value_code in (0, 1):
            value_bits = count * (1 + magnitude_width) if value_code == 0 else int(magnitudes.sum()) + count
            for b in range(index_width + 1):
                add(1, b, value_code, int((gaps >> b).sum()) + count * (b + 1) + value_bits)
            add(2, 0, value_code, count * index_width + value_bits)
            add(3, 0, value_code, d + value_bits)
            if count == d:
                add(4, 0, value_code, value_bits)
    add(5, 0, 0, d + int(magnitudes.sum()) + count)
    add(6, 0, 0, d * (2 * max_abs).bit_length())
    if not candidates:
        raise ValueError("requested codec mode does not represent this support")
    _, mode, b, value_code, predicted = min(candidates)
    writer = BitWriter()

    def write_value(value):
        magnitude = abs(int(value))
        if value_code == 0:
            writer.write(magnitude - 1, magnitude_width)
        else:
            writer.unary(magnitude - 1)
        writer.write(int(value < 0), 1)

    if mode in (1, 2, 4):
        for j, index in enumerate(indices):
            if mode == 1:
                gap = int(gaps[j])
                writer.unary(gap >> b)
                writer.write(gap & ((1 << b) - 1), b)
            elif mode == 2:
                writer.write(int(index), index_width)
            write_value(vector[index])
    elif mode == 3:
        for value in vector:
            writer.write(int(value != 0), 1)
        for index in indices:
            write_value(vector[index])
    elif mode == 5:
        for value in vector:
            writer.unary(abs(int(value)))
            if value:
                writer.write(int(value < 0), 1)
    elif mode == 6:
        width = (2 * max_abs).bit_length()
        for value in vector:
            writer.write(2 * abs(int(value)) - int(value > 0), width)
    if writer.bits != predicted:
        raise AssertionError("codec length formula differs from serialized bits")
    data = _HEADER.pack(b"EVS1", d, count, max_abs, mode | (value_code << 4), b) + writer.finish()
    return PulsePacket(data, f"{_NAMES[mode]}:{'fixed' if value_code == 0 else 'unary'}:{b}", count, predicted, (-predicted) % 8)


def decode_pulses(data: bytes, *, expected_dimension: int | None = None) -> np.ndarray:
    if len(data) < _HEADER.size:
        raise ValueError("truncated pulse header")
    magic, d, count, bound, flags, b = _HEADER.unpack_from(data)
    mode, value_code = flags & 15, flags >> 4
    if (magic != b"EVS1" or not 0 < d <= _MAX_DIMENSION or count > d
            or not 1 <= bound <= 32767 or mode >= len(_NAMES) or value_code > 1
            or b > (d - 1).bit_length() or (mode != 1 and b != 0)
            or (mode in (0, 5, 6) and value_code != 0)
            or (mode == 0 and count != 0) or (mode == 4 and count != d)
            or (expected_dimension is not None and d != expected_dimension)):
        raise ValueError("invalid pulse metadata")
    result = np.zeros(d, dtype=np.int32)
    reader = BitReader(data[_HEADER.size:])
    magnitude_width = (bound - 1).bit_length()

    def read_value():
        magnitude = (reader.read(magnitude_width) if value_code == 0 else reader.unary(bound - 1)) + 1
        if magnitude > bound:
            raise ValueError("invalid pulse magnitude")
        return -magnitude if reader.read(1) else magnitude

    if mode in (1, 2, 4):
        previous = -1
        for ordinal in range(count):
            if mode == 1:
                gap = (reader.unary(d >> b) << b) + reader.read(b)
                index = previous + 1 + gap
            elif mode == 2:
                index = reader.read((d - 1).bit_length())
            else:
                index = ordinal
            if not previous < index < d:
                raise ValueError("unordered or out-of-range pulse position")
            result[index] = read_value()
            previous = index
    elif mode == 3:
        indices = [i for i in range(d) if reader.read(1)]
        for index in indices:
            result[index] = read_value()
    elif mode == 5:
        for i in range(d):
            magnitude = reader.unary(bound)
            if magnitude:
                result[i] = -magnitude if reader.read(1) else magnitude
    elif mode == 6:
        for i in range(d):
            symbol = reader.read((2 * bound).bit_length())
            if symbol > 2 * bound:
                raise ValueError("invalid dense pulse symbol")
            result[i] = (symbol + 1) // 2 if symbol & 1 else -(symbol // 2)
    if int(np.count_nonzero(result)) != count:
        raise ValueError("pulse nonzero count mismatch")
    reader.finish()
    return result


def pulse_aggregate(pulses: np.ndarray, jump: float) -> np.ndarray:
    """The historical float32 operation order, including each nonzero sign."""
    aggregate = np.zeros(pulses.shape[1], dtype=np.float32)
    for vector in pulses:
        mask = vector != 0
        aggregate[mask] += jump * vector[mask].astype(np.float32)
    return aggregate


def _encode_ordered(pulses: np.ndarray) -> bytes:
    n, d = pulses.shape
    writer = BitWriter()
    for column in pulses.T:
        active = column[column != 0]
        writer.unary(len(active))
        for sign in active:
            writer.write(int(sign < 0), 1)
    return _ORDERED.pack(b"EVO1", d, n, int(np.count_nonzero(pulses))) + writer.finish()


@dataclass(frozen=True)
class EventDownlink:
    data: bytes
    mode: str
    counts: PulsePacket
    aggregate_exact_bits: int
    ordered_bits: int
    correction_nonzeros: int
    canonical_max_abs_error: float


def encode_event_downlink(pulses: np.ndarray, jump: float) -> EventDownlink:
    matrix = np.asarray(pulses)
    if (matrix.ndim != 2 or not 0 < matrix.shape[0] <= 32767
            or not math.isfinite(jump) or jump <= 0):
        raise ValueError("invalid pulse matrix or common amplitude")
    for row in matrix:
        _integers(row, 1)
    counts = encode_pulses(matrix.astype(np.int32).sum(axis=0, dtype=np.int32), max_abs=matrix.shape[0])
    canonical = jump * decode_pulses(counts.data).astype(np.float32)
    reference = pulse_aggregate(matrix, jump)
    exceptions = reference.view(np.uint32) != canonical.view(np.uint32)
    n_exceptions = int(exceptions.sum())
    patch = encode_pulses(exceptions.astype(np.int8)).data if n_exceptions else b""
    corrected = (_PATCH.pack(b"EVR1", len(counts.data), len(patch)) + counts.data + patch
                 + reference[exceptions].astype(">f4").tobytes())
    ordered = _encode_ordered(matrix)
    mode, chosen = min((("counts_patch", corrected), ("ordered", ordered)), key=lambda pair: (len(pair[1]), pair[0]))
    return EventDownlink(chosen, mode, counts, len(corrected) * 8, len(ordered) * 8,
                         n_exceptions, float(np.max(np.abs(reference.astype(float) - canonical))))


def decode_event_downlink(data: bytes, jump: float, *, expected_dimension: int) -> np.ndarray:
    if not math.isfinite(jump) or jump <= 0:
        raise ValueError("invalid common pulse amplitude")
    if data[:4] == b"EVR1":
        if len(data) < _PATCH.size:
            raise ValueError("truncated aggregate frame")
        _, count_bytes, patch_bytes = _PATCH.unpack_from(data)
        start = _PATCH.size
        counts = decode_pulses(data[start:start + count_bytes], expected_dimension=expected_dimension)
        start += count_bytes
        aggregate = jump * counts.astype(np.float32)
        if patch_bytes:
            mask = decode_pulses(data[start:start + patch_bytes], expected_dimension=expected_dimension)
            if np.any((mask != 0) & (mask != 1)):
                raise ValueError("invalid correction mask")
            start += patch_bytes
            size = int(np.count_nonzero(mask))
            if len(data) - start != 4 * size:
                raise ValueError("invalid aggregate correction length")
            values = np.frombuffer(data[start:], dtype=">f4").astype(np.float32)
            if not np.all(np.isfinite(values)):
                raise ValueError("nonfinite aggregate correction")
            aggregate[mask != 0] = values
        elif start != len(data):
            raise ValueError("trailing aggregate bytes")
        return aggregate
    if len(data) < _ORDERED.size:
        raise ValueError("truncated ordered-pulse frame")
    magic, d, n, total = _ORDERED.unpack_from(data)
    if magic != b"EVO1" or d != expected_dimension or not 0 < d <= _MAX_DIMENSION or not 0 < n <= 32767 or total > n * d:
        raise ValueError("invalid ordered-pulse metadata")
    reader = BitReader(data[_ORDERED.size:])
    aggregate = np.zeros(d, dtype=np.float32)
    seen = 0
    positive = (jump * np.ones(1, dtype=np.float32))[0]
    negative = (jump * -np.ones(1, dtype=np.float32))[0]
    for j in range(d):
        count = reader.unary(n)
        seen += count
        for _ in range(count):
            aggregate[j] += negative if reader.read(1) else positive
    if seen != total:
        raise ValueError("ordered-pulse count mismatch")
    reader.finish()
    return aggregate


def encode_stc_lossless(data: bytes, tensor_sizes: tuple[int, ...]) -> bytes:
    """Re-encode an already quantized native STC message, without requantizing."""
    vector = decode_stc(data, tensor_sizes)
    signs = np.sign(vector).astype(np.int8)
    scales = []
    start = 0
    for size in tensor_sizes:
        block = vector[start:start + size]
        active = np.abs(block[block != 0])
        scale = float(active[0]) if active.size else 0.0
        if active.size and not np.all(active == scale):
            raise ValueError("STC tensor does not have a shared magnitude")
        scales.append(scale)
        start += size
    return struct.pack(">4sI", b"SCL1", len(scales)) + struct.pack(">" + "f" * len(scales), *scales) + encode_pulses(signs).data


def decode_stc_lossless(data: bytes, tensor_sizes: tuple[int, ...]) -> np.ndarray:
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
