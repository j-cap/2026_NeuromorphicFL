"""Sparse Ternary Compression in a synchronous, fully participating FedAvg host.

Implements the client/server error feedback of Sattler et al., Algorithms 1--2,
and the power-of-two Golomb (Rice) position code of their Appendix A:
https://arxiv.org/abs/1903.02891

Each parameter tensor has its own top-k selection and float32 mean magnitude.
We use k=max(1, floor(p*n)), breaking magnitude ties by increasing coordinate
index. This fixes the paper's unspecified integer/tie convention. At p=1 the
operator still ternarizes; it is not an identity compressor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Sequence

import numpy as np


STC_PROTOCOL = "stc_fedavg_tensor_rice_v1"
_MAGIC = b"STC1"
_HEADER = struct.Struct(">4sI")  # 64 bits: format/version and tensor count
_TENSOR_HEADER = struct.Struct(">IfB")  # count, float32 magnitude, Rice exponent


def _check_fraction(fraction: float) -> None:
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("STC fraction must lie in (0, 1]")


def _tensor_sizes(sizes: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(size) for size in sizes)
    if (not result or any(size <= 0 or size >= 2**32 for size in result)
            or any(size != original for size, original in zip(result, sizes))):
        raise ValueError("STC needs positive integer tensor sizes below 2**32")
    return result


def sparse_ternary_compress(values: np.ndarray, fraction: float) -> np.ndarray:
    """Return a float32 sparse ternary vector with deterministic exact-k ties."""
    _check_fraction(fraction)
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or not vector.size or not np.all(np.isfinite(vector)):
        raise ValueError("STC input must be a finite, nonempty vector")
    k = max(1, int(math.floor(fraction * vector.size)))
    magnitudes = np.abs(vector)
    cutoff = np.partition(magnitudes, vector.size - k)[vector.size - k]
    larger = np.flatnonzero(magnitudes > cutoff)
    tied = np.flatnonzero(magnitudes == cutoff)[:k - larger.size]
    indices = np.concatenate((larger, tied))
    # Float64 reduction followed by a transmitted float32 scale is explicit.
    magnitude = np.float32(magnitudes[indices].mean(dtype=np.float64))
    compressed = np.zeros_like(vector)
    compressed[indices] = magnitude * np.sign(vector[indices])
    return compressed


def _rice_exponent(fraction: float, size: int) -> int:
    # k>=1 makes p<1/n equivalent to p=1/n for position coding. Clamp b at 0
    # for dense supports, where the analytic expression can be negative.
    density = max(fraction, 1.0 / size)
    if density == 1.0:
        return 0
    ratio = math.log((math.sqrt(5.0) - 1.0) / 2.0) / math.log1p(-density)
    return max(0, 1 + math.floor(math.log2(ratio)))


class _BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.buffer = 0
        self.pending = 0
        self.bits = 0

    def write(self, value: int, width: int) -> None:
        self.buffer = (self.buffer << width) | value
        self.pending += width
        self.bits += width
        while self.pending >= 8:
            self.pending -= 8
            self.data.append((self.buffer >> self.pending) & 255)
        self.buffer &= (1 << self.pending) - 1

    def finish(self) -> bytes:
        if self.pending:
            self.data.append(self.buffer << (8 - self.pending))
        return bytes(self.data)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def read(self, width: int) -> int:
        if self.position + width > 8 * len(self.data):
            raise ValueError("truncated STC position/sign stream")
        result = 0
        for _ in range(width):
            byte = self.data[self.position // 8]
            result = (result << 1) | ((byte >> (7 - self.position % 8)) & 1)
            self.position += 1
        return result


@dataclass(frozen=True)
class STCPacket:
    """An actual byte-aligned wire message, including its 64-bit frame header."""
    data: bytes
    nonzeros: int
    padding_bits: int

    @property
    def wire_bits(self) -> int:
        return 8 * len(self.data)

    @property
    def payload_bits(self) -> int:
        # Tensor metadata and byte padding are payload, not free side channels.
        return self.wire_bits - 8 * _HEADER.size


def compress_encode_stc(
    values: np.ndarray, tensor_sizes: Sequence[int], fraction: float,
) -> STCPacket:
    """Compress per tensor and encode positions, signs, scales and metadata.

The tensor layout is agreed with the model. Every tensor carries a 32-bit
nonzero count, 32-bit magnitude and 8-bit Rice exponent. A zero tensor carries
the same metadata and no symbols. Each tensor's symbol stream is zero padded
to a byte boundary. No entropy approximation is used for the message length.
"""
    _check_fraction(fraction)
    sizes = _tensor_sizes(tensor_sizes)
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or vector.size != sum(sizes):
        raise ValueError("STC vector does not match the tensor layout")
    wire = bytearray(_HEADER.pack(_MAGIC, len(sizes)))
    start = total_nonzeros = padding_bits = 0
    for size in sizes:
        compressed = sparse_ternary_compress(vector[start:start + size], fraction)
        indices = np.flatnonzero(compressed)
        magnitude = float(abs(compressed[indices[0]])) if indices.size else 0.0
        exponent = _rice_exponent(fraction, size)
        wire.extend(_TENSOR_HEADER.pack(int(indices.size), magnitude, exponent))
        writer = _BitWriter()
        previous = -1
        for index in indices:
            index = int(index)
            gap = index - previous - 1
            quotient, remainder = divmod(gap, 1 << exponent)
            writer.write((1 << quotient) - 1, quotient)
            writer.write(0, 1)
            writer.write(remainder, exponent)
            writer.write(int(compressed[index] < 0), 1)
            previous = index
        wire.extend(writer.finish())
        padding_bits += (-writer.bits) % 8
        total_nonzeros += int(indices.size)
        start += size
    return STCPacket(bytes(wire), total_nonzeros, padding_bits)


def decode_stc(data: bytes, tensor_sizes: Sequence[int]) -> np.ndarray:
    """Decode a message without consulting the uncompressed source update."""
    sizes = _tensor_sizes(tensor_sizes)
    if len(data) < _HEADER.size:
        raise ValueError("truncated STC frame header")
    magic, tensor_count = _HEADER.unpack_from(data)
    if magic != _MAGIC or tensor_count != len(sizes):
        raise ValueError("STC format/version or tensor count mismatch")
    decoded = np.zeros(sum(sizes), dtype=np.float32)
    cursor, start = _HEADER.size, 0
    for size in sizes:
        if cursor + _TENSOR_HEADER.size > len(data):
            raise ValueError("truncated STC tensor header")
        count, magnitude, exponent = _TENSOR_HEADER.unpack_from(data, cursor)
        cursor += _TENSOR_HEADER.size
        if (count > size or not math.isfinite(magnitude) or magnitude < 0
                or (count == 0) != (magnitude == 0) or exponent > 32):
            raise ValueError("invalid STC tensor metadata")
        reader = _BitReader(data[cursor:])
        previous = -1
        for _ in range(count):
            quotient = 0
            while reader.read(1):
                quotient += 1
                if (quotient << exponent) >= size:
                    raise ValueError("STC position exceeds the tensor")
            remainder = reader.read(exponent)
            index = previous + 1 + (quotient << exponent) + remainder
            if index >= size:
                raise ValueError("STC position exceeds the tensor")
            decoded[start + index] = -magnitude if reader.read(1) else magnitude
            previous = index
        padding = (-reader.position) % 8
        if reader.read(padding) != 0:
            raise ValueError("STC byte padding must be zero")
        cursor += reader.position // 8
        start += size
    if cursor != len(data):
        raise ValueError("unexpected trailing bytes in STC message")
    return decoded


@dataclass(frozen=True)
class STCRound:
    server_update: np.ndarray
    uplink: tuple[STCPacket, ...]
    downlink: STCPacket


@dataclass
class STCState:
    tensor_sizes: tuple[int, ...]
    upload_fraction: float
    download_fraction: float
    client_residuals: np.ndarray
    server_residual: np.ndarray

    @classmethod
    def zeros(
        cls, *, n_clients: int, tensor_sizes: Sequence[int],
        upload_fraction: float, download_fraction: float,
    ) -> STCState:
        sizes = _tensor_sizes(tensor_sizes)
        _check_fraction(upload_fraction)
        _check_fraction(download_fraction)
        if n_clients <= 0:
            raise ValueError("STC needs at least one client")
        return cls(
            sizes, upload_fraction, download_fraction,
            np.zeros((n_clients, sum(sizes)), dtype=np.float32),
            np.zeros(sum(sizes), dtype=np.float32),
        )

    def step(
        self, deltas: Sequence[np.ndarray], weights: np.ndarray, *,
        server_gain: float = 1.0,
    ) -> STCRound:
        """Communicate one full-participation round, returning the wire update.

        Client residuals are in unweighted local-model-delta units. Aggregation
        applies client weights exactly once. An optional common server gain is
        applied before server error feedback, whose residual is therefore in
        actual model-update units. Gain one recovers Algorithm 2.
        """
        updates = np.asarray(deltas, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float64)
        if (updates.shape != self.client_residuals.shape
                or not np.all(np.isfinite(updates))):
            raise ValueError("STC local deltas must match its finite client state")
        if (weights.shape != (updates.shape[0],) or not np.all(np.isfinite(weights))
                or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)):
            raise ValueError("STC weights must be nonnegative and sum to one")
        if not math.isfinite(server_gain) or server_gain <= 0:
            raise ValueError("STC server gain must be finite and positive")
        aggregate = np.zeros_like(self.server_residual)
        uplink = []
        for client, delta in enumerate(updates):
            corrected = self.client_residuals[client] + delta
            packet = compress_encode_stc(
                corrected, self.tensor_sizes, self.upload_fraction,
            )
            received = decode_stc(packet.data, self.tensor_sizes)
            self.client_residuals[client] = corrected - received
            aggregate += float(weights[client]) * received
            uplink.append(packet)
        aggregate *= server_gain
        corrected_server = self.server_residual + aggregate
        downlink = compress_encode_stc(
            corrected_server, self.tensor_sizes, self.download_fraction,
        )
        received_server = decode_stc(downlink.data, self.tensor_sizes)
        self.server_residual[:] = corrected_server - received_server
        return STCRound(received_server, tuple(uplink), downlink)
