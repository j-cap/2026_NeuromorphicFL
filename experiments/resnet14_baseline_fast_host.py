"""Isolated fast codec host preserving the frozen baseline update functions.

The optimizer, local learning, tensor reductions, gain, frame selection and
checkpoint code are loaded from their unchanged, source-pinned implementations.
Only the private instances' bound-one pulse codec functions are substituted.
Original imported modules and frozen package files remain unchanged.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_pulse_codec as codec
import resnet14_baseline_host as frozen


def _private_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wire = _private_module("_resnet14_fast_float_wire", Path(__file__).with_name("core_float_host.py"))
wire.encode_pulses = codec.encode_pulses
wire.decode_pulses = codec.decode_pulses
_implementation = _private_module("_resnet14_fast_baseline", Path(frozen.__file__))
_implementation.wire = wire
_implementation.encode_pulses = codec.encode_pulses
_implementation.decode_stc_lossless = codec.decode_stc_lossless

MAIN, WIRE_PROTOCOL = frozen.MAIN, frozen.WIRE_PROTOCOL
IMPLEMENTATION = codec.IMPLEMENTATION
run, step = _implementation.run, _implementation.step
native_step, traffic_summary = frozen.native_step, frozen.traffic_summary
