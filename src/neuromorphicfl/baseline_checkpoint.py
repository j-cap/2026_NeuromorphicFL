"""Atomic numeric checkpoints for the synchronous NumPy baseline runner."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BaselineCheckpoint:
    path: Path
    provenance: dict[str, Any]

    def load(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        with np.load(self.path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if (metadata.get("format") != "numpy_fedavg_checkpoint_v1"
                    or metadata.get("identity") != identity
                    or metadata.get("provenance") != self.provenance):
                raise ValueError(f"checkpoint identity/provenance mismatch: {self.path}")
            arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        return {"arrays": arrays, "progress": metadata["progress"]}

    def save(
        self, identity: dict[str, Any], arrays: dict[str, np.ndarray],
        progress: dict[str, Any],
    ) -> None:
        metadata = {
            "format": "numpy_fedavg_checkpoint_v1", "identity": identity,
            "provenance": self.provenance, "progress": progress,
        }
        if any(not np.all(np.isfinite(value)) for value in arrays.values()):
            raise FloatingPointError("nonfinite model/residual in checkpoint")
        try:
            encoded_metadata = json.dumps(metadata, allow_nan=False)
        except ValueError as exc:
            raise FloatingPointError("nonfinite checkpoint diagnostics") from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + f".tmp-{os.getpid()}")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle, metadata=np.array(encoded_metadata),
                    **arrays,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
