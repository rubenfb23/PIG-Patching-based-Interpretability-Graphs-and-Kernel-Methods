"""I/O helpers for experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


RESULTS_ROOT = Path(os.environ.get("CLMI_RESULTS_ROOT", "results"))


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(run_id: str) -> Path:
    """Return and create run directory."""
    return ensure_dir(RESULTS_ROOT / "runs" / run_id)


def checkpoints_dir(run_id: str) -> Path:
    """Return and create checkpoint directory."""
    return ensure_dir(RESULTS_ROOT / "checkpoints" / run_id)


def cache_dir(kind: str) -> Path:
    """Return and create cache directory for a specific artifact type."""
    return ensure_dir(RESULTS_ROOT / "cache" / kind)


def tables_dir() -> Path:
    """Return and create tables directory."""
    return ensure_dir(RESULTS_ROOT / "tables")


def figures_dir() -> Path:
    """Return and create figures directory."""
    return ensure_dir(RESULTS_ROOT / "figures")


def save_json(path: Path, payload: Any) -> None:
    """Write json file with deterministic formatting."""
    ensure_dir(path.parent)
    serializable = asdict(payload) if is_dataclass(payload) else payload
    path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=True)
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load json file."""
    return json.loads(path.read_text())


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    """Persist dataframe to csv without index."""
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    """Append a dataframe to csv creating header on first write.

    Uses a lock file to allow safe concurrent appends from multiple processes.
    """
    import fcntl

    ensure_dir(path.parent)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            exists = path.exists()
            frame.to_csv(path, mode="a", header=not exists, index=False)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def stable_hash(parts: list[str]) -> str:
    """Generate a stable short hash for cache keys."""
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
