"""Binary output and content identities for resumable stages."""
from __future__ import annotations

import csv
from contextvars import ContextVar
from functools import wraps
import hashlib
from inspect import signature
import io
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from ..io import atomic_text, fingerprint
from ..sampling import SampleSet

_LOCKED_OUTPUTS = ContextVar("locked_decoder_training_outputs", default=frozenset())


def single_writer(action):
    """Serialize writes to one output directory, including resumable stages."""
    parameters = signature(action)

    @wraps(action)
    def locked(*args, **kwargs):
        root = Path(parameters.bind(*args, **kwargs).arguments["output_dir"]).resolve()
        held = _LOCKED_OUTPUTS.get()
        if root in held:
            return action(*args, **kwargs)
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".writer.lock").open("a+b") as lock:
            try:
                if os.name == "nt":
                    from msvcrt import LK_NBLCK, locking

                    lock.seek(0)
                    locking(lock.fileno(), LK_NBLCK, 1)
                else:
                    from fcntl import LOCK_EX, LOCK_NB, flock

                    flock(lock, LOCK_EX | LOCK_NB)
            except OSError as error:
                raise ValueError(f"Another process is writing results in {root}") from error
            token = _LOCKED_OUTPUTS.set(held | {root})
            try:
                return action(*args, **kwargs)
            finally:
                _LOCKED_OUTPUTS.reset(token)

    return locked


def atomic_arrays(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temp = Path(handle.name)
        try:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_identity(samples: SampleSet, *, include_observables: bool = True) -> dict:
    result = {"config": samples.metadata["config"], "detectors_sha256": file_digest(samples.detector_path)}
    if include_observables:
        result["observables_sha256"] = file_digest(samples.observable_path)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    output = io.StringIO()
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_text(path, output.getvalue())


def model_identity(priors: np.ndarray) -> str:
    return fingerprint(np.asarray(priors, dtype=float))
