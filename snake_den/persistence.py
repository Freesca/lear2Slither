"""Persistence: the hub's data file (atomic load/save).

One JSON file the hub owns -- ``snake_den/hub_data.json`` -- holding its
cross-restart state: the model registry, a job history log, the user's
settings, and the saved eval suites. This module is the low-level store: it
knows the top-level keys exist and nothing about what is inside them
(registry.py / settings.py / suites.py own those). Two rules it enforces:

- **Fresh-clone safe:** a missing *or* corrupt file is not an error -- the hub
  starts from an empty skeleton rather than crashing (no-crash rule).
- **Atomic writes:** the data is written to a temp file in the same directory
  and then ``os.replace``-d into place. ``os.replace`` is atomic on POSIX and
  Windows, so a crash mid-write leaves the previous file intact -- never a
  torn, half-written JSON that would lose every registered model.
"""
import json
import os
import tempfile
from copy import deepcopy

_SKELETON = {"models": {}, "history": [], "settings": {}, "suites": {}}


def empty():
    """A fresh, empty hub-data dict (the fresh-clone state)."""
    return deepcopy(_SKELETON)


def load(path):
    """Load the hub data file, or an empty skeleton if missing/corrupt.

    Only the four known top-level keys are kept; anything else in the file is
    dropped, and any missing key is filled from the skeleton, so callers always
    get a well-shaped dict.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return empty()
    if not isinstance(data, dict):
        return empty()
    merged = empty()
    for key in merged:
        if key in data:
            merged[key] = data[key]
    return merged


def save(path, data):
    """Atomically write ``data`` as JSON (temp file + ``os.replace``).

    The temp file is created in the destination directory so the replace is a
    same-filesystem rename (atomic); on any failure the temp file is removed so
    no ``.tmp`` litter is left behind.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
