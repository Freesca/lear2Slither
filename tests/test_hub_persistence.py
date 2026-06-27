"""Tests for the hub data file (snake_den/persistence.py).

The store must be fresh-clone safe (a missing or corrupt file yields an empty
skeleton, never a crash) and atomic (a failed write leaves the previous file
intact and no .tmp litter, so a crash can't lose data).
"""
from pathlib import Path

import pytest

from snake_den import persistence


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


def test_load_missing_returns_empty(tmp_path):
    assert persistence.load(str(tmp_path / "nope.json")) == persistence.empty()


def test_load_corrupt_returns_empty(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    assert persistence.load(str(bad)) == persistence.empty()


def test_load_keeps_only_known_keys_and_fills_missing(tmp_path):
    path = tmp_path / "hub.json"
    path.write_text('{"models": {"m": 1}, "junk": 9}', encoding="utf-8")
    data = persistence.load(str(path))
    assert data == {"models": {"m": 1}, "history": [],
                    "settings": {}, "suites": {}}


def test_save_then_load_roundtrips(tmp_path):
    path = str(tmp_path / "hub.json")
    data = persistence.empty()
    data["models"]["m.json"] = {"sessions": 10, "scores": {}}
    data["history"].append({"job": "train", "model": "m.json"})
    data["settings"]["pool_size"] = 3
    persistence.save(path, data)
    assert persistence.load(path) == data


def test_save_is_atomic_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "hub.json"
    persistence.save(str(path), {"models": {"a": 1},
                                 "history": [], "settings": {}})
    original = path.read_text(encoding="utf-8")

    monkeypatch.setattr(persistence.json, "dump", _raise)
    with pytest.raises(RuntimeError):
        persistence.save(str(path), {"models": {"b": 2},
                                     "history": [], "settings": {}})

    assert path.read_text(encoding="utf-8") == original          # untouched
    leftovers = [p.name for p in tmp_path.iterdir()
                 if p.suffix == ".tmp"]
    assert leftovers == []                                        # no litter


def test_save_creates_missing_directories(tmp_path):
    path = tmp_path / "a" / "b" / "hub.json"
    persistence.save(str(path), persistence.empty())
    assert Path(path).exists()
