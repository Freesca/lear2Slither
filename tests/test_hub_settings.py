"""Tests for hub settings (snake_den/settings.py).

Defaults fill in for a fresh data file, writes land in the shared table, and
the whole thing survives a persistence round-trip. The eval profile no longer
lives here -- it moved onto the Eval page as named suites (suites.py).
"""
from snake_den import persistence, settings


def test_defaults_on_empty_table():
    s = settings.Settings(persistence.empty()["settings"])
    assert s.pool_size is None                  # auto -> JobManager decides
    assert s.models_path == "models"
    assert s.theme == "pixel"
    assert s.reduced_motion is False
    assert s.scanlines is False


def test_set_and_read_back_mutates_shared_table():
    table = {}
    s = settings.Settings(table)
    s.pool_size = 3
    s.theme = "dark"
    s.reduced_motion = True
    s.scanlines = True
    assert s.pool_size == 3
    assert s.theme == "dark"
    assert s.reduced_motion is True
    assert s.scanlines is True
    assert table["pool_size"] == 3              # wrote through to the dict


def test_roundtrip_through_persistence(tmp_path):
    path = str(tmp_path / "hub.json")
    data = persistence.empty()
    s = settings.Settings(data["settings"])
    s.pool_size = 2
    s.models_path = "trained"
    s.scanlines = True
    persistence.save(path, data)

    s2 = settings.Settings(persistence.load(path)["settings"])
    assert s2.pool_size == 2
    assert s2.models_path == "trained"
    assert s2.scanlines is True
