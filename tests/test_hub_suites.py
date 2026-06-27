"""Rework tests: named eval suites (snake_den/suites.py).

The standard suite is always first and immutable; custom suites validate, save,
list (sorted), survive a persistence round-trip, and reduce to the registry's
four-key profile.
"""
import pytest

from snake_den import persistence, registry, suites


def test_standard_is_always_present_and_first():
    s = suites.Suites(persistence.empty()["suites"])
    assert s.names()[0] == "standard"
    assert s.get("standard") == suites.STANDARD


def test_save_list_sorted_and_get():
    table = {}
    s = suites.Suites(table)
    s.save({"name": "zeta", "games": 50, "seed": 1,
            "board": 10, "target": 10, "step_cap": 500})
    s.save({"name": "alpha", "games": 200, "seed": 0,
            "board": 15, "target": 15, "step_cap": 1000})
    assert s.names() == ["standard", "alpha", "zeta"]     # standard, then sort
    assert s.get("alpha")["board"] == 15


def test_save_rejects_reserved_and_empty_names():
    s = suites.Suites({})
    with pytest.raises(ValueError):
        s.save({"name": "standard", "games": 10})
    with pytest.raises(ValueError):
        s.save({"name": "  ", "games": 10})


def test_save_rejects_bad_numbers():
    s = suites.Suites({})
    with pytest.raises(ValueError):
        s.save({"name": "bad", "games": 0})               # < 1
    with pytest.raises(ValueError):
        s.save({"name": "bad", "games": 10, "board": -1})


def test_remove_custom_suite():
    table = {}
    s = suites.Suites(table)
    s.save({"name": "tmp", "games": 10, "seed": 0,
            "board": 10, "target": 10, "step_cap": 1000})
    s.remove("tmp")
    assert s.names() == ["standard"]


def test_profile_is_the_registry_key_parts():
    profile = suites.profile(suites.STANDARD)
    assert profile == {"games": 100, "seed": 0, "board": 10, "target": 10}
    # a suite maps cleanly onto the registry's stable score key:
    assert registry.profile_key(profile) == registry.profile_key(
        suites.profile(suites.STANDARD))


def test_roundtrip_through_persistence(tmp_path):
    path = str(tmp_path / "hub.json")
    data = persistence.empty()
    suites.Suites(data["suites"]).save(
        {"name": "fast", "games": 20, "seed": 3,
         "board": 8, "target": 10, "step_cap": 400})
    persistence.save(path, data)

    reloaded = suites.Suites(persistence.load(path)["suites"])
    assert reloaded.get("fast")["games"] == 20
