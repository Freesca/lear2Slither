"""Tests for config defaults, file/override precedence, validation."""
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

import pytest

from slither import config as cfg

DEFAULT_TOML = (
    Path(__file__).resolve().parents[1] / "configs" / "default.toml"
)


def test_defaults_build_a_valid_config():
    c = cfg.load()
    assert c.board.size == 10
    assert c.board.initial_length == 3
    assert c.target_length == 10
    assert c.rewards.green == 20.0
    assert c.hyperparameters.gamma == 0.9
    assert c.evaluation.step_cap == 1000


def test_default_toml_mirrors_builtin_defaults():
    # The committed default.toml must equal the in-code DEFAULTS so the two
    # can never drift.
    with open(DEFAULT_TOML, "rb") as handle:
        on_disk = tomllib.load(handle)
    assert on_disk == cfg.DEFAULTS


# --- precedence: defaults < file < override dict ------------------------

def test_file_overrides_defaults(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[board]\nsize = 20\n")
    c = cfg.load(str(path))
    assert c.board.size == 20
    assert c.board.green_apples == 2          # untouched default


def test_override_dict_beats_file(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[board]\nsize = 20\n")
    c = cfg.load(str(path), {"board": {"size": 7}})
    assert c.board.size == 7


# --- typos fail loudly ---------------------------------------------------

def test_unknown_section_raises(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[nope]\nx = 1\n")
    with pytest.raises(ValueError):
        cfg.load(str(path))


def test_unknown_key_raises(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[board]\nwidth = 9\n")
    with pytest.raises(ValueError):
        cfg.load(str(path))


# --- validation ----------------------------------------------------------

def test_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        cfg.load(overrides={"board": {"size": 1}})        # < 2
    with pytest.raises(ValueError):
        cfg.load(overrides={"learning": {"gamma": 1.5}})  # > 1
    with pytest.raises(ValueError):
        cfg.load(overrides={"learning": {"alpha": 0.0}})  # not > 0


def test_capacity_check():
    # A 2x2 board (4 cells) cannot hold a length-3 snake plus 3 apples.
    with pytest.raises(ValueError):
        cfg.load(overrides={"board": {"size": 2}})


def test_unimplemented_strategy_names_its_gate():
    with pytest.raises(ValueError, match="T2"):
        cfg.load(overrides={"exploration": {"strategy": "optimistic"}})
    with pytest.raises(ValueError, match="T3"):
        cfg.load(overrides={"learning": {"alpha_strategy": "visit_count"}})
