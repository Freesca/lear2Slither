"""Milestone-A tests: the hub's boundary to ./snake (snake_den/snake_proc.py).

Covers the pure functions that must be exactly right before any subprocess is
spawned: the H8 TOML emitter (round-trips through tomllib AND is accepted by
slither.config.load, with int/float types preserved), the H9 argv builder (one
golden per job type), the eval-profile mapping, and the -progress parser (right
events out, malformed line raises so the caller can fail just that job).
"""
import os
import sys
import tomllib

import pytest

from slither import config, evaluate, progress
from slither.runner import SessionResult
from snake_den import snake_proc
from snake_den.snake_proc import JobSpec


# --- TOML emitter (H8) ------------------------------------------------------

def test_emit_toml_roundtrips_through_tomllib():
    text = snake_proc.emit_toml(config.DEFAULTS)
    assert tomllib.loads(text) == config.DEFAULTS


def test_emit_toml_is_accepted_by_config_load(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text(snake_proc.emit_toml(config.DEFAULTS), encoding="utf-8")
    # No ValueError, and the parsed config equals the built-in defaults.
    assert config.load(str(path)) == config.load()


def test_emit_toml_preserves_int_vs_float_types():
    # dict equality can't catch this (20.0 == 20 in Python), so check types.
    parsed = tomllib.loads(snake_proc.emit_toml(config.DEFAULTS))
    assert type(parsed["board"]["size"]) is int
    assert type(parsed["rewards"]["green"]) is float
    assert parsed["rewards"]["red"] == -10.0
    assert type(parsed["rewards"]["red"]) is float


def test_emit_toml_handles_every_scalar_type():
    section = {"flag": True, "off": False, "n": 3, "f": 1.5, "s": "hi"}
    parsed = tomllib.loads(snake_proc.emit_toml({"x": section}))
    assert parsed["x"] == section
    assert parsed["x"]["flag"] is True       # not the int 1
    assert type(parsed["x"]["n"]) is int
    assert type(parsed["x"]["f"]) is float


def test_emit_toml_rejects_unsupported_scalar():
    with pytest.raises(TypeError):
        snake_proc.emit_toml({"x": {"bad": [1, 2, 3]}})


def test_write_temp_config_is_loadable():
    path = snake_proc.write_temp_config(config.DEFAULTS)
    try:
        assert config.load(path) == config.load()
    finally:
        os.remove(path)


# --- eval profile -> config -------------------------------------------------

def test_eval_config_overrides_board_and_target():
    profile = {"games": 100, "seed": 0, "board": 6, "target": 15}
    cfg = snake_proc.eval_config(profile)
    assert cfg["board"]["size"] == 6
    assert cfg["goal"]["target_length"] == 15
    # Everything else is untouched defaults.
    assert cfg["rewards"] == config.DEFAULTS["rewards"]


def test_eval_config_does_not_mutate_defaults():
    before = config.DEFAULTS["board"]["size"]
    snake_proc.eval_config({"games": 1, "seed": 0, "board": 99, "target": 1})
    assert config.DEFAULTS["board"]["size"] == before


# --- argv builder (H9) ------------------------------------------------------

_PREFIX = [sys.executable, "-u", "-m", "slither", "-config", "C"]


def test_build_argv_train_full():
    spec = JobSpec("train", config.DEFAULTS, sessions=10, seed=5,
                   base_model="base.json", save_path="out.json")
    assert snake_proc.build_argv(spec, "C") == _PREFIX + [
        "-sessions", "10", "-seed", "5",
        "-load", "base.json", "-save", "out.json", "-progress"]


def test_build_argv_train_minimal():
    spec = JobSpec("train", config.DEFAULTS, sessions=1, seed=0)
    argv = snake_proc.build_argv(spec, "C")
    assert argv == _PREFIX + ["-sessions", "1", "-seed", "0", "-progress"]
    assert "-load" not in argv and "-save" not in argv


def test_build_argv_eval():
    spec = JobSpec("eval", config.DEFAULTS, sessions=100, seed=0,
                   base_model="m.json")
    assert snake_proc.build_argv(spec, "C") == _PREFIX + [
        "-load", "m.json", "-dontlearn",
        "-sessions", "100", "-seed", "0", "-progress"]


def test_build_argv_watch_is_greedy_and_has_no_progress():
    spec = JobSpec("watch", config.DEFAULTS, sessions=100, base_model="m.json")
    argv = snake_proc.build_argv(spec, "C")
    assert argv == _PREFIX + ["-load", "m.json", "-visual", "on",
                              "-dontlearn", "-sessions", "100"]
    assert "-progress" not in argv          # an interactive window, not parsed


def test_build_argv_eval_requires_base_model():
    spec = JobSpec("eval", config.DEFAULTS, base_model=None)
    with pytest.raises(ValueError):
        snake_proc.build_argv(spec, "C")


def test_build_argv_unknown_type_raises():
    spec = JobSpec("frobnicate", config.DEFAULTS)
    with pytest.raises(ValueError):
        snake_proc.build_argv(spec, "C")


# --- progress parser --------------------------------------------------------

def test_parse_line_accepts_real_producer_lines():
    # Build lines with the actual product serializer -> a true contract test.
    start = snake_proc.parse_line(progress.start("train", 3))
    assert start["type"] == "start" and start["total_sessions"] == 3

    result = SessionResult(max_length=7, duration=12,
                           death_cause="wall", won=False)
    session = snake_proc.parse_line(
        progress.session(0, result, epsilon=0.99, sessions_trained=1))
    assert session["type"] == "session"
    assert session["max_length"] == 7 and session["i"] == 0

    stats = evaluate.summarize(
        [SessionResult(12, 40, "self", False),
         SessionResult(4, 20, "wall", False)], target_length=10)
    summary = snake_proc.parse_line(progress.summary(stats))
    assert summary["type"] == "summary" and summary["games"] == 2


def test_parse_line_rejects_non_json():
    with pytest.raises(ValueError):
        snake_proc.parse_line("not json {")


def test_parse_line_rejects_non_object():
    with pytest.raises(ValueError):
        snake_proc.parse_line("[1, 2, 3]")


def test_parse_line_rejects_unknown_type():
    with pytest.raises(ValueError):
        snake_proc.parse_line('{"type": "bogus"}')


def test_parse_line_rejects_bad_format_version():
    with pytest.raises(ValueError):
        snake_proc.parse_line(
            '{"type":"start","format_version":999,"mode":"train",'
            '"total_sessions":1}')


def test_local_format_version_matches_product():
    # Pin the H4 mirror against drift, like test_config pins DEFAULTS.
    assert snake_proc.PROGRESS_FORMAT_VERSION == progress.FORMAT_VERSION
