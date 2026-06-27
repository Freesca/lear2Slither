"""Tests for CLI parsing, flag correctness, and end-to-end wiring."""
import pytest

from slither import cli


# --- parsing -------------------------------------------------------------

def test_defaults_when_no_flags():
    args = cli.build_parser().parse_args([])
    assert args.sessions == 1
    assert args.visual == "off"
    assert args.dontlearn is False
    assert args.step_by_step is False
    assert args.save is None
    assert args.load is None
    assert args.seed is None


def test_parses_subject_eval_example():
    args = cli.build_parser().parse_args(
        ["-visual", "on", "-load", "models/100sess.txt",
         "-sessions", "10", "-dontlearn", "-step-by-step"])
    assert args.visual == "on"
    assert args.load == "models/100sess.txt"
    assert args.sessions == 10
    assert args.dontlearn is True
    assert args.step_by_step is True


def test_state_spec_parses_into_overrides():
    assert cli._parse_state_spec("default") == cli._DEFAULT_STATE
    assert cli._parse_state_spec("warn,caution,green_far,red_far") == {
        "warn": True, "caution": True, "green_far": True, "red_far": True,
        "body_far": False}
    # listed = on, unlisted = off:
    assert cli._parse_state_spec("body_far") == {
        "warn": False, "caution": False, "green_far": False, "red_far": False,
        "body_far": True}


def test_state_spec_rejects_unknown_feature():
    with pytest.raises(ValueError):
        cli._parse_state_spec("warn,bogus")


def test_state_flag_runs_with_caution_scheme(tmp_path):
    model = tmp_path / "m.json"
    code = cli.main(["-sessions", "3", "-seed", "0",
                     "-state", "warn,caution,green_far,red_far",
                     "-save", str(model)])
    assert code == 0
    # The saved model records the caution scheme, and reloads with it.
    from slither import model_io
    assert model_io.load(str(model)).scheme.caution is True


def test_no_prefix_abbreviation():
    # allow_abbrev=False: -s must not resolve to -sessions/-save/-seed.
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["-s", "5"])


def test_sessions_must_be_positive():
    with pytest.raises(SystemExit):
        cli.main(["-sessions", "0"])


# --- end-to-end wiring (headless) ---------------------------------------

def test_main_runs_headless_and_prints_game_over(capsys):
    code = cli.main(["-sessions", "2", "-seed", "0", "-visual", "off"])
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("Game over, max length = ") == 2


def test_main_save_then_load_round_trip(tmp_path, capsys):
    model = tmp_path / "m.json"
    code = cli.main(["-sessions", "3", "-seed", "1", "-save", str(model)])
    assert code == 0
    out = capsys.readouterr().out
    assert f"Save learning state in {model}" in out
    assert model.exists()

    code = cli.main(["-sessions", "2", "-seed", "1",
                     "-load", str(model), "-dontlearn"])
    assert code == 0
    out = capsys.readouterr().out
    assert f"Load trained model from {model}" in out
    assert out.count("Game over") == 2


def test_seed_makes_the_run_reproducible(capsys):
    cli.main(["-sessions", "5", "-seed", "7"])
    first = capsys.readouterr().out
    cli.main(["-sessions", "5", "-seed", "7"])
    second = capsys.readouterr().out
    assert first == second


def test_many_sessions_run(capsys):
    # Smoke: a long headless run completes without error (and fast).
    code = cli.main(["-sessions", "100", "-seed", "0"])
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("Game over") == 100


def test_headless_run_never_imports_pygame():
    # The headless-safety invariant: -visual off must not load pygame, so the
    # test suite and headless training stay display-free. gui.Presenter is the
    # only pygame importer and is reached only via -visual on (lazy import).
    import sys
    cli.main(["-sessions", "3", "-seed", "0"])
    assert "pygame" not in sys.modules
    assert "slither.gui" not in sys.modules
