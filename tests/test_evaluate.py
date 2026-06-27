"""Phase 6 tests: the evaluation stats engine.

Covered: summarize math, outcome classification, the golden -stats report
text, and compare()'s determinism + headless quiet + baseline-is-random.
"""
from slither import cli, evaluate
from slither import config as config_module
from slither.runner import SessionResult


def _result(max_length, duration, death_cause, won=False):
    return SessionResult(max_length, duration, death_cause, won)


# --- summarize: counts, rates, distributions -----------------------------

def test_success_rate_counts_length_at_or_above_target():
    results = [
        _result(12, 40, "self"),     # >= 10  -> success
        _result(10, 20, "wall"),     # == 10  -> success (boundary)
        _result(9, 15, "wall"),      # <  10  -> miss
        _result(3, 5, "length"),     # miss
    ]
    stats = evaluate.summarize(results, target_length=10)
    assert stats.games == 4
    assert stats.successes == 2
    assert stats.success_rate == 0.5


def test_distribution_math_is_correct():
    results = [_result(4, 10, "wall"), _result(6, 30, "wall")]
    stats = evaluate.summarize(results, target_length=10)
    assert stats.length.mean == 5.0
    assert stats.length.median == 5.0
    assert stats.length.maximum == 6
    assert round(stats.length.std, 4) == 1.4142     # sample stdev of {4,6}
    assert stats.length.values == (4, 6)
    assert stats.duration.mean == 20.0


def test_std_is_zero_for_a_single_game():
    stats = evaluate.summarize([_result(7, 50, "self")], target_length=10)
    assert stats.length.std == 0.0
    assert stats.duration.std == 0.0


def test_empty_results_do_not_crash():
    stats = evaluate.summarize([], target_length=10)
    assert stats.games == 0
    assert stats.success_rate == 0.0
    assert stats.length.maximum == 0


def test_outcomes_classify_each_terminal_cause():
    results = [
        _result(5, 9, "wall"),
        _result(5, 9, "self"),
        _result(0, 9, "length"),       # red apple starved to length 0
        _result(15, 100, None, won=True),
        _result(4, 1000, None),        # alive at the cap -> truncated
    ]
    stats = evaluate.summarize(results, target_length=10)
    assert stats.outcomes == {
        "wall": 1, "self": 1, "length_zero": 1, "truncated": 1, "won": 1}


# --- the -stats report text (golden) -------------------------------------

def test_format_report_is_exact():
    results = [_result(12, 40, "self"), _result(4, 20, "wall")]
    stats = evaluate.summarize(results, target_length=10)
    assert evaluate.format_report(stats) == (
        "Evaluation over 2 games\n"
        "  Success (length >= 10): 1/2 = 50.0%\n"
        "  Length    mean 8.0  median 8.0  max 12  std 5.7\n"
        "  Duration  mean 30.0  median 30.0  max 40  std 14.1\n"
        "  Outcomes  wall 1  self 1  length-0 0  truncated 0  won 0"
    )


# --- compare(): the learning-curve suite ---------------------------------

def test_compare_is_quiet_and_deterministic(capsys):
    config = config_module.load(overrides={"evaluation": {"games": 15}})
    rows_a = evaluate.compare(config, [], games=15, seed=0)
    out = capsys.readouterr().out
    assert "Game over" not in out            # quiet: no per-game spam

    rows_b = evaluate.compare(config, [], games=15, seed=0)
    assert rows_a == rows_b                   # same seed -> identical stats


def test_compare_baseline_is_a_weak_random_policy():
    config = config_module.load()
    (label, stats), = evaluate.compare(config, [], games=30, seed=0)
    assert label == "baseline (random)"
    assert stats.games == 30
    # A random walker dies young: nowhere near the length-10 goal on average.
    assert stats.length.mean < config.target_length


def test_compare_evaluates_a_saved_model(tmp_path):
    model = tmp_path / "m.json"
    cli.main(["-sessions", "5", "-seed", "1", "-save", str(model)])

    config = config_module.load()
    rows = evaluate.compare(config, [str(model)], games=10, seed=0)
    labels = [label for label, _ in rows]
    assert labels == ["baseline (random)", str(model)]
    assert all(stats.games == 10 for _, stats in rows)


def test_format_curve_has_header_and_a_row_per_model():
    config = config_module.load()
    rows = evaluate.compare(config, [], games=10, seed=0)
    lines = evaluate.format_curve(rows).splitlines()
    assert lines[0].startswith("model")
    assert "success%" in lines[0]
    assert lines[1].startswith("baseline (random)")


# --- CLI integration -----------------------------------------------------

def test_stats_flag_prints_report(capsys):
    code = cli.main(["-sessions", "5", "-seed", "0", "-dontlearn", "-stats"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Evaluation over 5 games" in out
    assert "Success (length >= 10):" in out


def test_compare_flag_prints_table_and_skips_normal_run(capsys, tmp_path):
    model = tmp_path / "m.json"
    cli.main(["-sessions", "5", "-seed", "1", "-save", str(model)])
    capsys.readouterr()

    code = cli.main(["-compare", str(model), "-seed", "0",
                     "-config", _tiny_games_config(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "success%" in out
    assert "baseline (random)" in out
    assert "Game over" not in out             # compare path is quiet


def _tiny_games_config(tmp_path):
    """A config file with a small games count, to keep the test fast."""
    path = tmp_path / "fast.toml"
    path.write_text("[evaluation]\ngames = 8\n", encoding="utf-8")
    return str(path)
