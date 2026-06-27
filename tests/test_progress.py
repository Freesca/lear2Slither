"""Tests for the -progress JSON-lines stream (the hub contract).

Golden exact strings for the deterministic lines (start, session), structural
checks for the float-bearing summary, and a CLI integration test proving
-progress yields pure JSON with no human prose.
"""
import json

from slither import cli, evaluate, progress
from slither.runner import SessionResult


def test_start_line_is_exact():
    assert progress.start("train", 100) == (
        '{"format_version":1,"mode":"train","total_sessions":100,'
        '"type":"start"}')


def test_session_line_is_exact():
    result = SessionResult(max_length=3, duration=2,
                           death_cause="wall", won=False)
    line = progress.session(0, result, epsilon=0.99, sessions_trained=1)
    assert line == (
        '{"death_cause":"wall","duration":2,"epsilon":0.99,"i":0,'
        '"max_length":3,"sessions_trained":1,"type":"session","won":false}')


def test_session_line_handles_no_death_cause():
    result = SessionResult(max_length=15, duration=200,
                           death_cause=None, won=True)
    payload = json.loads(
        progress.session(7, result, epsilon=0.0, sessions_trained=42))
    assert payload["death_cause"] is None
    assert payload["won"] is True
    assert payload["i"] == 7


def test_summary_line_serializes_a_suitestats():
    results = [SessionResult(12, 40, "self", False),
               SessionResult(4, 20, "wall", False)]
    stats = evaluate.summarize(results, target_length=10)
    payload = json.loads(progress.summary(stats))
    assert payload["type"] == "summary"
    assert payload["games"] == 2
    assert payload["success_rate"] == 0.5
    assert payload["target_length"] == 10
    assert payload["length"]["mean"] == 8.0
    assert payload["length"]["max"] == 12
    assert payload["outcomes"] == {
        "wall": 1, "self": 1, "length_zero": 0, "truncated": 0, "won": 0}


# --- CLI integration: pure JSON, no human prose ------------------------------

def test_progress_flag_emits_pure_json(capsys):
    code = cli.main(["-sessions", "3", "-seed", "0", "-progress"])
    assert code == 0
    out = capsys.readouterr().out
    lines = out.splitlines()

    # Every line parses as JSON; no "Game over"/"Save"/"Load" prose leaks.
    events = [json.loads(line) for line in lines]
    assert "Game over" not in out
    assert "Save" not in out and "Load" not in out

    assert events[0]["type"] == "start"
    assert events[0]["mode"] == "train"
    assert events[0]["total_sessions"] == 3
    session_events = [e for e in events if e["type"] == "session"]
    assert len(session_events) == 3
    assert [e["i"] for e in session_events] == [0, 1, 2]
    assert events[-1]["type"] == "summary"
    assert events[-1]["games"] == 3


def test_progress_eval_mode_labels_start_as_eval(capsys, tmp_path):
    model = tmp_path / "m.json"
    cli.main(["-sessions", "2", "-seed", "1", "-save", str(model)])
    capsys.readouterr()

    cli.main(["-load", str(model), "-sessions", "2", "-seed", "0",
              "-dontlearn", "-progress"])
    out = capsys.readouterr().out
    start = json.loads(out.splitlines()[0])
    assert start["mode"] == "eval"
    # The -load human line must not leak into the JSON stream.
    assert "Load trained model" not in out


def test_progress_suppresses_human_stats(capsys):
    # -progress wins over -stats: JSON summary, not the human report block.
    cli.main(["-sessions", "2", "-seed", "0", "-progress", "-stats"])
    out = capsys.readouterr().out
    assert "Evaluation over" not in out
    assert json.loads(out.splitlines()[-1])["type"] == "summary"
