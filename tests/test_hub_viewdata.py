"""Tests for Q-table view data (snake_den/viewdata.py).

Q-table row assembly (sorted, argmax/greedy highlight) and the policy
fingerprint (greedy-Q-value sign per state) the Models list renders. Pure --
no pygame import.
"""
from types import SimpleNamespace

from slither import config as config_module
from slither import model_io
from slither.interpreter import DEFAULT_SCHEME, Scheme
from slither.model_io import ModelData
from snake_den import scheme as scheme_mod
from snake_den import viewdata


# --- qtable_rows ------------------------------------------------------------

def test_qtable_rows_sorted_with_argmax():
    md = ModelData(
        q={"BBB": [0.0, 0.0, 0.0, 0.0], "AAA": [1.0, 9.0, 2.0, 3.0]},
        n={"BBB": [0, 0, 0, 0], "AAA": [1, 5, 2, 3]},
        sessions_trained=10)
    rows = viewdata.qtable_rows(md)
    assert [r[0] for r in rows] == ["AAA", "BBB"]      # sorted by state
    assert rows[0][3] == 1                              # argmax of AAA = col 1
    assert rows[1][3] == 0                              # all-zero -> first col


def test_qtable_rows_empty_model():
    md = ModelData(q={}, n={}, sessions_trained=0)
    assert viewdata.qtable_rows(md) == []


def test_load_qtable_rows_from_a_real_file(tmp_path):
    cfg = config_module.load()
    agent = SimpleNamespace(q={"DgN": [1.0, -2.0, 3.0, 0.5]},
                            n={"DgN": [4, 1, 9, 2]}, sessions_trained=7)
    path = str(tmp_path / "m.json")
    model_io.save(path, agent, cfg)
    rows = viewdata.load_qtable_rows(path)
    assert rows == [("DgN", [1.0, -2.0, 3.0, 0.5], [4, 1, 9, 2], 2)]


# --- policy_summary (the Models-list fingerprint) ---------------------------

def test_policy_summary_counts_greedy_mix_and_value_signs():
    md = ModelData(
        q={"AAA": [9.0, 0.0, 0.0, 0.0],      # greedy F, best > 0  -> +1
           "BBB": [0.0, 9.0, 0.0, 0.0],      # greedy L, best > 0  -> +1
           "CCC": [-2.0, -1.0, -3.0, -4.0]},  # greedy L, best < 0  -> -1
        n={"AAA": [1, 0, 0, 0], "BBB": [0, 1, 0, 0], "CCC": [0, 1, 0, 0]},
        sessions_trained=10, scheme=DEFAULT_SCHEME)
    summary = viewdata.policy_summary(md)
    assert summary["coverage"] == 3
    assert summary["mix"] == [1, 2, 0, 0]              # one F, two L
    # fingerprint is the greedy-value SIGN per sorted state (value map).
    assert summary["fingerprint"] == [1, 1, -1]
    assert summary["total"] == scheme_mod.state_count(DEFAULT_SCHEME.as_dict())


def test_policy_summary_total_tracks_the_models_scheme():
    md = ModelData(q={}, n={}, sessions_trained=0,
                   scheme=Scheme(caution=True))
    summary = viewdata.policy_summary(md)
    assert summary["coverage"] == 0
    assert summary["total"] == scheme_mod.state_count(
        Scheme(caution=True).as_dict())            # k = 8 -> 288
