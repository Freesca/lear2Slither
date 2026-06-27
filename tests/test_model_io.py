"""Phase 5 tests: model save/load round-trip preserves Q exactly."""
import random

import pytest

from slither import config as cfg
from slither import model_io
from slither.agent import Agent
from slither.interpreter import DEFAULT_SCHEME, Scheme


def _agent_with_state():
    """An agent carrying a couple of hand-set Q-rows and a session count."""
    c = cfg.load()
    agent = Agent(c.hyperparameters, random.Random(0))
    agent.q = {"DgN": [1.2, -0.5, 3.1, -99.0], "Ggg": [0.0, 2.0, 0.0, 0.0]}
    agent.n = {"DgN": [12, 4, 30, 1], "Ggg": [0, 5, 0, 0]}
    agent.sessions_trained = 42
    return agent, c


def test_save_load_round_trip_preserves_q(tmp_path):
    agent, c = _agent_with_state()
    path = tmp_path / "m.json"
    model_io.save(str(path), agent, c)
    data = model_io.load(str(path))
    assert data.q == agent.q
    assert data.n == agent.n
    assert data.sessions_trained == 42


def test_resume_into_agent_restores_epsilon(tmp_path):
    agent, c = _agent_with_state()
    path = tmp_path / "m.json"
    model_io.save(str(path), agent, c)
    data = model_io.load(str(path))
    resumed = Agent(c.hyperparameters, random.Random(0),
                    q=data.q, n=data.n,
                    sessions_trained=data.sessions_trained)
    assert resumed.sessions_trained == 42
    # epsilon is a pure function of sessions_trained x hyperparameters.
    hp = c.hyperparameters
    expected = max(hp.epsilon_min, hp.epsilon_start * hp.epsilon_decay ** 42)
    assert resumed.epsilon == pytest.approx(expected)


def test_save_is_byte_stable(tmp_path):
    agent, c = _agent_with_state()
    path = tmp_path / "m.json"
    model_io.save(str(path), agent, c)
    first = path.read_text(encoding="utf-8")
    model_io.save(str(path), agent, c)
    assert path.read_text(encoding="utf-8") == first


def test_bad_format_version_raises(tmp_path):
    path = tmp_path / "m.json"
    path.write_text('{"format_version": 99, "q": {}, "visits": {},'
                    ' "sessions_trained": 0}', encoding="utf-8")
    with pytest.raises(ValueError):
        model_io.load(str(path))


def test_missing_key_raises(tmp_path):
    path = tmp_path / "m.json"
    path.write_text('{"format_version": 2, "visits": {},'
                    ' "sessions_trained": 0}', encoding="utf-8")
    with pytest.raises(ValueError):
        model_io.load(str(path))


def test_round_trip_preserves_the_scheme(tmp_path):
    agent, c = _agent_with_state()
    scheme = Scheme(caution=True, green_far=False)
    path = tmp_path / "m.json"
    model_io.save(str(path), agent, c, scheme)
    assert model_io.load(str(path)).scheme == scheme


def test_v1_file_loads_as_default_scheme(tmp_path):
    # The committed deliverable models are format v1 (no state block); they
    # must keep loading, as the legacy 7-letter scheme, with no retrain.
    path = tmp_path / "legacy.json"
    path.write_text('{"format_version": 1, "q": {}, "visits": {},'
                    ' "sessions_trained": 0}', encoding="utf-8")
    assert model_io.load(str(path)).scheme == DEFAULT_SCHEME


def test_curve_round_trips_and_is_downsampled(tmp_path):
    # v3 embeds the per-session max-length curve, downsampled to <=100 points.
    agent, c = _agent_with_state()
    path = tmp_path / "m.json"
    model_io.save(str(path), agent, c, curve=[3, 4, 5, 6])
    assert model_io.load(str(path)).curve == [3.0, 4.0, 5.0, 6.0]
    model_io.save(str(path), agent, c, curve=list(range(500)))
    assert len(model_io.load(str(path)).curve) == 100


def test_v1_v2_files_load_with_empty_curve(tmp_path):
    # Pre-v3 files have no "curve"; they load with an empty one (no crash).
    path = tmp_path / "legacy.json"
    path.write_text('{"format_version": 2, "q": {}, "visits": {},'
                    ' "sessions_trained": 0}', encoding="utf-8")
    assert model_io.load(str(path)).curve == []
