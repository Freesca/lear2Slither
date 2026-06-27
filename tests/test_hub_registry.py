"""Milestone-A tests: the model registry (snake_den/registry.py).

Proves the two contracts that matter: the registry reads sessions + config off
a *real* product-saved model file (sessions via model_io, config via raw JSON,
per H4), and eval scores are isolated per eval-profile key so one profile never
clobbers another's number (gate H2).
"""
import json
import os
import random

import pytest

from slither import config as config_module
from slither import evaluate, model_io, progress
from slither.agent import Agent
from slither.runner import SessionResult
from snake_den import persistence, registry

_P10 = {"games": 100, "seed": 0, "board": 10, "target": 10}
_P15 = {"games": 100, "seed": 0, "board": 10, "target": 15}


@pytest.fixture
def model_file(tmp_path):
    """A real model file written by the product's own model_io.save."""
    cfg = config_module.load()
    agent = Agent(cfg.hyperparameters, random.Random(0))
    agent.q = {"DgN": [1.0, 2.0, 3.0, 4.0]}
    agent.n = {"DgN": [1, 2, 3, 4]}
    agent.sessions_trained = 10
    path = str(tmp_path / "m.json")
    model_io.save(path, agent, cfg)
    return path


def _summary(results, target):
    """The summary dict a finished eval job would hold (real product path)."""
    stats = evaluate.summarize(results, target)
    return json.loads(progress.summary(stats))


def _half_target10():
    # one game >= 10, one below -> success_rate 0.5
    return _summary([SessionResult(12, 40, "self", False),
                     SessionResult(4, 20, "wall", False)], target=10)


def test_register_reads_sessions_and_config(model_file):
    reg = registry.Registry(persistence.empty())
    entry = reg.register(model_file)
    assert entry["sessions"] == 10
    assert entry["config"]["board"]["size"] == 10     # from cfg.as_dict()
    assert entry["scores"] == {}


def test_record_score_autoregisters(model_file):
    reg = registry.Registry(persistence.empty())
    reg.record_score(model_file, _P10, _half_target10())
    key = registry.normalize(model_file)         # keys are normalized
    assert key in reg.models()
    assert reg.models()[key]["sessions"] == 10
    assert reg.score(model_file, _P10)["success_pct"] == 50.0


def test_scores_isolated_per_profile(model_file):
    reg = registry.Registry(persistence.empty())
    reg.record_score(model_file, _P10, _half_target10())
    # a second profile (different target) must not overwrite the first
    reg.record_score(model_file, _P15, _summary(
        [SessionResult(8, 30, "wall", False)], target=15))

    assert reg.score(model_file, _P10)["target"] == 10
    assert reg.score(model_file, _P10)["success_pct"] == 50.0
    assert reg.score(model_file, _P15)["target"] == 15
    assert reg.score(model_file, _P15)["success_pct"] == 0.0


def test_profile_key_stable_and_distinct():
    assert registry.profile_key(_P10) == registry.profile_key(dict(_P10))
    assert registry.profile_key(_P10) != registry.profile_key(_P15)


def test_unknown_model_scores_none(model_file):
    reg = registry.Registry(persistence.empty())
    assert reg.score("never_seen.json", _P10) is None


def test_join_and_literal_collapse_to_one_key():
    # The slash bug: os.path.join (backslash on Windows) and the forward-slash
    # literal name the same file -- they must reduce to one registry key.
    assert (registry.normalize(os.path.join("models", "new.json"))
            == registry.normalize("models/new.json"))


def test_register_keys_by_normalized_path(model_file):
    # Registering the same file under a non-canonical spelling does not split
    # it: the entry lands under (and is found by) the normalized key.
    reg = registry.Registry(persistence.empty())
    reg.register(os.path.join(os.path.dirname(model_file) + os.sep,
                              ".", os.path.basename(model_file)))
    assert registry.normalize(model_file) in reg.models()
    assert len(reg.models()) == 1


def test_migrate_merges_equivalent_keys():
    # A data file carrying two spellings of one model collapses on load, and
    # their scores merge rather than one clobbering the other.
    data = persistence.empty()
    data["models"]["models/new.json"] = {"sessions": 10, "scores": {"a": 1}}
    data["models"]["models/./new.json"] = {"sessions": 10, "scores": {"b": 2}}
    reg = registry.Registry(data)
    models = reg.models()
    assert set(models) == {"models/new.json"}
    assert set(models["models/new.json"]["scores"]) == {"a", "b"}


def test_discover_returns_normalized_paths(tmp_path, model_file):
    # model_file lives in tmp_path; discovery of that dir finds it, normalized.
    found = registry.discover(str(tmp_path))
    assert registry.normalize(model_file) in found
    assert all(name == registry.normalize(name) for name in found)


def test_survives_persistence_roundtrip(model_file, tmp_path):
    path = str(tmp_path / "hub.json")
    data = persistence.empty()
    registry.Registry(data).record_score(model_file, _P10, _half_target10())
    persistence.save(path, data)

    reloaded = registry.Registry(persistence.load(path))
    assert reloaded.score(model_file, _P10)["success_pct"] == 50.0
