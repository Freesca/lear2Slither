"""Smoke test: the real ./snake end-to-end contract.

Unlike the unit tests (which use fake children), this spawns the actual
``python -m slither`` through the full JobManager + snake_proc path: a short
train job must produce a model file and a parsed summary, and an eval job on
that model must yield a score the registry can file. This is the true contract
the hub depends on; kept small (tiny board, few sessions) so it still runs in
about a second or two.
"""
import json
import time
from copy import deepcopy

from slither.config import DEFAULTS
from snake_den import jobs, persistence, registry, snake_proc
from snake_den.snake_proc import JobSpec


def _run_until(manager, predicate, timeout=30.0):
    """Poll the real subprocess until it finishes (generous timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        manager.poll()
        if predicate():
            return
        time.sleep(0.02)
    manager.poll()
    if not predicate():
        raise AssertionError("timed out waiting for the real ./snake job")


def test_train_then_eval_end_to_end(tmp_path):
    model = tmp_path / "smoke.json"

    train_config = deepcopy(DEFAULTS)
    train_config["board"]["size"] = 7          # small board -> fast sessions
    train = JobSpec("train", train_config, sessions=5, seed=0,
                    save_path=str(model))

    manager = jobs.JobManager(pool_size=1)
    tid = manager.submit(train)
    _run_until(manager, lambda: manager.job(tid).is_terminal)

    job = manager.job(tid)
    assert job.status == jobs.FINISHED, job.error
    assert model.exists()                       # -save produced the model
    assert job.summary["games"] == 5
    assert len(job.sessions) == 5               # the live curve, one per game

    # Evaluate the freshly trained model under a hub eval profile.
    profile = {"games": 4, "seed": 0, "board": 7, "target": 10}
    eval_spec = JobSpec("eval", snake_proc.eval_config(profile),
                        sessions=profile["games"], seed=profile["seed"],
                        base_model=str(model), eval_profile=profile)
    eid = manager.submit(eval_spec)
    _run_until(manager, lambda: manager.job(eid).is_terminal)

    eval_job = manager.job(eid)
    assert eval_job.status == jobs.FINISHED, eval_job.error
    assert eval_job.summary["games"] == 4

    reg = registry.Registry(persistence.empty())
    reg.record_score(str(model), profile, eval_job.summary)
    score = reg.score(str(model), profile)
    assert score is not None
    assert score["games"] == 4 and score["target"] == 10
    manager.shutdown()


def test_train_with_custom_scheme_records_it(tmp_path):
    # The Train tab's contract: a chosen [state] scheme rides into the temp
    # TOML and slither records it in the (v3) model, so it replays correctly.
    model = tmp_path / "scheme.json"
    config = deepcopy(DEFAULTS)
    config["board"]["size"] = 7
    config["state"] = {"warn": True, "caution": True,
                       "green_far": True, "red_far": True}
    spec = JobSpec("train", config, sessions=5, seed=0, save_path=str(model))

    manager = jobs.JobManager(pool_size=1)
    tid = manager.submit(spec)
    _run_until(manager, lambda: manager.job(tid).is_terminal)
    assert manager.job(tid).status == jobs.FINISHED, manager.job(tid).error

    data = json.loads(model.read_text(encoding="utf-8"))
    assert data["format_version"] == 3
    assert data["state"]["caution"] is True
    assert data["curve"]                 # 5 sessions -> a non-empty curve
    manager.shutdown()
