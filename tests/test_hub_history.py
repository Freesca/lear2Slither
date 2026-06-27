"""Run-history record tests (registry.history_record). Pure -- no pygame.

The Runs tab's source of truth: a terminal train/eval job becomes an archive
record. Train records are learning-honest (sessions + best/recent length, never
success%); eval records keep the score; failed jobs record their error.
"""
from types import SimpleNamespace

from snake_den import jobs as jobs_mod
from snake_den import persistence, registry


def _job(**kw):
    spec = SimpleNamespace(
        type=kw.pop("type"), save_path=kw.pop("save_path", None),
        base_model=kw.pop("base_model", None),
        sessions=kw.pop("spec_sessions", 0))
    return SimpleNamespace(
        id=kw.pop("id", 1), spec=spec,
        status=kw.pop("status", jobs_mod.FINISHED),
        sessions=kw.pop("sessions", []), summary=kw.pop("summary", None),
        error=kw.pop("error", None))


def test_train_record_is_learning_honest():
    job = _job(type="train", save_path="models/m.json", id=7,
               sessions=[{"max_length": 3}, {"max_length": 9},
                         {"max_length": 6}])
    rec = registry.history_record(job)
    assert rec["type"] == "train" and rec["id"] == 7
    assert rec["model"] == "models/m.json"
    assert rec["sessions"] == 3
    assert rec["outcome"] == {"best_len": 9, "recent_len": 6}
    assert "success_pct" not in rec["outcome"]    # never leaks into training
    assert rec["when"]


def test_eval_record_keeps_the_success_summary():
    summary = {"success_rate": 0.42, "target_length": 10,
               "length": {"mean": 7.5}}
    job = _job(type="eval", base_model="models/m.json", spec_sessions=100,
               summary=summary)
    rec = registry.history_record(job)
    assert rec["type"] == "eval" and rec["games"] == 100
    assert rec["outcome"]["success_pct"] == 42.0
    assert rec["outcome"]["target"] == 10
    assert rec["outcome"]["mean_length"] == 7.5


def test_failed_record_keeps_a_clipped_error():
    job = _job(type="train", save_path="models/m.json",
               status=jobs_mod.FAILED, error="boom" * 40)
    rec = registry.history_record(job)
    assert rec["outcome"]["error"].startswith("boom")
    assert len(rec["outcome"]["error"]) <= 80


def test_add_history_round_trips_through_the_registry():
    reg = registry.Registry(persistence.empty())
    reg.add_history({"type": "eval", "model": "m.json"})
    assert reg.history() == [{"type": "eval", "model": "m.json"}]
