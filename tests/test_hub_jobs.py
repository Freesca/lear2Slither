"""Milestone-A tests: the JobManager pool + lifecycle (snake_den/jobs.py).

The subprocess engine is the hub's highest-risk surface (the no-orphan /
no-crash rule), so it is tested headlessly with fast *fake* children -- a tiny
script driven by a JSON plan file (emit these lines, write this stderr, sleep,
exit with this code). The real ./snake is exercised once by the smoke test, not
here, so these stay deterministic and sub-second.

build_argv / write_temp_config are monkeypatched per test so the manager spawns
our fake child instead of slither; everything else (Popen, the reader threads,
poll/stop/shutdown) is the real code under test.
"""
import json
import os
import sys
import textwrap
import time
import uuid

import pytest

from slither import evaluate, progress
from slither.runner import SessionResult
from snake_den import jobs
from snake_den.snake_proc import JobSpec

# A fake ./snake: read a plan (a JSON file path in argv[1]), optionally sleep,
# echo the given stdout/stderr lines, then exit with the given code.
_CHILD = textwrap.dedent("""
    import json, sys, time
    with open(sys.argv[1], encoding="utf-8") as handle:
        plan = json.load(handle)
    time.sleep(plan.get("sleep", 0))
    for line in plan.get("stdout", []):
        print(line, flush=True)
    for line in plan.get("stderr", []):
        print(line, file=sys.stderr, flush=True)
    sys.exit(plan.get("exit", 0))
""")


@pytest.fixture
def child(tmp_path):
    """A factory: build a fake-child argv for a given behaviour plan."""
    script = tmp_path / "fake_snake.py"
    script.write_text(_CHILD, encoding="utf-8")

    def make_argv(**plan):
        plan_path = tmp_path / f"plan_{uuid.uuid4().hex}.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        return [sys.executable, "-u", str(script), str(plan_path)]

    return make_argv


def _use(monkeypatch, argv):
    """Make the manager spawn ``argv`` and skip the real temp-config write."""
    monkeypatch.setattr(jobs.snake_proc, "build_argv",
                        lambda spec, path: argv)
    monkeypatch.setattr(jobs.snake_proc, "write_temp_config",
                        lambda config: None)


def _spec():
    return JobSpec("train", {"board": {"size": 10}}, sessions=1, seed=0)


def _summary_line(results, target=10):
    return progress.summary(evaluate.summarize(results, target))


def _run_until(manager, predicate, timeout=5.0):
    """Poll the manager until ``predicate`` holds (or fail on timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        manager.poll()
        if predicate():
            return
        time.sleep(0.01)
    manager.poll()
    if not predicate():
        raise AssertionError("timed out waiting for the manager")


# --- defaults + admission ---------------------------------------------------

def test_default_pool_size_is_capped_at_four():
    manager = jobs.JobManager()
    assert manager.pool_size == min(4, os.cpu_count() or 1)


def test_pool_caps_running_and_queues_the_rest(child, monkeypatch):
    _use(monkeypatch, child(sleep=30))           # long-lived sleepers
    manager = jobs.JobManager(pool_size=2)
    ids = [manager.submit(_spec()) for _ in range(5)]
    manager.poll()
    running = [i for i in ids if manager.job(i).status == jobs.RUNNING]
    queued = [i for i in ids if manager.job(i).status == jobs.QUEUED]
    assert len(running) == 2 and len(queued) == 3
    manager.shutdown()
    for i in ids:                                # no orphans left behind
        proc = manager.job(i).proc
        assert proc is None or proc.poll() is not None


def test_queued_job_starts_when_a_slot_frees(child, monkeypatch):
    _use(monkeypatch, child(stdout=[progress.start("train", 1),
                                    _summary_line([SessionResult(3, 5,
                                                                 "wall",
                                                                 False)])]))
    manager = jobs.JobManager(pool_size=1)
    a = manager.submit(_spec())
    b = manager.submit(_spec())
    _run_until(manager,
               lambda: manager.job(a).is_terminal
               and manager.job(b).is_terminal)
    assert manager.job(a).status == jobs.FINISHED
    assert manager.job(b).status == jobs.FINISHED


# --- the happy path ---------------------------------------------------------

def test_finished_requires_exit0_and_a_summary(child, monkeypatch):
    results = [SessionResult(i, i * 2, "wall", False) for i in range(3)]
    lines = [progress.start("train", 3)]
    lines += [progress.session(i, r, epsilon=0.5, sessions_trained=i + 1)
              for i, r in enumerate(results)]
    lines.append(_summary_line(results))
    _use(monkeypatch, child(stdout=lines, exit=0))

    manager = jobs.JobManager(pool_size=2)
    jid = manager.submit(_spec())
    _run_until(manager, lambda: manager.job(jid).is_terminal)

    job = manager.job(jid)
    assert job.status == jobs.FINISHED
    assert job.exit_code == 0
    assert job.start_event["total_sessions"] == 3
    assert [s["i"] for s in job.sessions] == [0, 1, 2]   # the live curve
    assert job.summary["games"] == 3
    assert job.error is None


# --- failure modes ----------------------------------------------------------

def test_nonzero_exit_fails_with_stderr_tail(child, monkeypatch):
    _use(monkeypatch, child(
        stderr=["Traceback (most recent call last):", "ValueError: boom"],
        exit=1))
    manager = jobs.JobManager()
    jid = manager.submit(_spec())
    _run_until(manager, lambda: manager.job(jid).is_terminal)

    job = manager.job(jid)
    assert job.status == jobs.FAILED
    assert job.exit_code == 1
    assert "boom" in job.stderr_tail


def test_exit0_without_summary_fails(child, monkeypatch):
    _use(monkeypatch, child(stdout=[progress.start("train", 1)], exit=0))
    manager = jobs.JobManager()
    jid = manager.submit(_spec())
    _run_until(manager, lambda: manager.job(jid).is_terminal)

    job = manager.job(jid)
    assert job.status == jobs.FAILED
    assert "summary" in job.error


def test_malformed_line_fails_the_job_not_the_hub(child, monkeypatch):
    # A garbage line mid-stream: poll() must never raise; the job fails.
    lines = [progress.start("train", 1), "this is not json {",
             _summary_line([SessionResult(3, 5, "wall", False)])]
    _use(monkeypatch, child(stdout=lines, exit=0))
    manager = jobs.JobManager()
    jid = manager.submit(_spec())
    _run_until(manager, lambda: manager.job(jid).is_terminal)

    job = manager.job(jid)
    assert job.status == jobs.FAILED
    assert "bad progress line" in job.error


# --- stopping (no orphans) --------------------------------------------------

def test_stop_kills_a_running_child(child, monkeypatch):
    _use(monkeypatch, child(sleep=30))
    manager = jobs.JobManager(pool_size=1)
    jid = manager.submit(_spec())
    manager.poll()                               # admit + start
    job = manager.job(jid)
    assert job.status == jobs.RUNNING
    assert job.proc.poll() is None               # alive
    proc = job.proc

    manager.stop(jid)
    assert job.status == jobs.STOPPED
    assert proc.poll() is not None               # dead -> no orphan


def test_stop_a_queued_job_never_starts_it(child, monkeypatch):
    _use(monkeypatch, child(sleep=30))
    manager = jobs.JobManager(pool_size=1)
    manager.submit(_spec())                      # fills the only slot
    waiting = manager.submit(_spec())
    manager.poll()                               # first runs, second queued
    assert manager.job(waiting).status == jobs.QUEUED

    manager.stop(waiting)
    assert manager.job(waiting).status == jobs.STOPPED
    manager.poll()                               # must not resurrect it
    assert manager.job(waiting).status == jobs.STOPPED
    assert manager.job(waiting).proc is None
    manager.shutdown()


def _watch_spec():
    return JobSpec("watch", {"board": {"size": 10}}, base_model="m.json")


def test_watch_is_detached_and_torn_down(child, monkeypatch):
    _use(monkeypatch, child(sleep=30))
    manager = jobs.JobManager(pool_size=1)
    proc = manager.watch(_watch_spec())
    assert proc.poll() is None             # the watch window is alive
    assert manager.jobs == []              # not a pool job (no slot used)
    manager.shutdown()
    assert proc.poll() is not None         # torn down on exit -> no orphan


def test_finished_watch_is_reaped_on_poll(child, monkeypatch):
    _use(monkeypatch, child(sleep=0))      # exits immediately
    manager = jobs.JobManager()
    proc = manager.watch(_watch_spec())
    proc.wait(timeout=5)
    manager.poll()
    assert manager._watchers == []         # closed window is reaped


def test_shutdown_tears_down_all_children(child, monkeypatch):
    _use(monkeypatch, child(sleep=30))
    manager = jobs.JobManager(pool_size=3)
    ids = [manager.submit(_spec()) for _ in range(3)]
    manager.poll()
    procs = [manager.job(i).proc for i in ids]
    assert all(p.poll() is None for p in procs)  # all alive

    manager.shutdown()
    assert all(p.poll() is not None for p in procs)   # all dead
    assert all(manager.job(i).status == jobs.STOPPED for i in ids)
