"""JobManager: the worker pool that spawns and supervises ./snake.

The UI's main thread owns a 60 FPS loop and must never block on subprocess
I/O, so each running child gets daemon *reader threads* that do the blocking
reads and hand structured events to the main thread through a thread-safe
``queue.Queue``; the UI drains the queue each frame (``poll``). ``select()``
on pipes is unavailable on Windows, so a blocking read on a dedicated thread
is the portable choice (works the same on the Linux defense box).

Two reader threads per child, not one: stdout carries the parsed ``-progress``
stream, while stderr is drained into a bounded tail. Draining stderr on its own
thread matters -- if a crashing child dumps a large traceback and nobody reads
that pipe, its OS buffer fills and the child blocks forever (a hang violates
the no-crash rule). All *job-state* mutation happens on the main thread in
``poll``/``stop``/``shutdown``; the reader threads only ``put`` onto the queue
and ``append`` to a deque (both thread-safe), so no locks are needed.

Lifecycle: ``queued -> running -> finished | failed | stopped``.
**finished** = exit 0 *and* a summary parsed; **failed** = non-zero exit, no
summary, or a malformed line (caught -- a bad child fails only its own job,
never the hub); **stopped** = user kill (terminate -> kill after a grace
period -> join the readers, so no orphan survives).
"""
import collections
import os
import queue
import subprocess
import threading

from snake_den import snake_proc

QUEUED = "queued"
RUNNING = "running"
FINISHED = "finished"
FAILED = "failed"
STOPPED = "stopped"

_TERMINAL = frozenset({FINISHED, FAILED, STOPPED})

_GRACE = 2.0              # seconds between terminate() and a hard kill()
_JOIN_TIMEOUT = 2.0       # seconds to join a reader thread (instant after EOF)
_STDERR_TAIL_LINES = 20   # trailing stderr lines kept for a failure report


class _ParseError:
    """A reader-thread marker that a stdout line could not be parsed."""

    def __init__(self, message):
        self.message = message


def _read_stdout(stream, event_queue):
    """Daemon: parse each progress line onto ``event_queue`` until EOF.

    A malformed line becomes a ``_ParseError`` marker rather than an exception,
    so a bad child can only fail its own job (the no-crash rule). The thread
    ends naturally when the child closes stdout (``readline`` returns "").
    """
    while True:
        raw = stream.readline()
        if raw == "":
            return
        line = raw.strip()
        if not line:
            continue
        try:
            event_queue.put(snake_proc.parse_line(line))
        except ValueError as error:
            event_queue.put(_ParseError(str(error)))


def _read_stderr(stream, sink):
    """Daemon: drain the child's stderr into ``sink`` (a bounded deque)."""
    while True:
        raw = stream.readline()
        if raw == "":
            return
        sink.append(raw.rstrip("\n"))


class Job:
    """One ./snake run and everything the UI needs to show about it."""

    def __init__(self, job_id, spec):
        self.id = job_id
        self.spec = spec
        self.status = QUEUED
        self.proc = None
        self.config_path = None        # temp TOML to delete when the job ends
        self.start_event = None        # the opening -progress event
        self.sessions = []             # per-game events, in order (the curve)
        self.summary = None            # the closing summary (SuiteStats dict)
        self.exit_code = None
        self.stderr_tail = ""
        self.error = None              # a human reason when status == FAILED
        self._queue = queue.Queue()
        self._stderr = collections.deque(maxlen=_STDERR_TAIL_LINES)
        self._threads = []
        self._parse_failed = False

    @property
    def is_terminal(self):
        return self.status in _TERMINAL


class JobManager:
    """A FIFO pool of at most ``pool_size`` live ./snake subprocesses.

    ``submit`` only enqueues; ``poll`` (called every UI frame) does all the
    work -- drain running jobs' queues, reap exits, and admit queued jobs into
    freed slots -- so every state change happens on the one calling thread.
    """

    def __init__(self, pool_size=None):
        self.pool_size = pool_size or min(4, os.cpu_count() or 1)
        self.jobs = []                 # all jobs in submission order (the UI)
        self._by_id = {}
        self._waiting = collections.deque()   # ids of QUEUED jobs, FIFO
        self._next_id = 1
        self._watchers = []            # (Popen, temp-config) detached windows

    # --- public API -----------------------------------------------------

    def submit(self, spec):
        """Enqueue a job; it starts on the next ``poll`` if a slot is free."""
        job = Job(self._next_id, spec)
        self._next_id += 1
        self.jobs.append(job)
        self._by_id[job.id] = job
        self._waiting.append(job.id)
        return job.id

    def job(self, job_id):
        """The Job with this id."""
        return self._by_id[job_id]

    def poll(self):
        """One supervision tick: drain output, reap exits, admit queued jobs.

        Safe to call as often as the UI likes (every frame). Never blocks: the
        readers do the blocking I/O, and reaping uses the non-blocking
        ``Popen.poll``.
        """
        for job in self.jobs:
            if job.status == RUNNING:
                self._drain(job)
                if job.proc.poll() is not None:
                    self._finalize(job)
        self._reap_watchers()
        self._admit()

    def stop(self, job_id):
        """Stop a job cleanly. Queued: just drop it. Running: terminate it."""
        job = self._by_id[job_id]
        if job.status == QUEUED:
            self._unqueue(job_id)
            job.status = STOPPED
            return
        if job.status != RUNNING:
            return                     # already finished/failed/stopped
        self._terminate(job)
        job.status = STOPPED

    def watch(self, spec):
        """Spawn a detached, interactive ./snake -visual on window.

        Not a pool job: it takes no slot, has no reader threads and is not
        parsed (it has its own pygame window). Tracked loosely so it is torn
        down on hub exit -- no orphan -- and reaped once the user closes it.
        Returns the Popen.
        """
        config_path = snake_proc.write_temp_config(spec.config)
        argv = snake_proc.build_argv(spec, config_path)
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._watchers.append((proc, config_path))
        return proc

    def shutdown(self):
        """Tear down every child on hub exit -- no orphan, no traceback.

        Signals all running children first, then reaps each, so their grace
        windows overlap instead of summing.
        """
        running = [j for j in self.jobs if j.status == RUNNING]
        for job in running:
            job.proc.terminate()
        for job in running:
            self._reap(job)
            job.status = STOPPED
        for job in self.jobs:
            if job.status == QUEUED:
                job.status = STOPPED
        self._waiting.clear()
        for proc, config_path in self._watchers:
            proc.terminate()
            try:
                proc.wait(timeout=_GRACE)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            _remove_file(config_path)
        self._watchers = []

    # --- internals ------------------------------------------------------

    def _admit(self):
        """Start queued jobs while pool slots are free (FIFO order)."""
        while self._waiting and self._running_count() < self.pool_size:
            job = self._by_id[self._waiting.popleft()]
            if job.status == QUEUED:   # may have been stopped while waiting
                self._start(job)

    def _running_count(self):
        return sum(1 for job in self.jobs if job.status == RUNNING)

    def _start(self, job):
        """Spawn the child and its reader threads; mark the job RUNNING.

        A failure to even start (bad executable, un-emittable config) fails the
        job rather than raising into the UI loop.
        """
        try:
            job.config_path = snake_proc.write_temp_config(job.spec.config)
            argv = snake_proc.build_argv(job.spec, job.config_path)
            job.proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                bufsize=1)
        except (OSError, ValueError, TypeError) as error:
            job.error = f"failed to start: {error}"
            job.status = FAILED
            self._cleanup_config(job)
            return
        job.status = RUNNING
        out = threading.Thread(
            target=_read_stdout, args=(job.proc.stdout, job._queue),
            daemon=True)
        err = threading.Thread(
            target=_read_stderr, args=(job.proc.stderr, job._stderr),
            daemon=True)
        out.start()
        err.start()
        job._threads = [out, err]

    def _drain(self, job):
        """Move every queued reader event into the job's own fields."""
        while True:
            try:
                item = job._queue.get_nowait()
            except queue.Empty:
                return
            self._handle(job, item)

    def _handle(self, job, item):
        if isinstance(item, _ParseError):
            job._parse_failed = True
            if job.error is None:
                job.error = f"bad progress line: {item.message}"
            return
        kind = item.get("type")
        if kind == "start":
            job.start_event = item
        elif kind == "session":
            job.sessions.append(item)
        elif kind == "summary":
            job.summary = item

    def _finalize(self, job):
        """A child has exited on its own: collect the rest and classify it."""
        self._reap(job)
        if job.exit_code == 0 and job.summary is not None \
                and not job._parse_failed:
            job.status = FINISHED
        else:
            job.status = FAILED
            if job.error is None:
                if job.exit_code != 0:
                    job.error = f"exited with code {job.exit_code}"
                elif job.summary is None:
                    job.error = "no summary in the progress stream"

    def _terminate(self, job):
        """Stop a running child: terminate, hard-kill after the grace, reap."""
        job.proc.terminate()
        self._reap(job)

    def _reap(self, job):
        """Wait for the child (kill on timeout), join readers, drain, clean up.

        Assumes the child is exiting or has been signalled. Joining the readers
        after the pipes close guarantees every buffered event is captured
        before we read the final state.
        """
        proc = job.proc
        try:
            proc.wait(timeout=_GRACE)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        for thread in job._threads:
            thread.join(timeout=_JOIN_TIMEOUT)
        self._drain(job)
        job.exit_code = proc.returncode
        job.stderr_tail = "\n".join(job._stderr)
        self._cleanup_config(job)

    def _reap_watchers(self):
        """Drop watchers whose window the user closed; clean their config."""
        alive = []
        for proc, config_path in self._watchers:
            if proc.poll() is None:
                alive.append((proc, config_path))
            else:
                _remove_file(config_path)
        self._watchers = alive

    def _unqueue(self, job_id):
        try:
            self._waiting.remove(job_id)
        except ValueError:
            pass

    def _cleanup_config(self, job):
        _remove_file(job.config_path)
        job.config_path = None


def _remove_file(path):
    """Best-effort delete of a temp file (ignore if it is gone or in use)."""
    if path is None:
        return
    try:
        os.remove(path)
    except OSError:
        pass
