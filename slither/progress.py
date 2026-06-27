"""Structured progress output for the hub. (Phase 7, GATE H1)

When ``./snake`` runs with ``-progress`` it emits JSON-lines on stdout in place
of the human Game-over/Save/Load text, so the hub (a separate program, see
docs/hub-design.md) can build a live training curve and read final results
without scraping human prose. One object per line:

    {"type":"start",   "format_version":1, "mode":..., "total_sessions":N}
    {"type":"session", "i":k, "max_length":X, "duration":Y, ...}  # one/game
    {"type":"summary", ...}                                    # a SuiteStats

The schema is versioned (``format_version``) and every line is emitted with
sorted keys + compact separators, so the stream is byte-stable. This module
imports nothing else from ``slither`` -- it only serializes dicts -- so it is
a leaf with no import cycles: the runner emits ``start`` + ``session``, the CLI
emits ``summary`` (it owns ``evaluate``).
"""
import json

FORMAT_VERSION = 1


def _line(obj):
    """One compact, key-sorted JSON line (byte-stable across runs)."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def start(mode, total_sessions):
    """The opening line: the run's mode ('train'/'eval') and game count."""
    return _line({
        "type": "start",
        "format_version": FORMAT_VERSION,
        "mode": mode,
        "total_sessions": total_sessions,
    })


def session(i, result, *, epsilon, sessions_trained):
    """One finished game (0-based index ``i``) as a progress event."""
    return _line({
        "type": "session",
        "i": i,
        "max_length": result.max_length,
        "duration": result.duration,
        "death_cause": result.death_cause,
        "won": result.won,
        "epsilon": epsilon,
        "sessions_trained": sessions_trained,
    })


def summary(stats):
    """The closing line: a ``SuiteStats`` (evaluate.py) serialized in full."""
    return _line({
        "type": "summary",
        "games": stats.games,
        "success_rate": stats.success_rate,
        "target_length": stats.target_length,
        "length": _distribution(stats.length),
        "duration": _distribution(stats.duration),
        "outcomes": stats.outcomes,
    })


def _distribution(dist):
    """A ``Distribution`` as a plain dict (the raw values are omitted)."""
    return {
        "mean": dist.mean,
        "median": dist.median,
        "max": dist.maximum,
        "std": dist.std,
    }
