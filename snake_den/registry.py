"""Model registry: metadata + per-profile eval scores. (Phase A3)

The hub's memory of the models it has trained and evaluated (hub-design.md
sec. 6). Each entry is keyed by the model's file path and holds how many
sessions it was trained for, the config it was trained with, and a map of eval
scores. Scores are filed under an **eval-profile key**
(games/seed/board/target) so the registry never silently compares numbers
measured on different yardsticks (gate H2) -- a model evaluated at target 10
and at target 15 keeps both scores.

Metadata comes from the model file itself: ``sessions`` via ``model_io.load``
(the product's own validated parser, so the hub does not re-implement it), and
``config`` from the raw model JSON -- ``model_io.load`` deliberately does not
surface config. ``model_io`` and ``config.DEFAULTS`` are the only product
modules the hub imports (H4); execution always goes through spawning ./snake.
"""
import json
import os
from datetime import datetime

from slither import model_io
from snake_den import jobs as jobs_mod
from snake_den import viewdata


def profile_key(profile):
    """A stable string key for an eval profile (H2 comparability)."""
    return (f"games={profile['games']};seed={profile['seed']};"
            f"board={profile['board']};target={profile['target']}")


def normalize(path):
    """The canonical registry key for a model path: one form per file.

    A model can be named two ways for the same file -- the saved
    ``models/new.json`` (forward slash) and a discovered
    ``models\\new.json`` (``os.path.join`` on Windows) -- which would split it
    into two list entries. Collapsing through ``normpath`` + forward slashes
    gives every reference one key.
    """
    return os.path.normpath(path).replace(os.sep, "/")


def discover(models_path):
    """Normalized paths of the ``*.json`` files in ``models_path`` (sorted)."""
    found = []
    try:
        for name in os.listdir(models_path):
            if name.endswith(".json"):
                found.append(normalize(os.path.join(models_path, name)))
    except OSError:
        pass
    return sorted(set(found))


class Registry:
    """The ``models`` + ``history`` tables of the hub data file, with helpers.

    Constructed over the shared data dict from ``persistence.load``; every
    method mutates that dict in place, and the app persists it via
    ``persistence.save``.
    """

    def __init__(self, data):
        self._models = data["models"]
        self._history = data["history"]
        self._migrate_keys()

    def _migrate_keys(self):
        """Collapse any pre-existing un-normalized keys (the slash bug)."""
        for old in list(self._models):
            new = normalize(old)
            if new == old:
                continue
            entry = self._models.pop(old)
            existing = self._models.get(new)
            if existing is None:
                self._models[new] = entry
            else:                                # merge a duplicate into one
                existing.setdefault("scores", {}).update(
                    entry.get("scores", {}))
                for field in ("sessions", "config", "scheme", "qsummary",
                              "curve"):
                    existing.setdefault(field, entry.get(field))

    def register(self, model_path):
        """Record or refresh ``model_path``'s metadata; return its entry.

        Re-registering keeps existing scores -- the file is unchanged, only its
        metadata is refreshed. Raises ``ValueError`` on a malformed model file
        (from ``model_io.load``); the app layer turns it into a clean message.
        """
        model_path = normalize(model_path)
        loaded = model_io.load(model_path)
        entry = self._models.get(model_path, {})
        entry["sessions"] = loaded.sessions_trained
        entry["config"] = _read_config(model_path)
        entry["scheme"] = loaded.scheme.as_dict()
        entry["qsummary"] = viewdata.policy_summary(loaded)
        entry.setdefault("scores", {})
        self._models[model_path] = entry
        return entry

    def record_score(self, model_path, profile, summary):
        """File an eval ``summary`` under ``profile``'s key for the model."""
        model_path = normalize(model_path)
        if model_path not in self._models:
            self.register(model_path)
        key = profile_key(profile)
        self._models[model_path]["scores"][key] = _score(summary)

    def record_curve(self, model_path, curve):
        """Persist a model's (downsampled) training curve for its detail."""
        model_path = normalize(model_path)
        if model_path in self._models:
            self._models[model_path]["curve"] = list(curve)

    def score(self, model_path, profile):
        """The stored score for (model, profile), or ``None``."""
        entry = self._models.get(normalize(model_path))
        if entry is None:
            return None
        return entry.get("scores", {}).get(profile_key(profile))

    def models(self):
        """A snapshot ``{path: entry}`` of registered models (for the UI)."""
        return dict(self._models)

    def forget(self, model_path):
        """Drop a model from the registry (does not touch the file)."""
        self._models.pop(normalize(model_path), None)

    def add_history(self, record):
        """Append a job record to the history log."""
        self._history.append(record)

    def history(self):
        """A snapshot of the job-history log."""
        return list(self._history)


def history_record(job):
    """A learning-honest archive record for a terminal train/eval job (D3).

    Train records carry a curve-honest outcome (sessions + best/recent length)
    and never success%/target -- that is an eval concept (epsilon > 0 during
    training understates the policy). Eval records keep the success summary; a
    failed job records its error. The Runs tab reads these defensively. Pure --
    builds a dict, mutates nothing.
    """
    spec = job.spec
    model = normalize(spec.save_path or spec.base_model or "")
    record = {"id": job.id, "type": spec.type, "model": model,
              "when": datetime.now().isoformat(timespec="seconds")}
    if job.status == jobs_mod.FAILED:
        record["outcome"] = {"error": (job.error or "")[:80]}
    elif spec.type == "train":
        lengths = [e["max_length"] for e in job.sessions]
        record["sessions"] = len(job.sessions)
        record["outcome"] = {"best_len": max(lengths, default=0),
                             "recent_len": lengths[-1] if lengths else 0}
    else:
        summary = job.summary or {}
        record["games"] = spec.sessions
        record["outcome"] = {
            "success_pct": round(summary.get("success_rate", 0) * 100, 1),
            "target": summary.get("target_length"),
            "mean_length": round(summary.get("length", {}).get("mean", 0), 1)}
    return record


def _read_config(model_path):
    """The model file's ``config`` block (raw JSON), or ``{}`` if absent."""
    try:
        with open(model_path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(loaded, dict) and isinstance(loaded.get("config"), dict):
        return loaded["config"]
    return {}


def _score(summary):
    """A model's score for one profile: headline fields + the full summary.

    ``summary`` is a -progress ``summary`` event / serialized ``SuiteStats``
    (slither/progress.py) -- exactly what a finished eval job holds. The five
    headline fields stay flat (the Models list reads them directly); the whole
    event is kept under ``full`` so the detail view can show the length /
    duration distributions and the failure-cause breakdown.
    """
    return {
        "success_pct": round(summary["success_rate"] * 100, 1),
        "games": summary["games"],
        "mean_length": summary["length"]["mean"],
        "mean_duration": summary["duration"]["mean"],
        "target": summary["target_length"],
        "full": summary,
    }
