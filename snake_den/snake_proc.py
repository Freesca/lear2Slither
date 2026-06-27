"""snake_proc: turn a JobSpec into a ./snake invocation, parse its progress.

This is the hub's whole boundary to the shipped product. A JobSpec is turned
into:

  1. a temp TOML config -- the stdlib's ``tomllib`` is read-only and adding a
     writer (``tomli-w``) is a fresh-clone install risk on the defense box, so
     the hub hand-writes it; and
  2. an argv invoked as the Python *module* form, with unbuffered stdout so the
     -progress stream arrives live:
     ``[sys.executable, "-u", "-m", "slither", ...]`` -- this runs identically
     on Windows dev and the Linux box, which the POSIX ``./snake`` does not.

Nothing about the agent crosses this boundary: only config in, model JSON + the
-progress JSON-lines stream out. This module imports ``slither.config`` solely
for ``DEFAULTS`` (the config schema); it never imports the runner/agent/
environment/gui.
"""
import json
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass

from slither.config import DEFAULTS

# The -progress schema version this parser understands. A local mirror of
# slither.progress.FORMAT_VERSION, kept here rather than imported so the hub's
# import surface into the product stays exactly {config, model_io};
# test_hub_snake_proc pins the two equal so the duplication cannot drift.
PROGRESS_FORMAT_VERSION = 1

_EVENT_TYPES = ("start", "session", "summary")


@dataclass
class JobSpec:
    """A complete description of one ./snake run the hub will spawn.

    ``config`` is the full, DEFAULTS-shaped dict written to the temp TOML. For
    train jobs it comes from the New-Job editor; for eval/watch jobs it is
    built from an eval profile by :func:`eval_config`, and ``eval_profile``
    records that profile so a finished eval's score files under the right
    profile key. ``sessions``/``seed`` become CLI flags, not config keys.
    """

    type: str                          # "train" | "eval" | "watch"
    config: dict                       # full nested config for the temp TOML
    sessions: int = 1
    seed: int = 0
    base_model: str | None = None      # -load source (resume/eval/watch)
    save_path: str | None = None       # -save target (train only)
    eval_profile: dict | None = None   # eval/watch: profile config came from


# --- TOML emitter -----------------------------------------------------------

def emit_toml(config):
    """Serialize a two-level config dict to TOML text.

    The config is a flat table-of-tables of scalars, so the grammar needed is
    tiny: a ``[section]`` header per table, then ``key = value`` lines. Values
    are emitted by Python type so the file round-trips through ``tomllib`` back
    to the same dict -- crucially ``int -> str`` but ``float -> repr`` (keeps
    ``20.0`` a float rather than silently becoming the int ``20``), ``bool ->
    true/false``, ``str -> "quoted"``.
    """
    lines = []
    for section, table in config.items():
        lines.append(f"[{section}]")
        for key, value in table.items():
            lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")
    return "\n".join(lines)


def _toml_scalar(value):
    """One TOML scalar literal, dispatched on Python type.

    ``bool`` is checked before ``int`` because ``bool`` is an ``int`` subclass;
    emitting ``True`` as ``1`` would lose the type on round-trip.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _toml_string(value)
    raise TypeError(
        f"cannot emit TOML scalar of type {type(value).__name__}")


def _toml_string(value):
    """A TOML basic string: double-quoted with the spec's backslash escapes."""
    escaped = (value.replace("\\", "\\\\").replace('"', '\\"')
               .replace("\n", "\\n").replace("\t", "\\t")
               .replace("\r", "\\r"))
    return f'"{escaped}"'


def write_temp_config(config):
    """Write ``config`` to a temp TOML file and return its path.

    The caller (the JobManager) deletes the file once the job ends.
    ``delete=False`` because on Windows a still-open NamedTemporaryFile cannot
    be reopened by the spawned child; we close it here and hand back the path.
    """
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", prefix="snake_den_",
        delete=False, encoding="utf-8")
    with handle:
        handle.write(emit_toml(config))
    return handle.name


# --- eval profile -> config -------------------------------------------------

def eval_config(profile):
    """Build the config for an eval/watch job from an eval profile.

    The profile fixes ``board`` + ``target`` so every model is scored on one
    yardstick; rewards/learning are irrelevant under
    ``-dontlearn`` so they stay at defaults. ``games``/``seed`` are CLI flags
    (not config keys), so they are not written here. Validating a model on a
    different board size -- the bonus's proof -- is just a profile with a
    different ``board``.
    """
    config = deepcopy(DEFAULTS)
    config["board"]["size"] = profile["board"]
    config["goal"]["target_length"] = profile["target"]
    return config


# --- argv builder -----------------------------------------------------------

def build_argv(spec, config_path):
    """The argv to spawn ``spec``, using ``config_path`` as the temp config.

    Module form + unbuffered: ``[sys.executable, "-u", "-m", "slither",
    ...]``. Per type:

      train: -config C -sessions N -seed S [-load base] -save out -progress
      eval:  -config C -load M -dontlearn -sessions N -seed S -progress
      watch: -config C -load M -visual on -dontlearn -sessions N
             (no -progress / -seed: an interactive window showing the model's
             greedy policy across N fresh games until the user closes it)
    """
    if spec.type in ("eval", "watch") and spec.base_model is None:
        raise ValueError(f"{spec.type!r} job requires base_model")

    base = [sys.executable, "-u", "-m", "slither", "-config", config_path]

    if spec.type == "train":
        argv = base + ["-sessions", str(spec.sessions),
                       "-seed", str(spec.seed)]
        if spec.base_model is not None:
            argv += ["-load", spec.base_model]
        if spec.save_path is not None:
            argv += ["-save", spec.save_path]
        argv.append("-progress")
        return argv

    if spec.type == "eval":
        return base + ["-load", spec.base_model, "-dontlearn",
                       "-sessions", str(spec.sessions),
                       "-seed", str(spec.seed), "-progress"]

    if spec.type == "watch":
        return base + ["-load", spec.base_model, "-visual", "on",
                       "-dontlearn", "-sessions", str(spec.sessions)]

    raise ValueError(f"unknown job type {spec.type!r}")


# --- progress parser --------------------------------------------------------

def parse_line(line):
    """Parse one -progress JSON-lines record into an event dict.

    Recognizes the three event types (slither/progress.py) and validates the
    schema version on the opening ``start`` event (only that line carries
    ``format_version``). Raises ``ValueError`` on anything malformed -- the
    caller (the job's reader) wraps this so a bad line fails only its own job,
    never the hub (the no-crash rule).
    """
    try:
        event = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"progress line is not JSON: {line!r}") from error
    if not isinstance(event, dict):
        raise ValueError(f"progress line is not an object: {line!r}")

    kind = event.get("type")
    if kind not in _EVENT_TYPES:
        raise ValueError(f"unknown progress event type {kind!r}")
    if kind == "start":
        version = event.get("format_version")
        if version != PROGRESS_FORMAT_VERSION:
            raise ValueError(
                f"unsupported progress format_version {version!r} "
                f"(expected {PROGRESS_FORMAT_VERSION})")
    return event
