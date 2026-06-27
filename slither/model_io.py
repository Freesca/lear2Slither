"""Model persistence: save / load the model JSON. (Phase 5)

Keeps the Agent file-format-ignorant -- the agent never imports json. The
format is implementation-plan.md sec. 6: one human-readable JSON capturing the
sparse Q-table, the visit counts, the session count (so training can resume)
and the config used (traceability only; the active run uses the *current*
config, never the stored one).
"""
import json
from dataclasses import dataclass, field

from slither.interpreter import DEFAULT_SCHEME, Scheme

FORMAT_VERSION = 3          # v3 adds "curve"; v2 added "state"; v1 still loads


@dataclass
class ModelData:
    """What a loaded model restores into an Agent (plus its training curve)."""

    q: dict
    n: dict
    sessions_trained: int
    scheme: Scheme = field(default=DEFAULT_SCHEME)
    curve: list = field(default_factory=list)


def downsample(values, n=100):
    """At most ``n`` evenly-spaced samples of ``values`` (a compact curve).

    A pure copy of ``snake_den.viewdata.downsample`` kept here so slither never
    imports the hub (the layering + -42 firewall): the per-session max-length
    series is decimated small enough to store in the model file while keeping
    its shape. A short series is returned unchanged.
    """
    values = list(values)
    if len(values) <= n:
        return values
    step = len(values) / n
    return [values[min(len(values) - 1, int(i * step))] for i in range(n)]


def save(path, agent, config, scheme=None, curve=None):
    """Write the agent's learning state to ``path`` as JSON.

    The top-level ``state`` block records the *effective* ``scheme`` (the one
    the table was actually built with -- the loaded model's on a resume, else
    the config's), so the model always reloads with a matching state alphabet.
    ``curve`` is the per-session max-length series (training telemetry, not
    agent vision -- the -42 firewall is untouched); it is stored downsampled so
    every model carries its own learning curve. Q-rows and visit rows are
    emitted in sorted-key order so re-saving an unchanged agent yields a
    byte-identical file -- clean diffs for the committed deliverable models.
    """
    if scheme is None:
        scheme = config.scheme
    data = {
        "format_version": FORMAT_VERSION,
        "sessions_trained": agent.sessions_trained,
        "state": scheme.as_dict(),
        "config": config.as_dict(),
        "curve": downsample(curve or []),
        "q": {state: agent.q[state] for state in sorted(agent.q)},
        "visits": {state: agent.n[state] for state in sorted(agent.n)},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load(path):
    """Read a model file into ``ModelData`` (q, visits, sessions, scheme).

    Coerces the JSON numbers to the agent's expected types and checks the
    format version and required keys, raising ``ValueError`` on a malformed
    file (Phase 9 wraps this into a clean top-level message). A v1 file (no
    ``state`` block) loads as the default 7-letter scheme, and a v1/v2 file (no
    ``curve``) loads with an empty curve, so the committed deliverable models
    keep working without a retrain.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    version = data.get("format_version")
    if version not in (1, 2, FORMAT_VERSION):
        raise ValueError(
            f"unsupported model format_version {version!r} "
            f"(expected 1, 2 or {FORMAT_VERSION})")
    for key in ("q", "visits", "sessions_trained"):
        if key not in data:
            raise ValueError(f"malformed model file: missing {key!r}")

    q = {state: [float(x) for x in row]
         for state, row in data["q"].items()}
    n = {state: [int(x) for x in row]
         for state, row in data["visits"].items()}
    curve = [float(x) for x in data.get("curve", [])]
    return ModelData(
        q=q, n=n, sessions_trained=int(data["sessions_trained"]),
        scheme=_read_scheme(data.get("state")), curve=curve)


def _read_scheme(block):
    """The scheme from a model's ``state`` block (legacy default if absent)."""
    if not isinstance(block, dict):
        return DEFAULT_SCHEME
    return Scheme(
        warn=bool(block.get("warn", True)),
        caution=bool(block.get("caution", False)),
        green_far=bool(block.get("green_far", True)),
        red_far=bool(block.get("red_far", True)),
        body_far=bool(block.get("body_far", False)),
    )
