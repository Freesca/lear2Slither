"""View data: pure prep for the Q-table views. (Phase B3 / rework)

Pygame-free, so the screens' data logic is unit-testable without a display (and
the hub suite stays display-free -- slither's rule). The pygame screens only
render what these functions return. Reads model files via
``slither.model_io`` (pure, H4) and mirrors the state alphabet via
``snake_den.scheme``; no execution code is touched.
"""
from slither import model_io
from snake_den import scheme as scheme_mod

SUMMARY_VERSION = 2          # bump when policy_summary's shape/meaning changes


def qtable_rows(model_data):
    """Sorted Q-table rows for the viewer: ``(state, qvals, nvals, argmax)``.

    ``model_data`` is a ``slither.model_io.ModelData`` (q, n, sessions).
    One row per visited state, sorted by state; ``argmax`` is the greedy column
    (ties -> first), so the viewer can box the policy's choice.
    """
    rows = []
    for state in sorted(model_data.q):
        qvals = model_data.q[state]
        nvals = model_data.n.get(state, [0] * len(qvals))
        argmax = max(range(len(qvals)), key=qvals.__getitem__)
        rows.append((state, qvals, nvals, argmax))
    return rows


def load_qtable_rows(model_path):
    """Load a model file and build its Q-table rows (may raise ValueError)."""
    return qtable_rows(model_io.load(model_path))


def downsample(values, n=100):
    """At most ``n`` evenly-spaced samples of ``values`` (a compact curve).

    Used to persist a training curve small enough to store and redraw. A short
    series is returned unchanged; a long one is decimated, keeping the shape.
    """
    values = list(values)
    if len(values) <= n:
        return values
    step = len(values) / n
    return [values[min(len(values) - 1, int(i * step))] for i in range(n)]


def policy_summary(model_data):
    """Compact policy fingerprint for the Models list (pure, pygame-free).

    Returns ``{coverage, total, mix, fingerprint}``. ``fingerprint`` is the
    **sign of the greedy Q-value** per *visited* state in sorted order
    (``+1 / 0 / -1``) -- a value map the minimap/list tint green/dim/magenta
    (green where the policy expects to do well, magenta where even its best
    move is bad). ``mix`` counts greedy actions as ``[F, L, R, B]``;
    ``coverage`` is the visited-state count and ``total`` the canonical-state
    count for the model's recorded scheme.
    """
    rows = qtable_rows(model_data)
    fingerprint = []
    mix = [0, 0, 0, 0]
    for _state, qvals, _n, argmax in rows:
        mix[argmax] += 1
        best = qvals[argmax]
        fingerprint.append(1 if best > 0 else -1 if best < 0 else 0)
    total = scheme_mod.state_count(model_data.scheme.as_dict())
    return {"v": SUMMARY_VERSION, "coverage": len(rows), "total": total,
            "mix": mix, "fingerprint": fingerprint}
