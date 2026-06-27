"""Eval suites: named, reusable evaluation profiles. (rework)

A *suite* is one set of evaluation parameters -- ``games``, ``seed``,
``board``, ``target`` and ``step_cap`` -- under which models are scored. The
built-in
``STANDARD`` suite is a constant comparison yardstick (every model's headline
badge is its score here); custom suites are saved from the Eval page and stored
in the hub data file's ``suites`` table.

Scores stay keyed by the registry's ``profile_key`` (games/seed/board/target),
so a suite is just a *named* profile: two suites with the same four numbers map
to the same stored score. ``step_cap`` is a config knob carried by the suite
but not part of the key (it rarely changes; default 1000).
"""

STANDARD = {"name": "standard", "games": 100, "seed": 0,
            "board": 10, "target": 10, "step_cap": 1000}

_INT_FIELDS = ("games", "seed", "board", "target", "step_cap")


def profile(suite):
    """The four-key eval profile (registry.profile_key parts) of a suite."""
    return {"games": suite["games"], "seed": suite["seed"],
            "board": suite["board"], "target": suite["target"]}


def normalize(suite):
    """A validated suite dict; raises ``ValueError`` on a bad field.

    ``seed`` may be any non-negative int; the rest must be >= 1. The name is
    stripped and must be non-empty and not the reserved ``standard``.
    """
    name = str(suite.get("name", "")).strip()
    if not name:
        raise ValueError("suite name must not be empty")
    if name == STANDARD["name"]:
        raise ValueError("'standard' is reserved")
    out = {"name": name}
    for field in _INT_FIELDS:
        value = suite.get(field, STANDARD[field])
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"suite {field} must be an integer")
        if value < (0 if field == "seed" else 1):
            raise ValueError(f"suite {field} out of range")
        out[field] = value
    return out


class Suites:
    """Accessors over the shared ``data["suites"]`` table (name -> suite)."""

    def __init__(self, table):
        self._table = table          # the shared data["suites"] subdict

    def all(self):
        """``STANDARD`` first, then the custom suites sorted by name."""
        custom = [self._table[name] for name in sorted(self._table)]
        return [dict(STANDARD)] + custom

    def names(self):
        return [suite["name"] for suite in self.all()]

    def get(self, name):
        """The suite called ``name`` (``STANDARD`` for 'standard'), or None."""
        if name == STANDARD["name"]:
            return dict(STANDARD)
        suite = self._table.get(name)
        return dict(suite) if suite else None

    def save(self, suite):
        """Validate and store ``suite`` (overwrites a same-named custom)."""
        clean = normalize(suite)
        self._table[clean["name"]] = clean
        return clean

    def remove(self, name):
        """Drop a custom suite (the standard suite cannot be removed)."""
        self._table.pop(name, None)
