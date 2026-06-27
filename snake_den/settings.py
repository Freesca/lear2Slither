"""Settings: the hub's remembered preferences.

A typed, default-filled view over the data file's ``settings`` table: the
worker-pool size, where models live, the UI theme, and the two motion toggles.
Reads fall back to defaults so a fresh data file (no settings yet) still yields
values; writes go straight into the shared dict the app persists via
``persistence.save``.

The eval profile no longer lives here: evaluation parameters moved onto the
Eval page as named *suites* (suites.py), so Settings is paths + pool + theme
only.

``pool_size`` defaults to ``None`` meaning *auto* -- the JobManager then picks
``min(4, cpu_count)`` (jobs.py), which is machine-appropriate rather than a
number frozen into the file.
"""

_DEFAULTS = {
    "pool_size": None,
    "models_path": "models",
    "theme": "pixel",
    "reduced_motion": False,
    "scanlines": False,
}


class Settings:
    """Accessors over the shared ``data["settings"]`` table, with defaults."""

    def __init__(self, table):
        self._table = table          # the shared data["settings"] subdict

    @property
    def pool_size(self):
        return self._table.get("pool_size", _DEFAULTS["pool_size"])

    @pool_size.setter
    def pool_size(self, value):
        self._table["pool_size"] = value

    @property
    def models_path(self):
        return self._table.get("models_path", _DEFAULTS["models_path"])

    @models_path.setter
    def models_path(self, value):
        self._table["models_path"] = value

    @property
    def theme(self):
        return self._table.get("theme", _DEFAULTS["theme"])

    @theme.setter
    def theme(self, value):
        self._table["theme"] = value

    @property
    def reduced_motion(self):
        return bool(self._table.get(
            "reduced_motion", _DEFAULTS["reduced_motion"]))

    @reduced_motion.setter
    def reduced_motion(self, value):
        self._table["reduced_motion"] = bool(value)

    @property
    def scanlines(self):
        return bool(self._table.get("scanlines", _DEFAULTS["scanlines"]))

    @scanlines.setter
    def scanlines(self, value):
        self._table["scanlines"] = bool(value)
