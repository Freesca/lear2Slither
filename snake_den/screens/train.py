"""Train tab: the training-job form + the dynamic state-letters picker.

The train half of the old New-Job screen, plus a **state letters** section: the
user toggles the perceptual distinctions for this run (snake_den.scheme mirrors
the product's alphabet) and sees the resulting alphabet + state count live --
the trade-off (more letters = bigger table = slower but richer) made visible.
The chosen ``[state]`` block rides into the temp TOML, so slither trains with
it. Resuming from a base model locks the scheme to that model's (slither uses
the loaded model's scheme regardless), so the toggles grey out.
"""
import os
from copy import deepcopy

from slither import config as config_module
from snake_den import registry
from snake_den import scheme as scheme_mod
from snake_den import widgets
from snake_den.screens.base import MARGIN, Screen
from snake_den.snake_proc import JobSpec

# Editable numeric config fields, in display order: (section, key, kind).
FIELDS = (
    ("board", "size", int),
    ("board", "green_apples", int),
    ("board", "red_apples", int),
    ("board", "initial_length", int),
    ("rewards", "green", float),
    ("rewards", "red", float),
    ("rewards", "step", float),
    ("rewards", "death", float),
    ("rewards", "win", float),
    ("exploration", "epsilon_start", float),
    ("exploration", "epsilon_min", float),
    ("exploration", "epsilon_decay", float),
    ("learning", "alpha", float),
    ("learning", "gamma", float),
    ("evaluation", "step_cap", int),
)
_PER_COL = 8


def _name(path):
    return os.path.basename(path)


class TrainScreen(Screen):
    """The training-job launcher."""

    def __init__(self, app):
        super().__init__(app, "train")
        self.prefill(None)

    def prefill(self, spec):
        """(Re)seed every field from DEFAULTS, overlaid with a cloned spec."""
        self.config = deepcopy(config_module.DEFAULTS)
        source = spec.config if spec else None
        if source:
            for sec, key, _ in FIELDS:
                if isinstance(source.get(sec), dict) and key in source[sec]:
                    self.config[sec][key] = source[sec][key]
        self.values = {(sec, key): str(self.config[sec][key])
                       for sec, key, _ in FIELDS}
        self.scheme = scheme_mod.normalize(
            (source or {}).get("state") if source else None)
        self.sessions = str(spec.sessions) if spec else "100"
        self.seed = str(spec.seed) if spec else "0"
        self.name = (_name(spec.save_path) if spec and spec.save_path
                     else "new.json")
        self.base_index = 0
        self.error = ""

    def _model_choices(self):
        found = set(self.app.registry.models())
        found |= set(registry.discover(self.app.settings.models_path))
        return ["(none)"] + sorted(found)

    def _save_path(self):
        """The full save path: ``models_path`` (Settings) + the name field."""
        name = self.name.strip()
        if not name:
            return ""
        if not name.endswith(".json"):
            name += ".json"
        return os.path.join(self.app.settings.models_path, name)

    def _resuming(self, choices):
        return choices[self.base_index % len(choices)] != "(none)"

    def draw(self):
        surface, ui = self.app.surface, self.app.ui
        self.heading("train a model")
        choices = self._model_choices()
        self._fields(surface, ui)
        self._state_section(surface, ui, choices)
        self._run_params(surface, ui, choices)

    def _fields(self, surface, ui):
        col_x = (MARGIN, 312)
        top = widgets.CONTENT_TOP + 30
        for i, (sec, key, _kind) in enumerate(FIELDS):
            x = col_x[i // _PER_COL]
            y = top + (i % _PER_COL) * 30
            widgets.label(surface, ui, (x, y + 5), key, widgets.MUTED)
            self.values[(sec, key)] = widgets.text_field(
                surface, ui, (x + 176, y, 86, 24), self.values[(sec, key)],
                field_id=f"tr.{sec}.{key}", numeric=True)

    def _state_section(self, surface, ui, choices):
        x = 600
        y = widgets.CONTENT_TOP + 30
        widgets.label(surface, ui, (x, y), "state letters",
                      widgets.CABINET_GOLD)
        resuming = self._resuming(choices)
        if resuming:
            widgets.label(surface, ui, (x, y + 24),
                          "locked to base model's scheme", widgets.MUTED)
        else:
            for i, feature in enumerate(scheme_mod.FEATURES):
                ty = y + 28 + i * 30
                self.scheme[feature] = widgets.toggle(
                    surface, ui, (x, ty, 22, 22), self.scheme[feature],
                    scheme_mod.LABELS[feature])
        # Place the readout below the last toggle row, whatever the count.
        alpha_y = y + 28 + len(scheme_mod.FEATURES) * 30 + 8
        alpha = scheme_mod.alphabet(self.scheme)
        widgets.label(surface, ui, (x, alpha_y),
                      f"alphabet  {alpha}", widgets.SILVER)
        widgets.label(surface, ui, (x, alpha_y + 22),
                      f"k={len(alpha)}  ->  "
                      f"{scheme_mod.state_count(self.scheme)} states / "
                      f"{scheme_mod.qvalue_count(self.scheme)} Q",
                      widgets.MUTED)

    def _run_params(self, surface, ui, choices):
        y = widgets.CONTENT_BOTTOM - 80
        widgets.label(surface, ui, (MARGIN, y + 5), "sessions", widgets.MUTED)
        self.sessions = widgets.text_field(
            surface, ui, (116, y, 64, 24), self.sessions,
            field_id="tr.sessions", numeric=True)
        widgets.label(surface, ui, (192, y + 5), "seed", widgets.MUTED)
        self.seed = widgets.text_field(
            surface, ui, (250, y, 56, 24), self.seed,
            field_id="tr.seed", numeric=True)
        widgets.label(surface, ui, (322, y + 5), "name", widgets.MUTED)
        self.name = widgets.text_field(
            surface, ui, (372, y, 150, 24), self.name, field_id="tr.name")
        widgets.label(surface, ui, (538, y + 5), "resume", widgets.MUTED)
        if widgets.button(surface, ui, (612, y - 1, 24, 26), "<"):
            self.base_index = (self.base_index - 1) % len(choices)
        if widgets.button(surface, ui, (768, y - 1, 24, 26), ">"):
            self.base_index = (self.base_index + 1) % len(choices)
        choice = choices[self.base_index % len(choices)]
        widgets.label(surface, ui, (644, y + 5),
                      choice if choice == "(none)" else _name(choice),
                      widgets.SILVER)
        launch_y = y + 38
        if widgets.button(surface, ui, (MARGIN, launch_y, 150, 32), "Launch"):
            self._launch(choices)
        widgets.label(surface, ui, (180, launch_y + 9),
                      "-> " + (self._save_path() or "?"), widgets.MUTED)
        if self.error:
            widgets.label(surface, ui, (560, launch_y + 9), self.error,
                          widgets.APPLE_RED)

    def _launch(self, choices):
        try:
            cfg = self._parse_config()
        except ValueError as exc:
            self.error = f"number error: {exc}"
            return
        try:
            config_module.load(overrides=cfg)
        except ValueError as exc:
            self.error = str(exc)
            return
        try:
            sessions, seed = int(self.sessions), int(self.seed)
        except ValueError:
            self.error = "sessions and seed must be integers"
            return
        if sessions < 1:
            self.error = "sessions must be >= 1"
            return
        save_path = self._save_path()
        if not save_path:
            self.error = "model name must not be empty"
            return
        choice = choices[self.base_index % len(choices)]
        base_model = None if choice == "(none)" else choice
        spec = JobSpec("train", cfg, sessions=sessions, seed=seed,
                       base_model=base_model, save_path=save_path)
        self.app.jobs.submit(spec)       # archived to history on finish (D3)
        self.error = ""
        self.app.open_models()

    def _parse_config(self):
        cfg = deepcopy(self.config)
        for sec, key, kind in FIELDS:
            cfg[sec][key] = kind(self.values[(sec, key)].strip())
        cfg["state"] = dict(self.scheme)
        return cfg
