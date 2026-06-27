"""Eval tab: suites + multi-model batch evaluation.

The eval half of the old New-Job screen, now suite-driven. The evaluation
parameters (games / seed / board / target / step_cap) moved here from Settings
as named *suites*: pick a saved suite to load its numbers, edit them, and
**Save** the set under a name. Tick any number of models and **Eval selected**
submits one eval job per model against the current parameters -- they run
concurrently in the pool ("at the same time"). Each model is replayed under its
own recorded scheme (slither enforces that on -load), so no scheme picker here.
"""
import os
from copy import deepcopy

import pygame

from snake_den import registry, snake_proc, suites, widgets
from snake_den.screens.base import MARGIN, Screen
from snake_den.snake_proc import JobSpec

_PARAMS = ("games", "seed", "board", "target", "step_cap")
_ROW_H = 28
_LIST_TOP = widgets.CONTENT_TOP + 134


def _name(path):
    return os.path.basename(path)


class EvalScreen(Screen):
    """Suite editor + a checklist of models to evaluate together."""

    def __init__(self, app):
        super().__init__(app, "eval")
        self.suite_index = 0
        self.params = {key: str(suites.STANDARD[key]) for key in _PARAMS}
        self.suite_name = ""
        self.selected = set()
        self.list_scroll = 0
        self.message = ""

    def preselect(self, model):
        if model:
            self.selected = {model}

    def _select_suite(self, delta):
        names = self.app.suites.names()
        self.suite_index = (self.suite_index + delta) % len(names)
        suite = self.app.suites.get(names[self.suite_index])
        self.params = {key: str(suite[key]) for key in _PARAMS}

    def _model_choices(self):
        found = set(self.app.registry.models())
        found |= set(registry.discover(self.app.settings.models_path))
        return sorted(found)

    def draw(self):
        surface, ui = self.app.surface, self.app.ui
        self.heading("eval by suites")
        self._suite_row(surface, ui)
        self._params_row(surface, ui)
        self._save_row(surface, ui)
        self._model_list(surface, ui)

    def _suite_row(self, surface, ui):
        y = widgets.CONTENT_TOP + 42
        names = self.app.suites.names()
        self.suite_index %= len(names)
        widgets.label(surface, ui, (MARGIN, y + 4), "suite", widgets.MUTED)
        if widgets.button(surface, ui, (80, y, 24, 24), "<"):
            self._select_suite(-1)
        if widgets.button(surface, ui, (300, y, 24, 24), ">"):
            self._select_suite(1)
        widgets.label(surface, ui, (114, y + 4), names[self.suite_index],
                      widgets.CABINET_GOLD)

    def _params_row(self, surface, ui):
        y = widgets.CONTENT_TOP + 42
        x = 360
        for key in _PARAMS:
            widgets.label(surface, ui, (x, y - 18), key, widgets.MUTED)
            self.params[key] = widgets.text_field(
                surface, ui, (x, y, 88, 24), self.params[key],
                field_id=f"ev.{key}", numeric=True)
            x += 100

    def _save_row(self, surface, ui):
        y = widgets.CONTENT_TOP + 80
        widgets.label(surface, ui, (MARGIN, y + 5), "save as", widgets.MUTED)
        self.suite_name = widgets.text_field(
            surface, ui, (110, y, 200, 24), self.suite_name,
            field_id="ev.name")
        if widgets.button(surface, ui, (324, y, 132, 26), "Save suite"):
            self._save_suite()
        if self.message:
            widgets.label(surface, ui, (470, y + 5), self.message,
                          widgets.MUTED)

    def _model_list(self, surface, ui):
        choices = self._model_choices()
        widgets.label(surface, ui, (MARGIN, _LIST_TOP - 22),
                      f"models  ({len(self.selected)} selected)",
                      widgets.CABINET_GOLD)
        if widgets.button(surface, ui, (724, _LIST_TOP - 26, 220, 26),
                          f"Eval selected ({len(self.selected)})"):
            self._eval_selected()
        if not choices:
            widgets.label(surface, ui, (MARGIN, _LIST_TOP + 6),
                          "no models found - train one first.", widgets.MUTED)
            return
        view_h = widgets.CONTENT_BOTTOM - _LIST_TOP
        visible = view_h // _ROW_H
        self.list_scroll = max(0, min(self.list_scroll - ui.scroll,
                                      max(0, len(choices) - visible)))
        clip = surface.get_clip()
        surface.set_clip(pygame.Rect(0, _LIST_TOP, widgets.WIDTH, view_h))
        end = min(len(choices), self.list_scroll + visible)
        for i in range(self.list_scroll, end):
            path = choices[i]
            y = _LIST_TOP + (i - self.list_scroll) * _ROW_H
            checked = widgets.toggle(surface, ui, (MARGIN, y, 22, 22),
                                     path in self.selected)
            if checked:
                self.selected.add(path)
            else:
                self.selected.discard(path)
            widgets.label(surface, ui, (MARGIN + 32, y + 2), _name(path),
                          widgets.SILVER)
        surface.set_clip(clip)

    def _read_params(self):
        return {key: int(self.params[key]) for key in _PARAMS}

    def _save_suite(self):
        try:
            params = self._read_params()
        except ValueError:
            self.message = "params must be integers"
            return
        try:
            self.app.suites.save({"name": self.suite_name, **params})
        except ValueError as exc:
            self.message = str(exc)
            return
        self.app.save()
        self.message = f"saved suite '{self.suite_name.strip()}'"

    def _eval_selected(self):
        try:
            params = self._read_params()
        except ValueError:
            self.message = "params must be integers"
            return
        if not self.selected:
            self.message = "select at least one model"
            return
        profile = {key: params[key]
                   for key in ("games", "seed", "board", "target")}
        config = snake_proc.eval_config(profile)
        config["evaluation"]["step_cap"] = params["step_cap"]
        for path in sorted(self.selected):
            spec = JobSpec("eval", deepcopy(config), sessions=params["games"],
                           seed=params["seed"], base_model=path,
                           eval_profile=profile)
            self.app.jobs.submit(spec)       # archived to history on finish
        self.message = f"queued {len(self.selected)} eval job(s)"
