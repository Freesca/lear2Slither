"""Models tab: the model list + a model's detail view. (rework)

Merges the old Home + Registry. The **list** shows every registered model with
a policy-fingerprint glyph and an evaluated badge under the selected suite
(Compare is gone). Clicking a model opens its **detail**: full statistics, the
training config + letter scheme, the persisted training curve, the action
buttons (Eval / Watch / Clone / Remove -- no Compare), and the enhanced Q-table
with a scheme-driven legend. Models on disk are auto-registered so the
committed deliverables appear.
"""
import os

import pygame

from snake_den import registry, scheme as scheme_mod
from snake_den import snake_proc, suites, viewdata, widgets
from snake_den.screens.base import MARGIN, Screen

_ARROWS = ("^F", "<L", ">R", "vB")          # FORWARD / LEFT / RIGHT / BACK
_SORTS = ("state", "greedy", "visits")
_ROW_H = 54
_QROW_H = 22
_LIST_TOP = widgets.CONTENT_TOP + 34


def _name(path):
    """Just the model's file name (the directory is a Settings concern)."""
    return os.path.basename(path)


def _summary_current(entry):
    """Whether an entry's cached policy summary matches the current schema."""
    return (entry.get("qsummary") or {}).get("v") == viewdata.SUMMARY_VERSION


def _glyph_color(letter):
    """The decoded colour of one state symbol (the legend, calm tints).

    Calmed 2026-06-24 to match the Q-value tints: green family -> Q_POS,
    danger/red family -> Q_NEG, soft warnings (d/c) -> MUTED. Gold is no longer
    spent on a glyph -- it stays reserved for active/greedy marks (DESIGN 5.2).
    """
    if letter in "Gg":
        return widgets.Q_POS
    if letter in "DRr":
        return widgets.Q_NEG
    if letter in "dc":
        return widgets.MUTED
    return widgets.DIM                       # N / clear


class ModelsScreen(Screen):
    """List of trained models; click one for its detail + Q-table."""

    def __init__(self, app):
        super().__init__(app, "models")
        self.suite_index = 0
        self.list_scroll = 0
        self.detail = None                   # open model's path, or None
        self.rows = []                       # cached Q-table rows of detail
        self.legend = {}
        self.curve = []                      # detail model's training curve
        self.qfocus = None                   # state the minimap last jumped to
        self.qscroll = 0
        self.sort_index = 0
        self.error = ""
        self.message = ""

    # --- navigation entry points ----------------------------------------

    def open_detail(self, path):
        self.detail = path
        self.qscroll = 0
        self.message = ""
        self.rows, self.legend, self.error = [], {}, ""
        self.curve, self.qfocus = [], None
        if path is None:
            return
        try:
            from slither import model_io
            model = model_io.load(path)
            self.rows = viewdata.qtable_rows(model)
            self.legend = scheme_mod.legend(model.scheme.as_dict())
            self.curve = model.curve          # v3 carries its own curve
            self._resort()
        except (ValueError, OSError) as exc:
            self.error = f"cannot read model: {exc}"

    # --- frame dispatch -------------------------------------------------

    def draw(self):
        if self.detail is None:
            self._draw_list()
        else:
            self._draw_detail()

    # --- the list -------------------------------------------------------

    def _selected_suite(self):
        all_suites = self.app.suites.all()
        self.suite_index %= len(all_suites)
        return all_suites[self.suite_index]

    def _discover(self):
        """Auto-register any *.json in the models path not yet known, and
        refresh a known model whose cached policy summary is stale (the
        fingerprint semantics changed -- viewdata.SUMMARY_VERSION). Registering
        rewrites the cache, so each model is reloaded at most once per upgrade.
        """
        reg = self.app.registry
        models = reg.models()
        for path in registry.discover(self.app.settings.models_path):
            entry = models.get(path)
            if entry is not None and _summary_current(entry):
                continue
            try:
                reg.register(path)
            except (ValueError, OSError):
                pass

    def _draw_list(self):
        surface, ui = self.app.surface, self.app.ui
        self._discover()
        self.heading("models")
        suite = self._selected_suite()
        self._suite_selector(surface, ui, suite)

        models = self.app.registry.models()
        paths = sorted(models)
        if not paths:
            widgets.label(surface, ui, (MARGIN, _LIST_TOP + 10),
                          "no models yet - train one from the Train tab.",
                          widgets.MUTED)
            return
        profile = suites.profile(suite)
        view_h = widgets.CONTENT_BOTTOM - _LIST_TOP
        visible = view_h // _ROW_H
        self.list_scroll = max(0, min(self.list_scroll - ui.scroll,
                                      max(0, len(paths) - visible)))
        clip = surface.get_clip()
        surface.set_clip(pygame.Rect(0, _LIST_TOP, widgets.WIDTH, view_h))
        end = min(len(paths), self.list_scroll + visible)
        for i in range(self.list_scroll, end):
            path = paths[i]
            self._row(surface, ui, path, models[path], profile,
                      _LIST_TOP + (i - self.list_scroll) * _ROW_H)
        surface.set_clip(clip)

    def _suite_selector(self, surface, ui, suite):
        y = widgets.CONTENT_TOP
        widgets.label(surface, ui, (508, y), "suite", widgets.MUTED)
        if widgets.button(surface, ui, (576, y - 4, 26, 24), "<"):
            self.suite_index -= 1
        if widgets.button(surface, ui, (716, y - 4, 26, 24), ">"):
            self.suite_index += 1
        widgets.label(surface, ui, (610, y), suite["name"],
                      widgets.CABINET_GOLD)
        widgets.label(surface, ui, (752, y),
                      f"g{suite['games']} t{suite['target']} "
                      f"b{suite['board']}", widgets.MUTED)

    def _row(self, surface, ui, path, entry, profile, y):
        rect = widgets.panel(surface, (MARGIN, y, widgets.WIDTH - 2 * MARGIN,
                                       _ROW_H - 6))
        widgets.label(surface, ui, (rect.x + 8, y + 4), _name(path),
                      widgets.SILVER)
        widgets.label(surface, ui, (rect.x + 8, y + 26),
                      f"{entry.get('sessions', '?')} sessions", widgets.MUTED)
        score = self.app.registry.score(path, profile)
        if score is None:
            widgets.label(surface, ui, (rect.x + 360, y + 14), "not evaluated",
                          widgets.MUTED)
        else:
            widgets.label(surface, ui, (rect.x + 360, y + 14),
                          f"{score['success_pct']:.0f}%  "
                          f"@ len>={profile['target']}", widgets.APPLE_GREEN)
        summary = entry.get("qsummary") or {}
        widgets.label(surface, ui, (rect.x + 560, y + 14),
                      f"cover {summary.get('coverage', 0)}/"
                      f"{summary.get('total', 0)}", widgets.MUTED)
        widgets.fingerprint(
            surface, (rect.right - 196, y + 4, 188, _ROW_H - 14),
            summary.get("fingerprint", []))
        if rect.collidepoint(ui.mouse) and ui.click:
            self.open_detail(path)

    # --- the detail -----------------------------------------------------

    def _resort(self):
        mode = _SORTS[self.sort_index]
        if mode == "state":
            self.rows.sort(key=lambda r: r[0])
        elif mode == "greedy":
            self.rows.sort(key=lambda r: (r[3], r[0]))
        else:                                # visits, descending
            self.rows.sort(key=lambda r: -sum(r[2]))

    def _draw_detail(self):
        surface, ui = self.app.surface, self.app.ui
        path = self.detail
        entry = self.app.registry.models().get(path, {})
        self.heading(_name(path))
        back = (widgets.WIDTH - 110, widgets.CONTENT_TOP - 6, 94, 26)
        if widgets.button(surface, ui, back, "< back"):
            self.detail = None
            return
        self._detail_actions(surface, ui, path, entry)
        if self.detail is None:              # Remove closed the detail view
            return
        self._detail_stats(surface, ui, entry)
        if self.error:
            widgets.label(surface, ui, (MARGIN, widgets.CONTENT_TOP + 150),
                          self.error, widgets.APPLE_RED)
            return
        self._curve(surface, ui, entry)      # "how it learned"
        self._qtable(surface, ui)            # "what it learned" + minimap

    def _detail_actions(self, surface, ui, path, entry):
        y = widgets.CONTENT_TOP + 28
        if widgets.button(surface, ui, (MARGIN, y, 90, 30), "Eval"):
            self.app.open_eval(model=path)
        if widgets.button(surface, ui, (MARGIN + 98, y, 90, 30), "Watch"):
            self._watch(path)
        if widgets.button(surface, ui, (MARGIN + 196, y, 90, 30), "Clone"):
            self._clone(path, entry)
        if widgets.button(surface, ui, (MARGIN + 294, y, 90, 30), "Remove"):
            self.app.registry.forget(path)
            self.app.save()
            self.detail = None
        if self.message:
            widgets.label(surface, ui, (MARGIN + 396, y + 8), self.message,
                          widgets.MUTED)

    def _watch(self, path):
        suite = self._selected_suite()
        config = snake_proc.eval_config(suites.profile(suite))
        config["evaluation"]["step_cap"] = suite["step_cap"]
        spec = snake_proc.JobSpec(
            "watch", config, sessions=max(1, suite["games"]),
            base_model=path, eval_profile=suites.profile(suite))
        self.app.jobs.watch(spec)
        self.message = "watch window opened"

    def _clone(self, path, entry):
        config = entry.get("config")
        if not config:
            self.message = "no config recorded to clone"
            return
        spec = snake_proc.JobSpec("train", config, save_path="models/new.json")
        self.app.open_train(spec)

    def _detail_stats(self, surface, ui, entry):
        x, y = MARGIN, widgets.CONTENT_TOP + 70
        scheme = entry.get("scheme") or scheme_mod.DEFAULT
        widgets.label(surface, ui, (x, y),
                      f"alphabet  {scheme_mod.alphabet(scheme)}   "
                      f"({scheme_mod.state_count(scheme)} states)",
                      widgets.SILVER)
        suite = self._selected_suite()
        score = self.app.registry.score(self.detail, suites.profile(suite))
        if score is None:
            widgets.label(surface, ui, (x, y + 24),
                          f"not evaluated on suite '{suite['name']}'",
                          widgets.MUTED)
            return
        full = score.get("full", {})
        length = full.get("length", {})
        duration = full.get("duration", {})
        outcomes = full.get("outcomes", {})
        widgets.label(surface, ui, (x, y + 24),
                      f"success {score['success_pct']:.1f}%   "
                      f"games {score['games']}   target {score['target']}",
                      widgets.APPLE_GREEN)
        widgets.label(surface, ui, (x, y + 44),
                      f"LENGTH:  mean {length.get('mean', 0):.1f}  "
                      f"med {length.get('median', 0):.1f}  "
                      f"max {length.get('max', 0)}  "
                      f"std {length.get('std', 0):.1f}", widgets.MUTED)
        widgets.label(surface, ui, (x, y + 64),
                      f"DURATION:  mean {duration.get('mean', 0):.0f}  "
                      f"max {duration.get('max', 0)}", widgets.MUTED)
        widgets.label(surface, ui, (x, y + 84),
                      "OUTCOMES:  " + "  ".join(
                          f"{k} {outcomes.get(k, 0)}" for k in
                          ("wall", "self", "length_zero", "truncated", "won")),
                      widgets.MUTED)

    def _curve(self, surface, ui, entry):
        """The training curve as its own labelled full-width band.

        Reads the curve from the loaded model (v3 carries it, so every model
        shows one), falling back to the legacy hub cache for v1/v2 files.
        """
        curve = self.curve or entry.get("curve") or []
        y = widgets.CONTENT_TOP + 170
        widgets.label(surface, ui, (MARGIN, y),
                      "max length per session",
                      widgets.CABINET_GOLD)
        rect = (MARGIN, y + 18, widgets.WIDTH - 2 * MARGIN, 42)
        if curve:
            widgets.line_chart(surface, rect, curve)
        else:
            widgets.panel(surface, rect)
            widgets.label(surface, ui, (MARGIN + 8, y + 30),
                          "no curve recorded (pre-v3 model)", widgets.MUTED)

    # --- the enhanced Q-table -------------------------------------------

    _QX = (MARGIN, 150, 240, 330, 420, 520, 600)     # state/F/L/R/B/move/n
    _MINIMAP_X = 656                                  # right-hand minimap zone

    def _qtable(self, surface, ui):
        sort_rect = (360, widgets.CONTENT_TOP + 232, 120, 24)
        if widgets.button(surface, ui, sort_rect,
                          f"by {_SORTS[self.sort_index]}"):
            self.sort_index = (self.sort_index + 1) % len(_SORTS)
            self._resort()
        legend_y = widgets.CONTENT_TOP + 262
        head_y = self._legend(surface, ui, legend_y) + 6
        for x, head in zip(self._QX, ("state", "F", "L", "R", "B", "move",
                                      "n")):
            widgets.label(surface, ui, (x, head_y), head, widgets.MUTED)
        first = head_y + 24
        view_h = widgets.CONTENT_BOTTOM - first
        visible = view_h // _QROW_H
        self._minimap(surface, ui, head_y)
        self.qscroll = max(0, min(self.qscroll - ui.scroll,
                                  max(0, len(self.rows) - visible)))
        clip = surface.get_clip()
        surface.set_clip(pygame.Rect(0, first, self._MINIMAP_X - 8, view_h))
        end = min(len(self.rows), self.qscroll + visible)
        for i in range(self.qscroll, end):
            self._qrow(surface, ui, self.rows[i],
                       first + (i - self.qscroll) * _QROW_H)
        surface.set_clip(clip)

    def _minimap(self, surface, ui, top):
        """Interactive policy map beside the table (D2): the same value-tint as
        the list glyph, built live from the rows in the current sort so a click
        jumps the table to that state under any sort. One block per state:
        green where the greedy value is positive, magenta where it is negative.
        """
        x = self._MINIMAP_X
        widgets.label(surface, ui, (x, top), "policy map", widgets.MUTED)
        signs = [1 if q[a] > 0 else -1 if q[a] < 0 else 0
                 for _s, q, _n, a in self.rows]
        keys = [s for s, _q, _n, _a in self.rows]
        rect = (x, top + 20, widgets.WIDTH - x - MARGIN,
                widgets.CONTENT_BOTTOM - top - 22)
        clicked = widgets.fingerprint(surface, rect, signs, block=10,
                                      ui=ui, keys=keys)
        if clicked is not None:
            self.qfocus = keys[clicked]      # follow this state on re-sort
            self.qscroll = max(0, clicked - 3)

    def _legend(self, surface, ui, y):
        """Draw the symbol legend, wrapping to extra rows; return next y."""
        x, row_y = MARGIN, y
        limit = widgets.WIDTH - MARGIN
        for letter, meaning in self.legend.items():
            text = f"{letter}={meaning}"
            width = ui.font.size(text)[0]
            if x > MARGIN and x + width > limit:
                x, row_y = MARGIN, row_y + 22
            widgets.label(surface, ui, (x, row_y), letter,
                          _glyph_color(letter))
            widgets.label(surface, ui, (x + 12, row_y), f"={meaning}",
                          widgets.MUTED)
            x += width + 18
        return row_y + 22

    def _qrow(self, surface, ui, row, y):
        state, qvals, nvals, argmax = row
        if state == self.qfocus:             # the state the minimap jumped to
            widgets.dither(surface, (8, y - 1, 640, _QROW_H - 2),
                           widgets.DITHER_BLUE)
        gx = self._QX[0]
        for letter in state:
            widgets.label(surface, ui, (gx, y), letter, _glyph_color(letter))
            gx += 12
        for col in range(4):
            value = qvals[col]
            color = (widgets.Q_POS if value > 0 else
                     widgets.Q_NEG if value < 0 else widgets.DIM)
            widgets.label(surface, ui, (self._QX[col + 1], y),
                          f"{value:+.1f}", color)
            if col == argmax:
                pygame.draw.rect(surface, widgets.CABINET_GOLD,
                                 pygame.Rect(self._QX[col + 1] - 3, y - 1, 76,
                                             _QROW_H - 2), 1)
        widgets.label(surface, ui, (self._QX[5], y), _ARROWS[argmax],
                      widgets.CABINET_GOLD)
        widgets.label(surface, ui, (self._QX[6], y), str(sum(nvals)),
                      widgets.MUTED)
