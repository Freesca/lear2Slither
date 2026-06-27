"""Runs tab: the persistent archive of completed train/eval runs.

The footer is the live tail -- what is running, queued, or just finished; this
tab is the home of the ``history`` store: every terminal train/eval run,
newest first, read from ``registry.history()``. A train run shows its
learning-honest outcome (sessions + best/recent length, never success% -- that
is an eval concept); an eval run shows its success summary; a failed
run shows its error. Clicking a run opens its model's detail (the curve is
reliably present from format v3). Records are read defensively so a pre-rework
or partial record renders sparse rather than crashing (the no-crash rule).
"""
import os

import pygame

from snake_den import widgets
from snake_den.screens.base import MARGIN, Screen

_ROW_H = 30
_LIST_TOP = widgets.CONTENT_TOP + 22


def _name(path):
    return os.path.basename(path) if path else "?"


def _when(iso):
    """A compact ``MM-DD HH:MM`` from an iso8601 string (or it unchanged)."""
    return iso[5:16].replace("T", " ") if len(iso) >= 16 else iso


class RunsScreen(Screen):
    """Reverse-chronological list of finished runs (train + eval)."""

    def __init__(self, app):
        super().__init__(app, "runs")
        self.scroll = 0

    def draw(self):
        surface, ui = self.app.surface, self.app.ui
        self.heading("runs history")
        records = list(reversed(self.app.registry.history()))
        if not records:
            widgets.label(surface, ui, (MARGIN, _LIST_TOP + 10),
                          "no runs yet - launch one from Train or Eval",
                          widgets.MUTED)
            return
        view_h = widgets.CONTENT_BOTTOM - _LIST_TOP
        visible = view_h // _ROW_H
        self.scroll = max(0, min(self.scroll - ui.scroll,
                                 max(0, len(records) - visible)))
        clip = surface.get_clip()
        surface.set_clip(pygame.Rect(0, _LIST_TOP, widgets.WIDTH, view_h))
        end = min(len(records), self.scroll + visible)
        for i in range(self.scroll, end):
            self._row(surface, ui, records[i],
                      _LIST_TOP + (i - self.scroll) * _ROW_H)
        surface.set_clip(clip)

    def _row(self, surface, ui, rec, y):
        kind = rec.get("type") or rec.get("job") or "?"
        model = rec.get("model", "")
        rect = widgets.panel(surface, (MARGIN, y, widgets.WIDTH - 2 * MARGIN,
                                       _ROW_H - 6))
        color = (widgets.Q_POS if kind == "train" else
                 widgets.CABINET_GOLD if kind == "eval" else widgets.MUTED)
        widgets.label(surface, ui, (rect.x + 8, y + 4), kind.upper(), color)
        widgets.label(surface, ui, (rect.x + 96, y + 4), _name(model),
                      widgets.SILVER)
        widgets.label(surface, ui, (rect.x + 312, y + 4),
                      self._outcome(rec), widgets.MUTED)
        widgets.label(surface, ui, (rect.right - 140, y + 4),
                      _when(rec.get("when", "")), widgets.MUTED)
        if model and rect.collidepoint(ui.mouse) and ui.click:
            self.app.open_models(detail=model)

    def _outcome(self, rec):
        """A defensive one-line outcome for any history record shape."""
        outcome = rec.get("outcome") or {}
        kind = rec.get("type") or rec.get("job")
        if "error" in outcome:
            return f"failed: {str(outcome['error'])[:32]}"
        if kind == "train":
            return (f"{rec.get('sessions', '?')} sessions   "
                    f"best len {outcome.get('best_len', '?')}   "
                    f"recent {outcome.get('recent_len', '?')}")
        if kind == "eval":
            pct = outcome.get("success_pct")
            if pct is None:
                return f"{rec.get('games', '?')} games"
            return (f"{pct}%  @ len>={outcome.get('target')}   "
                    f"({rec.get('games', '?')} games)")
        n = rec.get("sessions", rec.get("games", "?"))     # pre-rework record
        return f"{n} sessions/games"
