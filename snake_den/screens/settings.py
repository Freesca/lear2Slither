"""Settings tab: paths, pool size, theme, motion toggles.

Edits the persisted hub settings (snake_den/settings.py). The eval profile no
longer lives here -- it moved to the Eval page as named suites -- so this is
just the worker-pool size, where models live, the theme name, and the two
motion toggles (reduced motion / scanlines). Apply validates, writes through
the Settings accessor, updates the JobManager, and persists.
"""
import os

from snake_den import widgets
from snake_den.screens.base import MARGIN, Screen


class SettingsScreen(Screen):
    """Form over the persisted hub settings."""

    def __init__(self, app):
        super().__init__(app, "settings")
        s = app.settings
        self.pool = "" if s.pool_size is None else str(s.pool_size)
        self.models_path = s.models_path
        self.theme = s.theme
        self.reduced_motion = s.reduced_motion
        self.scanlines = s.scanlines
        self.message = ""

    def draw(self):
        surface, ui = self.app.surface, self.app.ui
        self.heading("settings")
        y = widgets.CONTENT_TOP + 36
        widgets.label(surface, ui, (MARGIN, y + 5),
                      "pool size (blank = auto)", widgets.MUTED)
        self.pool = widgets.text_field(
            surface, ui, (336, y, 200, 26), self.pool,
            field_id="set.pool", numeric=True)
        widgets.label(surface, ui, (MARGIN, y + 45), "models path",
                      widgets.MUTED)
        self.models_path = widgets.text_field(
            surface, ui, (336, y + 40, 360, 26), self.models_path,
            field_id="set.models")
        widgets.label(surface, ui, (MARGIN, y + 85), "theme", widgets.MUTED)
        self.theme = widgets.text_field(
            surface, ui, (336, y + 80, 200, 26), self.theme,
            field_id="set.theme")
        self.reduced_motion = widgets.toggle(
            surface, ui, (MARGIN, y + 120, 22, 22), self.reduced_motion,
            "reduced motion (no blink)")
        self.scanlines = widgets.toggle(
            surface, ui, (MARGIN, y + 152, 22, 22), self.scanlines,
            "CRT scanlines overlay")
        if widgets.button(surface, ui, (MARGIN, y + 196, 150, 34), "Apply"):
            self._apply()
        if self.message:
            color = widgets.APPLE_GREEN if self.message == "saved" \
                else widgets.APPLE_RED
            widgets.label(surface, ui, (180, y + 204), self.message, color)

    def _apply(self):
        pool_text = self.pool.strip()
        try:
            pool = None if pool_text == "" else int(pool_text)
        except ValueError:
            self.message = "pool size must be an integer or blank"
            return
        if pool is not None and pool < 1:
            self.message = "pool size must be >= 1 (or blank for auto)"
            return
        settings = self.app.settings
        settings.pool_size = pool
        settings.models_path = self.models_path.strip() or "models"
        settings.theme = self.theme.strip() or "pixel"
        settings.reduced_motion = self.reduced_motion
        settings.scanlines = self.scanlines
        self.app.jobs.pool_size = pool or min(4, os.cpu_count() or 1)
        self.app.save()
        self.message = "saved"
