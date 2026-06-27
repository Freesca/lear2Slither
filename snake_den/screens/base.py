"""Screen base, shared by the hub's tab screens. (rework)

A screen is driven by the App's 60 FPS loop: ``draw`` is called every frame and
both renders and handles this frame's input (immediate mode), reaching the app
through ``self.app`` (tab navigation, JobManager, registry, suites, settings,
surface, ui). The gold title bar and the nav tabs are drawn by the app, so a
screen only fills the content band (``widgets.CONTENT_TOP`` ..
``widgets.CONTENT_BOTTOM``). The base imports no concrete screen, so the
screens form no import cycle.
"""
from snake_den import widgets

MARGIN = 16


class Screen:
    """Base: an app reference, a title, and a per-frame ``draw``."""

    def __init__(self, app, title=""):
        self.app = app
        self.title = title

    def draw(self):
        raise NotImplementedError

    def heading(self, text, color=widgets.CABINET_GOLD):
        """Draw a content heading at the top of the content band; return y."""
        widgets.label(self.app.surface, self.app.ui,
                      (MARGIN, widgets.CONTENT_TOP), text, color)
        return widgets.CONTENT_TOP + 26
