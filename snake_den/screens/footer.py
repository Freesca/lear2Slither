"""The persistent job-pool footer, visible on every tab.

Absorbs the old Dashboard's role: a bottom strip showing the pool status and a
compact chip per recent job -- live curve, status colour, a Stop button while
it runs, and (for a finished train job) a click that opens the model's detail.
Drawn by the App after the active tab's content, every frame. Reads the
JobManager; mutates nothing but job state via ``stop``.
"""
from snake_den import jobs as jobs_mod
from snake_den import widgets

_STATUS_COLOR = {
    jobs_mod.RUNNING: widgets.CABINET_GOLD,
    jobs_mod.QUEUED: widgets.MUTED,
    jobs_mod.FINISHED: widgets.APPLE_GREEN,
    jobs_mod.FAILED: widgets.APPLE_RED,
    jobs_mod.STOPPED: widgets.MUTED,
}

_TOP = widgets.CONTENT_BOTTOM
_CHIPS = 4
_CHIP_W = 228
_CHIP_H = 60
_GAP = 8


class Footer:
    """The bottom pool strip; one App owns one Footer."""

    def __init__(self, app):
        self.app = app

    def draw(self):
        surface, ui = self.app.surface, self.app.ui
        mgr = self.app.jobs
        widgets.dither(surface, (0, _TOP, widgets.WIDTH, 2),
                       widgets.DITHER_BLUE)
        surface.fill(widgets.PANEL, (0, _TOP + 2, widgets.WIDTH,
                                     widgets.FOOTER_H - 2))
        surface.fill(widgets.CABINET_GOLD,
                     (0, widgets.HEIGHT - 2, widgets.WIDTH, 2))
        running = sum(1 for j in mgr.jobs if j.status == jobs_mod.RUNNING)
        queued = sum(1 for j in mgr.jobs if j.status == jobs_mod.QUEUED)
        widgets.label(surface, ui, (16, _TOP + 8),
                      f"POOL {mgr.pool_size or 'auto'}    RUNNING {running}"
                      f"    QUEUED {queued}", widgets.SILVER)
        if not mgr.jobs:
            widgets.label(surface, ui, (16, _TOP + 34),
                          "no jobs yet - launch one from Train or Eval",
                          widgets.MUTED)
            return
        newest = list(reversed(mgr.jobs))[:_CHIPS]
        for i, job in enumerate(newest):
            self._chip(surface, ui, job, 16 + i * (_CHIP_W + _GAP),
                       _TOP + 30)

    def _chip(self, surface, ui, job, x, y):
        rect = (x, y, _CHIP_W, _CHIP_H)
        body = widgets.panel(surface, rect)
        color = _STATUS_COLOR.get(job.status, widgets.MUTED)
        widgets.label(surface, ui, (x + 8, y + 6),
                      f"#{job.id} {job.spec.type}", color)
        widgets.label(surface, ui, (x + 8, y + 28), self._detail(job),
                      widgets.MUTED)
        curve = [event["max_length"] for event in job.sessions]
        widgets.line_chart(
            surface, (x + _CHIP_W - 96, y + 6, 88, _CHIP_H - 12), curve)
        if job.status in (jobs_mod.RUNNING, jobs_mod.QUEUED):
            if widgets.button(surface, ui, (x + _CHIP_W - 60, y + 42, 52, 14),
                              "stop"):
                self.app.jobs.stop(job.id)
        elif (job.status == jobs_mod.FINISHED
              and job.spec.type == "train" and job.spec.save_path):
            if body.collidepoint(ui.mouse) and ui.click:
                self.app.open_models(detail=job.spec.save_path)

    def _detail(self, job):
        if job.status == jobs_mod.FAILED:
            return f"failed: {(job.error or '')[:22]}"
        if job.spec.type == "train":
            # Training is a process: judged by its curve, not success%@target
            # (epsilon > 0 during training would understate the policy).
            if job.status == jobs_mod.FINISHED:
                lengths = [e["max_length"] for e in job.sessions]
                best = max(lengths, default=0)
                return f"done  {len(job.sessions)} sessions  best len {best}"
            return f"{len(job.sessions)}/{job.spec.sessions} sessions"
        if job.status == jobs_mod.FINISHED and job.summary:
            pct = job.summary["success_rate"] * 100   # success%: an eval idea
            return f"done  {pct:.0f}%  len {job.summary['length']['max']}"
        return f"{len(job.sessions)} games"
