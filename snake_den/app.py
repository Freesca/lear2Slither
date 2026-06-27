"""The hub's pygame application: tabbed shell + the 60 FPS loop.

Owns the window, the one ~60 FPS loop, and a flat **three-tab + settings**
navigation (Models / Train / Eval / Settings) under a gold cabinet title bar,
with a persistent job-pool footer visible on every tab. Each frame it drains
the JobManager (non-blocking), pumps pygame events into the UI context, draws
the active tab's screen and the footer. Closing the window tears every child
down cleanly (``JobManager.shutdown``) and persists the data file before
quitting -- no orphan ``./snake`` process, no traceback (the no-crash rule).

Like the product's gui.py this module imports pygame and is never imported by a
test; the dummy SDL driver is used for a headless smoke (``run`` takes an
optional ``max_frames``). Cross-tab actions (Clone -> Train, Eval -> Eval,
open a model's detail) go through the small ``open_*`` helpers here.
"""
import pygame
from pathlib import Path

from snake_den import jobs as jobs_mod
from snake_den import viewdata, widgets
from snake_den.jobs import JobManager
from snake_den.persistence import load, save
from snake_den.registry import Registry, history_record
from snake_den.screens.eval import EvalScreen
from snake_den.screens.footer import Footer
from snake_den.screens.models import ModelsScreen
from snake_den.screens.runs import RunsScreen
from snake_den.screens.settings import SettingsScreen
from snake_den.screens.train import TrainScreen
from snake_den.settings import Settings
from snake_den.suites import Suites

DATA_PATH = str(Path(__file__).with_name("hub_data.json"))

_TABS = ("models", "train", "eval", "runs", "settings")
_TAB_LABELS = ("MODELS", "TRAIN", "EVAL", "RUNS", "SETTINGS")


class App:
    """The hub: data + JobManager + four tab screens + the pool footer."""

    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.data = load(data_path)
        self.settings = Settings(self.data["settings"])
        self.registry = Registry(self.data)
        self.suites = Suites(self.data["suites"])
        self.jobs = JobManager(pool_size=self.settings.pool_size)
        self.tab = "models"
        self.running = True
        self.window = None               # the real (resizable) window surface
        self.surface = None              # fixed logical canvas screens draw on
        self._scaled = (widgets.WIDTH, widgets.HEIGHT)
        self._origin = (0, 0)
        self.ui = None
        self.clock = None
        self.screens = {}
        self.footer = None
        self._handled = set()            # job ids already folded into registry

    def _setup(self):
        """Open the window, build the UI context, screens and footer."""
        pygame.init()
        pygame.key.set_repeat(400, 40)        # held Backspace etc. in fields
        pygame.display.set_caption("snake_den")
        init = (int(widgets.WIDTH * widgets.INIT_ZOOM),
                int(widgets.HEIGHT * widgets.INIT_ZOOM))
        self.window = pygame.display.set_mode(init, pygame.RESIZABLE)
        # screens draw to a fixed logical canvas; the loop scales it to window
        self.surface = pygame.Surface((widgets.WIDTH, widgets.HEIGHT))
        self.ui = widgets.UI(widgets.make_font())
        self.clock = pygame.time.Clock()
        self.screens = {
            "models": ModelsScreen(self),
            "train": TrainScreen(self),
            "eval": EvalScreen(self),
            "runs": RunsScreen(self),
            "settings": SettingsScreen(self),
        }
        self.footer = Footer(self)

    def run(self, max_frames=None):
        """Open the window and run the loop; tear everything down on exit."""
        self._setup()
        try:
            self._loop(max_frames)
        finally:
            self.jobs.shutdown()
            self.save()
            pygame.quit()
        return 0

    def _loop(self, max_frames):
        frames = 0
        while self.running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.window = pygame.display.set_mode(
                        (max(480, event.w), max(320, event.h)),
                        pygame.RESIZABLE)
            self._update_viewport()
            self.jobs.poll()
            self._reap_finished()
            self.ui.begin_frame(events)
            self.surface.fill(widgets.BG)
            self._draw_chrome()
            self.screens[self.tab].draw()
            self.footer.draw()
            if self.settings.scanlines:
                widgets.scanline_overlay(self.surface)
            widgets.vignette(self.surface)
            self._present()
            self.clock.tick(60)
            frames += 1
            if max_frames is not None and frames >= max_frames:
                self.running = False

    def _update_viewport(self):
        """Fit the logical canvas into the window at an *integer* zoom.

        Integer-only scaling keeps every pixel square and crisp: pick the
        largest whole multiple that fits, centre it, letterbox
        the rest. The UI gets the same integer zoom so mouse hit-tests map
        window pixels back to logical pixels.
        """
        win_w, win_h = self.window.get_size()
        zoom = max(1, int(min(win_w / widgets.WIDTH,
                              win_h / widgets.HEIGHT)))
        self._scaled = (widgets.WIDTH * zoom, widgets.HEIGHT * zoom)
        self._origin = ((win_w - self._scaled[0]) // 2,
                        (win_h - self._scaled[1]) // 2)
        self.ui.scale = zoom
        self.ui.offset = self._origin

    def _present(self):
        """Blit the zoomed canvas into the window (crisp, no smoothing)."""
        self.window.fill((0, 0, 0))
        self.window.blit(
            pygame.transform.scale(self.surface, self._scaled), self._origin)
        pygame.display.flip()

    def _draw_chrome(self):
        """The gold title bar + the nav tab row (drawn before content)."""
        widgets.title_bar(self.surface, self.ui, "snake_den")
        active = _TABS.index(self.tab)
        chosen = widgets.tabs(
            self.surface, self.ui,
            (0, widgets.TITLE_H + 2, widgets.WIDTH, widgets.TAB_H),
            list(_TAB_LABELS), active)
        widgets.side_rails(self.surface)
        if chosen != active:
            self.tab = _TABS[chosen]
            self.ui.focus = None              # leaving a tab drops field focus

    # --- cross-tab navigation -------------------------------------------

    def open_models(self, detail=None):
        self.tab = "models"
        self.screens["models"].open_detail(detail)

    def open_train(self, spec=None):
        self.tab = "train"
        self.screens["train"].prefill(spec)

    def open_eval(self, model=None):
        self.tab = "eval"
        self.screens["eval"].preselect(model)

    # --- persistence ----------------------------------------------------

    def save(self):
        save(self.data_path, self.data)

    def _reap_finished(self):
        """Fold newly-finished jobs into the registry so the views populate.

        A finished **train** job's model is registered and its live curve is
        persisted (downsampled) so the detail page can show it after the job is
        gone. A finished **eval** job's summary is recorded as that model's
        score under the profile it ran with. Every terminal train/eval job
        (finished *or* failed) is also archived to the ``history`` store -- the
        Runs tab's source of truth. Once each.
        """
        changed = False
        for job in self.jobs.jobs:
            if job.id in self._handled or not job.is_terminal:
                continue
            self._handled.add(job.id)
            if (job.status not in (jobs_mod.FINISHED, jobs_mod.FAILED)
                    or job.spec.type not in ("train", "eval")):
                continue         # cancelled / watch: not an archived run
            try:
                if job.status == jobs_mod.FINISHED:
                    self._fold_finished(job)
                self.registry.add_history(history_record(job))
                changed = True
            except (ValueError, OSError):
                pass             # a vanished/bad model file must not crash us
        if changed:
            self.save()

    def _fold_finished(self, job):
        """Register a finished train model + curve, or file an eval score."""
        spec = job.spec
        if spec.type == "train" and spec.save_path:
            self.registry.register(spec.save_path)
            curve = [e["max_length"] for e in job.sessions]
            self.registry.record_curve(
                spec.save_path, viewdata.downsample(curve))
        elif spec.type == "eval" and spec.base_model and job.summary:
            self.registry.record_score(
                spec.base_model,
                spec.eval_profile or _profile_from_spec(spec),
                job.summary)


def _profile_from_spec(spec):
    """The eval-profile key parts implied by an eval job's spec."""
    return {"games": spec.sessions, "seed": spec.seed,
            "board": spec.config["board"]["size"],
            "target": spec.config["goal"]["target_length"]}


def main(argv=None):
    """Run the hub. (``python -m snake_den`` / ``./hub``.)"""
    return App().run()
