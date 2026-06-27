"""GUI rendering.

The product's only module that imports pygame; ``-visual off`` and the test
suite must never load it. This module is a *pure renderer*: given a board it
draws it. It owns no game logic, no input handling and no stdout -- the
runner and the human harness (``tests/play.py``) drive it.
"""
import pygame

# Snake-Byte cabinet palette, duplicated here by hand: the -42 firewall keeps
# this module from sharing a theme import with the hub, so these constants are
# mirrored rather than imported.
BLACK = (0, 0, 0)
CABINET_GOLD = (173, 139, 58)
DITHER_BLUE = (92, 92, 200)
SNAKE_LIME = (140, 165, 45)              # less fluorescent than the original
SNAKE_HEAD_LIME = (200, 224, 110)
APPLE_GREEN = (80, 175, 55)              # dimmer than the original bright dot
APPLE_RED = (208, 72, 140)
SILVER = (200, 200, 192)
DIM = (110, 110, 120)

BG = BLACK
GRID = (14, 16, 28)                       # barely-there dark blue (debug aid)
GREEN_APPLE = APPLE_GREEN
RED_APPLE = APPLE_RED
SNAKE_BODY = SNAKE_LIME
SNAKE_HEAD = SNAKE_HEAD_LIME            # marked by lightness, still lime
TEXT = SILVER

FRAME_PX = 16                           # side dither-rail width
BAR_PX = 6                              # thin gold top/bottom bar
OVERSCAN = 28                           # black TV overscan around the cabinet
OX = OVERSCAN + FRAME_PX        # field origin x (inside left rail)
OY = OVERSCAN + BAR_PX          # field origin y (below top bar)
BAR_H = 96                      # status area: a 2-column readout, <=4 rows

# Q-table view (second tab): current-state band, value tints, greedy mark.
# Q-value tints -- calm/dimmed, NOT the bright apple colors: a Q's sign is
# data, so it reads as a quiet tint beside the (primary) signed number, not a
# signal competing for attention.
POS = (110, 170, 90)
NEG = (190, 110, 150)
ZERO = DIM
MARK = CABINET_GOLD
BAND = DITHER_BLUE
QPANEL_W = 540                            # panel width: state + 4 Q + visits
QCOLS_X = (8, 96, 186, 276, 366, 456)     # x of: state, F, L, R, B, n(visits)
QBOX_W = 82                               # greedy-action box (fits "-100.0")
QHEAD_Y = 6
QTOP = 30
QLINE_H = 24
_STATUS_MIN_W = 480                       # board-only floor: fits the readout


def create_window(size, cell_px):
    pygame.init()
    pygame.display.set_caption("Learn2Slither")
    board_px = size * cell_px
    # Board-only by default; pressing `Q` grows the window for the Q-table
    # panel. The floor keeps the status readout legible on tiny boards.
    width = max(board_px + 2 * OX, _STATUS_MIN_W)
    height = OY + board_px + BAR_PX + BAR_H + OVERSCAN
    return pygame.display.set_mode((width, height))


def _dither(screen, rect, color, cell=2, bg=BLACK):
    """Fill ``rect`` with a 2px checkerboard of ``color``."""
    x0, y0, w, h = rect
    screen.fill(bg, (x0, y0, w, h))
    for yy in range(0, h, cell):
        for xx in range((yy // cell) % 2 * cell, w, cell * 2):
            screen.fill(color, (x0 + xx, y0 + yy, cell, cell))


# --- 8x8 bitmap font: the Atari 8-bit OS ROM character set ($E000), thinned -
# The font Snake Byte itself used; its ROM strokes are 2px (bold), thinned to
# 1px here to match the original's hairline weight. Printable ASCII 0x20..0x7E,
# pre-mapped from Atari internal/screen-code order into ASCII, bit-reversed
# (LSB = leftmost). 8x8 bitmap data is uncopyrightable (data, not outlines);
# via kenjennings/Atari-Font-To-Code. Duplicated verbatim in
# snake_den/widgets.py (the -42 firewall bars a shared import).
_FONT_ATARI = (
    "00000000000000000008080808000800002222220000000000227f22227f2200083c021c"
    "201e080000221208042222001824180c72224c0000080808000000000030180808183000"
    "00060c08080c060000221c7f1c2200000008083e0808000000000000000808040000003e"
    "0000000000000000000808000020100804020200001c223226221c0000080c0808083e00"
    "001c221008043e00003e100810221c000010181c123e1000003e021e20221c00001c021e"
    "22221c00003e201008040400001c221c22221c00001c223c20100c000000080800080800"
    "0000080800080804201008040810200000003e00003e00000204081008040200001c2210"
    "08000800001c223232023c0000081c22223e2200001e221e22221e00001c220202221c00"
    "000e122222120e00003e021e02023e00003e021e02020200003c020232223c000022223e"
    "22222200003e080808083e000020202020221c000022120e0e1222000002020202023e00"
    "0042667e524242000022263e3e322200001c222222221c00001e22221e020200001c2222"
    "22122400001e22221e122200001c021c20201c00003e0808080808000022222222223e00"
    "00222222221c0800004242527e6642000022221c1c2222000022221c08080800003e1008"
    "04023e0000380808080838000002020408102000000e080808080e000010182442000000"
    "0000000000007f0000081c3e3e1c080000001c203c223c000002021e22221e0000001c02"
    "02021c000020203c22223c0000001c223e021c000030083c0808080000003c22223c201e"
    "0002021e222222000008000c08081c00002000202020201c000202120e122200000c0808"
    "08081c000000227e7e52420000001e222222220000001c2222221c0000001e22221e0202"
    "00003c22223c202000001e220202020000003c021c201e0000083e080808300000002222"
    "22223c0000002222221c0800000042527e3c24000000221c081c220000002222223c100e"
    "00003e1008043e0000081c3e3e081c000808080808080808003e0e1e3222200010080c0e"
    "0c081000"
)


class BitmapFont:
    """A chunky 8x8 pixel font blitted as scaled hard pixels (no AA).

    Mimics ``pygame.font.Font`` (``render``/``size``/``get_height``) so every
    call site is unchanged. ``advance`` is the per-glyph cell width in source
    pixels (< 8 overlaps the 1-2 blank right columns of wide glyphs, keeping
    text tight); ``scale`` is the integer pixel size. Renders are cached.
    """

    def __init__(self, scale=2, advance=7):
        self.scale = scale
        self.advance = advance
        self.cw = advance * scale
        self.ch = 8 * scale
        self._cache = {}

    def get_height(self):
        return self.ch

    def size(self, text):
        return (len(text) * self.cw, self.ch)

    def render(self, text, antialias=False, color=(255, 255, 255)):
        cached = self._cache.get((text, color))
        if cached is not None:
            return cached
        scale, cw = self.scale, self.cw
        surf = pygame.Surface((max(1, len(text) * cw), self.ch),
                              pygame.SRCALPHA)
        for n, char in enumerate(text):
            index = ord(char) - 32
            if not 0 <= index < 95:
                index = ord("?") - 32
            base, ox = index * 16, n * cw
            for row in range(8):
                byte = int(_FONT_ATARI[base + row * 2:base + row * 2 + 2], 16)
                top = row * scale
                for x in range(8):
                    if byte & (1 << x):
                        surf.fill(color, (ox + x * scale, top, scale, scale))
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[(text, color)] = surf
        return surf


def make_font():
    return BitmapFont(scale=2, advance=7)


def make_qfont():
    return BitmapFont(scale=2, advance=6)


def _cross(screen, color, r, c, cell_px):
    """A small +-shaped apple mark (Snake Byte), centered on the cell.

    The original's apples are tiny sparks, not filled tiles -- the long lime
    circuit, the tiny colored target and the large black field are the object
    hierarchy. The cell is a positioning lattice, not the sprite size.
    """
    cx = OX + c * cell_px + cell_px // 2
    cy = OY + r * cell_px + cell_px // 2
    arm = max(3, cell_px // 9)               # half-length of each arm
    th = max(1, cell_px // 18)               # half-thickness of each bar
    screen.fill(color, (cx - arm, cy - th, 2 * arm, 2 * th))
    screen.fill(color, (cx - th, cy - arm, 2 * th, 2 * arm))


def _cell_center(r, c, cell_px):
    return (OX + c * cell_px + cell_px // 2,
            OY + r * cell_px + cell_px // 2)


def _draw_snake(screen, body, cell_px):
    """Draw the snake as a thin hollow tube, Snake-Byte style.

    A lime polyline through the segment centres, overdrawn with a black
    polyline, leaves two lime walls -- a continuous hollow pipe that turns
    corners like the original. The head is marked with a small light-lime cap.
    """
    if not body:
        return
    pts = [_cell_center(r, c, cell_px) for (r, c) in body]
    outer = max(4, cell_px // 5)             # thin trace, not a fat tile
    inner = max(1, outer - 4)                # leaves ~2px lime rails
    if len(pts) >= 2:
        pygame.draw.lines(screen, SNAKE_BODY, False, pts, outer)
        pygame.draw.lines(screen, BG, False, pts, inner)
    else:                                    # single-cell snake: hollow square
        x, y = pts[0]
        pygame.draw.rect(screen, SNAKE_BODY,
                         (x - outer // 2, y - outer // 2, outer, outer))
        pygame.draw.rect(screen, BG,
                         (x - inner // 2, y - inner // 2, inner, inner))
    hx, hy = pts[0]
    mark = max(4, outer - 2)
    pygame.draw.rect(screen, SNAKE_HEAD,
                     (hx - mark // 2, hy - mark // 2, mark, mark))


def _draw_status(screen, font, fields, x0, y0):
    """Lay out (label, value) pairs as a 2-column readout.

    A quiet aligned readout, not one bold HUD line: two balanced columns
    filled top-to-bottom, aligned by the monospace font's advance
    (``font.cw``); all SILVER. The row count grows with the field count --
    watch adds STATE + ACTION -- so the readout stays two columns.
    """
    cw = font.cw
    row_h = font.get_height() + 6
    col_w, val_x = 14 * cw, 9 * cw
    rows = max(1, -(-len(fields) // 2))            # ceil: two balanced columns
    for i, (label, value) in enumerate(fields):
        x = x0 + (i // rows) * col_w
        y = y0 + (i % rows) * row_h
        screen.blit(font.render(f"{label}:", False, TEXT), (x, y))
        screen.blit(font.render(value, False, TEXT), (x + val_x, y))


def render(screen, env, font, status, cell_px):
    size = env.size
    board_px = size * cell_px
    screen.fill(BG)                          # black; overscan is just black

    # Cabinet inside the black TV overscan: thin gold top/bottom bars; side
    # rails dithered-blue on top, solid gold below. The field never touches
    # the window edge.
    bottom = OY + board_px
    rx = OX + board_px
    cab_w = 2 * FRAME_PX + board_px
    split = OY + 2 * board_px // 3
    screen.fill(CABINET_GOLD, (OVERSCAN, OVERSCAN, cab_w, BAR_PX))
    screen.fill(CABINET_GOLD, (OVERSCAN, bottom, cab_w, BAR_PX))
    for rail_x in (OVERSCAN, rx):
        _dither(screen, (rail_x, OY, FRAME_PX, split - OY), DITHER_BLUE)
        screen.fill(CABINET_GOLD, (rail_x, split, FRAME_PX, bottom - split))

    for i in range(size + 1):
        gx, gy = OX + i * cell_px, OY + i * cell_px
        pygame.draw.line(screen, GRID, (gx, OY), (gx, bottom))
        pygame.draw.line(screen, GRID, (OX, gy), (rx, gy))

    for (r, c) in env.green_apples:
        _cross(screen, GREEN_APPLE, r, c, cell_px)
    for (r, c) in env.red_apples:
        _cross(screen, RED_APPLE, r, c, cell_px)

    _draw_snake(screen, list(env.snake), cell_px)

    # Status below the bottom bar: a 2-column readout when given (label,
    # value) fields, or a single SILVER line for a plain string.
    sy = bottom + BAR_PX + 12
    if isinstance(status, str):
        screen.blit(font.render(status.upper(), False, TEXT), (OX, sy))
    else:
        _draw_status(screen, font, status, OX, sy)


def draw_action(screen, env, delta, cell_px):
    """Overlay an arrow on the snake's head pointing in the chosen move.

    ``delta`` is an absolute ``Direction.value`` (dr, dc), or ``None`` to draw
    nothing. A debug overlay only -- the caller decides when to call it.
    """
    if delta is None or not env.snake:
        return
    hr, hc = env.snake[0]
    cx = OX + hc * cell_px + cell_px // 2
    cy = OY + hr * cell_px + cell_px // 2
    dr, dc = delta
    reach = cell_px // 2 - 4
    tip = (cx + dc * reach, cy + dr * reach)
    pygame.draw.line(screen, MARK, (cx, cy), tip, 3)
    pygame.draw.circle(screen, MARK, tip, 4)


def render_qtable(screen, font, rows, current_state):
    """Draw the sparse Q-table, auto-following the current state.

    ``rows`` is a caller-prepared list of ``(state, qvals, nvals)`` already in
    display order. The viewport keeps ``current_state`` centred; that row is
    highlighted, and every row boxes its greedy (argmax) action so the policy
    is readable at a glance. Pure rendering -- the caller owns the data order.
    """
    width, height = screen.get_size()
    screen.fill(BG)

    for x, label in zip(QCOLS_X, ("state", "F", "L", "R", "B", "n")):
        screen.blit(font.render(label, False, TEXT), (x, QHEAD_Y))
    pygame.draw.line(screen, GRID, (0, QTOP - 4), (width, QTOP - 4))

    if not rows:
        screen.blit(font.render("(no states visited yet)", False, ZERO),
                    (QCOLS_X[0], QTOP + 4))
        return

    visible = max(1, (height - QTOP) // QLINE_H)
    index = next((i for i, row in enumerate(rows)
                  if row[0] == current_state), 0)
    scroll = min(max(0, index - visible // 2), max(0, len(rows) - visible))

    for i in range(scroll, min(len(rows), scroll + visible)):
        state, qvals, nvals = rows[i]
        y = QTOP + (i - scroll) * QLINE_H
        if state == current_state:
            _dither(screen, (0, y, width, QLINE_H), BAND)
        screen.blit(font.render(state, False, TEXT), (QCOLS_X[0], y))
        best_col = qvals.index(max(qvals))
        for col, value in enumerate(qvals):
            color = POS if value > 0 else NEG if value < 0 else ZERO
            x = QCOLS_X[col + 1]
            screen.blit(font.render(f"{value:+.1f}", False, color), (x, y))
            if col == best_col:
                pygame.draw.rect(screen, MARK,
                                 pygame.Rect(x - 3, y - 1, QBOX_W, QLINE_H), 1)
        screen.blit(font.render(str(sum(nvals)), False, ZERO), (QCOLS_X[5], y))


# --- Analog-CRT post-process ------------------------------------------------
# The game board can carry a full Atari-on-a-CRT signal: it is sparse (snake +
# apples + one status line), so the glow costs no data legibility. Pipeline,
# in render order: horizontal smear (composite bandwidth) -> chroma fringe
# (channel offset) -> bloom (phosphor glow, additive) -> scanlines -> vignette.
# Pure pygame-ce (smoothscale + BLEND_RGB_*), no numpy. Overlays are cached;
# `C` toggles the pass off.
_SCANLINES_CRT = None
_VIGNETTE_CRT = None


def _smear(surf, factor=2, alpha=70):
    """Soften vertical edges: a horizontal-only blur laid back over it."""
    w, h = surf.get_size()
    small = pygame.transform.smoothscale(surf, (max(1, w // factor), h))
    blur = pygame.transform.smoothscale(small, (w, h))
    blur.set_alpha(alpha)
    surf.blit(blur, (0, 0))


def _chroma(surf, shift=1):
    """Composite chroma fringe: green stays put, red/blue shift +/-`shift`px.

    Each channel is re-laid (not added over the full image), so luminance is
    preserved -- edges get the red/blue fringe without washing the frame out.
    """
    base = surf.copy()
    red = base.copy()
    red.fill((255, 0, 0), special_flags=pygame.BLEND_RGB_MULT)
    green = base.copy()
    green.fill((0, 255, 0), special_flags=pygame.BLEND_RGB_MULT)
    blue = base.copy()
    blue.fill((0, 0, 255), special_flags=pygame.BLEND_RGB_MULT)
    surf.fill(BLACK)
    surf.blit(green, (0, 0))
    surf.blit(red, (shift, 0), special_flags=pygame.BLEND_RGB_ADD)
    surf.blit(blue, (-shift, 0), special_flags=pygame.BLEND_RGB_ADD)


def _bloom(surf, downscale=4, strength=75):
    """Phosphor glow: a dimmed blur added back, so bright-on-black haloes."""
    w, h = surf.get_size()
    lo = (max(1, w // downscale), max(1, h // downscale))
    glow = pygame.transform.smoothscale(
        pygame.transform.smoothscale(surf, lo), (w, h))
    glow.fill((strength, strength, strength),
              special_flags=pygame.BLEND_RGB_MULT)
    surf.blit(glow, (0, 0), special_flags=pygame.BLEND_RGB_ADD)


def _scanlines(surf, gap=3, dark=48):
    """Stripe every `gap`px with a dark line (cached at the output size)."""
    global _SCANLINES_CRT
    size = surf.get_size()
    if _SCANLINES_CRT is None or _SCANLINES_CRT.get_size() != size:
        w, h = size
        _SCANLINES_CRT = pygame.Surface(size, pygame.SRCALPHA)
        for y in range(0, h, gap):
            _SCANLINES_CRT.fill((0, 0, 0, dark), (0, y, w, 1))
    surf.blit(_SCANLINES_CRT, (0, 0))


def _vignette(surf, strength=110, n=48):
    """Darken the corners (cached): a small radial gradient scaled up."""
    global _VIGNETTE_CRT
    size = surf.get_size()
    if _VIGNETTE_CRT is None or _VIGNETTE_CRT.get_size() != size:
        grad = pygame.Surface((n, n), pygame.SRCALPHA)
        c = (n - 1) / 2
        far = (2 * c * c) ** 0.5
        for y in range(n):
            for x in range(n):
                d = ((x - c) ** 2 + (y - c) ** 2) ** 0.5 / far
                a = min(255, int(strength * d * d))
                grad.set_at((x, y), (0, 0, 0, a))
        _VIGNETTE_CRT = pygame.transform.smoothscale(grad, size)
    surf.blit(_VIGNETTE_CRT, (0, 0))


def crt_process(surf):
    """Run the full CRT pass in place."""
    _smear(surf)
    _chroma(surf)
    _bloom(surf)
    _scanlines(surf)
    _vignette(surf)


# --- Boot marquee + phosphor persistence ------------------------------------
# The -visual window opens like an arcade cabinet powering on: a CRT warm-up
# that snaps a SNAKE BYTE marquee open, then the live board carries
# long-persistence phosphor -- the snake leaves a decaying comet-glow trail.
# Both are defeatable (any key skips the boot; `T` toggles the trail), the
# analog of prefers-reduced-motion, with the static marquee as the fallback.
PHOSPHOR_DECAY = 0.80          # per-frame trail decay


def _ease_out(t):
    """Cubic ease-out clamped to [0, 1] (no bounce) for the warm-up reveal."""
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def phosphor_step(phosphor, screen, field_rect, decay=PHOSPHOR_DECAY):
    """Accumulate the play-field into a decaying phosphor buffer.

    ``phosphor = max(phosphor * decay, current_field)`` -- BLEND_RGB_MAX so a
    *static* pixel (grid, a still snake, an apple) holds its own brightness and
    never trails, while a pixel the bright snake just *left* fades over a few
    frames: a comet tail behind motion, nothing behind stillness. The result is
    painted back over the field only, so the gold frame and the status numbers
    (drawn outside ``field_rect``) stay crisp. Cheap: one mul + two blits.
    """
    d = int(decay * 255)
    phosphor.fill((d, d, d), special_flags=pygame.BLEND_RGB_MULT)
    phosphor.blit(screen, (0, 0), pygame.Rect(field_rect),
                  special_flags=pygame.BLEND_RGB_MAX)
    screen.blit(phosphor, (field_rect[0], field_rect[1]))


def draw_marquee(screen, title_font, font, *, open_frac=1.0, blink_on=True,
                 flash=0.0, info=""):
    """Draw the power-on title marquee (Snake Byte homage).

    A gold ``SLITHER`` wordmark over a silver ``AFTER SNAKE BYTE 1982`` line, a
    blinking gold prompt and a dim status line. ``open_frac`` < 1 clips the
    image to a centred horizontal band with bright beam edges -- the classic
    "the tube is warming up" snap; ``flash`` overlays a fading white bloom at
    the instant it snaps fully open. Pure render (screenshot-testable).
    """
    w, h = screen.get_size()
    cy = h // 2
    screen.fill(BLACK)
    title = title_font.render("SLITHER", False, CABINET_GOLD)
    screen.blit(title, title.get_rect(center=(w // 2, cy - title_font.ch)))
    sub = font.render("AFTER  SNAKE  BYTE  1982", False, SILVER)
    screen.blit(sub, sub.get_rect(center=(w // 2, cy + 6)))
    if blink_on:
        prompt = font.render("> PRESS  SPACE", False, CABINET_GOLD)
        screen.blit(prompt, prompt.get_rect(center=(w // 2, cy + 44)))
    if info:
        line = font.render(info, False, DIM)
        screen.blit(line, line.get_rect(center=(w // 2, cy + 70)))
    if open_frac < 1.0:
        band = int(open_frac * h)
        top = (h - band) // 2
        screen.fill(BLACK, (0, 0, w, top))
        screen.fill(BLACK, (0, top + band, w, h - top - band))
        edge = (235, 235, 228)
        if band >= 2:
            screen.fill(edge, (0, top, w, 1))
            screen.fill(edge, (0, top + band - 1, w, 1))
        else:
            screen.fill(edge, (0, cy, w, 1))
    if flash > 0:
        glow = pygame.Surface((w, h))
        glow.fill((255, 255, 255))
        glow.set_alpha(int(max(0.0, min(1.0, flash)) * 255))
        screen.blit(glow, (0, 0))


class Presenter:
    """The interactive ``-visual on`` window.

    The presenter is the *whole pygame boundary* for the product path: it owns
    the window, the event pump and the speed/pause/step/overlay state. The
    runner drives it by calling :meth:`present` once per move and only ever
    learns an *intent* back (``"advance"`` / ``"quit"``) -- it never sees a
    keycode, so ``runner.py`` stays pygame-free and ``gui`` remains the sole
    importer of pygame on the product path.

    Rendering at ~60 FPS is decoupled from the move cadence: each frame draws
    and pumps events, and a move is released only when the speed-ladder
    interval elapses (continuous) or SPACE is pressed (paused / step-by-step).
    So the window stays responsive and the close button is instant at any
    speed -- it never hangs or crashes.

    On construction it plays a one-time arcade power-on marquee (``_boot``);
    any key or ~2.4s drops into the game.

    Keys: ``-``/``+`` and ``1``-``5`` set speed; ``P`` pauses; ``SPACE`` steps
    one move when paused or step-by-step; ``TAB`` toggles the debug overlay
    (the chosen-move arrow); ``C`` toggles the CRT pass; ``T`` toggles the
    phosphor trail (on by default); ``Q`` toggles the live Q-table panel (the
    window grows to fit it); ``Esc`` or the window close button ends the run
    cleanly.
    """

    SPEEDS = (2, 4, 8, 15, 30)        # moves per second; a human-readable 2/s

    def __init__(self, size, cell_px, speed_ms, step_by_step):
        self.screen = create_window(size, cell_px)
        self.font = make_font()
        self.clock = pygame.time.Clock()
        self.size = size
        self.cell_px = cell_px
        self.step_by_step = step_by_step
        self.paused = False
        self.overlay = False
        self.crt = False      # geometry carries the look; `C` toggles CRT on
        self.phosphor = True              # long-persistence trail; `T` toggles
        self._phosphor = None             # lazily-sized field accumulation buf
        self.quit = False
        self.show_qtable = False          # `Q` toggles the panel + a resize
        # The board-only size: `Q` grows the window by QPANEL_W and the panel
        # is drawn from x = base_w (the board sits to its left).
        self.base_w, self.win_h = self.screen.get_size()
        self.current_state = None         # state to highlight in the panel
        self.qrows = []                   # panel rows, supplied by the runner
        target = 1000 / max(1, speed_ms)
        self.speed_index = min(
            range(len(self.SPEEDS)),
            key=lambda i: abs(self.SPEEDS[i] - target))
        self._boot()

    def _boot(self):
        """Play the arcade power-on marquee once, before the first session.

        Self-contained: the Presenter owns the window + pump, so this needs no
        runner hook. Any key (or ~2.4s) drops into the game; closing the window
        sets ``quit`` so the run ends cleanly on the first ``present``.
        Skippable + a static fallback = the prefers-reduced-motion analog here.
        """
        width = self.screen.get_size()[0]
        tscale = max(3, min(8, (width - 64) // (7 * 8)))
        title_font = BitmapFont(scale=tscale, advance=8)
        info = f"{self.size} X {self.size}   AGENT READY"
        warm, hold = 450, 2400
        start = pygame.time.get_ticks()
        skipped = False
        while not self.quit and not skipped:
            now = pygame.time.get_ticks() - start
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit = True
                elif event.type == pygame.KEYDOWN:
                    skipped = True
            if now >= hold:
                break
            opened = now >= warm
            frac = 1.0 if opened else _ease_out(now / warm)
            flash = 0.5 * (1 - (now - warm) / 120) if opened and \
                now < warm + 120 else 0.0
            blink = opened and (now // 380) % 2 == 0
            draw_marquee(self.screen, title_font, self.font,
                         open_frac=frac, blink_on=blink, flash=flash,
                         info=info)
            _bloom(self.screen)
            _scanlines(self.screen)
            _vignette(self.screen)
            pygame.display.flip()
            self.clock.tick(60)

    def present(self, env, status, action_delta, current_state=None,
                qrows=None):
        """Show the board and block (rendering) until the next move or quit.

        Returns ``"advance"`` when it is time to apply the pending move, or
        ``"quit"`` if the user asked to stop. Once quit, it returns ``"quit"``
        immediately on every later call. ``current_state`` + ``qrows`` feed the
        ``Q``-revealed live Q-table panel (data only -- the runner builds the
        rows; the firewall is untouched, this merely *displays* them).
        """
        self.current_state = current_state
        self.qrows = qrows or []
        if self.quit:
            return "quit"
        start = pygame.time.get_ticks()
        step_requested = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit = True
                elif event.type == pygame.KEYDOWN:
                    step_requested = self._on_key(event.key) or step_requested
            if self.quit:
                return "quit"
            self._draw(env, status, action_delta)
            self.clock.tick(60)
            if self.step_by_step or self.paused:
                if step_requested:
                    return "advance"
            elif pygame.time.get_ticks() - start >= 1000 // self._speed():
                return "advance"

    def _on_key(self, key):
        """Apply a keypress; return True if it requests a single step."""
        if key == pygame.K_ESCAPE:
            self.quit = True
        elif key == pygame.K_SPACE:
            return True
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_TAB:
            self.overlay = not self.overlay
        elif key == pygame.K_c:
            self.crt = not self.crt
        elif key == pygame.K_t:
            self.phosphor = not self.phosphor
            self._phosphor = None         # drop the buffer; restart clean
        elif key == pygame.K_q:
            self.show_qtable = not self.show_qtable
            self._resize()
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_index = max(0, self.speed_index - 1)
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed_index = min(len(self.SPEEDS) - 1, self.speed_index + 1)
        elif pygame.K_1 <= key <= pygame.K_5:
            self.speed_index = min(key - pygame.K_1, len(self.SPEEDS) - 1)
        return False

    def _speed(self):
        return self.SPEEDS[self.speed_index]

    def _resize(self):
        """Grow/shrink the window for the Q-table panel.

        The board-only window grows by ``QPANEL_W`` when the table is shown and
        shrinks back when hidden. ``set_mode`` returns the new surface; the CRT
        caches are size-keyed, so they rebuild themselves on the next pass.
        """
        width = self.base_w + (QPANEL_W if self.show_qtable else 0)
        self.screen = pygame.display.set_mode((width, self.win_h))

    def _draw(self, env, status, action_delta):
        if self.step_by_step:
            speed = "STEP"
        elif self.paused:
            speed = "PAUSED"
        else:
            speed = f"{self._speed()}/S"
        fields = list(status) + [("SPEED", speed)]
        render(self.screen, env, self.font, fields, self.cell_px)
        if self.phosphor:                   # comet-trail the field, not chrome
            self._apply_phosphor(env)
        if self.overlay and action_delta is not None:
            draw_action(self.screen, env, action_delta, self.cell_px)
        if self.crt:
            crt_process(self.screen)        # the board is the CRT surface
        if self.show_qtable:                # the panel stays crisp (post-CRT)
            self._draw_qpanel()
        pygame.display.flip()

    def _apply_phosphor(self, env):
        """Update + composite the field's long-persistence phosphor buffer.

        The buffer is sized to the play-field and rebuilt if the board size
        changes; ``phosphor_step`` decays it and paints the comet trail back
        over the field only (the chrome + status readout stay crisp).
        """
        board_px = env.size * self.cell_px
        if (self._phosphor is None
                or self._phosphor.get_size() != (board_px, board_px)):
            self._phosphor = pygame.Surface((board_px, board_px))
            self._phosphor.fill(BLACK)
        phosphor_step(self._phosphor, self.screen,
                      (OX, OY, board_px, board_px))

    def _draw_qpanel(self):
        """Draw the live Q-table in the panel to the right of the board.

        Auto-follows the current state (``render_qtable`` owns that). Crisp by
        design -- drawn after the CRT pass so the data stays legible.
        """
        panel = self.screen.subsurface((self.base_w, 0, QPANEL_W, self.win_h))
        render_qtable(panel, self.font, self.qrows, self.current_state)

    def close(self):
        pygame.quit()
