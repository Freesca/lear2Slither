"""Immediate-mode widget kit, pixel-art skin. (Phase B1 / Milestone C, H5)

Hand-rolled widgets so the hub keeps its runtime dependency to pygame-ce only.
"Immediate mode": a screen rebuilds its UI every frame by calling widget
functions that both draw and report interaction, reading this frame's input
from a shared :class:`UI` context. Retained state some widgets need (text-field
focus, which dropdown is open) lives in ``UI`` keyed by a caller-supplied id.

Retro skin (Milestone C): the identity comes from **blocky beveled widgets**
(no rounded corners; light/dark 1px bevels) and a retro palette -- not from the
font. Text is the bundled font rendered crisply at full size (no asset file ->
fresh-clone safe, no licensing). (An earlier nearest-neighbour upscale of a
tiny font was too rough to read, so it was dropped -- the blocks carry the
look, the text stays legible.) The skin is a pure render-layer change: every
signature is unchanged, so the screens -- and the jobs -- are untouched.

This module imports pygame, so -- like gui.py -- no test imports it; the
geometry it draws is the pygame-free charts.py, which is tested.
"""
import pygame

from snake_den import charts

# Snake-Byte cabinet palette (DESIGN.md sec. 2). Duplicated here by hand (the
# -42 firewall forbids a shared theme import with slither/gui.py); kept in sync
# against DESIGN.md sec. 2 -- ~14 constants, cheap to mirror. The legacy widget
# names (BG/TEXT/ACCENT/GOOD/BAD) keep their roles, only their values change,
# so every widget signature and call site is untouched (a pure render swap).
BLACK = (0, 0, 0)
CABINET_GOLD = (173, 139, 58)
DITHER_BLUE = (92, 92, 200)
SNAKE_LIME = (140, 165, 45)              # retuned 2026-06-24: less fluorescent
SNAKE_HEAD = (200, 224, 110)
APPLE_GREEN = (80, 175, 55)              # dimmer than the original bright dot
APPLE_RED = (208, 72, 140)
SILVER = (200, 200, 192)
DIM = (110, 110, 120)

BG = BLACK                  # the field: every surface bottoms out on black
PANEL = (16, 16, 26)        # beveled panel / card / field fill
PANEL_HI = (32, 32, 58)     # hover / open / pressed
TEXT = SILVER               # primary text
MUTED = (140, 140, 156)     # secondary text
ACCENT = CABINET_GOLD       # active/selected: tab underline, focus ring, knob
GOOD = APPLE_GREEN          # success status / positive bars (bright, sparse)
BAD = APPLE_RED             # failure status / negative bars
Q_POS = (110, 170, 90)      # calm green: a positive Q-value tint (calm tints)
Q_NEG = (190, 110, 150)     # calm magenta: a negative Q-value tint
LINE = (42, 44, 70)         # hairlines, chart frames, slider track
_BEVEL_HI = (70, 72, 122)   # 1px top/left highlight on raised blocks
_BEVEL_LO = BLACK           # 1px bottom/right shadow

# Window + chrome geometry (the tabbed shell; app.py / screens read these).
# WIDTH/HEIGHT are the *logical* canvas: every widget is placed in these fixed
# coordinates, then app.py scales the whole canvas up to a larger, resizable
# window (INIT_ZOOM is the start magnification). So "bigger" = a bigger zoom,
# not a re-layout -- the pixels just get larger, crisp (nearest-neighbour).
WIDTH, HEIGHT = 960, 640
INIT_ZOOM = 1             # opens at native 1x (960x640); resize snaps integer
TITLE_H = 30                # gold title bar
TAB_H = 32                  # nav tab row beneath it
CONTENT_TOP = TITLE_H + TAB_H + 14      # first y a screen may draw content
FOOTER_H = 96               # persistent job-pool strip at the bottom
CONTENT_BOTTOM = HEIGHT - FOOTER_H
RAIL_W = 8                  # thin dithered cabinet side rail (sec. 4.1/5.3)

# --- 8x8 bitmap font: the Atari 8-bit OS ROM character set ($E000), thinned -
# The font Snake Byte itself used; its ROM strokes are 2px (bold), so they are
# thinned to 1px here to match the original's hairline weight (snake_byte.png).
# Printable ASCII 0x20..0x7E, pre-mapped from Atari internal/screen-code order
# into ASCII, bit-reversed (LSB = leftmost). 8x8 bitmap data is uncopyrightable
# (data, not outlines); via kenjennings/Atari-Font-To-Code. Duplicated verbatim
# in slither/gui.py (the -42 firewall bars a shared import) -- static data.
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

    def __init__(self, scale=2, advance=6):
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


def make_font(size=2, advance=6):
    """A bitmap font at integer ``size`` (pixel scale) and ``advance``.

    ``advance`` is the per-glyph source-cell width: 6 packs dense data tight, 7
    gives chrome (tabs/title/buttons) the machine-output breathing room
    (DESIGN 3/5.2). The board splits 7/6 the same way (make_font/make_qfont).
    """
    return BitmapFont(scale=size, advance=advance)


def render_text(font, text, color):
    """A crisp aliased (hard-pixel) text surface; the font caches by text."""
    return font.render(text, False, color)


class UI:
    """Per-frame input + retained widget state for the immediate-mode kit."""

    def __init__(self, font):
        self.font = font
        self.chrome = BitmapFont(font.scale, 7)   # wider chrome font
        self.mouse = (0, 0)
        self.click = False          # left button released this frame (a click)
        self.mouse_down = False     # left button currently held
        self.scroll = 0             # mouse-wheel delta this frame
        self.keydowns = []          # KEYDOWN events this frame
        self.text = ""              # unicode typed this frame
        self.focus = None           # id of the focused text field
        self.open_dropdown = None   # id of the currently open dropdown
        self.scale = 1.0            # window/logical zoom (app sets each frame)
        self.offset = (0, 0)        # letterbox origin of canvas in the window

    def begin_frame(self, events):
        """Refresh per-frame input from this frame's pygame events.

        The mouse comes back in *window* pixels; map it into the fixed logical
        canvas (undo the app's zoom + letterbox) so every widget's hit-test
        works regardless of window size.
        """
        mx, my = pygame.mouse.get_pos()
        self.mouse = (int((mx - self.offset[0]) / self.scale),
                      int((my - self.offset[1]) / self.scale))
        self.mouse_down = pygame.mouse.get_pressed()[0]
        self.click = False
        self.scroll = 0
        self.keydowns = []
        self.text = ""
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.click = True
            elif event.type == pygame.MOUSEWHEEL:
                self.scroll += event.y
            elif event.type == pygame.KEYDOWN:
                self.keydowns.append(event)
            elif event.type == pygame.TEXTINPUT:
                self.text += event.text


# --- drawing helpers --------------------------------------------------------

def _bevel(surface, rect, raised=True):
    """A 1px light/dark bevel for a chunky pixel-3D edge."""
    light, dark = ((_BEVEL_HI, _BEVEL_LO) if raised
                   else (_BEVEL_LO, _BEVEL_HI))
    x, y, w, h = rect
    pygame.draw.line(surface, light, (x, y), (x + w - 1, y))
    pygame.draw.line(surface, light, (x, y), (x, y + h - 1))
    pygame.draw.line(surface, dark, (x, y + h - 1), (x + w - 1, y + h - 1))
    pygame.draw.line(surface, dark, (x + w - 1, y), (x + w - 1, y + h - 1))


def label(surface, ui, pos, text, color=TEXT, font=None):
    """Draw left-aligned pixel text at ``pos``."""
    surface.blit(render_text(font or ui.font, text, color), pos)


def _centered(surface, font, text, rect, color):
    img = render_text(font, text, color)
    surface.blit(img, img.get_rect(center=rect.center))


def dither(surface, rect, color, cell=2, bg=BLACK):
    """Fill ``rect`` with a 2px checkerboard of ``color`` on ``bg``.

    The Snake-Byte border fill (DESIGN.md sec. 4.2): side panels, the
    current-state band, selection bands. Graphic only -- never under text.
    """
    x0, y0, w, h = pygame.Rect(rect)
    surface.fill(bg, (x0, y0, w, h))
    for yy in range(0, h, cell):
        for xx in range((yy // cell) % 2 * cell, w, cell * 2):
            surface.fill(color, (x0 + xx, y0 + yy, cell, cell))


def title_bar(surface, ui, text):
    """A thin gold marquee rail + the app name in gold on black (DESIGN 5.3).

    Thinned 2026-06-24: a 4px ``CABINET_GOLD`` rail (not a 30px solid block),
    the name below it in gold on black -- an arcade marquee, not window chrome
    -- then the 2px ``DITHER_BLUE`` divider before the tab row.
    """
    surface.fill(CABINET_GOLD, (0, 0, WIDTH, 4))
    label(surface, ui, (16, 9), text.upper(), CABINET_GOLD, font=ui.chrome)
    dither(surface, (0, TITLE_H, WIDTH, 2), DITHER_BLUE)


_SCANLINES = None


def scanline_overlay(surface):
    """Blit a static 1px-every-3px dark CRT overlay (DESIGN.md sec. 7).

    Off by default (a Settings toggle); static, never animated -- the homage,
    not eye strain (PRODUCT.md anti-ref). Cached: built once, blitted cheaply.
    """
    global _SCANLINES
    if _SCANLINES is None:
        _SCANLINES = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for y in range(0, HEIGHT, 3):
            _SCANLINES.fill((0, 0, 0, 64), (0, y, WIDTH, 1))
    surface.blit(_SCANLINES, (0, 0))


_VIGNETTE = None


def vignette(surface, strength=90, n=48):
    """Gentle CRT corner-darkening for the cockpit (DESIGN.md sec. 7).

    The *calm* half of the two-cabinet CRT decision (2026-06-24): the dense
    data screens keep razor-sharp text (no bloom/scanlines) -- but a soft
    static vignette gives the glass its curve. It only shades the outer margins
    (data lives inside MARGIN), so it costs no legibility and needs no toggle.
    Cached: a small radial gradient smooth-scaled to the canvas, blitted once.
    """
    global _VIGNETTE
    if _VIGNETTE is None:
        grad = pygame.Surface((n, n), pygame.SRCALPHA)
        c = (n - 1) / 2
        far = (2 * c * c) ** 0.5
        for y in range(n):
            for x in range(n):
                d = ((x - c) ** 2 + (y - c) ** 2) ** 0.5 / far
                a = min(255, int(strength * d * d))
                grad.set_at((x, y), (0, 0, 0, a))
        _VIGNETTE = pygame.transform.smoothscale(grad, (WIDTH, HEIGHT))
    surface.blit(_VIGNETTE, (0, 0))


def side_rails(surface):
    """Thin dithered-blue cabinet rails down the content-band side gutters.

    The Snake-Byte side panels (DESIGN.md sec. 4.1) at *tool* density: 8px
    dither rails in the existing side margins -- content draws from MARGIN in,
    so they frame the working area without touching a widget. The cabinet read
    the cockpit was missing; graphic only, never under text.
    """
    top = TITLE_H + 2 + TAB_H
    height = CONTENT_BOTTOM - top
    dither(surface, (0, top, RAIL_W, height), DITHER_BLUE)
    dither(surface, (WIDTH - RAIL_W, top, RAIL_W, height), DITHER_BLUE)


def _sign_hue(sign):
    """Tint for a greedy-Q-value sign: green up, magenta down, dim flat."""
    if sign > 0:
        return Q_POS
    if sign < 0:
        return Q_NEG
    return DIM


def fingerprint(surface, rect, cells, *, block=4, ui=None, keys=None):
    """A model's policy fingerprint: one block per visited state, value-tinted.

    ``cells`` is the greedy-Q-value **sign** (``+1/0/-1``) per visited state,
    in the order ``viewdata.policy_summary`` returns -- a compact value map
    (green where the policy expects to do well, magenta where even its best
    move is bad). Blocks tile left-to-right then wrap.

    Inert by default (the list-row identity glyph). When ``ui`` is given it is
    an interactive minimap: the hovered block is boxed (and, with ``keys``, its
    state key shown as a tooltip below), and the clicked block index is
    returned (else ``None``) so the caller can navigate the table beside it.
    """
    rect = pygame.Rect(rect)
    pygame.draw.rect(surface, BG, rect)
    pygame.draw.rect(surface, LINE, rect, 1)
    cols = max(1, (rect.width - 4) // block)
    hovered = None
    for i, sign in enumerate(cells):
        cx = rect.x + 2 + (i % cols) * block
        cy = rect.y + 2 + (i // cols) * block
        if cy + block > rect.bottom - 2:
            break               # ran out of room (coverage exceeds the cell)
        surface.fill(_sign_hue(sign), (cx, cy, block - 1, block - 1))
        if ui is not None and pygame.Rect(
                cx, cy, block, block).collidepoint(ui.mouse):
            hovered = i
    if ui is None or hovered is None:
        return None
    hx = rect.x + 2 + (hovered % cols) * block
    hy = rect.y + 2 + (hovered // cols) * block
    pygame.draw.rect(surface, CABINET_GOLD,
                     (hx - 1, hy - 1, block + 1, block + 1), 1)
    if keys and hovered < len(keys):
        label(surface, ui, (ui.mouse[0] + 10, ui.mouse[1] - 12),
              keys[hovered], CABINET_GOLD)
    return hovered if ui.click else None


# --- controls ---------------------------------------------------------------

def button(surface, ui, rect, text, *, enabled=True):
    """A clickable button; returns True on the frame it is clicked.

    Hover lights it with the blue selection dither -- the same graphic language
    as the selected Q-table row and the active tab; pressed sinks the bevel.
    """
    rect = pygame.Rect(rect)
    hovered = enabled and rect.collidepoint(ui.mouse)
    pressed = hovered and ui.mouse_down
    if hovered:
        dither(surface, rect, DITHER_BLUE)
    else:
        pygame.draw.rect(surface, PANEL, rect)
    _bevel(surface, rect, raised=not pressed)
    _centered(surface, ui.chrome, text, rect, TEXT if enabled else MUTED)
    return bool(hovered and ui.click)


def toggle(surface, ui, rect, value, text=""):
    """A checkbox; returns the (possibly flipped) boolean value."""
    rect = pygame.Rect(rect)
    box = pygame.Rect(rect.x, rect.y, rect.height, rect.height)
    if rect.collidepoint(ui.mouse) and ui.click:
        value = not value
    pygame.draw.rect(surface, PANEL, box)
    _bevel(surface, box, raised=False)
    if value:
        pygame.draw.rect(surface, ACCENT, box.inflate(-8, -8))
    if text:
        label(surface, ui, (box.right + 8, rect.y + 2), text, font=ui.chrome)
    return value


def slider(surface, ui, rect, value, lo, hi):
    """A horizontal slider; returns the (possibly dragged) value."""
    rect = pygame.Rect(rect)
    track = pygame.Rect(rect.x, rect.centery - 2, rect.width, 4)
    pygame.draw.rect(surface, LINE, track)
    if ui.mouse_down and rect.collidepoint(ui.mouse):
        frac = (ui.mouse[0] - rect.x) / max(1, rect.width)
        value = lo + max(0.0, min(1.0, frac)) * (hi - lo)
    frac = 0.0 if hi <= lo else (value - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    knob = pygame.Rect(rect.x + int(frac * rect.width) - 5,
                       rect.centery - 8, 10, 16)
    pygame.draw.rect(surface, ACCENT, knob)
    _bevel(surface, knob)
    return value


def text_field(surface, ui, rect, value, *, field_id, numeric=False,
               disabled=False):
    """A single-line text field; returns its (possibly edited) string.

    Click to focus; typing appends, Backspace deletes, Enter unfocuses. With
    ``numeric`` only digits, ``.`` and ``-`` are accepted. ``disabled`` renders
    it greyed and inert (ignores clicks/typing); the value passes through.
    """
    rect = pygame.Rect(rect)
    if disabled:
        if ui.focus == field_id:
            ui.focus = None
        pygame.draw.rect(surface, BG, rect)
        _bevel(surface, rect, raised=False)
        img = render_text(ui.font, value, MUTED)
        surface.blit(img, (rect.x + 6, rect.centery - img.get_height() // 2))
        return value
    if rect.collidepoint(ui.mouse) and ui.click:
        ui.focus = field_id
    focused = ui.focus == field_id
    if focused:
        typed = ui.text
        if numeric:
            typed = "".join(c for c in typed if c in "0123456789.-")
        value += typed
        for event in ui.keydowns:
            if event.key == pygame.K_BACKSPACE:
                value = value[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                ui.focus = None
    pygame.draw.rect(surface, PANEL, rect)
    _bevel(surface, rect, raised=False)
    if focused:
        pygame.draw.rect(surface, ACCENT, rect, 1)
    img = render_text(ui.font, value + ("|" if focused else ""), TEXT)
    surface.blit(img, (rect.x + 6, rect.centery - img.get_height() // 2))
    return value


def dropdown(surface, ui, rect, options, value, *, field_id):
    """A select box; returns the (possibly newly chosen) option.

    Draw dropdowns *after* other widgets so the open list overlays them. Only
    one dropdown is open at a time (tracked on the UI).
    """
    rect = pygame.Rect(rect)
    is_open = ui.open_dropdown == field_id
    if rect.collidepoint(ui.mouse) and ui.click:
        ui.open_dropdown = None if is_open else field_id
        is_open = not is_open
    pygame.draw.rect(surface, PANEL, rect)
    _bevel(surface, rect, raised=not is_open)
    label(surface, ui, (rect.x + 6, rect.centery - 9), str(value))
    if is_open:
        for i, option in enumerate(options):
            row = pygame.Rect(rect.x, rect.bottom + i * rect.height,
                              rect.width, rect.height)
            hovered = row.collidepoint(ui.mouse)
            pygame.draw.rect(surface, PANEL_HI if hovered else PANEL, row)
            _bevel(surface, row)
            label(surface, ui, (row.x + 6, row.centery - 9), str(option))
            if hovered and ui.click:
                value = option
                ui.open_dropdown = None
    return value


def tabs(surface, ui, rect, labels, active):
    """A row of tabs; returns the (possibly newly selected) active index.

    The active tab wears the blue selection dither + a gold underline -- the
    same graphic language as the selected Q-table row and a hovered button;
    gold stays the active marker (DESIGN 5.2/5.3). Hovering another tab shows a
    faint band so the row reads as selectable, not a static menu line.
    """
    rect = pygame.Rect(rect)
    width = rect.width // max(1, len(labels))
    for i, text in enumerate(labels):
        tab = pygame.Rect(rect.x + i * width, rect.y, width, rect.height)
        hovered = tab.collidepoint(ui.mouse)
        if hovered and ui.click:
            active = i
        if i == active:
            dither(surface, tab, DITHER_BLUE)
            pygame.draw.rect(surface, ACCENT,
                             (tab.x, tab.bottom - 3, tab.width, 3))
        elif hovered:
            dither(surface, tab, LINE)         # faint preview of the band
        _centered(surface, ui.chrome, text, tab,
                  TEXT if i == active else MUTED)
    return active


def panel(surface, rect):
    """A flat panel: black fill + 1px hairline, no bevel (passive region).

    De-framed 2026-06-24 (Snake Byte fidelity): passive backgrounds and chart
    frames read as quiet hairline boxes on black, not raised purple-black menu
    cards. Interactive controls keep their bevels; these do not (DESIGN 5.2).
    """
    rect = pygame.Rect(rect)
    pygame.draw.rect(surface, BG, rect)
    pygame.draw.rect(surface, LINE, rect, 1)
    return rect


# --- charts (geometry from charts.py, drawn here) ---------------------------

def line_chart(surface, rect, values, *, color=SNAKE_LIME, vmin=None,
               vmax=None):
    """Draw a line chart of ``values`` inside ``rect`` (a framed panel)."""
    frame = panel(surface, rect)
    inner = (frame.x + 4, frame.y + 4, frame.width - 8, frame.height - 8)
    pts = [(int(x), int(y))
           for x, y in charts.line_points(values, inner, vmin, vmax)]
    if len(pts) >= 2:
        pygame.draw.lines(surface, color, False, pts, 2)
    elif len(pts) == 1:
        pygame.draw.rect(surface, color, (pts[0][0] - 2, pts[0][1] - 2, 4, 4))


def bar_chart(surface, rect, values, *, vmin=None, vmax=None):
    """Draw a bar chart (e.g. a Q-row) inside ``rect``."""
    frame = panel(surface, rect)
    inner = (frame.x + 4, frame.y + 4, frame.width - 8, frame.height - 8)
    for (x, y, w, h), value in zip(
            charts.bar_rects(values, inner, vmin, vmax), values):
        color = GOOD if value >= 0 else BAD
        pygame.draw.rect(surface, color,
                         pygame.Rect(int(x), int(y), int(w), max(1, int(h))))
