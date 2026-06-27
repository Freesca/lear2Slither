"""Chart geometry: pure data -> coordinate math (no pygame).

The hub draws live training curves, Q-row bars and a results matrix. The
*geometry* -- turning data values into pixel coordinates inside a plot rect --
is pure and lives here so it is unit-testable without a display; the actual
``pygame.draw`` calls live in widgets.py, which consumes these points/rects.
Keeping this module pygame-free is also what lets the whole hub test-suite stay
display-free, so slither's "headless never imports pygame" invariant
(test_cli.py) is never threatened by a hub test.

A rect is a plain ``(x, y, w, h)`` tuple of numbers; every function returns
plain tuples/lists, so callers convert to ``pygame.Rect`` / int points at the
draw site. The y axis is screen-style (0 at the top), so larger values plot
*higher* (smaller y).
"""


def value_range(values, vmin=None, vmax=None):
    """A non-empty ``(lo, hi)`` covering ``values`` (defaults to its min/max).

    ``vmin``/``vmax`` override either bound; the range is forced non-zero-width
    so a flat series still plots on a sensible mid-line instead of dividing by
    zero.
    """
    lo = min(values) if vmin is None and values else (vmin or 0.0)
    hi = max(values) if vmax is None and values else (vmax or 1.0)
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def line_points(values, rect, vmin=None, vmax=None):
    """Pixel points for a line chart of ``values`` inside ``rect``.

    ``x`` spans the rect left-to-right across the samples; ``y`` maps the value
    range to the rect height (top = max). One sample plots at the horizontal
    centre; no samples -> no points.
    """
    x0, y0, w, h = rect
    n = len(values)
    if n == 0:
        return []
    lo, hi = value_range(values, vmin, vmax)
    span = hi - lo
    if n == 1:
        xs = [x0 + w / 2]
    else:
        xs = [x0 + w * i / (n - 1) for i in range(n)]
    return [(x, y0 + h * (1 - (v - lo) / span)) for x, v in zip(xs, values)]


def bar_rects(values, rect, vmin=None, vmax=None):
    """Bar ``(x, y, w, h)`` rects for ``values`` inside ``rect``.

    Bars grow from a zero baseline when the range spans zero (so negative
    Q-values hang below it), else from the bottom of the rect. Bars are evenly
    spaced across the width with a small gap.
    """
    x0, y0, w, h = rect
    n = len(values)
    if n == 0:
        return []
    lo, hi = value_range(values, vmin, vmax)
    span = hi - lo
    baseline = min(max(0.0, lo), hi)            # zero if in range, else near
    base_y = y0 + h * (1 - (baseline - lo) / span)
    gap = w / n * 0.2
    bw = w / n - gap
    rects = []
    for i, v in enumerate(values):
        x = x0 + i * (w / n) + gap / 2
        vy = y0 + h * (1 - (v - lo) / span)
        top = min(vy, base_y)
        rects.append((x, top, bw, abs(base_y - vy)))
    return rects


def matrix_layout(rect, n_rows, n_cols, gap=2):
    """A grid of ``(row, col, (x, y, w, h))`` cells filling ``rect``.

    Used for the compare results matrix (models x metrics). Cells share the gap
    evenly; an empty grid yields no cells.
    """
    x0, y0, w, h = rect
    if n_rows <= 0 or n_cols <= 0:
        return []
    cw = (w - gap * (n_cols - 1)) / n_cols
    ch = (h - gap * (n_rows - 1)) / n_rows
    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            x = x0 + c * (cw + gap)
            y = y0 + r * (ch + gap)
            cells.append((r, c, (x, y, cw, ch)))
    return cells
