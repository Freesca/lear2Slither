"""Tests for chart geometry (snake_den/charts.py).

Pure data->pixel math, so it is tested without a display (no pygame import --
keeps slither's headless invariant safe). Covers the data->screen-coord
mapping: endpoints, the inverted y axis, single/empty series, flat ranges, the
zero baseline for bars, and the matrix grid.
"""
import subprocess
import sys

from snake_den import charts


def test_importing_charts_never_loads_pygame():
    # Order-independent guard (a fresh process): charts stays pygame-free so
    # the whole hub suite is display-free and slither's invariant holds.
    code = ("import importlib, sys; "
            "importlib.import_module('snake_den.charts'); "
            "sys.exit(1 if 'pygame' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# --- value_range ------------------------------------------------------------

def test_value_range_defaults_to_min_max():
    assert charts.value_range([2, 5, 3]) == (2.0, 5.0)


def test_value_range_is_never_zero_width():
    lo, hi = charts.value_range([4, 4, 4])
    assert lo == 4.0 and hi == 5.0


def test_value_range_overrides():
    assert charts.value_range([2, 5], vmin=0, vmax=10) == (0.0, 10.0)


# --- line_points ------------------------------------------------------------

def test_line_points_spans_rect_width():
    pts = charts.line_points([0, 10], (0, 0, 100, 50))
    assert pts[0][0] == 0 and pts[-1][0] == 100


def test_line_points_inverts_y():
    # max value -> top (y=0), min value -> bottom (y=h)
    pts = charts.line_points([0, 10], (0, 0, 100, 50))
    assert pts[0][1] == 50.0        # value 0 at the bottom
    assert pts[1][1] == 0.0         # value 10 at the top


def test_line_points_single_sample_centers():
    pts = charts.line_points([7], (10, 0, 100, 50))
    assert len(pts) == 1
    assert pts[0][0] == 60.0        # 10 + 100/2


def test_line_points_empty():
    assert charts.line_points([], (0, 0, 100, 50)) == []


def test_line_points_flat_series_does_not_divide_by_zero():
    pts = charts.line_points([3, 3, 3], (0, 0, 100, 60))
    assert all(0.0 <= y <= 60.0 for _, y in pts)


# --- bar_rects --------------------------------------------------------------

def test_bar_rects_count_and_within_rect():
    rects = charts.bar_rects([1, 2, 3, 4], (0, 0, 80, 40))
    assert len(rects) == 4
    for x, y, w, h in rects:
        assert 0 <= x and x + w <= 80 + 1e-9
        assert 0 <= y and y + h <= 40 + 1e-9


def test_bar_rects_zero_baseline_splits_pos_neg():
    # values spanning zero: a positive bar sits above a negative one
    rects = charts.bar_rects([1.0, -1.0], (0, 0, 40, 100), vmin=-1, vmax=1)
    pos_top = rects[0][1]
    neg_top = rects[1][1]
    assert pos_top < neg_top        # positive bar starts higher (smaller y)


def test_bar_rects_empty():
    assert charts.bar_rects([], (0, 0, 40, 40)) == []


# --- matrix_layout ----------------------------------------------------------

def test_matrix_layout_grid_shape():
    cells = charts.matrix_layout((0, 0, 100, 100), 2, 3)
    assert len(cells) == 6
    assert {(r, c) for r, c, _ in cells} == {
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}


def test_matrix_layout_cells_fit_width():
    cells = charts.matrix_layout((0, 0, 100, 100), 1, 4, gap=2)
    last = cells[-1][2]
    assert last[0] + last[2] <= 100 + 1e-9


def test_matrix_layout_empty():
    assert charts.matrix_layout((0, 0, 100, 100), 0, 3) == []
