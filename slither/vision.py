"""Vision: the snake's 4-direction perception and its cross print.

This is the sole board-reading module on the product path. Every signal the
agent ever receives is derived from the four ray strings produced here, so
the -42 rule is enforced structurally: the state and reward code in
interpreter.py consumes only this output and never touches the board.
"""
from slither.environment import Direction

WALL = "W"
HEAD = "H"
BODY = "S"
GREEN = "G"
RED = "R"
EMPTY = "0"


def vision(env):
    """Return the snake's 4-direction vision as raw symbol strings.

    Maps each absolute ``Direction`` to the ray of cells from the head
    outward to the wall. A ray starts at the cell *adjacent* to the head
    (distance 1) and ends with the terminating ``"W"``; the head itself
    is never part of a ray -- it is shared at the cross centre when
    printed. Symbols: ``W`` wall, ``S`` body, ``G`` green apple, ``R``
    red apple, ``0`` empty.

    This is the only board read in the whole program: everything the
    agent later learns from is derived from these four strings, which is
    exactly how the -42 rule is enforced structurally.
    """
    # One snapshot of the body cells; membership is O(1) and the head is
    # included harmlessly (a ray never revisits the head's own cell).
    body = set(env.snake)
    return {direction: _ray(env, body, direction) for direction in Direction}


def _ray(env, body, direction):
    """Walk one direction from the head to the wall, classifying cells.

    Apples are checked before body so an apple is never mislabelled, and
    the loop stops only at the wall -- the full ray to the edge is the
    raw material the state design reads (no truncation here).
    """
    dr, dc = direction.value
    r, c = env.snake[0]
    symbols = []
    while True:
        r, c = r + dr, c + dc
        if not (0 <= r < env.size and 0 <= c < env.size):
            symbols.append(WALL)
            return "".join(symbols)
        cell = (r, c)
        if cell in env.green_apples:
            symbols.append(GREEN)
        elif cell in env.red_apples:
            symbols.append(RED)
        elif cell in body:
            symbols.append(BODY)
        else:
            symbols.append(EMPTY)


def format_vision(rays):
    """Render the vision as the subject's cross of W/H/S/G/R/0 symbols.

    The head's column is printed vertically (top wall to bottom wall) and
    its row horizontally (left wall to right wall), sharing ``H`` at the
    intersection. Each ray is stored head-outward, so the up/left arms are
    reversed to read from the wall inward to the head. The vertical arm is
    left-padded by ``len(left ray)`` spaces -- the head's offset inside the
    horizontal line -- so every symbol sits directly under ``H``. Returns
    the multi-line string; the caller does the printing.
    """
    up = rays[Direction.UP][::-1]
    down = rays[Direction.DOWN]
    left = rays[Direction.LEFT]
    right = rays[Direction.RIGHT]

    horizontal = left[::-1] + HEAD + right
    pad = " " * len(left)

    lines = [pad + ch for ch in up]
    lines.append(horizontal)
    lines.extend(pad + ch for ch in down)
    return "\n".join(lines)
