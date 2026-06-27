"""Interpreter: vision -> canonical state + frame, and outcome -> reward.

A pure function of the vision (the ``{Direction: ray}`` dict from
``vision.py``) and the ``StepOutcome`` -- it never reads the board, so the
-42 firewall is structural: everything downstream (the agent) can only ever
receive the ``(canonical_state, reward)`` this module emits.

State design: each ray is reduced to one nearest-object
symbol; the body anchors a relative frame that removes the board's 4 rotations
and a mirror flag removes its 2 reflections. The canonical key is the
lexicographically smallest 3-symbol (Forward, Left, Right) string over all
admissible frames; the winning frame is returned so the agent's relative
action can be translated back to an absolute Direction.
"""
from dataclasses import dataclass
from enum import IntEnum

from slither.environment import Direction
from slither.vision import BODY, EMPTY, GREEN, RED, WALL

# Clockwise order of the absolute directions (screen view, row increasing
# downward): up -> right -> down -> left. Turns a "forward" direction into
# its relative left/right neighbours.
_CW = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)


class RelativeAction(IntEnum):
    """The Q-table's four columns, in the order fixed by the model format.

    The values double as the column index into the agent's Q-row, so the
    agent can stay purely numeric (it never imports this enum); only the
    Frame and the runner use the names for readability.
    """

    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    BACKWARDS = 3


def _opposite(direction):
    dr, dc = direction.value
    return Direction((-dr, -dc))


def _relative_lr(forward):
    """Absolute (left, right) when facing ``forward`` (before mirroring)."""
    i = _CW.index(forward)
    return _CW[(i - 1) % 4], _CW[(i + 1) % 4]


@dataclass
class Frame:
    """A body-relative orientation: which absolute direction is BACK, and
    whether left/right are mirrored. Converts between the agent's relative
    actions and the environment's absolute directions.
    """

    back: Direction
    mirror: bool

    def _fwd_left_right(self):
        forward = _opposite(self.back)
        left, right = _relative_lr(forward)
        if self.mirror:
            left, right = right, left
        return forward, left, right

    def encode(self, symbols):
        """The 3-symbol (F, L, R) key of this frame over classified rays."""
        forward, left, right = self._fwd_left_right()
        return symbols[forward] + symbols[left] + symbols[right]

    def to_absolute(self, action):
        """Map a relative action (RelativeAction or int 0-3) to a Direction."""
        forward, left, right = self._fwd_left_right()
        return (forward, left, right, self.back)[int(action)]


@dataclass(frozen=True)
class Scheme:
    """Which perceptual distinctions the ray classifier draws (dynamic state).

    The state stays *one symbol per ray*; each flag only splits or merges a
    symbol, so the alphabet size k -- and therefore the state count
    k**2*(k+1)/2 -- is a pure function of this object. ``D`` (danger at 1) and
    ``N`` (clear) are always present. Recorded in the model file so a model is
    always replayed with the scheme it trained on; built from the config's
    ``[state]`` section.
    """

    warn: bool = True        # d: obstacle at distance 2
    caution: bool = False    # c: obstacle at distance 3
    green_far: bool = True   # split green into close G / far g
    red_far: bool = True     # split red into close R / far r
    body_far: bool = False   # b: a far obstacle that is *body*, not wall

    def as_dict(self):
        """The scheme as a plain dict (for the model file's state block)."""
        return {"warn": self.warn, "caution": self.caution,
                "green_far": self.green_far, "red_far": self.red_far,
                "body_far": self.body_far}


DEFAULT_SCHEME = Scheme()    # the legacy 7-letter alphabet (D d G g R r N)


def alphabet(scheme=DEFAULT_SCHEME):
    """The active symbols for ``scheme``, in a fixed display order.

    ``len(alphabet(scheme))`` is the k that sizes the state space; the string
    also drives the viewer's legend ordering. D and N always bracket the set.
    """
    letters = ["D"]
    if scheme.warn:
        letters.append("d")
    if scheme.caution:
        letters.append("c")
    letters.append("G")
    if scheme.green_far:
        letters.append("g")
    letters.append("R")
    if scheme.red_far:
        letters.append("r")
    if scheme.body_far:
        letters.append("b")
    letters.append("N")
    return "".join(letters)


def legend(scheme=DEFAULT_SCHEME):
    """``{symbol: human meaning}`` for ``scheme`` -- the viewer's key."""
    items = {"D": "danger@1"}
    if scheme.warn:
        items["d"] = "danger@2"
    if scheme.caution:
        items["c"] = "danger@3"
    items["G"] = "green<=3" if scheme.green_far else "green"
    if scheme.green_far:
        items["g"] = "green>=4"
    items["R"] = "red<=3" if scheme.red_far else "red"
    if scheme.red_far:
        items["r"] = "red>=4"
    if scheme.body_far:
        items["b"] = "body, far"
    items["N"] = "clear"
    return items


def classify(ray, scheme=DEFAULT_SCHEME):
    """Reduce one ray to its nearest-object symbol.

    Scans outward from distance 1 (``ray[0]``) to the first non-empty cell and
    buckets it under ``scheme``: an obstacle (wall/body) is ``D`` at distance
    1, ``d`` at 2 (if ``warn``), ``c`` at 3 (if ``caution``), else ``N``; a
    green apple is split into close ``G`` / far ``g`` at distance 3 (if
    ``green_far``) else a single ``G``; red likewise (``R``/``r``, if
    ``red_far``). A far obstacle that falls through to ``N`` is reported as
    ``b`` when it is *body* rather than wall (if ``body_far``) -- a distant
    body segment means that direction leads back into the snake's own coils,
    which a far wall (open boundary) does not. Objects behind the nearest one
    are masked (accepted limit).
    """
    for index, cell in enumerate(ray):
        if cell == EMPTY:
            continue
        distance = index + 1
        if cell in (WALL, BODY):
            if distance == 1:
                return "D"
            if distance == 2 and scheme.warn:
                return "d"
            if distance == 3 and scheme.caution:
                return "c"
            if scheme.body_far and cell == BODY:
                return "b"
            return "N"
        if cell == GREEN:
            return "G" if not scheme.green_far or distance <= 3 else "g"
        if cell == RED:
            return "R" if not scheme.red_far or distance <= 3 else "r"
    return "N"  # unreachable: every ray ends in a wall


def state(rays, scheme=DEFAULT_SCHEME):
    """Vision -> (canonical 3-symbol key, winning Frame).

    Admissible anchors are the directions with body adjacent to the head
    (the ray starts with ``S``); with none (length 1) every direction is
    admissible. The canonical key is the lexicographic min of the encodings
    over all (anchor, mirror) frames; the frame that achieves it is returned.
    """
    symbols = {d: classify(rays[d], scheme) for d in Direction}

    anchors = [d for d in Direction if rays[d][:1] == BODY]
    if not anchors:
        anchors = list(Direction)

    best_key = None
    best_frame = None
    for back in anchors:
        for mirror in (False, True):
            frame = Frame(back, mirror)
            key = frame.encode(symbols)
            if best_key is None or key < best_key:
                best_key = key
                best_frame = frame
    return best_key, best_frame


@dataclass(frozen=True)
class Rewards:
    """The reward values, supplied by the runner."""

    green: float
    red: float
    step: float
    death: float
    win: float


def reward(outcome, rewards):
    """Price one ``StepOutcome`` into a scalar.

    Terminal events dominate, which makes the cases mutually exclusive: a win
    is the board-full victory (it also set ``ate_green``), death covers all
    three game-overs, and otherwise exactly one of green / red / empty
    happened. Values come from config; the rule (structure) is fixed here.
    """
    if outcome.won:
        return rewards.win
    if outcome.died:
        return rewards.death
    if outcome.ate_green:
        return rewards.green
    if outcome.ate_red:
        return rewards.red
    return rewards.step
