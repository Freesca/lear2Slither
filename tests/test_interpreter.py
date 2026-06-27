"""Phase 3 tests: the bug-magnet (state-design.md sec. 7).

Covers ray classification, the 196-state key space, the frame round-trip
(bijection + anchors), mirror equivariance, and the reward cascade.
"""
from itertools import product

import pytest

from slither.environment import Direction, StepOutcome
from slither.interpreter import (
    DEFAULT_SCHEME,
    Frame,
    RelativeAction,
    Rewards,
    Scheme,
    alphabet,
    classify,
    legend,
    reward,
    state,
)

ALPHABET = "DdGgRrN"

OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

# Horizontal mirror (reflect across the vertical axis): up/down fixed,
# left<->right swapped.
HMIRROR = {
    Direction.UP: Direction.UP,
    Direction.DOWN: Direction.DOWN,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


# --- ray classification -------------------------------------------------

@pytest.mark.parametrize("ray, expected", [
    ("W", "D"),         # wall at distance 1
    ("SW", "D"),        # body at 1 (masks the wall behind it)
    ("0W", "d"),        # wall at 2
    ("0SW", "d"),       # body at 2
    ("00W", "N"),       # wall at 3 -> nothing relevant
    ("000SW", "N"),     # body at 4 -> N
    ("GW", "G"),        # green at 1
    ("00GW", "G"),      # green at 3 (close boundary)
    ("000GW", "g"),     # green at 4 (far boundary)
    ("RW", "R"),        # red at 1
    ("00RW", "R"),      # red at 3
    ("0000RW", "r"),    # red at 5
    ("RGW", "R"),       # masking: nearer red hides the green
    ("SGW", "D"),       # masking: body hides the green
])
def test_classify(ray, expected):
    assert classify(ray) == expected


# --- the 196-state key space --------------------------------------------

def test_196_canonical_keys():
    # All 7**3 symbol triples, mirror-canonicalised (L<->R), must collapse
    # to exactly 196 distinct keys: k**2 * (k + 1) / 2 with k = 7.
    keys = {min(f + lt + rt, f + rt + lt)
            for f, lt, rt in product(ALPHABET, repeat=3)}
    assert len(keys) == 196


def _canonical_count(letters):
    return len({min(f + lt + rt, f + rt + lt)
                for f, lt, rt in product(letters, repeat=3)})


def test_default_scheme_is_the_legacy_seven():
    assert alphabet(DEFAULT_SCHEME) == ALPHABET


@pytest.mark.parametrize("scheme", [
    DEFAULT_SCHEME,                              # k = 7 -> 196
    Scheme(caution=True),                        # k = 8 -> 288
    Scheme(warn=False),                          # k = 6 -> 126
    Scheme(green_far=False, red_far=False),      # k = 5 -> 75
    Scheme(warn=False, caution=False,
           green_far=False, red_far=False),      # k = 4 -> 40
    Scheme(body_far=True),                        # k = 8 -> 288
])
def test_state_count_follows_the_alphabet(scheme):
    # The canonical-state count is a pure function of the active alphabet:
    # k**2 * (k + 1) / 2. Changing the scheme changes k, hence the table size.
    k = len(alphabet(scheme))
    assert _canonical_count(alphabet(scheme)) == k * k * (k + 1) // 2


@pytest.mark.parametrize("scheme, ray, expected", [
    (Scheme(warn=False), "0W", "N"),           # d disabled: wall@2 -> N
    (Scheme(caution=True), "00W", "c"),        # caution: wall@3 -> c
    (Scheme(caution=True), "000W", "N"),       # wall@4 still N
    (Scheme(green_far=False), "000GW", "G"),   # merged green: far -> G
    (Scheme(red_far=False), "0000RW", "R"),    # merged red: far -> R
    (DEFAULT_SCHEME, "00W", "N"),              # default: wall@3 -> N
    (Scheme(body_far=True), "0000SW", "b"),    # far body -> b
    (Scheme(body_far=True), "0000W", "N"),     # far wall stays N
    (Scheme(body_far=True), "0SW", "d"),       # near body still merged (d@2)
    (DEFAULT_SCHEME, "0000SW", "N"),           # default: far body -> N
])
def test_classify_honours_the_scheme(scheme, ray, expected):
    assert classify(ray, scheme) == expected


def test_legend_tracks_the_scheme():
    assert set(legend(DEFAULT_SCHEME)) == set(ALPHABET)
    assert "c" in legend(Scheme(caution=True))
    assert "g" not in legend(Scheme(green_far=False))
    assert "b" in legend(Scheme(body_far=True))
    assert "b" not in legend(DEFAULT_SCHEME)


# --- frame round-trip ---------------------------------------------------

def test_frame_is_a_bijection_with_fixed_anchors():
    for back in Direction:
        for mirror in (False, True):
            frame = Frame(back, mirror)
            mapped = [frame.to_absolute(a) for a in RelativeAction]
            assert set(mapped) == set(Direction)            # bijection
            assert frame.to_absolute(RelativeAction.BACKWARDS) == back
            assert (frame.to_absolute(RelativeAction.FORWARD)
                    == OPPOSITE[back])


# --- mirror equivariance ------------------------------------------------

def _hmirror_rays(rays):
    return {HMIRROR[d]: ray for d, ray in rays.items()}


def test_mirror_board_gives_same_state_and_mirrored_actions():
    # Asymmetric board so the winning frame is unique on both sides.
    rays = {
        Direction.UP: "GW",      # green ahead
        Direction.DOWN: "SW",    # body behind -> the anchor
        Direction.LEFT: "00RW",  # red to the left (distance 3)
        Direction.RIGHT: "0W",   # wall close on the right (distance 2)
    }
    key, frame = state(rays)
    mkey, mframe = state(_hmirror_rays(rays))

    assert key == mkey  # same canonical state for a board and its mirror
    for a in RelativeAction:
        assert frame.to_absolute(a) == HMIRROR[mframe.to_absolute(a)]


# --- reward cascade -----------------------------------------------------

def test_reward_cascade_terminal_dominates():
    rw = Rewards(green=20.0, red=-10.0, step=-1.0, death=-100.0, win=100.0)
    assert reward(StepOutcome(), rw) == -1.0                      # empty step
    assert reward(StepOutcome(ate_green=True), rw) == 20.0
    assert reward(StepOutcome(ate_red=True), rw) == -10.0
    assert reward(StepOutcome(died=True, death_cause="wall"), rw) == -100.0
    # terminal events dominate any simultaneous eat:
    assert reward(
        StepOutcome(ate_red=True, died=True, death_cause="length"), rw
    ) == -100.0
    assert reward(StepOutcome(ate_green=True, won=True), rw) == 100.0
