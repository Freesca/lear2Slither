"""Mirror of slither's dynamic state alphabet (the -42 firewall).

The hub lets the user pick the state distinctions for a training run and must
show the resulting alphabet, state count and legend -- but it may not import
slither's interpreter (only ``config.DEFAULTS`` + ``model_io`` cross the
boundary). So the tiny alphabet/legend math is duplicated here, the same way
the palette is duplicated in each render module and
``snake_proc.PROGRESS_FORMAT_VERSION`` mirrors the product's constant.
``test_hub_scheme`` pins this equal to ``slither.interpreter`` (a test may
import the product freely) so the copy cannot drift.

A scheme is a plain ``{feature: bool}`` dict. ``D`` (danger at distance 1) and
``N`` (clear) are always present; the four features only split or merge the
other symbols, so the alphabet size k -- and the state count k**2*(k+1)/2 --
is a pure function of the dict.
"""

# The selectable distinctions, in display order, each with a UI label.
FEATURES = ("warn", "caution", "green_far", "red_far", "body_far")
DEFAULT = {"warn": True, "caution": False, "green_far": True,
           "red_far": True, "body_far": False}
LABELS = {
    "warn": "warn @2  (d)",
    "caution": "caution @3  (c)",
    "green_far": "green near/far  (G/g)",
    "red_far": "red near/far  (R/r)",
    "body_far": "far body  (b)",
}


def normalize(scheme):
    """A full scheme dict from a partial/None one (missing keys -> default)."""
    scheme = scheme or {}
    return {feature: bool(scheme.get(feature, DEFAULT[feature]))
            for feature in FEATURES}


def alphabet(scheme):
    """The active symbols for ``scheme``, in fixed display order (a string)."""
    flags = normalize(scheme)
    letters = ["D"]
    if flags["warn"]:
        letters.append("d")
    if flags["caution"]:
        letters.append("c")
    letters.append("G")
    if flags["green_far"]:
        letters.append("g")
    letters.append("R")
    if flags["red_far"]:
        letters.append("r")
    if flags["body_far"]:
        letters.append("b")
    letters.append("N")
    return "".join(letters)


def state_count(scheme):
    """Canonical states = k**2 * (k + 1) / 2 for alphabet size k."""
    k = len(alphabet(scheme))
    return k * k * (k + 1) // 2


def qvalue_count(scheme):
    """Q-values = states * 4 actions (FORWARD/LEFT/RIGHT/BACKWARDS)."""
    return state_count(scheme) * 4


def legend(scheme):
    """``{symbol: human meaning}`` for ``scheme`` -- the viewer's key."""
    flags = normalize(scheme)
    items = {"D": "danger@1"}
    if flags["warn"]:
        items["d"] = "danger@2"
    if flags["caution"]:
        items["c"] = "danger@3"
    items["G"] = "green<=3" if flags["green_far"] else "green"
    if flags["green_far"]:
        items["g"] = "green>=4"
    items["R"] = "red<=3" if flags["red_far"] else "red"
    if flags["red_far"]:
        items["r"] = "red>=4"
    if flags["body_far"]:
        items["b"] = "body, far"
    items["N"] = "clear"
    return items
