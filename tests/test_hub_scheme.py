"""Rework tests: the hub's state-alphabet mirror (snake_den/scheme.py).

The hub duplicates slither's alphabet/legend math (the -42 firewall: the hub
may not import the interpreter). This pins the copy equal to the real thing
across several schemes, so the two cannot drift -- the same guard
test_hub_snake_proc applies to the progress-format version.
"""
import pytest

from slither import interpreter
from slither.interpreter import Scheme
from snake_den import scheme as mirror

_SCHEMES = [
    {},                                                   # default (k=7)
    {"caution": True},                                    # k=8
    {"warn": False},                                      # k=6
    {"green_far": False, "red_far": False},               # k=5
    {"warn": False, "caution": False,
     "green_far": False, "red_far": False},               # k=4
    {"warn": True, "caution": True,
     "green_far": True, "red_far": True},                 # k=8
    {"body_far": True},                                   # k=8
    {"caution": True, "body_far": True},                  # k=9
]


@pytest.mark.parametrize("flags", _SCHEMES)
def test_alphabet_matches_interpreter(flags):
    real = interpreter.alphabet(Scheme(**mirror.normalize(flags)))
    assert mirror.alphabet(flags) == real


@pytest.mark.parametrize("flags", _SCHEMES)
def test_legend_matches_interpreter(flags):
    real = interpreter.legend(Scheme(**mirror.normalize(flags)))
    assert mirror.legend(flags) == real


@pytest.mark.parametrize("flags", _SCHEMES)
def test_state_count_follows_formula(flags):
    k = len(mirror.alphabet(flags))
    assert mirror.state_count(flags) == k * k * (k + 1) // 2
    assert mirror.qvalue_count(flags) == mirror.state_count(flags) * 4


def test_default_is_the_legacy_seven():
    assert mirror.alphabet({}) == "DdGgRrN"
    assert mirror.state_count({}) == 196
