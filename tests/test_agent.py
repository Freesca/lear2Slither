"""Tests for the Bellman update math and the policy invariants.

The two highest-risk facts get hand-computed checks before any training: the
non-terminal update (with bootstrap) and the terminal update (no bootstrap).
Plus: random tie-breaking actually randomises, exploration can reach every
column, eval mode never mutates Q, and epsilon is a pure function of the
session count (resume behaviour).
"""
import copy
import random

import pytest

from slither.agent import Agent, Hyperparameters

# alpha/gamma match the worked examples in the tests below.
HP = Hyperparameters(alpha=0.1, gamma=0.9, epsilon_start=1.0,
                     epsilon_min=0.01, epsilon_decay=0.99)


def make_agent(**kwargs):
    return Agent(HP, random.Random(0), **kwargs)


# --- the Bellman update (hand-computed) ---------------------------------

def test_update_non_terminal_matches_hand_computation():
    agent = make_agent()
    agent.q["s_next"] = [2.0, 5.0, 1.0, -100.0]
    agent.n["s_next"] = [0, 0, 0, 0]
    # s unseen -> zero row; take FORWARD (col 0) onto a green, r = 20.
    agent.update("s", 0, 20.0, "s_next", terminal=False)
    # target = 20 + 0.9 * max(2, 5, 1, -100) = 24.5
    # Q[s][0] = 0 + 0.1 * (24.5 - 0) = 2.45
    assert agent.q["s"][0] == pytest.approx(2.45)
    assert agent.n["s"][0] == 1
    assert agent.q["s"][1:] == [0.0, 0.0, 0.0]   # other columns untouched


def test_update_terminal_has_no_bootstrap():
    agent = make_agent()
    # A tempting successor: if the terminal target wrongly bootstrapped, the
    # +999 would dominate. It must be ignored.
    agent.q["ignored"] = [999.0, 999.0, 999.0, 999.0]
    agent.update("s", 2, -100.0, "ignored", terminal=True)
    # target = r = -100; Q[s][2] = 0 + 0.1 * (-100 - 0) = -10
    assert agent.q["s"][2] == pytest.approx(-10.0)


def test_update_bootstrap_uses_max_not_the_action_taken():
    # Off-policy: the target reads max(Q[s_next]), regardless of which column
    # we updated. Updating col 3 still bootstraps on the best (col 1 = 5).
    agent = make_agent()
    agent.q["s_next"] = [2.0, 5.0, 1.0, 0.0]
    agent.update("s", 3, 0.0, "s_next", terminal=False)
    # target = 0 + 0.9 * 5 = 4.5; Q[s][3] = 0 + 0.1 * 4.5 = 0.45
    assert agent.q["s"][3] == pytest.approx(0.45)


# --- the policy: tie-breaking and exploration ---------------------------

def test_argmax_breaks_ties_randomly_over_all_maxima():
    agent = make_agent()
    agent.q["tie"] = [1.0, 1.0, 1.0, 1.0]        # everything ties
    seen = {agent.choose("tie", eval_mode=True) for _ in range(100)}
    assert seen == {0, 1, 2, 3}                  # every column reachable


def test_argmax_only_chooses_among_the_maxima():
    agent = make_agent()
    agent.q["two"] = [3.0, 1.0, 3.0, 0.0]        # maxima at cols 0 and 2
    seen = {agent.choose("two", eval_mode=True) for _ in range(100)}
    assert seen == {0, 2}


def test_exploration_can_pick_any_column_including_backwards():
    agent = make_agent()                         # epsilon_start = 1.0
    # random() in [0, 1) is always < 1.0 -> every choose explores.
    seen = {agent.choose("anything") for _ in range(100)}
    assert seen == {0, 1, 2, 3}                  # BACKWARDS (3) reachable


# --- eval mode is frozen ------------------------------------------------

def test_eval_mode_never_mutates_q():
    agent = make_agent()
    agent.q["known"] = [1.0, 2.0, 3.0, 4.0]
    agent.n["known"] = [5, 5, 5, 5]
    before_q = copy.deepcopy(agent.q)
    before_n = copy.deepcopy(agent.n)
    for _ in range(200):
        agent.choose("known", eval_mode=True)
        agent.choose("unseen", eval_mode=True)   # must not be stored
    assert agent.q == before_q
    assert agent.n == before_n
    assert "unseen" not in agent.q               # reads stay side-effect free


# --- epsilon schedule (resume behaviour) --------------------------------

def test_epsilon_is_a_pure_function_of_sessions():
    fresh = make_agent()
    assert fresh.epsilon == pytest.approx(1.0)   # start, no decay yet
    fresh.end_session()
    assert fresh.epsilon == pytest.approx(0.99)  # one decay step
    # A model loaded at 1 session reconstructs the identical epsilon.
    resumed = make_agent(sessions_trained=1)
    assert resumed.epsilon == pytest.approx(fresh.epsilon)


def test_epsilon_floors_at_the_minimum():
    agent = make_agent(sessions_trained=10_000)
    assert agent.epsilon == pytest.approx(HP.epsilon_min)
