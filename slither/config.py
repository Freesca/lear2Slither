"""Configuration: built-in defaults, TOML load, merge, validation.

The only module that knows TOML exists. It produces one frozen ``Config``
from three layers, in increasing precedence:

    built-in DEFAULTS  <  -config file  <  CLI override dict

The defaults mirror ``configs/default.toml`` exactly -- a test pins them
together so they cannot drift. The frozen Config hands the rest of the
program ready-made ``interpreter.Rewards`` and ``agent.Hyperparameters``
objects, so env / interpreter / agent each receive exactly what they need
and nothing more.

The [rewards], [exploration] and [learning] values are provisional defaults
so the program runs end-to-end. Only the implemented strategies
("epsilon_greedy", constant alpha) are accepted; the candidate alternatives
raise a clear error.
"""
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 (school machines): tomli backport
    import tomli as tomllib
from copy import deepcopy
from dataclasses import dataclass, field

from slither.agent import Hyperparameters
from slither.interpreter import Rewards, Scheme

# Built-in defaults, mirroring configs/default.toml (test_config pins them).
DEFAULTS = {
    "board": {
        "size": 10,
        "green_apples": 2,
        "red_apples": 1,
        "initial_length": 3,
    },
    "goal": {
        "target_length": 10,
    },
    "state": {
        "warn": True,        # d: obstacle at distance 2
        "caution": False,    # c: obstacle at distance 3
        "green_far": True,   # split green into close G / far g
        "red_far": True,     # split red into close R / far r
        "body_far": False,   # b: a far obstacle that is body, not wall
    },
    "rewards": {
        "green": 20.0,
        "red": -10.0,
        "step": -1.0,
        "death": -100.0,
        "win": 100.0,
    },
    "exploration": {
        "strategy": "epsilon_greedy",
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.99,
    },
    "learning": {
        "alpha_strategy": "constant",
        "alpha": 0.1,
        "gamma": 0.9,
    },
    "evaluation": {
        "games": 100,
        "step_cap": 1000,
    },
    "gui": {
        "cell_px": 40,
        "speed_ms": 150,
    },
}

# Strategies the code actually implements; others raise a clear error.
_IMPLEMENTED_EXPLORATION = {"epsilon_greedy"}
_IMPLEMENTED_ALPHA = {"constant"}


@dataclass(frozen=True)
class Board:
    size: int
    green_apples: int
    red_apples: int
    initial_length: int


@dataclass(frozen=True)
class Evaluation:
    games: int
    step_cap: int


@dataclass(frozen=True)
class Gui:
    cell_px: int
    speed_ms: int


@dataclass(frozen=True)
class Config:
    """One frozen, validated configuration for a run.

    ``_raw`` is the merged source dict, kept for the model file's traceability
    field; it is excluded from equality so two Configs compare on their typed
    values, and reached only via ``as_dict`` (which deep-copies it).
    """

    board: Board
    target_length: int
    scheme: Scheme
    rewards: Rewards
    hyperparameters: Hyperparameters
    exploration_strategy: str
    alpha_strategy: str
    evaluation: Evaluation
    gui: Gui
    _raw: dict = field(compare=False, repr=False)

    def as_dict(self):
        """The merged config as a plain nested dict (for the model file)."""
        return deepcopy(self._raw)


def load(path=None, overrides=None):
    """Build a Config from defaults < ``path`` file < ``overrides``, validated.

    ``path`` is an optional TOML file; ``overrides`` an optional nested dict
    (the highest-precedence layer). Unknown sections/keys and out-of-range
    values raise ``ValueError``.
    """
    merged = deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "rb") as handle:
            _merge(merged, tomllib.load(handle))
    if overrides:
        _merge(merged, overrides)
    return _build(merged)


def _merge(base, incoming):
    """Overlay ``incoming`` onto ``base`` in place (two-level tables).

    Unknown sections or keys raise -- a typo in a config file should fail
    loudly, not be silently ignored.
    """
    for section, values in incoming.items():
        if section not in base:
            raise ValueError(f"unknown config section [{section}]")
        if not isinstance(values, dict):
            raise ValueError(f"[{section}] must be a table")
        for key, value in values.items():
            if key not in base[section]:
                raise ValueError(f"unknown config key {section}.{key}")
            base[section][key] = value


def _build(c):
    """Validate the merged dict and assemble the typed, frozen Config."""
    board = Board(
        size=_int(c, "board", "size", minimum=2),
        green_apples=_int(c, "board", "green_apples", minimum=0),
        red_apples=_int(c, "board", "red_apples", minimum=0),
        initial_length=_int(c, "board", "initial_length", minimum=1),
    )
    _check_capacity(board)

    target = _int(c, "goal", "target_length", minimum=1)

    scheme = Scheme(
        warn=_bool(c, "state", "warn"),
        caution=_bool(c, "state", "caution"),
        green_far=_bool(c, "state", "green_far"),
        red_far=_bool(c, "state", "red_far"),
        body_far=_bool(c, "state", "body_far"),
    )

    rewards = Rewards(
        green=_num(c, "rewards", "green"),
        red=_num(c, "rewards", "red"),
        step=_num(c, "rewards", "step"),
        death=_num(c, "rewards", "death"),
        win=_num(c, "rewards", "win"),
    )

    exploration_strategy = _strategy(
        c, "exploration", "strategy", _IMPLEMENTED_EXPLORATION, "T2")
    alpha_strategy = _strategy(
        c, "learning", "alpha_strategy", _IMPLEMENTED_ALPHA, "T3")

    alpha = _num(c, "learning", "alpha", low=0.0, high=1.0, low_open=True)
    gamma = _num(c, "learning", "gamma", low=0.0, high=1.0)
    eps_start = _num(c, "exploration", "epsilon_start", low=0.0, high=1.0)
    eps_min = _num(c, "exploration", "epsilon_min", low=0.0, high=1.0)
    eps_decay = _num(c, "exploration", "epsilon_decay",
                     low=0.0, high=1.0, low_open=True)
    if eps_min > eps_start:
        raise ValueError(
            "exploration.epsilon_min must be <= epsilon_start")
    hyperparameters = Hyperparameters(
        alpha=alpha, gamma=gamma, epsilon_start=eps_start,
        epsilon_min=eps_min, epsilon_decay=eps_decay,
    )

    evaluation = Evaluation(
        games=_int(c, "evaluation", "games", minimum=1),
        step_cap=_int(c, "evaluation", "step_cap", minimum=1),
    )
    gui = Gui(
        cell_px=_int(c, "gui", "cell_px", minimum=1),
        speed_ms=_int(c, "gui", "speed_ms", minimum=0),
    )

    return Config(
        board=board,
        target_length=target,
        scheme=scheme,
        rewards=rewards,
        hyperparameters=hyperparameters,
        exploration_strategy=exploration_strategy,
        alpha_strategy=alpha_strategy,
        evaluation=evaluation,
        gui=gui,
        _raw=c,
    )


def _bool(c, section, key):
    value = c[section][key]
    if not isinstance(value, bool):
        raise ValueError(f"{section}.{key} must be a boolean")
    return value


def _int(c, section, key, *, minimum=None):
    value = c[section][key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{section}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{section}.{key} must be >= {minimum}")
    return value


def _num(c, section, key, *, low=None, high=None, low_open=False):
    value = c[section][key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{section}.{key} must be a number")
    value = float(value)
    if low is not None and (value <= low if low_open else value < low):
        bound = ">" if low_open else ">="
        raise ValueError(f"{section}.{key} must be {bound} {low}")
    if high is not None and value > high:
        raise ValueError(f"{section}.{key} must be <= {high}")
    return value


def _strategy(c, section, key, implemented, gate):
    value = c[section][key]
    if not isinstance(value, str):
        raise ValueError(f"{section}.{key} must be a string")
    if value not in implemented:
        raise ValueError(
            f"{section}.{key}={value!r} is not implemented yet "
            f"(GATE {gate}); supported: {sorted(implemented)}")
    return value


def _check_capacity(board):
    """The board must hold the initial snake plus all apples at reset."""
    cells = board.size * board.size
    needed = board.initial_length + board.green_apples + board.red_apples
    if needed > cells:
        raise ValueError(
            "board too small: size**2 must hold the snake and apples")
