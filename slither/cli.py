"""Command-line interface: subject flags -> mode + config -> runner.

Parses the subject's single-dash flags, loads the configuration, builds the
run options, and hands both to the runner. No game logic lives here.
``allow_abbrev=False`` so the several -s* flags (-sessions, -save,
-step-by-step, -seed) never collide via prefix matching.
"""
import argparse

from slither import config as config_module
from slither import evaluate
from slither import progress as progress_module
from slither import runner
from slither.runner import RunOptions


def build_parser():
    """The argparse parser for the subject's CLI flags."""
    parser = argparse.ArgumentParser(
        prog="snake", allow_abbrev=False,
        description="Learn2Slither: a Q-learning snake.",
    )
    parser.add_argument("-sessions", type=int, default=1,
                        help="number of games to run (default 1)")
    parser.add_argument("-save", metavar="PATH",
                        help="export the model after the run")
    parser.add_argument("-load", metavar="PATH",
                        help="import a model before the run")
    parser.add_argument("-visual", choices=("on", "off"), default="off",
                        help="GUI window on/off")
    parser.add_argument("-dontlearn", action="store_true",
                        help="evaluation: no Q update and epsilon = 0")
    parser.add_argument("-step-by-step", dest="step_by_step",
                        action="store_true",
                        help="wait for Enter before each move")
    parser.add_argument("-config", metavar="PATH",
                        help="TOML config file (overrides built-in defaults)")
    parser.add_argument("-state", metavar="SPEC",
                        help="state alphabet: 'default', or a comma list of "
                             "the enabled distinctions among "
                             "warn,caution,green_far,red_far,body_far "
                             "(unlisted ones are turned off)")
    parser.add_argument("-seed", type=int, default=None,
                        help="seed the RNG for a reproducible run")
    parser.add_argument("-stats", action="store_true",
                        help="print an aggregate report after the run")
    parser.add_argument("-compare", nargs="+", metavar="PATH",
                        help="evaluate the random baseline + each model over "
                             "[evaluation].games seeded greedy games, print a "
                             "learning-curve table, and exit")
    parser.add_argument("-progress", action="store_true",
                        help="emit JSON-lines progress instead of human text "
                             "(for the hub)")
    return parser


_STATE_FEATURES = ("warn", "caution", "green_far", "red_far", "body_far")
_DEFAULT_STATE = {"warn": True, "caution": False,
                  "green_far": True, "red_far": True, "body_far": False}


def _parse_state_spec(spec):
    """A ``-state`` SPEC string -> a ``[state]`` overrides dict.

    ``'default'`` is the built-in 7-letter alphabet; otherwise the listed
    features are enabled and every unlisted one is disabled. Raises
    ``ValueError`` on an unknown feature name.
    """
    spec = spec.strip()
    if spec == "default":
        return dict(_DEFAULT_STATE)
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    unknown = [token for token in tokens if token not in _STATE_FEATURES]
    if unknown:
        raise ValueError(
            f"-state: unknown feature(s) {unknown}; "
            f"choose from {list(_STATE_FEATURES)} or 'default'")
    return {feature: (feature in tokens) for feature in _STATE_FEATURES}


def main(argv=None):
    """Parse ``argv``, run the sessions, return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sessions < 1:
        parser.error("-sessions must be >= 1")

    overrides = None
    if args.state is not None:
        try:
            overrides = {"state": _parse_state_spec(args.state)}
        except ValueError as exc:
            parser.error(str(exc))
    config = config_module.load(args.config, overrides)

    if args.compare:
        # Comparative greedy suite: its own code path, not a normal run. A
        # fixed default seed makes the learning-curve table reproducible.
        seed = args.seed if args.seed is not None else 0
        rows = evaluate.compare(
            config, args.compare,
            games=config.evaluation.games, seed=seed)
        print(evaluate.format_curve(rows))
        return 0

    options = RunOptions(
        sessions=args.sessions,
        save=args.save,
        load=args.load,
        visual=(args.visual == "on"),
        dontlearn=args.dontlearn,
        step_by_step=args.step_by_step,
        seed=args.seed,
    )
    if args.progress:
        # Machine mode: runner streams start + per-session events; we close
        # with the summary. Human -stats output is suppressed (pure JSON).
        results = runner.run(config, options, progress=True)
        print(progress_module.summary(
            evaluate.summarize(results, config.target_length)))
        return 0

    results = runner.run(config, options)
    if args.stats:
        print(evaluate.format_report(
            evaluate.summarize(results, config.target_length)))
    return 0
