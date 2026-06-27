"""Evaluation suite: aggregate stats over runner results.

Pure statistics over the ``SessionResult`` list the runner already returns
(runner.py) -- this module reads boards, agents and pygame *not at all*. It is
the single source of truth for "how good is this model", consumed both by the
``-stats`` terminal report and by the GUI stats screens: build the numbers
once, render them twice.

Design rationale:

- A frozen model is measured with learning OFF and epsilon = 0
  (``-dontlearn``), so every move is the policy's genuine argmax. The runner
  is driven in that mode here.
- Headline metric = **success rate at length >= target** (the project goal). It
  is a binomial proportion: at ~100 games it carries a ~+/-10% band, so close
  models want more games, not a single run.
- Duration is reported but never alone: a "don't die" snake circles forever at
  length 3 (huge duration, zero success). The step cap turns that livelock into
  a visible high-duration / low-length outlier instead of a hang.
- The random baseline gives the numbers scale. It needs no special code: a
  fresh agent has an empty Q-table, so under eval mode every row ties at zero
  and the tie-break picks uniformly at random (agent.py). So "no -load,
  -dontlearn" *is* the random policy; ``compare`` prepends it automatically.
"""
import statistics
from dataclasses import dataclass

from slither import runner
from slither.runner import RunOptions

# The five terminal outcomes a session can end in. "length_zero" is the red
# apple starving the snake to length 0; "truncated" is the step cap firing
# while still alive (a livelock outlier, not a death).
_OUTCOME_KEYS = ("wall", "self", "length_zero", "truncated", "won")


@dataclass(frozen=True)
class Distribution:
    """Summary of one metric across the games, plus the raw values.

    ``values`` is kept so the GUI can draw a histogram later; the scalar
    fields are what the terminal report shows. ``std`` is the sample standard
    deviation (0.0 for fewer than two games, where it is undefined).
    """

    mean: float
    median: float
    maximum: int
    std: float
    values: tuple[int, ...]


@dataclass(frozen=True)
class SuiteStats:
    """Aggregate of one N-game suite for one (frozen) model."""

    games: int
    successes: int
    success_rate: float
    length: Distribution
    duration: Distribution
    outcomes: dict
    target_length: int


def summarize(results, target_length):
    """Aggregate a list of ``SessionResult`` into a ``SuiteStats``.

    ``results`` is what ``runner.run`` returns; ``target_length`` is the goal
    a game must reach to count as a success (config.target_length).
    """
    if not results:
        empty = _distribution([])
        return SuiteStats(
            games=0, successes=0, success_rate=0.0,
            length=empty, duration=empty,
            outcomes={key: 0 for key in _OUTCOME_KEYS},
            target_length=target_length,
        )

    lengths = [r.max_length for r in results]
    durations = [r.duration for r in results]
    successes = sum(1 for length in lengths if length >= target_length)

    outcomes = {key: 0 for key in _OUTCOME_KEYS}
    for result in results:
        outcomes[_classify(result)] += 1

    return SuiteStats(
        games=len(results),
        successes=successes,
        success_rate=successes / len(results),
        length=_distribution(lengths),
        duration=_distribution(durations),
        outcomes=outcomes,
        target_length=target_length,
    )


def _classify(result):
    """Which of the five terminal outcomes this session ended in."""
    if result.won:
        return "won"
    if result.death_cause == "wall":
        return "wall"
    if result.death_cause == "self":
        return "self"
    if result.death_cause == "length":
        return "length_zero"
    return "truncated"          # alive at the step cap


def _distribution(values):
    """Mean / median / max / sample-std of ``values`` (ints), kept raw too."""
    if not values:
        return Distribution(0.0, 0.0, 0, 0.0, ())
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return Distribution(
        mean=float(statistics.mean(values)),
        median=float(statistics.median(values)),
        maximum=max(values),
        std=float(std),
        values=tuple(values),
    )


def compare(config, model_paths, *, games, seed):
    """Run the *same* seeded greedy suite for the baseline and each model.

    Returns ``[(label, SuiteStats), ...]``, baseline first. Every suite shares
    one ``seed``, so each model faces the identical board sequence -- the
    comparison is fair and fully reproducible. The runner is driven quietly
    (no per-game lines) so only the final table
    reaches stdout.
    """
    rows = [("baseline (random)",
             _evaluate_one(config, None, games=games, seed=seed))]
    for path in model_paths:
        rows.append((path,
                     _evaluate_one(config, path, games=games, seed=seed)))
    return rows


def _evaluate_one(config, load_path, *, games, seed):
    """One frozen-model suite: ``games`` greedy games, summarized."""
    options = RunOptions(
        sessions=games, load=load_path, dontlearn=True, seed=seed)
    results = runner.run(config, options, quiet=True)
    return summarize(results, config.target_length)


def format_report(stats):
    """The ``-stats`` terminal block for one suite (golden-tested text)."""
    out = stats.outcomes
    return "\n".join((
        f"Evaluation over {stats.games} games",
        f"  Success (length >= {stats.target_length}): "
        f"{stats.successes}/{stats.games} = {stats.success_rate * 100:.1f}%",
        f"  Length    mean {stats.length.mean:.1f}  "
        f"median {stats.length.median:.1f}  "
        f"max {stats.length.maximum}  std {stats.length.std:.1f}",
        f"  Duration  mean {stats.duration.mean:.1f}  "
        f"median {stats.duration.median:.1f}  "
        f"max {stats.duration.maximum}  std {stats.duration.std:.1f}",
        f"  Outcomes  wall {out['wall']}  self {out['self']}  "
        f"length-0 {out['length_zero']}  truncated {out['truncated']}  "
        f"won {out['won']}",
    ))


_CURVE_HEADER = (
    f"{'model':<30}{'success%':>9}{'mean len':>10}"
    f"{'mean dur':>10}{'games':>7}"
)


def format_curve(rows):
    """The learning-curve table for ``compare`` output (baseline + models)."""
    lines = [_CURVE_HEADER]
    lines.extend(_curve_row(label, stats) for label, stats in rows)
    return "\n".join(lines)


def _curve_row(label, stats):
    if len(label) > 29:
        label = ".." + label[-27:]      # keep the filename, hold the column
    return (
        f"{label:<30}"
        f"{stats.success_rate * 100:>8.1f}%"
        f"{stats.length.mean:>10.1f}"
        f"{stats.duration.mean:>10.1f}"
        f"{stats.games:>7}"
    )
