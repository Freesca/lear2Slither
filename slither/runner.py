"""Runner: session loop, stats, terminal output, mode switches. (Phase 5)

The only module that sees the whole cast (env, vision, interpreter, agent,
model_io). It owns the tick loop from implementation-plan.md sec. 4.2 and the
session loop around it, and is where the run's mode flags take effect:

- learning      -- update Q (off under -dontlearn)
- eval_mode     -- epsilon = 0 greedy (on under -dontlearn; eval-design sec. 1)
- demo          -- print the vision cross + chosen action each move (GATE O1:
                   only under -visual on or -step-by-step)
- step_by_step  -- wait before each move (Enter in the terminal; SPACE in the
                   -visual window, handled by the gui.Presenter)

The step cap (config.evaluation.step_cap) bounds every session in every mode
(livelock guard, evaluation-design.md sec. 4). Hitting it is a *truncation*,
not a death: the final transition still bootstraps, so we never teach the
agent that surviving long leads to a dead end.

GUI note (Phase 7): -visual on builds a ``gui.Presenter`` (imported lazily, so
``-visual off`` and pytest never import pygame -- the headless-safety
invariant). The presenter owns the window/events; the runner only sends it the
board + a status line and reads back an intent (advance / quit).
"""
import random
from dataclasses import dataclass

from slither import interpreter, model_io, vision
from slither.agent import Agent
from slither.environment import Environment
from slither.progress import session as progress_session
from slither.progress import start as progress_start


@dataclass
class RunOptions:
    """The run-level parameters parsed from the CLI (not config keys)."""

    sessions: int = 1
    save: str | None = None
    load: str | None = None
    visual: bool = False
    dontlearn: bool = False
    step_by_step: bool = False
    seed: int | None = None


@dataclass
class SessionResult:
    """One game's headline facts (Phase 6's -stats aggregates these)."""

    max_length: int
    duration: int
    death_cause: str | None
    won: bool


def run(config, options, *, quiet=False, progress=False):
    """Run ``options.sessions`` games and return their results.

    Builds the shared RNG, environment and agent (loading a model first if
    asked), plays each session, prints the per-session ``Game over`` line, and
    saves at the end if asked. Returns the list of SessionResult for callers
    (Phase 6's -stats) to aggregate.

    ``quiet`` suppresses every stdout line (per-game, load, save) without
    changing the run itself -- used by ``evaluate.compare`` so a multi-model
    suite prints only its final table, not hundreds of games.

    ``progress`` (GATE H1, the hub contract) replaces the human lines with
    JSON-lines: a ``start`` event, one ``session`` event per game (the CLI adds
    the closing ``summary``). It also forces the per-move demo print off so the
    stream stays pure JSON. ``quiet`` and ``progress`` are mutually exclusive
    in practice; either one silences the human output.
    """
    rng = random.Random(options.seed)
    env = Environment(
        size=config.board.size,
        snake_length=config.board.initial_length,
        n_green_apples=config.board.green_apples,
        n_red_apples=config.board.red_apples,
    )
    quiet_human = quiet or progress
    agent, scheme, prior_curve = _build_agent(
        config, options, rng, quiet=quiet_human)

    learning = not options.dontlearn
    eval_mode = options.dontlearn
    demo = (options.visual or options.step_by_step) and not progress
    presenter = _make_presenter(config, options, progress)

    if progress:
        print(progress_start("eval" if eval_mode else "train",
                             options.sessions))

    results = []
    for i in range(options.sessions):
        result = _play_session(
            env, agent, config, rng, scheme,
            learning=learning, eval_mode=eval_mode,
            demo=demo, step_by_step=options.step_by_step,
            presenter=presenter, session_index=i,
        )
        results.append(result)
        if progress:
            print(progress_session(i, result, epsilon=agent.epsilon,
                                   sessions_trained=agent.sessions_trained))
        elif not quiet:
            print(f"Game over, max length = {result.max_length}, "
                  f"max duration = {result.duration}")
        if presenter is not None and presenter.quit:
            break               # window closed: stop cleanly, save below

    if presenter is not None:
        presenter.close()

    if options.save is not None:
        curve = prior_curve + [result.max_length for result in results]
        model_io.save(options.save, agent, config, scheme, curve=curve)
        if not quiet_human:
            print(f"Save learning state in {options.save}")

    return results


def _make_presenter(config, options, progress):
    """Build the -visual window, or None when headless.

    pygame is imported here, lazily, so ``-visual off`` and pytest never load
    it (the headless-safety invariant). ``-progress`` keeps stdout pure JSON,
    so it never opens a window either.
    """
    if not options.visual or progress:
        return None
    from slither import gui  # lazy import: pygame only under -visual on
    return gui.Presenter(
        config.board.size, config.gui.cell_px,
        config.gui.speed_ms, options.step_by_step)


def _build_agent(config, options, rng, *, quiet=False):
    """Return ``(agent, scheme)``: fresh (config scheme) or resumed (-load).

    A loaded model carries the scheme it was trained with; that scheme wins
    over the config's ``[state]`` so its state keys keep their meaning (a model
    cannot be replayed under a different alphabet). A fresh agent uses the
    config's scheme.
    """
    if options.load is None:
        return Agent(config.hyperparameters, rng), config.scheme, []
    model = model_io.load(options.load)
    if not quiet:
        print(f"Load trained model from {options.load}")
    agent = Agent(
        config.hyperparameters, rng,
        q=model.q, n=model.n, sessions_trained=model.sessions_trained,
    )
    return agent, model.scheme, list(model.curve)


def _play_session(env, agent, config, rng, scheme, *, learning, eval_mode,
                  demo, step_by_step, presenter=None, session_index=0):
    """Play one game to death / win / step cap; learn along the way.

    Carries ``(rays, state, frame)`` across iterations: the Bellman update
    prices the pair (state, action) by the state reached *after* it, and the
    frame translates the agent's relative action back to an absolute move.
    ``scheme`` selects the state alphabet used to classify each vision.

    With a ``presenter`` (visual mode) each move shows the board and waits for
    the window's intent; ``"quit"`` ends the session early (the run stops).
    """
    env.reset(rng)
    rays = vision.vision(env)
    state, frame = interpreter.state(rays, scheme)

    max_length = len(env.snake)
    steps = 0
    cap = config.evaluation.step_cap

    while steps < cap:
        action = agent.choose(state, eval_mode=eval_mode)
        absolute = frame.to_absolute(action)
        if demo:
            _print_move(rays, absolute)
        if presenter is not None:
            status = _status_fields(
                session_index, env, agent, eval_mode, steps, state, action)
            if presenter.present(env, status, absolute.value,
                                 current_state=state,
                                 qrows=_qrows(agent)) == "quit":
                return _end(agent, learning, max_length, steps, None, False)
        elif step_by_step:
            _wait_for_step()

        outcome = env.step(absolute)
        steps += 1
        max_length = max(max_length, len(env.snake))
        reward = interpreter.reward(outcome, config.rewards)
        terminal = outcome.died or outcome.won

        if terminal:
            next_state = state          # ignored: terminal target = reward
        else:
            rays = vision.vision(env)
            next_state, frame = interpreter.state(rays, scheme)

        if learning:
            agent.update(state, action, reward, next_state,
                         terminal=terminal)
        state = next_state

        if terminal:
            if presenter is not None:   # show the final frame before reset
                status = _status_fields(session_index, env, agent,
                                        eval_mode, steps, state, None)
                presenter.present(env, status, None,
                                  current_state=state, qrows=_qrows(agent))
            return _end(agent, learning, max_length, steps,
                        outcome.death_cause, outcome.won)

    # Step cap reached while still alive: a truncation, not a death.
    return _end(agent, learning, max_length, steps, None, False)


_REL_ARROWS = ("^F", "<L", ">R", "vB")     # relative action: F / L / R / B


def _status_fields(session_index, env, agent, eval_mode, steps, state, action):
    """The window's status as (label, value) pairs; gui lays out the readout.

    Structured (not a pre-joined string) so the renderer can align it into the
    quiet two-column readout of the original (DESIGN 5.1); the Presenter adds
    its own SPEED field. ``EPS`` is the agent's exploration rate (train only).
    ``STATE`` + ``ACTION`` (the relative arrow matching the Q-table's boxed
    column) show what the agent perceived and chose this step -- display only.
    """
    arrow = _REL_ARROWS[action] if action is not None else "--"
    return [
        ("SESSION", str(session_index + 1)),
        ("LENGTH", str(len(env.snake))),
        ("STEP", str(steps)),
        ("MODE", "EVAL" if eval_mode else "TRAIN"),
        ("EPS", "--" if eval_mode else f"{agent.epsilon:.2f}"),
        ("STATE", state),
        ("ACTION", arrow),
    ]


def _qrows(agent):
    """The agent's Q-table as state-sorted ``(state, qvals, nvals)`` rows for
    the watch panel -- mirrors the hub's ``viewdata.qtable_rows`` order. Pure
    data: the runner never touches a surface (gui owns rendering)."""
    rows = []
    for state in sorted(agent.q):
        qvals = agent.q[state]
        rows.append((state, qvals, agent.n.get(state, [0] * len(qvals))))
    return rows


def _end(agent, learning, max_length, steps, death_cause, won):
    """Close the session: decay epsilon (only while training) and report."""
    if learning:
        agent.end_session()
    return SessionResult(max_length, steps, death_cause, won)


def _print_move(rays, absolute):
    """Print the subject's vision cross and the chosen absolute action."""
    print(vision.format_vision(rays))
    print(f"Action: {absolute.name}")
    print()


def _wait_for_step():
    """Block until the user presses Enter (terminal step-by-step mode).

    SPACE-in-the-GUI is Phase 7; here we read stdin. On EOF (piped or
    non-interactive input) we simply proceed rather than crash or hang.
    """
    try:
        input()
    except EOFError:
        pass
