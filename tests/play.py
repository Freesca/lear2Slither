"""Phase-1/4 dev harness: drive the snake by hand AND watch the agent learn.

A *dev tool*, not part of the product: it lives in ``tests/`` but is **not** a
pytest test (its name is not ``test_*``, so pytest never collects it). Launch
it by hand from the repo root (``python -m tests.play``); it is never imported
by ``-visual off`` runs or by pytest, so it may import pygame even though
gui.py is the product's sole pygame importer.

It steps the Environment one move at a time -- by your key (arrows/WASD) or by
the agent (SPACE) -- and for **every** move it runs the real Bellman update and
prints a learning trace: the canonical state and frame, the agent's current
greedy preference, the Q-row before, the reward and target, and the Q-row
after. The single ``Agent`` persists across deaths and restarts, so you can
watch Q-values grow and epsilon decay over many hand-played sessions.

    python -m tests.play [--seed N] [--size N] [--cell PX]

Controls: arrows/WASD = you move, SPACE = agent moves, R = restart, Esc = quit.
"""
import argparse
import random

import pygame

from slither import gui, vision, interpreter
from slither.agent import Agent, Hyperparameters
from slither.environment import Direction, Environment, StepOutcome

KEYMAP = {
    pygame.K_UP: Direction.UP,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_w: Direction.UP,
    pygame.K_s: Direction.DOWN,
    pygame.K_a: Direction.LEFT,
    pygame.K_d: Direction.RIGHT,
}

# Provisional agent settings -- mirror configs/default.toml until the config
# loader (Phase 5) exists. These values are NOT decided (gates T1-T4); they are
# fine for a dev harness whose point is to watch the mechanism, not to tune it.
REWARDS = interpreter.Rewards(
    green=20.0, red=-10.0, step=-1.0, death=-100.0, win=100.0,
)
HYPERPARAMS = Hyperparameters(
    alpha=0.1, gamma=0.9,
    epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99,
)


def _read_state(env):
    """The canonical state + frame the agent currently perceives."""
    return interpreter.state(vision.vision(env))


def _format_qrow(row):
    """The four Q-values as F/L/R/B cells, starring the argmax (ties too)."""
    best = max(row)
    cells = []
    for label, value in zip("FLRB", row):
        star = "*" if value == best else " "
        cells.append(f"{label}:{value:+6.2f}{star}")
    return " ".join(cells)


def _to_relative(frame, direction):
    """Invert the frame: the relative column whose absolute move is
    ``direction`` (``to_absolute`` is a bijection over the 4 directions)."""
    for action in range(4):
        if frame.to_absolute(action) == direction:
            return action
    raise ValueError(direction)  # unreachable: every direction is covered


def _status(env, outcome, agent, last_dir):
    length = len(env.snake)
    move = last_dir.name if last_dir else "-"
    tail = f"s{agent.sessions_trained} eps{agent.epsilon:.2f} move:{move}"
    if outcome.won:
        return f"YOU WIN - len {length} - {tail} - press R"
    if outcome.died:
        return (f"GAME OVER ({outcome.death_cause}) - len {length}"
                f" - {tail} - press R")
    return f"len {length} - {tail}"


def _step(agent, env, state_key, frame, a_rel, a_abs, source):
    """Apply a chosen action, run the Q-update, print a learning trace.

    Returns ``(outcome, next_state_key, next_frame)``. The action is already
    decided (the caller picked it, human or agent); this is the shared
    "act + learn + show" path so both input sources behave identically.
    """
    print("=" * 60)
    print(vision.format_vision(vision.vision(env)))
    before = list(agent.q.get(state_key, [0.0] * 4))
    greedy = interpreter.RelativeAction(
        agent.choose(state_key, eval_mode=True)).name
    print(f"state {state_key!r}  back={frame.back.name} "
          f"mirror={frame.mirror}  eps={agent.epsilon:.3f}")
    print(f"  Q before  {_format_qrow(before)}  greedy={greedy}")

    outcome = env.step(a_abs)
    r = interpreter.reward(outcome, REWARDS)
    terminal = outcome.died or outcome.won

    next_key, next_frame = state_key, frame
    if terminal:
        target = r
    else:
        next_key, next_frame = _read_state(env)
        target = r + HYPERPARAMS.gamma * max(agent.q.get(next_key, [0.0] * 4))

    agent.update(state_key, a_rel, r, next_key, terminal=terminal)
    after = agent.q[state_key]

    rel = interpreter.RelativeAction(a_rel).name
    print(f"move {rel} (abs {a_abs.name}) by {source}")
    print(f"  reward {r:+.2f}  terminal={terminal}  target={target:+.2f}")
    print(f"  Q after   {_format_qrow(after)}  "
          f"Q[{rel[:1]}]: {before[a_rel]:+.2f} -> {after[a_rel]:+.2f}")
    print(f"  len={len(env.snake)} green={int(outcome.ate_green)} "
          f"red={int(outcome.ate_red)} died={int(outcome.died)}"
          f"({outcome.death_cause}) won={int(outcome.won)}")
    return outcome, next_key, next_frame


def _after_step(agent, outcome):
    """End-of-step bookkeeping: on game-over close the session (decay eps)."""
    if outcome.died or outcome.won:
        agent.end_session()
        print(f"--- session {agent.sessions_trained} over,"
              f" eps now {agent.epsilon:.3f} ---")
        return False
    return True


def run(size=10, cell_px=40, seed=None):
    screen = gui.create_window(size, cell_px)
    font = gui.make_font()
    qfont = gui.make_qfont()
    clock = pygame.time.Clock()

    rng = random.Random(seed)
    env = Environment(size=size)
    env.reset(rng)
    agent = Agent(HYPERPARAMS, rng)        # one agent, persists across games
    state_key, frame = _read_state(env)
    outcome = StepOutcome()
    alive = True
    tab = "board"                          # TAB switches to "qtable"
    last_dir = None                        # last absolute move, for the arrow

    print("controls: arrows/WASD = you move, SPACE = agent moves, "
          "TAB = board/q-table, R = restart, Esc = quit")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    tab = "qtable" if tab == "board" else "board"
                elif event.key == pygame.K_r:
                    env.reset(rng)
                    state_key, frame = _read_state(env)
                    outcome = StepOutcome()
                    alive = True
                    last_dir = None
                elif alive and event.key == pygame.K_SPACE:
                    a_rel = agent.choose(state_key)      # explores via eps
                    a_abs = frame.to_absolute(a_rel)
                    outcome, state_key, frame = _step(
                        agent, env, state_key, frame, a_rel, a_abs, "AGENT")
                    last_dir = a_abs
                    alive = _after_step(agent, outcome)
                elif alive and event.key in KEYMAP:
                    a_abs = KEYMAP[event.key]
                    a_rel = _to_relative(frame, a_abs)
                    outcome, state_key, frame = _step(
                        agent, env, state_key, frame, a_rel, a_abs, "HUMAN")
                    last_dir = a_abs
                    alive = _after_step(agent, outcome)
        if tab == "qtable":
            rows = [(k, agent.q[k], agent.n[k]) for k in sorted(agent.q)]
            gui.render_qtable(screen, qfont, rows, state_key)
        else:
            status = _status(env, outcome, agent, last_dir)
            gui.render(screen, env, font, status, cell_px)
            gui.draw_action(screen, env,
                            last_dir.value if last_dir else None, cell_px)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 1/4 dev harness")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--cell", type=int, default=40)
    args = parser.parse_args(argv)
    run(size=args.size, cell_px=args.cell, seed=args.seed)


if __name__ == "__main__":
    main()
