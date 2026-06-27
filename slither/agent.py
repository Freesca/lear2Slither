"""Agent: sparse Q-table, epsilon-greedy policy, Bellman update. (Phase 4)

The agent receives only ``(canonical_state, reward)`` and answers with a
column index 0-3 -- it is ignorant of boards, directions, pygame, and files,
which is the structural half of the -42 firewall (state-design.md sec. 5).
The four columns match ``interpreter.RelativeAction`` (FORWARD, LEFT, RIGHT,
BACKWARDS), but the agent never imports that enum: it stays purely numeric.

Theory (training-design.md, implementation-plan.md sec. 4.4):

- A Q-value estimates the expected discounted return of taking an action in a
  state and acting greedily afterwards. Learning nudges each value toward the
  Bellman target by a fraction ``alpha`` (the learning rate).
- Non-terminal target = ``r + gamma * max(Q[s_next])``: the immediate reward
  plus the discounted best future. Iterated, these one-step updates make a
  reward n steps away worth gamma**n -- the discount compounds across updates.
- Terminal target = ``r``: a dead/won state has no successor, so no bootstrap.
- Q-learning is off-policy -- the target uses ``max`` over the next state, not
  the action actually taken -- so exploratory (even fatal) moves do not bias
  convergence. That is why BACKWARDS = suicide is *learned*, not hard-coded.
- Exploration is epsilon-greedy with per-session decay. Epsilon is a pure
  function of ``sessions_trained``, so a loaded model recomputes it from that
  count: nothing about epsilon is stored in the model file.
"""
from dataclasses import dataclass

# One column per relative action (FORWARD, LEFT, RIGHT, BACKWARDS); mirrors
# interpreter.RelativeAction. A bare constant so the agent imports nothing.
NUM_ACTIONS = 4


@dataclass(frozen=True)
class Hyperparameters:
    """The agent's learning + exploration knobs (from config in Phase 5)."""

    alpha: float           # learning rate: fraction of the TD error applied
    gamma: float           # discount: weight on the bootstrapped future
    epsilon_start: float   # exploration rate at session 0 (before decay)
    epsilon_min: float     # floor the decay never drops below
    epsilon_decay: float   # per-session multiplier (0 < decay <= 1)


class Agent:
    """Tabular Q-learning agent keyed by canonical state strings."""

    def __init__(self, hyperparameters, rng, *,
                 q=None, n=None, sessions_trained=0):
        self.hp = hyperparameters
        self.rng = rng                       # one seeded Random for the run
        # Sparse tables: a row exists only once the agent has met that state.
        # model_io (Phase 5) passes loaded dicts here to resume training.
        self.q = q if q is not None else {}
        self.n = n if n is not None else {}
        self.sessions_trained = sessions_trained
        self.epsilon = self._epsilon_for(sessions_trained)

    # --- exploration schedule ------------------------------------------

    def _epsilon_for(self, sessions):
        """Epsilon after ``sessions`` decays. A pure function of the count, so
        loading a model reconstructs it exactly -- epsilon is never stored."""
        decayed = self.hp.epsilon_start * (self.hp.epsilon_decay ** sessions)
        return max(self.hp.epsilon_min, decayed)

    def end_session(self):
        """One game ended: advance the decay by one step. The runner calls
        this once per game-over while training (never in eval mode)."""
        self.sessions_trained += 1
        self.epsilon = self._epsilon_for(self.sessions_trained)

    # --- table access ---------------------------------------------------

    def _row(self, state):
        """``state``'s Q-row if known, else a fresh zero row (NOT stored).

        Read-only -- used for decisions and the bootstrap lookup. Zero is the
        neutral default: an unseen successor contributes 0 to the target, and
        an unseen state yields a uniform-random greedy choice via the tie
        break. Keeping it read-only is what makes eval mode side-effect free.
        """
        return self.q.get(state, [0.0] * NUM_ACTIONS)

    def _ensure(self, state):
        """Materialise ``state``'s Q-row and visit counters (zero-init) so
        they can be mutated. Called only by ``update``, so the table stays
        sparse: a stored row means the agent has actually learned that state.
        """
        if state not in self.q:
            self.q[state] = [0.0] * NUM_ACTIONS
            self.n[state] = [0] * NUM_ACTIONS

    # --- policy ---------------------------------------------------------

    def choose(self, state, *, eval_mode=False):
        """Pick a column 0-3 for ``state`` (epsilon-greedy).

        With probability epsilon, explore uniformly over all four columns
        (BACKWARDS included -- its lethality is learned, not forbidden);
        otherwise exploit the greedy action. ``eval_mode`` forces epsilon = 0
        for frozen-model evaluation (evaluation-design.md sec. 1).
        """
        epsilon = 0.0 if eval_mode else self.epsilon
        if self.rng.random() < epsilon:
            return self.rng.randrange(NUM_ACTIONS)
        return self._argmax(self._row(state))

    def _argmax(self, row):
        """Greedy column, breaking ties at random: with zero-init many values
        tie, and a naive argmax would always take column 0, skewing early
        exploration (training-design.md hygiene)."""
        best = max(row)
        winners = [i for i, value in enumerate(row) if value == best]
        return self.rng.choice(winners)

    # --- learning -------------------------------------------------------

    def update(self, s, a, r, s_next, *, terminal):
        """One Bellman step on (s, a): nudge Q[s][a] toward its target.

        Non-terminal target = r + gamma * max(Q[s_next]) (bootstrap on the
        best next value); terminal target = r (no successor -> no bootstrap,
        so ``s_next`` is ignored). Constant-alpha update; the N counters kept
        here also fund the GATE-T3 alternative alpha = 1/N(s, a) later.
        """
        self._ensure(s)
        if terminal:
            target = r
        else:
            target = r + self.hp.gamma * max(self._row(s_next))
        self.n[s][a] += 1
        self.q[s][a] += self.hp.alpha * (target - self.q[s][a])
