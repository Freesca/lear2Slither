"""Environment: board rules, spawns, step(), game-over. (Phase 1)

Knows nothing about vision, rewards, or Q-values. It is a pure, silent
world simulator: it owns the board and the rules and does no I/O. See
docs/environment-design.md for the design decisions behind this module.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    """The four absolute moves.

    The value is the ``(row, col)`` delta on the board, with row
    increasing downward (environment-design.md §1).
    """

    UP = (-1, 0)
    LEFT = (0, -1)
    DOWN = (1, 0)
    RIGHT = (0, 1)


@dataclass
class StepOutcome:
    """Facts about one step, for the Interpreter to price into a reward.

    Events only -- no vision, no reward, no board. ``death_cause`` is one
    of ``"wall"``, ``"self"``, ``"length"`` (None while alive). ``won``
    flags the GATE E2 victory: a green apple was eaten but the board is
    now full, so there is nowhere left to respawn.
    """

    ate_green: bool = False
    ate_red: bool = False
    died: bool = False
    death_cause: str | None = None
    won: bool = False


class Environment:
    def __init__(
        self, size=10, snake_length=3, n_green_apples=2, n_red_apples=1
    ):
        self.size = size
        self.snake_length = snake_length
        self.n_green_apples = n_green_apples
        self.n_red_apples = n_red_apples
        self.snake: deque[tuple[int, int]] = deque()
        self.green_apples: set[tuple[int, int]] = set()
        self.red_apples: set[tuple[int, int]] = set()

    def reset(self, rng) -> None:
        self.rng = rng
        self.snake.clear()
        self.green_apples.clear()
        self.red_apples.clear()

        self._spawn_snake()
        self._spawn_apples()

    def step(self, direction: Direction) -> StepOutcome:
        """Advance one tick in ``direction``; mutate the board in place.

        Check order (environment-design.md §5): wall, then self-collision
        honouring the tail-vacating rule (GATE E1), then the move itself
        with growth / shrink and the length-0 and E2 terminal cases.
        """
        dr, dc = direction.value
        hr, hc = self.snake[0]
        new_head = (hr + dr, hc + dc)

        if not self._in_bounds(new_head):
            return StepOutcome(died=True, death_cause="wall")

        ate_green = new_head in self.green_apples
        ate_red = new_head in self.red_apples

        # Self-collision. When the snake does not grow, the tail vacates
        # this tick, so the cell it leaves is legal to enter (GATE E1).
        body = set(self.snake)
        if not ate_green:
            body.discard(self.snake[-1])
        if new_head in body:
            return StepOutcome(died=True, death_cause="self")

        self.snake.appendleft(new_head)

        if ate_green:
            return self._eat_green(new_head)
        if ate_red:
            return self._eat_red(new_head)

        self.snake.pop()  # ordinary move: head advances, tail follows
        return StepOutcome()

    def _eat_green(self, head: tuple[int, int]) -> StepOutcome:
        # Grew: the tail stayed, so length += 1. Respawn the green apple;
        # a full board (no free cell) is the GATE E2 victory.
        self.green_apples.discard(head)
        cell = self._random_free_cell()
        if cell is None:
            return StepOutcome(ate_green=True, won=True)
        self.green_apples.add(cell)
        return StepOutcome(ate_green=True)

    def _eat_red(self, head: tuple[int, int]) -> StepOutcome:
        # length -= 1: the ordinary tail-vacate plus one extra segment.
        # After appendleft the deque holds >= 2 cells, so both pops are
        # safe; emptying it means length hit 0 -> game over.
        self.red_apples.discard(head)
        self.snake.pop()
        self.snake.pop()
        if not self.snake:
            return StepOutcome(
                ate_red=True, died=True, death_cause="length"
            )
        cell = self._random_free_cell()
        if cell is not None:
            self.red_apples.add(cell)
        return StepOutcome(ate_red=True)

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.size and 0 <= c < self.size

    def _random_free_cell(self):
        occupied = set(self.snake) | self.green_apples | self.red_apples
        free = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in occupied
        ]
        if not free:
            return None
        return self.rng.choice(free)

    def _free_cell_or_raise(self) -> tuple[int, int]:
        cell = self._random_free_cell()
        if cell is None:
            raise ValueError("no free cell available at reset")
        return cell

    def _random_adjacent_cell(self, r: int, c: int):
        adjacent = [
            (r + dr, c + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if self._in_bounds((r + dr, c + dc))
            and (r + dr, c + dc) not in self.snake
        ]
        if not adjacent:
            return None
        return self.rng.choice(adjacent)

    def _spawn_snake(self) -> None:
        cell = self._free_cell_or_raise()
        self.snake.append(cell)
        for _ in range(1, self.snake_length):
            cell = self._random_adjacent_cell(*cell)
            if cell is None:
                raise ValueError("snake start trapped: board too small")
            self.snake.append(cell)

    def _spawn_apples(self) -> None:
        while len(self.green_apples) < self.n_green_apples:
            self.green_apples.add(self._free_cell_or_raise())
        while len(self.red_apples) < self.n_red_apples:
            self.red_apples.add(self._free_cell_or_raise())
