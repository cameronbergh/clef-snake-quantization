"""Scripted policies for offline validation and demos.

Test fixtures only: these are never model fallbacks, never recommendations to a
model, and never mixed into model experiment results. Each policy is
deterministic given its seed.
"""
import hashlib
from .maze import DIRECTIONS, ACTION_ORDER, bfs_path


def _stream(tag):
    counter = 0
    while True:
        digest = hashlib.sha256(f"clef-maze-policy-v1:{tag}:{counter}".encode("ascii")).digest()
        for offset in range(0, 32, 8):
            yield int.from_bytes(digest[offset:offset + 8], "big")
        counter += 1


class RandomPolicy:
    """Uniform random action; exercises invalid-move and cap handling."""

    name = "random"

    def __init__(self, seed="default"):
        self.stream = _stream(f"random:{seed}")

    def reset(self):
        pass

    def __call__(self, maze):
        return ACTION_ORDER[next(self.stream) % 4]


class GreedyPolicy:
    """Move minimizing Manhattan distance to the goal (ties: fixed action order).

    Uses only the visible state any policy could see; still myopic (walks into
    dead ends). A fixture for generating non-trivial trajectories, not a solver.
    """

    name = "greedy"

    def reset(self):
        pass

    def __call__(self, maze):
        gx, gy = maze.goal
        best, best_key = None, None
        for action in ACTION_ORDER:
            info = maze._analyze(action)
            x, y = info["next_position"]
            key = (abs(gx - x) + abs(gy - y), ACTION_ORDER.index(action))
            if best_key is None or key < best_key:
                best, best_key = action, key
        return best


class WallFollower:
    """Left-hand-rule wall follower with heading memory."""

    name = "wallfollower"
    _left = {"up": "left", "left": "down", "down": "right", "right": "up"}
    _right = {"up": "right", "right": "down", "down": "left", "left": "up"}
    _back = {"up": "down", "down": "up", "left": "right", "right": "left"}

    def __init__(self):
        self.heading = "up"

    def reset(self):
        self.heading = "up"

    def __call__(self, maze):
        for candidate in (self._left[self.heading], self.heading,
                          self._right[self.heading], self._back[self.heading]):
            if maze._analyze(candidate)["valid"]:
                self.heading = candidate
                return candidate
        return self.heading  # fully boxed in; the invalid move is logged


class OptimalPolicy:
    """First step of a BFS shortest path. TEST FIXTURE ONLY.

    Never a model fallback, never exposed to a model. Used to validate the
    generator's recorded shortest_path_length and the replay auditor.
    """

    name = "optimal"

    def reset(self):
        pass

    def __call__(self, maze):
        layout = maze.layout
        path = bfs_path(maze.position, maze.goal, layout["walls"], layout["width"], layout["height"])
        assert path is not None and len(path) >= 2, "Goal must be reachable in fixtures"
        x, y = maze.position
        nx, ny = path[1]
        for action, (dx, dy) in DIRECTIONS.items():
            if [x + dx, y + dy] == [nx, ny]:
                return action
        raise AssertionError("BFS step is not a cardinal move")


POLICIES = {"random": RandomPolicy, "greedy": GreedyPolicy,
            "wallfollower": WallFollower, "optimal": OptimalPolicy}


def make_policy(name, seed="default"):
    if name not in POLICIES:
        raise ValueError(f"Unknown policy: {name!r}")
    if name == "random":
        return RandomPolicy(seed)
    return POLICIES[name]()
