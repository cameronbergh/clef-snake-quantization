"""Deterministic maze-navigation environment for CLEF quantization comparisons.

Second sequential decision environment beyond Snake. Fully observable structured
state, four cardinal actions, no opponent and no real-time penalty. Generation is
deterministic and versioned from explicit seeds and parameters; exact layouts are
saved in manifests so comparisons never rely on regenerating seeds.

Model-blind: this module never loads weights, calls inference, or recommends
actions. Optimal-path computation exists for validation/reporting only and is
never exposed to the model.
"""
import hashlib
from collections import deque

GENERATOR_VERSION = "maze-v1"
DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
ACTION_ORDER = ("up", "right", "down", "left")
QUESTION = {
    "move": {
        "type": "choice",
        "instructions": (
            "Select the next move to navigate the maze from your current position "
            "to the goal. Walls block movement; moving into a wall or outside the "
            "grid leaves your position unchanged, is logged, and consumes one attempt. "
            "Reaching the goal ends the run in success. No program selects or overrides your answer."
        ),
        "criteria": {
            "up": "Move north (y - 1)",
            "down": "Move south (y + 1)",
            "left": "Move west (x - 1)",
            "right": "Move east (x + 1)",
        },
    }
}
RULES = [
    "Four cardinal moves: up, down, left, right. No diagonal moves.",
    "The full maze layout, your position and the goal are always visible.",
    "An invalid move (into a wall or outside the grid) leaves your position unchanged, is logged, and consumes one attempt.",
    "Moving onto the goal ends the run in success; the goal move counts as an attempt.",
    "Exhausting the attempt cap without reaching the goal ends the run as a capped failure.",
]


def _keystream(seed_tag):
    """Infinite SHA256 counter-mode stream of 64-bit ints; stable across runs/platforms."""
    counter = 0
    while True:
        digest = hashlib.sha256(f"clef-maze-gen-v1:{seed_tag}:{counter}".encode("ascii")).digest()
        for offset in range(0, 32, 8):
            yield int.from_bytes(digest[offset:offset + 8], "big")
        counter += 1


def _shuffled(items, stream):
    """Fisher-Yates shuffle driven by the keyed stream; returns a new list."""
    items = list(items)
    for index in range(len(items) - 1, 0, -1):
        other = next(stream) % (index + 1)
        items[index], items[other] = items[other], items[index]
    return items


def _interior_edges(width, height):
    """All wall segments between orthogonally adjacent cells, canonical order."""
    edges = []
    for y in range(height):
        for x in range(width):
            if x + 1 < width:
                edges.append((x, y, x + 1, y))
            if y + 1 < height:
                edges.append((x, y, x, y + 1))
    return edges


def _carve_spanning_tree(width, height, stream):
    """Randomized iterative DFS; returns the set of carved (removed) edges."""
    start = (next(stream) % width, next(stream) % height)
    visited = {start}
    carved = set()
    stack = [start]
    while stack:
        x, y = stack[-1]
        options = []
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nb = (x + dx, y + dy)
            if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in visited:
                options.append(nb)
        if not options:
            stack.pop()
            continue
        nb = _shuffled(options, stream)[0]
        carved.add((min(x, nb[0]), min(y, nb[1]), max(x, nb[0]), max(y, nb[1])))
        visited.add(nb)
        stack.append(nb)
    return carved


def bfs_distances(start, walls, width, height):
    """Shortest-path distances from start over open edges; validation/reporting only."""
    wall_set = {tuple(w) for w in walls}
    dist = {tuple(start): 0}
    pending = deque([tuple(start)])
    while pending:
        x, y = pending.popleft()
        for dx, dy in DIRECTIONS.values():
            nb = (x + dx, y + dy)
            if not (0 <= nb[0] < width and 0 <= nb[1] < height):
                continue
            edge = (min(x, nb[0]), min(y, nb[1]), max(x, nb[0]), max(y, nb[1]))
            if edge in wall_set or nb in dist:
                continue
            dist[nb] = dist[(x, y)] + 1
            pending.append(nb)
    return dist


def bfs_path(start, goal, walls, width, height):
    """One shortest path from start to goal; test fixture / reporting only."""
    wall_set = {tuple(w) for w in walls}
    start, goal = tuple(start), tuple(goal)
    prev = {start: None}
    pending = deque([start])
    while pending:
        node = pending.popleft()
        if node == goal:
            break
        x, y = node
        for dx, dy in DIRECTIONS.values():
            nb = (x + dx, y + dy)
            if not (0 <= nb[0] < width and 0 <= nb[1] < height) or nb in prev:
                continue
            if (min(x, nb[0]), min(y, nb[1]), max(x, nb[0]), max(y, nb[1])) in wall_set:
                continue
            prev[nb] = node
            pending.append(nb)
    if goal not in prev:
        return None
    path, node = [], goal
    while node is not None:
        path.append([node[0], node[1]])
        node = prev[node]
    return path[::-1]


def is_wall_between(cell_a, cell_b, walls):
    ax, ay = cell_a
    bx, by = cell_b
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)) in {tuple(w) for w in walls}


def generate_layout(seed, width=9, height=9, loop_fraction=0.08, target_distance="far"):
    """Deterministic maze layout. Returns a JSON-serializable manifest entry.

    target_distance: "far" (default) picks a farthest-from-start cell as goal;
    an int picks the reachable cell whose shortest-path distance is closest to it.
    """
    seed = str(seed)
    if width < 2 or height < 2:
        raise ValueError("Maze must be at least 2x2")
    if not 0.0 <= loop_fraction < 1.0:
        raise ValueError("loop_fraction must be in [0, 1)")
    stream = _keystream(seed)
    carved = _carve_spanning_tree(width, height, stream)
    remaining = [e for e in _interior_edges(width, height) if e not in carved]
    extra = _shuffled(remaining, stream)[: round(loop_fraction * len(remaining))]
    walls = sorted(set(_interior_edges(width, height)) - carved - set(extra))
    walls = [list(w) for w in walls]

    start = [next(stream) % width, next(stream) % height]
    distances = bfs_distances(start, walls, width, height)
    assert len(distances) == width * height, "Generator must keep every cell reachable"
    order = _shuffled(sorted(distances), stream)
    if target_distance == "far":
        goal = list(max(order, key=lambda cell: distances[cell]))
    else:
        target = int(target_distance)
        if target < 1:
            raise ValueError("target_distance must be 'far' or a positive int")
        rank = {cell: i for i, cell in enumerate(order)}
        goal = list(min(order, key=lambda cell: (abs(distances[cell] - target), rank[cell])))
    shortest = distances[tuple(goal)]
    return {
        "maze_id": None,  # assigned by the manifest writer
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "width": width,
        "height": height,
        "loop_fraction": loop_fraction,
        "target_distance": target_distance,
        "walls": walls,
        "start": start,
        "goal": goal,
        "shortest_path_length": shortest,
    }


class Maze:
    """One maze episode. Model-blind; transitions depend only on recorded actions."""

    def __init__(self, layout, attempt_cap):
        for key in ("width", "height", "walls", "start", "goal"):
            if key not in layout:
                raise ValueError(f"Layout missing required key: {key}")
        if attempt_cap < 1:
            raise ValueError("attempt_cap must be positive")
        self.layout = layout
        self.width = layout["width"]
        self.height = layout["height"]
        self.walls = {tuple(w) for w in layout["walls"]}
        self.start = list(layout["start"])
        self.goal = list(layout["goal"])
        self.attempt_cap = attempt_cap
        self.position = list(self.start)
        self.attempts = 0
        self.invalid_moves = 0
        self.done = False
        self.reached = False
        self.history = []

    def _analyze(self, action):
        dx, dy = DIRECTIONS[action]
        x, y = self.position
        target = [x + dx, y + dy]
        if not (0 <= target[0] < self.width and 0 <= target[1] < self.height):
            return {"valid": False, "reason": "bounds", "next_position": [x, y]}
        if (min(x, target[0]), min(y, target[1]), max(x, target[0]), max(y, target[1])) in self.walls:
            return {"valid": False, "reason": "wall", "next_position": [x, y]}
        return {"valid": True, "reason": "open", "next_position": target}

    def request(self):
        """Deterministic structured model request; fixed schema across conditions."""
        analysis = {name: self._analyze(name) for name in DIRECTIONS}
        state = {
            "grid": {
                "width": self.width,
                "height": self.height,
                "coordinates": "x increases right, y increases down",
                "walls": sorted([list(w) for w in self.walls]),
                "start": self.start[:],
                "goal": self.goal[:],
            },
            "position": self.position[:],
            "attempts_used": self.attempts,
            "attempt_cap": self.attempt_cap,
            "invalid_moves": self.invalid_moves,
            "move_analysis": analysis,
            "rules": list(RULES),
        }
        return {"model": "clef-flash", "state": state, "questions": QUESTION}

    def apply(self, action):
        """Apply one action. Invalid moves hold position, are logged, consume an attempt."""
        if self.done:
            raise ValueError("Cannot act on a terminal maze")
        if action not in DIRECTIONS:
            raise ValueError(f"Unknown action: {action!r}")
        info = self._analyze(action)
        before = self.position[:]
        self.attempts += 1
        if info["valid"]:
            self.position = info["next_position"]
        else:
            self.invalid_moves += 1
        self.history.append({"attempt": self.attempts, "action": action,
                             "valid": info["valid"], "reason": info["reason"],
                             "from": before, "position": self.position[:]})
        if self.position == self.goal:
            self.done = True
            self.reached = True
        elif self.attempts >= self.attempt_cap:
            self.done = True

    def terminal_reason(self):
        if not self.done:
            return None
        return "goal" if self.reached else "capped"

    def snapshot(self):
        return {"position": self.position[:], "attempts": self.attempts,
                "invalid_moves": self.invalid_moves, "done": self.done,
                "reached": self.reached, "terminal_reason": self.terminal_reason()}
