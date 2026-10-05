"""Independent Python implementation of the documented experiment protocol.

The original browser UI is not redistributed. Requests are verified against all
published browser-run records by the offline audit, including factual features.
"""
from collections import deque
from functools import lru_cache
import hashlib

GRID = 12
DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
QUESTION = {
    "move": {
        "type": "choice",
        "instructions": "Select the next move for the snake head. Avoid collisions and try to reach the food while preserving open space. No program selects or overrides your answer.",
        "criteria": {"up": "Move north (y - 1)", "down": "Move south (y + 1)", "left": "Move west (x - 1)", "right": "Move east (x + 1)"},
    }
}


@lru_cache(maxsize=8192)
def priorities(seed, event):
    """SHA256-keyed Fisher-Yates; modulo mapping exactly matches saved protocol."""
    cells = list(range(GRID * GRID))
    for index in range(len(cells) - 1, 0, -1):
        raw = hashlib.sha256(f"clef-snake-food-v1:{seed}:{event}:{index}".encode("ascii")).digest()
        other = int.from_bytes(raw, "big") % (index + 1)
        cells[index], cells[other] = cells[other], cells[index]
    return tuple(cells)


def reachable(start, obstacles):
    if start in obstacles or not all(0 <= c < GRID for c in start):
        return 0
    visited = {start}
    pending = deque([start])
    while pending:
        x, y = pending.popleft()
        for dx, dy in DIRECTIONS.values():
            point = (x + dx, y + dy)
            if all(0 <= c < GRID for c in point) and point not in visited and point not in obstacles:
                visited.add(point)
                pending.append(point)
    return len(visited)


class Game:
    def __init__(self, seed):
        self.seed = seed
        self.body = [[5, 6], [4, 6], [3, 6], [2, 6]]
        self.direction = "right"
        self.score = self.steps = 0
        self.alive = True
        self.food_events = []
        self.food = self.spawn()

    def spawn(self):
        occupied = {y * GRID + x for x, y in self.body}
        order = priorities(self.seed, self.score)
        rank = next((i for i, cell in enumerate(order) if cell not in occupied), -1)
        cell = order[rank] if rank >= 0 else 0
        chosen = [cell % GRID, cell // GRID]
        self.food_events.append({"event": self.score, "rank": rank, "selected": chosen, "occupied": sorted(occupied)})
        return chosen

    def request(self):
        x, y = self.body[0]
        fx, fy = self.food
        dx, dy = fx - x, fy - y
        obstacles = {tuple(p) for p in self.body[:-1]}
        neck = self.body[1] if len(self.body) > 1 else [-1, -1]
        safe = []
        analysis = {}
        for name, (vx, vy) in DIRECTIONS.items():
            target = (x + vx, y + vy)
            reason = "safe"
            if list(target) == neck:
                reason = "neck"
            elif not all(0 <= c < GRID for c in target):
                reason = "wall"
            elif target in obstacles:
                reason = "body"
            is_safe = reason == "safe"
            distance = abs(fx - target[0]) + abs(fy - target[1])
            if is_safe:
                safe.append(name)
            analysis[name] = {"safe": is_safe, "reason": reason, "manhattan_to_food": distance,
                              "open_space": reachable(target, obstacles) if is_safe else 0,
                              "reduces_distance": distance < abs(dx) + abs(dy)}
        state = {
            "grid": {"width": GRID, "height": GRID, "coordinates": "x increases right, y increases down"},
            "body": [p[:] for p in self.body], "direction": self.direction, "head": [x, y], "food": self.food[:],
            "food_delta": {"dx": dx, "dy": dy,
                           "preferred_horizontal": "right" if dx > 0 else "left" if dx < 0 else "aligned",
                           "preferred_vertical": "down" if dy > 0 else "up" if dy < 0 else "aligned"},
            "safe_moves": safe, "move_analysis": analysis,
        }
        return {"model": "clef-flash", "state": state, "questions": QUESTION}

    def apply(self, move):
        if not self.alive:
            raise ValueError("Cannot move a terminal game")
        vx, vy = DIRECTIONS[move]
        target = [self.body[0][0] + vx, self.body[0][1] + vy]
        if not all(0 <= c < GRID for c in target) or target in self.body[:-1]:
            self.alive = False
            return
        self.body = [target] + self.body
        self.direction = move
        if target == self.food:
            self.score += 1
            self.food = self.spawn()
        else:
            self.body.pop()
        self.steps += 1

    def snapshot(self):
        return {"snake": [p[:] for p in self.body], "food": self.food[:], "score": self.score,
                "steps": self.steps, "alive": self.alive}
