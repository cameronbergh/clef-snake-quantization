import unittest
from clef_maze.maze import generate_layout, bfs_distances, bfs_path, GENERATOR_VERSION


def open_neighbors(cell, walls, width, height):
    x, y = cell
    out = []
    wall_set = {tuple(w) for w in walls}
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nb = (x + dx, y + dy)
        if not (0 <= nb[0] < width and 0 <= nb[1] < height):
            continue
        if (min(x, nb[0]), min(y, nb[1]), max(x, nb[0]), max(y, nb[1])) not in wall_set:
            out.append(nb)
    return out


class GenerationTests(unittest.TestCase):
    def test_deterministic_and_versioned(self):
        first = generate_layout("seed-a", 7, 7, 0.1, "far")
        second = generate_layout("seed-a", 7, 7, 0.1, "far")
        self.assertEqual(first, second)
        self.assertEqual(first["generator"], GENERATOR_VERSION)

    def test_distinct_seeds_differ(self):
        a = generate_layout("seed-a", 7, 7, 0.1, "far")
        b = generate_layout("seed-b", 7, 7, 0.1, "far")
        self.assertNotEqual(a["walls"], b["walls"])

    def test_all_cells_reachable(self):
        layout = generate_layout("reach", 9, 9, 0.15, "far")
        dist = bfs_distances(layout["start"], layout["walls"], 9, 9)
        self.assertEqual(len(dist), 81)

    def test_perfect_maze_has_dead_ends(self):
        layout = generate_layout("deadend", 7, 7, 0.0, "far")
        leaves = [c for c in [(x, y) for y in range(7) for x in range(7)]
                  if len(open_neighbors(c, layout["walls"], 7, 7)) == 1]
        self.assertGreaterEqual(len(leaves), 2)

    def test_loop_fraction_removes_walls(self):
        plain = generate_layout("loops", 9, 9, 0.0, "far")
        loopy = generate_layout("loops", 9, 9, 0.3, "far")
        self.assertGreater(len(plain["walls"]), len(loopy["walls"]))
        # Reachability is preserved with loops.
        self.assertEqual(len(bfs_distances(loopy["start"], loopy["walls"], 9, 9)), 81)

    def test_target_distance_honored(self):
        layout = generate_layout("dist", 9, 9, 0.1, 10)
        self.assertEqual(layout["shortest_path_length"],
                         len(bfs_path(layout["start"], layout["goal"], layout["walls"], 9, 9)) - 1)
        self.assertLessEqual(abs(layout["shortest_path_length"] - 10), 4)

    def test_far_goal_is_maximal(self):
        layout = generate_layout("farmax", 7, 7, 0.0, "far")
        dist = bfs_distances(layout["start"], layout["walls"], 7, 7)
        self.assertEqual(layout["shortest_path_length"], max(dist.values()))

    def test_minimum_size_and_bad_params(self):
        layout = generate_layout("tiny", 2, 2, 0.0, "far")
        self.assertEqual(len(bfs_distances(layout["start"], layout["walls"], 2, 2)), 4)
        with self.assertRaises(ValueError):
            generate_layout("bad", 1, 5, 0.0, "far")
        with self.assertRaises(ValueError):
            generate_layout("bad", 5, 5, 1.0, "far")
        with self.assertRaises(ValueError):
            generate_layout("bad", 5, 5, 0.0, 0)

    def test_walls_canonical_json(self):
        layout = generate_layout("canon", 7, 7, 0.1, "far")
        walls = layout["walls"]
        self.assertEqual(walls, sorted(walls))
        for w in walls:
            self.assertEqual(len(w), 4)
            self.assertLess((w[0], w[1]), (w[2], w[3]))


if __name__ == "__main__":
    unittest.main()
