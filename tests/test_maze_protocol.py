import unittest
from clef_maze.maze import Maze, DIRECTIONS


def open3x3(**over):
    layout = {"maze_id": "t", "generator": "maze-v1", "seed": "s", "width": 3, "height": 3,
              "loop_fraction": 0.0, "target_distance": "far", "walls": [],
              "start": [0, 0], "goal": [2, 2], "shortest_path_length": 4}
    layout.update(over)
    return layout


class ProtocolTests(unittest.TestCase):
    def test_invalid_wall_move_holds_position_and_consumes_attempt(self):
        maze = Maze(open3x3(walls=[[0, 0, 1, 0]]), attempt_cap=10)
        maze.apply("right")
        self.assertEqual(maze.position, [0, 0])
        self.assertEqual((maze.attempts, maze.invalid_moves), (1, 1))
        self.assertFalse(maze.done)
        self.assertEqual(maze.history[-1]["reason"], "wall")

    def test_invalid_bounds_move_at_corner(self):
        maze = Maze(open3x3(), attempt_cap=10)
        maze.apply("up")
        maze.apply("left")
        self.assertEqual(maze.position, [0, 0])
        self.assertEqual(maze.invalid_moves, 2)
        self.assertEqual(maze.history[-1]["reason"], "bounds")

    def test_valid_move_updates_position(self):
        maze = Maze(open3x3(), attempt_cap=10)
        maze.apply("right")
        self.assertEqual(maze.position, [1, 0])
        self.assertEqual(maze.invalid_moves, 0)

    def test_goal_terminal_and_counts_attempt(self):
        maze = Maze(open3x3(), attempt_cap=10)
        for action in ("right", "right", "down", "down"):
            maze.apply(action)
        self.assertTrue(maze.done and maze.reached)
        self.assertEqual(maze.terminal_reason(), "goal")
        self.assertEqual(maze.attempts, 4)
        snap = maze.snapshot()
        self.assertEqual(snap["position"], [2, 2])

    def test_cap_terminal(self):
        maze = Maze(open3x3(), attempt_cap=2)
        maze.apply("up")
        maze.apply("up")
        self.assertTrue(maze.done)
        self.assertFalse(maze.reached)
        self.assertEqual(maze.terminal_reason(), "capped")

    def test_goal_on_final_attempt_counts_as_success(self):
        layout = open3x3(width=2, height=1, start=[0, 0], goal=[1, 0], shortest_path_length=1)
        maze = Maze(layout, attempt_cap=1)
        maze.apply("right")
        self.assertTrue(maze.reached)
        self.assertEqual(maze.terminal_reason(), "goal")

    def test_apply_after_terminal_raises(self):
        maze = Maze(open3x3(), attempt_cap=1)
        maze.apply("up")
        with self.assertRaises(ValueError):
            maze.apply("right")

    def test_unknown_action_raises(self):
        maze = Maze(open3x3(), attempt_cap=10)
        with self.assertRaises(ValueError):
            maze.apply("diagonal")

    def test_paired_trajectories_are_isolated(self):
        layout = open3x3()
        first, second = Maze(layout, 10), Maze(layout, 10)
        first.apply("right")
        second.apply("down")
        second.apply("down")
        self.assertEqual(first.position, [1, 0])
        self.assertEqual(second.position, [0, 2])
        self.assertEqual(first.attempts, 1)
        self.assertEqual(second.attempts, 2)

    def test_request_schema_fixed_and_deterministic(self):
        maze = Maze(open3x3(), attempt_cap=10)
        first, second = maze.request(), maze.request()
        self.assertEqual(first, second)
        self.assertEqual(first["model"], "clef-flash")
        self.assertEqual(first["questions"]["move"]["type"], "choice")
        self.assertEqual(set(first["state"]["move_analysis"]), set(DIRECTIONS))
        self.assertIn("walls", first["state"]["grid"])
        self.assertNotIn("distance", str(first["state"]))

    def test_move_analysis_is_factual(self):
        maze = Maze(open3x3(walls=[[0, 0, 1, 0]]), attempt_cap=10)
        analysis = maze.request()["state"]["move_analysis"]
        self.assertEqual(analysis["right"], {"valid": False, "reason": "wall", "next_position": [0, 0]})
        self.assertEqual(analysis["up"]["reason"], "bounds")
        self.assertEqual(analysis["down"], {"valid": True, "reason": "open", "next_position": [0, 1]})


if __name__ == "__main__":
    unittest.main()
