import unittest
from clef_snake.catalog import OFFICIAL_REPO, OFFICIAL_REV
from clef_maze.adapter import build_request, choice
from clef_maze.maze import Maze


def layout():
    return {"maze_id": "t", "generator": "maze-v1", "seed": "s", "width": 3, "height": 3,
            "loop_fraction": 0.0, "target_distance": "far", "walls": [],
            "start": [0, 0], "goal": [2, 2], "shortest_path_length": 4}


def good_response(choice_name="right"):
    probs = {"up": 0.1, "down": 0.2, "left": 0.05, "right": 0.65}
    return {
        "provenance": {"repo": OFFICIAL_REPO, "revision": OFFICIAL_REV,
                       "fallback": False, "safety_override": False},
        "answers": {"move": {"choice": choice_name, "probabilities": dict(probs)}},
    }


class AdapterTests(unittest.TestCase):
    def test_build_request_matches_environment(self):
        maze = Maze(layout(), attempt_cap=10)
        self.assertEqual(build_request(maze), maze.request())

    def test_choice_accepts_argmax(self):
        self.assertEqual(choice(good_response("right")), "right")

    def test_choice_rejects_wrong_provenance(self):
        bad = good_response()
        bad["provenance"]["repo"] = "someone/else"
        with self.assertRaises(AssertionError):
            choice(bad)

    def test_choice_rejects_fallback(self):
        bad = good_response()
        bad["provenance"]["fallback"] = True
        with self.assertRaises(AssertionError):
            choice(bad)

    def test_choice_rejects_missing_action(self):
        bad = good_response()
        del bad["answers"]["move"]["probabilities"]["left"]
        with self.assertRaises(AssertionError):
            choice(bad)

    def test_choice_rejects_non_argmax(self):
        with self.assertRaises(AssertionError):
            choice(good_response("up"))

    def test_choice_rejects_unknown_action(self):
        with self.assertRaises(AssertionError):
            choice(good_response("diagonal"))


if __name__ == "__main__":
    unittest.main()
