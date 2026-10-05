import unittest
from clef_snake.game import Game, priorities, reachable


class ProtocolTests(unittest.TestCase):
    def test_seed_reproducibility_and_body_independence(self):
        order = priorities("b145b41017ce4bce48bf75df6719f853", 0)
        self.assertEqual(len(set(order)), 144)
        game = Game("b145b41017ce4bce48bf75df6719f853")
        self.assertEqual(game.food, [0, 3])  # Recorded browser-run first state.
        self.assertNotEqual(order, priorities("6f5d8daa754d8a534cd54a77a298624b", 0))

    def test_unsafe_choice_is_not_overridden(self):
        game = Game("b145b41017ce4bce48bf75df6719f853")
        self.assertEqual(game.request()["state"]["move_analysis"]["left"]["reason"], "neck")
        game.apply("left")
        self.assertFalse(game.alive)
        self.assertEqual(game.steps, 0)
        self.assertEqual(game.body[0], [5, 6])

    def test_food_growth_and_event(self):
        game = Game("b145b41017ce4bce48bf75df6719f853")
        game.food = [6, 6]
        game.apply("right")
        self.assertEqual((game.score, game.steps, len(game.body)), (1, 1, 5))
        self.assertEqual(game.food_events[-1]["event"], 1)
        self.assertNotIn(game.food, game.body)

    def test_tail_is_vacated_and_reachability_is_factual(self):
        game = Game("b145b41017ce4bce48bf75df6719f853")
        game.body = [[1, 1], [1, 2], [0, 2], [0, 1]]
        game.apply("left")
        self.assertTrue(game.alive)
        self.assertEqual(game.body[0], [0, 1])
        self.assertEqual(reachable((0, 0), {(1, 0), (0, 1)}), 1)


if __name__ == "__main__":
    unittest.main()
