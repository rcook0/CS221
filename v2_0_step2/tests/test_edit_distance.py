import unittest

from ai_toolkit.dp import edit_distance_topdown, edit_distance_bottomup


class TestEditDistance(unittest.TestCase):
    def test_small(self):
        pairs = [
            ("", "", 0),
            ("a", "", 1),
            ("", "abc", 3),
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
            ("gumbo", "gambol", 2),
        ]
        for s, t, d in pairs:
            self.assertEqual(edit_distance_topdown(s, t), d)
            self.assertEqual(edit_distance_bottomup(s, t), d)

    def test_equivalence_random(self):
        import random, string
        rng = random.Random(0)
        for _ in range(50):
            s = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(0, 12)))
            t = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(0, 12)))
            self.assertEqual(edit_distance_topdown(s, t), edit_distance_bottomup(s, t))


if __name__ == "__main__":
    unittest.main()
