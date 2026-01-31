import unittest

from ai_toolkit.ml import dot, train_perceptron


class TestML(unittest.TestCase):
    def test_dot(self):
        a = {"x": 2.0, "y": 1.0}
        b = {"x": 3.0, "z": 5.0}
        self.assertEqual(dot(a, b), 6.0)

    def test_perceptron_toy(self):
        # Need a bias feature to separate classes in this tiny setup.
        def fe(x):
            return {"bias": 1.0, "hasA": 1.0 if "A" in x else 0.0}

        train = [(+1, "A"), (-1, "B")]
        dev = [(+1, "A"), (-1, "B")]
        res = train_perceptron(train, dev, fe, iters=10, seed=0)
        self.assertEqual(res.dev_error, 0.0)


if __name__ == "__main__":
    unittest.main()
