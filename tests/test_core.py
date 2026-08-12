import unittest

import numpy as np

from demo_mini_chat.config import DemoConfig
from demo_mini_chat.data import NgramData
from demo_mini_chat.engine import MiniLanguageEngine


class NgramDataTests(unittest.TestCase):
    def test_transform_uses_three_tokens_to_predict_one(self) -> None:
        corpus = ["la inteligencia artificial aprende patrones"]
        data = NgramData(order=4, vocab_size=64)
        data.fit(corpus)
        x, y = data.transform(corpus)
        self.assertEqual(x.shape[1], 3)
        self.assertEqual(x.shape[0], y.shape[0])


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = DemoConfig(max_epochs=2, patience=1, vocab_size=128, hidden_dim=32, embedding_dim=16)
        cls.engine = MiniLanguageEngine.build(config)

    def test_distribution_is_probability_distribution(self) -> None:
        probabilities, context = self.engine.distribution("la inteligencia artificial")
        self.assertEqual(len(context), 3)
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=5)
        self.assertTrue(np.all(probabilities >= 0))

    def test_generation_step_returns_candidates(self) -> None:
        step = self.engine.next_step("el modelo aprende", deterministic=True)
        self.assertTrue(step.candidates)
        self.assertGreaterEqual(step.chosen.probability, 0.0)


if __name__ == "__main__":
    unittest.main()
