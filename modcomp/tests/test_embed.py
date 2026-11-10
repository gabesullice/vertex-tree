import unittest

import numpy as np

from modcomp import embed as E


class EmbedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, cls.q = E.synthetic_embeddings(n=3000, n_queries=50, d=64, topics=20)

    def run_method(self, m):
        m.fit(self.db)
        codes = m.encode(self.db)
        self.assertEqual(len(codes), len(self.db))
        self.assertTrue(np.issubdtype(codes.dtype, np.integer))
        X_hat = m.decode(codes)
        self.assertEqual(X_hat.shape, self.db.shape)
        return E.mean_cosine(self.db, X_hat)

    def test_synthetic_is_unit_norm(self):
        np.testing.assert_allclose(np.linalg.norm(self.db, axis=1), 1, atol=1e-5)

    def test_all_methods_roundtrip_shapes_and_bits(self):
        for m, bits in [(E.Float16(), 16 * 64), (E.Grid01(2), 7 * 64),
                        (E.Binary(), 64), (E.Scalar(4), 4 * 64),
                        (E.PQ(8), 64), (E.CoarsePlus(16, E.Scalar(2)), 4 + 128),
                        (E.CoarsePlus(16, E.PQ(8)), 4 + 64)]:
            cos = self.run_method(m)
            self.assertEqual(m.bits, bits, m.name)
            self.assertGreater(cos, 0.5, m.name)

    def test_more_bits_is_better(self):
        cos = [self.run_method(E.Scalar(b)) for b in (2, 4, 8)]
        self.assertLess(cos[0], cos[1])
        self.assertLess(cos[1], cos[2])
        self.assertGreater(cos[2], 0.999)

    def test_learned_sector_beats_axis_sectors_at_same_correction_bits(self):
        plain = self.run_method(E.Scalar(2))
        coarse = self.run_method(E.CoarsePlus(64, E.Scalar(2)))
        self.assertGreater(coarse, plain)

    def test_recall_metric(self):
        self.assertEqual(E.recall_at_k(self.db, self.db, self.q, 10), 1.0)
        noise = np.random.default_rng(0).standard_normal(self.db.shape).astype(np.float32)
        self.assertLess(E.recall_at_k(self.db, noise, self.q, 10), 0.1)

    def test_kmeans_handles_k_larger_than_data(self):
        C = E.kmeans(self.db[:5], 16, iters=2)
        self.assertEqual(len(C), 5)


if __name__ == "__main__":
    unittest.main()
