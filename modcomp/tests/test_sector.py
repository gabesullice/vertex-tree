import unittest

import numpy as np

from modcomp import sector as S


class GridTest(unittest.TestCase):
    def test_inclusive_endpoints(self):
        g = S.SectorGrid.decimal(16, 2)
        Sx, C = g.split(np.array([[0, g.Q]]))
        self.assertEqual(Sx.tolist(), [[0, 99]])  # 1.0 lands in the last sector
        self.assertEqual(C[0, 0], 0)

    def test_split_merge_exhaustive(self):
        for bits, K in [(8, 10), (8, 100), (10, 7), (12, 1000), (4, 16), (6, 1)]:
            g = S.SectorGrid(bits, K)
            v = np.arange(g.Q + 1)[:, None]
            Sx, C = g.split(v)
            np.testing.assert_array_equal(g.merge(Sx, C), v)
            self.assertTrue(np.all((Sx >= 0) & (Sx < K)))
            self.assertTrue(np.all((C >= 0) & (C <= g.max_correction)))
            self.assertEqual(int(C.max()), g.max_correction)
            # Sector j really covers [j/K, (j+1)/K] in real terms.
            x = v[:, 0] / g.Q
            self.assertTrue(np.all(x >= Sx[:, 0] / K - 1e-12))
            self.assertTrue(np.all(x <= (Sx[:, 0] + 1) / K + 1e-12))

    def test_validation(self):
        g = S.SectorGrid(8, 10)
        with self.assertRaises(ValueError):
            g.split(np.array([[256]]))
        with self.assertRaises(TypeError):
            g.split(np.array([[0.5]]))
        with self.assertRaises(ValueError):
            S.to_fixed([[1.5]], 8)
        with self.assertRaises(ValueError):
            S.SectorGrid(8, 1000)


class ContainerTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        X = np.clip(rng.random((7, 3))[rng.integers(0, 7, 2000)]
                    + 0.01 * rng.standard_normal((2000, 3)), 0, 1)
        X[0] = [0, 1, 1]
        self.X = X

    def test_lossless_modes(self):
        for bits in (8, 16, 32):
            V = S.to_fixed(self.X, bits)
            for digits in (1, 2, 3):
                if 10 ** digits > (1 << bits):
                    continue
                g = S.SectorGrid.decimal(bits, digits)
                for codec in ("none", "zlib", "lzma"):
                    for mode in ("ordered", "grouped+perm"):
                        blob, sizes = S.encode(V, g, mode, codec)
                        self.assertEqual(sizes["total"], len(blob))
                        np.testing.assert_array_equal(S.decode(blob), V)
                    blob, _ = S.encode(V, g, "grouped", codec)
                    out = S.decode(blob)
                    key = lambda A: A[np.lexsort(A.T[::-1])]
                    np.testing.assert_array_equal(key(out), key(V))

    def test_lossy_error_bound(self):
        V = S.to_fixed(self.X, 16)
        for digits in (1, 2, 3):
            g = S.SectorGrid.decimal(16, digits)
            out = S.decode(S.encode(V, g, "sectors")[0])
            err = np.abs(S.from_fixed(out, 16) - self.X).max()
            self.assertLessEqual(err, g.precision / 2 + 2 / g.Q)

    def test_grouping_beats_ordered_on_clusters(self):
        V = S.to_fixed(self.X, 16)
        g = S.SectorGrid.decimal(16, 1)
        ordered = S.encode(V, g, "ordered")[1]["sectors"]
        grouped = S.encode(V, g, "grouped")[1]["sectors"]
        self.assertLess(grouped, ordered)

    def test_empty_and_single(self):
        g = S.SectorGrid.decimal(16, 2)
        for V in (np.zeros((0, 4), dtype=np.int64), np.array([[123, 0, 65535]])):
            for mode in ("ordered", "grouped", "grouped+perm"):
                np.testing.assert_array_equal(S.decode(S.encode(V, g, mode)[0]), V)


if __name__ == "__main__":
    unittest.main()
