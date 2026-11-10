import unittest

import numpy as np

from modcomp import analysis as A
from modcomp import core


class SplitMergeTest(unittest.TestCase):
    def test_identity_and_residue_range(self):
        rng = np.random.default_rng(1)
        V = rng.integers(-10_000, 10_000, size=(500, 4))
        m = [1, 7, 64, 1000]
        c = [0, -3, 5, 999]
        R, Q = core.split(V, m, c)
        self.assertTrue(np.all(R >= 0))
        self.assertTrue(np.all(R < np.array(m)))
        np.testing.assert_array_equal(core.merge(R, Q, m, c), V)

    def test_rejects_floats_and_bad_modulus(self):
        with self.assertRaises(TypeError):
            core.split(np.zeros((3, 2)), 4)
        with self.assertRaises(ValueError):
            core.split(np.zeros((3, 2), dtype=int), 0)


class PrimitivesTest(unittest.TestCase):
    def test_zigzag(self):
        x = np.array([0, -1, 1, -2, 2, 2**62, -(2**62)], dtype=np.int64)
        np.testing.assert_array_equal(core.zigzag(x)[:5], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(core.unzigzag(core.zigzag(x)), x)

    def test_varint(self):
        u = np.array([0, 1, 127, 128, 300, 2**35, 2**63 + 5], dtype=np.uint64)
        buf = core.varint_encode(u)
        self.assertEqual(buf[:5], bytes([0, 1, 127, 0x80, 1]))
        np.testing.assert_array_equal(core.varint_decode(buf, len(u)), u)
        with self.assertRaises(ValueError):
            core.varint_decode(buf, len(u) + 1)

    def test_bitpack(self):
        rng = np.random.default_rng(2)
        m = np.array([1, 2, 5, 256, 1000])
        R = rng.integers(0, m, size=(333, 5))
        w = core.residue_widths(m)
        np.testing.assert_array_equal(w, [0, 1, 3, 8, 10])
        np.testing.assert_array_equal(core.bitunpack(core.bitpack(R, w), 333, w), R)


class ContainerTest(unittest.TestCase):
    def roundtrip(self, V, m, c=0, **kw):
        blob, sizes = core.encode(V, m, c, **kw)
        self.assertEqual(sizes["total"], len(blob))
        self.assertEqual(sizes["header"] + sizes["residues"] + sizes["corrections"],
                         len(blob))
        expected = np.asarray(V)
        if expected.ndim == 1:
            expected = expected[:, None]
        np.testing.assert_array_equal(core.decode(blob), expected)
        return blob

    def test_roundtrip_variants(self):
        rng = np.random.default_rng(3)
        V = np.cumsum(rng.integers(-5, 6, size=(1000, 3)), axis=0) - 400
        for codec in core.CODECS:
            for cr in (False, True):
                self.roundtrip(V, [16, 3, 1], [2, -1, 0], codec=codec,
                               compress_residues=cr)

    def test_edge_shapes(self):
        self.roundtrip(np.zeros((0, 3), dtype=np.int64), 8)
        self.roundtrip(np.array([[5]]), 4)
        self.roundtrip(np.arange(10), 3)  # 1-D input becomes (N, 1)
        self.roundtrip(np.array([[2**40, -(2**40)]] * 4), 2**20)

    def test_random_access(self):
        rng = np.random.default_rng(4)
        V = rng.integers(0, 50_000, size=(200, 3))
        m = [256, 100, 1]
        blob = self.roundtrip(V, m, [7, 0, 0])
        R, _ = core.split(V, m, [7, 0, 0])
        for i in (0, 1, 57, 199):
            np.testing.assert_array_equal(core.residue_at(blob, i), R[i])

    def test_corrupt_input(self):
        blob, _ = core.encode(np.arange(30).reshape(10, 3), 4)
        with self.assertRaises(ValueError):
            core.decode(b"XXXX" + blob[4:])
        with self.assertRaises(Exception):
            core.decode(blob[:-3])


class AnalysisTest(unittest.TestCase):
    def test_transitions_by_offset_matches_brute_force(self):
        rng = np.random.default_rng(5)
        col = np.cumsum(rng.integers(-9, 10, size=400))
        for m in (1, 2, 7, 16, 50):
            fast = A.transitions_by_offset(col, m)
            slow = [np.count_nonzero(np.diff((col - c) // m)) for c in range(m)]
            np.testing.assert_array_equal(fast, slow)

    def test_offset_avoids_boundary(self):
        # Values jitter around 64, right on a boundary for offset 0.
        col = np.array([63, 64, 63, 64, 65, 63, 64] * 20)
        self.assertGreater(A.transitions_by_offset(col, 64)[0], 0)
        c = A.best_offset(col, 64)
        self.assertEqual(np.count_nonzero(np.diff((col - c) // 64)), 0)

    def test_limits(self):
        rng = np.random.default_rng(6)
        V = np.cumsum(rng.integers(-3, 4, size=(2000, 2)), axis=0)
        # m == 1 is exactly delta coding under the entropy model.
        self.assertAlmostEqual(A.estimate(V, 1)["total_bits"],
                               A.baselines(V)["delta_entropy_bits"])
        # Huge m leaves a constant correction (only the first row is nonzero).
        self.assertEqual(A.estimate(V - V.min(0), 2**30)["correction_nonzero"], 0)

    def test_choose_parameters_never_worse_than_endpoints(self):
        rng = np.random.default_rng(7)
        V = np.cumsum(rng.integers(-3, 4, size=(3000, 3)), axis=0)
        m, c, _ = A.choose_parameters(V)
        best = A.estimate(V, m, c)["total_bits"]
        self.assertLessEqual(best, A.estimate(V, 1)["total_bits"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
