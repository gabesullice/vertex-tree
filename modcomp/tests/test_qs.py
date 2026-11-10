import unittest
from urllib.parse import quote

import numpy as np

from modcomp import embed as E
from modcomp import qs


class BitsTest(unittest.TestCase):
    def test_roundtrip_and_length(self):
        rng = np.random.default_rng(0)
        for width in (1, 3, 6, 7, 16):
            vals = rng.integers(0, 1 << width, size=37)
            w = qs.BitWriter()
            w.write_array(vals, width)
            s = w.to_str()
            self.assertEqual(len(s), qs.chars_for_bits(37 * width))
            np.testing.assert_array_equal(qs.BitReader(s).read_array(37, width), vals)

    def test_url_safe(self):
        self.assertEqual(len(set(qs.ALPHABET)), 64)
        self.assertEqual(quote(qs.ALPHABET, safe=""), qs.ALPHABET)

    def test_errors(self):
        with self.assertRaises(ValueError):
            qs.BitReader("ab+c")
        with self.assertRaises(ValueError):
            qs.BitReader("A").read(7)


class CodecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, cls.q = E.synthetic_embeddings(n=3000, n_queries=20, d=64, topics=20)

    def cosines(self, codec):
        out = []
        for x in self.q:
            s = codec.encode(x)
            self.assertEqual(len(s), qs.chars_for_bits(codec.bits))
            self.assertEqual(quote(s, safe=""), s)
            out.append(float(codec.decode(s) @ x))
        return np.array(out)

    def test_rotated_precision_increases_with_bits(self):
        means = [self.cosines(qs.RotScalar(64, b)).mean() for b in (1, 2, 4, 8)]
        self.assertTrue(all(a < b for a, b in zip(means, means[1:])), means)
        self.assertGreater(means[-1], 0.9999)

    def test_rotation_is_orthogonal_and_deterministic(self):
        A = qs.seeded_rotation(32, 7)
        np.testing.assert_allclose(A.T @ A, np.eye(32), atol=1e-12)
        np.testing.assert_array_equal(A, qs.seeded_rotation(32, 7))
        # Must not coincide with the synthetic generator's own basis (seed 0).
        self.assertFalse(np.allclose(np.abs(qs.seeded_rotation(64, qs.ROTATION_SEED)),
                                     np.abs(qs.seeded_rotation(64, 0))))

    def test_sector_beats_plain_at_same_bits(self):
        plain = self.cosines(qs.RotScalar(64, 2)).mean()
        sector = self.cosines(qs.SectorRot(64, 64, 2).fit(self.db)).mean()
        self.assertGreater(sector, plain)

    def test_progressive_prefixes(self):
        c = qs.Progressive(64, [8, 8]).fit(self.db)
        self.assertEqual(c.bits, 6 + 64 + 64)
        x = self.q[0]
        s = c.encode(x)
        cos = [float(c.decode(s[:L]) @ x) for L in c.prefix_chars()]
        self.assertEqual(c.decode(s[:c.prefix_chars()[-1]]).tolist(), c.decode(s).tolist())
        self.assertLess(cos[0], cos[-1])
        # A cut mid-stage decodes like the previous complete stage.
        mid = c.prefix_chars()[1] + 3
        np.testing.assert_allclose(c.decode(s[:mid]), c.decode(s[:c.prefix_chars()[1]]))


if __name__ == "__main__":
    unittest.main()
