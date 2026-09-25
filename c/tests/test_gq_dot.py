"""The integer GGUF matmul computes what it claims, for every block type.

The activation is rounded to int8 by the kernel itself (that rounding is the
fast path's one approximation); given the SAME rounded activation, the
product must match double precision to float accuracy. Anything above 1e-5
of sum|w*x| is a kernel bug, not rounding. Shapes cover a single row and
prefill-sized batches, rows of 32-element and 256-element blocks.
"""
import struct, subprocess, sys, unittest
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
BIN = HERE / "tests" / ("test_gq_dot.exe" if sys.platform == "win32" else "test_gq_dot")
sys.path.insert(0, str(HERE / "tests"))
try:
    from gguf.constants import GGMLQuantizationType as QT
    from test_ggml_quants import make_blocks, TYPES
except ImportError:  # pragma: no cover
    QT = None

@unittest.skipIf(QT is None, "gguf-py is not installed")
@unittest.skipUnless(BIN.is_file(), "tests/test_gq_dot is not built")
class GqDot(unittest.TestCase):
    def test_every_type_matches_double_precision(self):
        for name in TYPES:
            blck = TYPES[name][0]
            for (O, I, S) in ((7, 512, 1), (33, 2048, 5), (16, 256, 64)):
                if I % blck: continue
                with self.subTest(type=name, O=O, I=I, S=S):
                    rng = np.random.default_rng(O * 1000 + S)
                    raw = make_blocks(name, O * I // blck, rng)
                    if name == "MXFP4":   # block exponents of real magnitudes, not 2^127
                        raw[:, 0] = rng.integers(110, 140, raw.shape[0], dtype=np.uint8)
                    x = rng.standard_normal((S, I)).astype(np.float32)
                    x[:, :3] *= 20   # a few large activations, as real hidden states have
                    p = subprocess.run([str(BIN)], input=struct.pack("<iiii", int(QT[name]), O, I, S) + raw.tobytes() + x.tobytes(),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    self.assertEqual(p.returncode, 0, p.stderr)
                    err = float(p.stdout.decode())
                    self.assertLess(err, 1e-5, f"{name} O={O} I={I} S={S}: relative error {err:.2e}")

if __name__ == "__main__":
    unittest.main()
