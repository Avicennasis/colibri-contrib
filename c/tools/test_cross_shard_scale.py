#!/usr/bin/env python3
"""
Synthetic test: prove that dequant() can resolve a _scale_inv tensor that lives
in a DIFFERENT safetensors shard from the weight.

Creates two tiny safetensors files:
  shard_a.safetensors  ->  "t.weight"            (float8_e4m3fn, shape [128,128])
  shard_b.safetensors  ->  "t.weight_scale_inv"  (float32, shape [1,1])

Sets up the module-level globals (_SHARD_DIR, _WEIGHT_MAP, _SAFE_OPEN_CACHE)
so that dequant() resolves the scale from shard_b, then verifies:
  - no crash / no SafetensorError
  - the returned tensor is finite float32
  - the scale was actually read from shard_b (not shard_a)

Clean up temp files afterwards.
"""

import os, sys, tempfile, shutil

# Pull the three globals from the converter module so we can inject test state
sys.path.insert(0, os.path.dirname(__file__))
import convert_fp8_to_int4 as conv

def test_cross_shard_scale():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    tmpdir = tempfile.mkdtemp(prefix="cross_shard_test_")
    try:
        # --- 1. Create the FP8 weight in shard_a ---
        O, I = 128, 128  # exactly 128 so scale shape = [1,1]
        weight_val = torch.ones(O, I, dtype=torch.float8_e4m3fn)  # all zeros
        save_file({"t.weight": weight_val}, os.path.join(tmpdir, "shard_a.safetensors"))

        # --- 2. Create the scale_inv in shard_b ---
        scale_val = torch.tensor([[3.0]], dtype=torch.float32)  # scale for the single 128x128 block
        save_file({"t.weight_scale_inv": scale_val}, os.path.join(tmpdir, "shard_b.safetensors"))

        # --- 3. Set up the module-level globals exactly as main() would ---
        conv._SHARD_DIR = tmpdir
        conv._WEIGHT_MAP = {
            "t.weight": "shard_a.safetensors",
            "t.weight_scale_inv": "shard_b.safetensors",
        }
        conv._SAFE_OPEN_CACHE = {}

        # --- 4. Open shard_a and call dequant (scale lives in shard_b) ---
        with safe_open(os.path.join(tmpdir, "shard_a.safetensors"), framework="pt") as f:
            keys = set(f.keys())
            assert "t.weight" in keys, "weight should be in shard_a"
            assert "t.weight_scale_inv" not in keys, "scale should NOT be in shard_a"

            result = conv.dequant(f, "t.weight", keys)

        # --- 5. Verify ---
        import numpy as np
        assert isinstance(result, np.ndarray), f"expected ndarray, got {type(result)}"
        assert result.dtype == np.float32, f"expected float32, got {result.dtype}"
        assert result.shape == (O, I), f"expected shape ({O},{I}), got {result.shape}"
        assert np.all(np.isfinite(result)), "result contains non-finite values"

        # The FP8 weight is all-ones (value 1.0 in e4m3), scale is 3.0 -> result should be all ~3.0
        assert np.allclose(result, 3.0, atol=0.1), f"expected all-~3.0 result, got range [{result.min():.3f}, {result.max():.3f}]"

        # --- 6. Verify that shard_b was actually opened (cross-shard resolution happened) ---
        assert len(conv._SAFE_OPEN_CACHE) > 0, "cross-shard cache should have entries"
        assert "shard_b.safetensors" in conv._SAFE_OPEN_CACHE, "shard_b should be cached"

        print("[cross-shard-test] PASS: dequant resolved scale_inv from shard_b successfully")
        print(f"  result shape={result.shape}, dtype={result.dtype}, all_finite={np.all(np.isfinite(result))}")
        print(f"  cross-shard cache keys: {list(conv._SAFE_OPEN_CACHE.keys())}")

    finally:
        # Clean up
        shutil.rmtree(tmpdir, ignore_errors=True)
        conv._SHARD_DIR = None
        conv._WEIGHT_MAP = {}
        conv._SAFE_OPEN_CACHE = {}

if __name__ == "__main__":
    test_cross_shard_scale()
    print("\nALL TESTS PASSED")
