"""GGUF -> colibri directory: both modes give the same model, the planner agrees.

A small but complete qwen35moe GGUF is written with gguf-py (every tensor the
engine reads, mixed ggml types as a UD file has them, a byte tokenizer), then
imported with tools/gguf_import.py in link and in copy mode. The engine must
produce the same tokens from the bare file, the link directory and the copy
container, on the float and on the integer path; the planner must see the
same experts in both directories; and in the container every expert slot must
start on a 4 KiB boundary of the file.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:   # the CI's Python tests run without numpy
    raise unittest.SkipTest("numpy not installed: %s" % exc)

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
sys.path.insert(0, str(HERE / "tests"))
sys.path.insert(0, str(HERE))
try:
    import gguf
    from gguf.constants import GGMLQuantizationType as QT
    from test_ggml_quants import make_blocks, TYPES
except ImportError:  # pragma: no cover
    gguf = None

H, LAYERS, E, TOPK, I, VOCAB = 256, 4, 8, 2, 256, 300
QH, KVH, HD, KH, VH, KD = 4, 2, 64, 4, 8, 32


def blocks(name, shape, rng, std=0.02):
    """Synthetic blocks of `name` with realistic magnitude (centred)."""
    blck, size, f16 = TYPES[name]
    rows = int(np.prod(shape[:-1])); nb = shape[-1] // blck
    raw = make_blocks(name, rows * nb, rng)
    probe = gguf.quants.dequantize(raw.reshape(-1), QT[name])
    s = float(np.std(probe)) or 1.0
    for off in f16:
        v = np.frombuffer(raw[:, off:off + 2].tobytes(), dtype=np.float16).astype(np.float32) * (std / s)
        raw[:, off:off + 2] = v.astype(np.float16).view(np.uint8).reshape(-1, 2)
    return raw.reshape(*shape[:-1], nb * size)


def write_model(path, rng):
    w = gguf.GGUFWriter(str(path), "qwen35moe")
    p = "qwen35moe."
    for k, v in (("block_count", LAYERS), ("embedding_length", H), ("attention.head_count", QH),
                 ("attention.head_count_kv", KVH), ("attention.key_length", HD), ("attention.value_length", HD),
                 ("expert_count", E), ("expert_used_count", TOPK), ("expert_feed_forward_length", I),
                 ("expert_shared_feed_forward_length", I), ("full_attention_interval", 4),
                 ("rope.dimension_count", HD // 4), ("ssm.conv_kernel", 4), ("ssm.state_size", KD),
                 ("ssm.group_count", KH), ("ssm.time_step_rank", VH), ("ssm.inner_size", VH * KD),
                 ("context_length", 4096)):
        w.add_uint32(p + k, v)
    w.add_float32(p + "rope.freq_base", 1e7); w.add_float32(p + "attention.layer_norm_rms_epsilon", 1e-6)
    toks = [f"<0x{b:02X}>" if b < 256 else f"t{b}" for b in range(VOCAB - 3)] + ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    w.add_tokenizer_model("gpt2"); w.add_token_list(toks)
    w.add_token_types([1] * (VOCAB - 3) + [3, 3, 3]); w.add_token_merges([])
    w.add_eos_token_id(VOCAB - 1); w.add_bos_token_id(VOCAB - 3)

    def f32(name, shape): w.add_tensor(name, (rng.standard_normal(shape) * 0.02).astype(np.float32))
    def q(name, shape, t): w.add_tensor(name, blocks(t, shape, rng), raw_dtype=QT[t])
    q("token_embd.weight", (VOCAB, H), "Q8_0"); q("output.weight", (VOCAB, H), "Q6_K")
    w.add_tensor("output_norm.weight", np.ones(H, np.float32))
    conv = 2 * KH * KD + VH * KD
    for L in range(LAYERS):
        b = f"blk.{L}."
        w.add_tensor(b + "attn_norm.weight", np.ones(H, np.float32))
        w.add_tensor(b + "post_attention_norm.weight", np.ones(H, np.float32))
        f32(b + "ffn_gate_inp.weight", (E, H)); f32(b + "ffn_gate_inp_shexp.weight", (H,))
        q(b + "ffn_gate_shexp.weight", (I, H), "Q8_0"); q(b + "ffn_up_shexp.weight", (I, H), "Q8_0")
        q(b + "ffn_down_shexp.weight", (H, I), "Q8_0")
        q(b + "ffn_gate_exps.weight", (E, I, H), "IQ2_XXS" if L == 0 else "Q4_K")
        q(b + "ffn_up_exps.weight", (E, I, H), "Q4_K")
        q(b + "ffn_down_exps.weight", (E, H, I), "Q6_K" if L % 2 else "Q5_K")
        if (L + 1) % 4 == 0:
            q(b + "attn_q.weight", (QH * HD * 2, H), "Q8_0"); q(b + "attn_k.weight", (KVH * HD, H), "Q8_0")
            q(b + "attn_v.weight", (KVH * HD, H), "Q8_0"); q(b + "attn_output.weight", (H, QH * HD), "Q8_0")
            w.add_tensor(b + "attn_q_norm.weight", np.ones(HD, np.float32)); w.add_tensor(b + "attn_k_norm.weight", np.ones(HD, np.float32))
        else:
            q(b + "attn_qkv.weight", (conv, H), "Q8_0"); q(b + "attn_gate.weight", (VH * KD, H), "Q8_0")
            w.add_tensor(b + "ssm_a", -np.exp(rng.standard_normal(VH).astype(np.float32) * 0.1))
            f32(b + "ssm_alpha.weight", (VH, H)); f32(b + "ssm_beta.weight", (VH, H))
            f32(b + "ssm_conv1d.weight", (conv, 4)); w.add_tensor(b + "ssm_dt.bias", np.zeros(VH, np.float32))
            w.add_tensor(b + "ssm_norm.weight", np.ones(KD, np.float32))
            q(b + "ssm_out.weight", (H, VH * KD), "Q8_0")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def engine_tokens(snap, ref, env=None):
    p = subprocess.run([str(ENGINE), "2", "8", str(ref)], env=dict(os.environ, SNAP=str(snap), OMP_NUM_THREADS="2",
                       COLI_NO_OMP_TUNE="1", **(env or {})), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode("utf-8", "replace")
    line = [l for l in out.splitlines() if l.startswith("C engine")]
    if not line:
        raise AssertionError(f"engine produced no tokens (rc {p.returncode}):\n{out[-2000:]}")
    return line[-1].split(":", 1)[1].split("Text")[0].split()


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
@unittest.skipUnless(ENGINE.is_file(), "qwen36 is not built")
class GgufImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = Path(cls.tmp.name)
        cls.src = d / "tiny-qwen35moe.gguf"
        write_model(cls.src, np.random.default_rng(11))
        cls.ref = d / "ref.json"
        cls.ref.write_text(json.dumps({"prompt_ids": [5, 6, 7, 8], "full_ids": [5, 6, 7, 8] + [0] * 12}))
        for mode in ("link", "copy"):
            p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(cls.src), str(d / mode), "--mode", mode],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert p.returncode == 0, p.stdout.decode()
        cls.dir = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_directories_are_complete(self):
        for mode in ("link", "copy"):
            d = self.dir / mode
            for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "gguf.json"):
                self.assertTrue((d / f).is_file(), f"{mode}: {f}")
            cfg = json.loads((d / "config.json").read_text())
            self.assertEqual(cfg["model_type"], "qwen3_5_moe_text")
            self.assertEqual((cfg["num_hidden_layers"], cfg["num_experts"], cfg["hidden_size"]), (LAYERS, E, H))
            self.assertEqual(cfg["layer_types"][3], "full_attention")
            tok = json.loads((d / "tokenizer.json").read_text())
            self.assertNotIn("<|im_end|>", tok["model"]["vocab"])          # added tokens live in added_tokens only
            self.assertTrue(any(a["content"] == "<|im_end|>" and a["special"] for a in tok["added_tokens"]))
        self.assertTrue((self.dir / "copy" / "model.colibri.gguf").is_file())
        self.assertFalse((self.dir / "link" / "model.colibri.gguf").exists())

    def test_same_tokens_from_file_link_and_copy(self):
        for env in ({}, {"GQ_EXACT": "1"}):
            with self.subTest(path="exact" if env else "integer"):
                a = engine_tokens(self.src, self.ref, env)
                self.assertEqual(engine_tokens(self.dir / "link", self.ref, env), a)
                self.assertEqual(engine_tokens(self.dir / "copy", self.ref, env), a)

    def test_container_slots_are_4k_aligned(self):
        r = gguf.GGUFReader(str(self.dir / "copy" / "model.colibri.gguf"))
        self.assertEqual(r.fields["colibri.layout"].contents(), "expert-packed-v1")
        packed = [t for t in r.tensors if t.name.endswith("ffn_exps_colibri")]
        self.assertEqual(len(packed), LAYERS)
        for t in packed:
            self.assertEqual(t.data_offset % 4096, 0, t.name)
            self.assertEqual(int(t.shape[0]) % 4096, 0, t.name)      # slot stride
        # the dense tensors are the source's, byte for byte
        src = gguf.GGUFReader(str(self.src))
        s = {t.name: t for t in src.tensors}
        for t in r.tensors:
            if t.name in s:
                self.assertTrue(np.array_equal(np.array(t.data).view(np.uint8), np.array(s[t.name].data).view(np.uint8)), t.name)

    def test_planner_sees_the_same_experts(self):
        import resource_plan as rp
        a, b = rp.analyze_model(str(self.dir / "link")), rp.analyze_model(str(self.dir / "copy"))
        for k in ("expert_count", "expert_layers", "expert_bytes", "per_cap_bytes", "dense_bytes"):
            self.assertEqual(a[k], b[k], k)
        self.assertEqual(a["expert_count"], LAYERS * E)


class Qwen4ExpLayers(unittest.TestCase):
    """Which layers attend, from either way a qwen4exp GGUF says it: unsloth's Qwen3.8-Flash-Next
    files give full_attention_interval (no attention.recurrent_layers), newer llama.cpp
    writes recurrent_layers; compress_ratios, when not all 0, must agree."""
    def meta(self, **extra):
        p = "qwen4exp."
        m = {p + k: v for k, v in {
            "block_count": 8, "embedding_length": 2560, "attention.head_count": 24, "attention.head_count_kv": 2,
            "attention.key_length": 256, "rope.dimension_count": 64, "expert_count": 512, "expert_used_count": 10,
            "expert_feed_forward_length": 640, "ssm.conv_kernel": 4, "ssm.state_size": 128, "ssm.group_count": 16,
            "ssm.time_step_rank": 48, "ssm.inner_size": 6144, "hyper_connection.count": 4,
            "hyper_connection.low_rank": 320, "attention.indexer.head_count": 4, "attention.indexer.key_length": 128,
            "attention.indexer.top_k": 2048, "ple.layers": [1], "ple.ngram_size": 3, "ple.heads_per_ngram": 8,
            "ple.conv_kernel": 4, "embedding_length_per_layer_input": 160}.items()}
        m.update({p + k: v for k, v in extra.items()})
        return m

    def config(self, meta):
        import tools.gguf_import as gi
        tensors = {"token_embd.weight": {"dims": [2560, 248320]},
                   **{f"blk.{i}.indexer.k_proj.weight": {"dims": [2560, 128]} for i in range(8)}}
        return gi.qwen4exp_config(meta, tensors)[0]

    def test_interval_and_recurrent_layers_agree(self):
        ratios = [0, 0, 0, 4, 0, 0, 0, 4]
        old = self.config(self.meta(full_attention_interval=4, **{"attention.compress_ratios": ratios}))
        new = self.config(self.meta(**{"attention.recurrent_layers": [1, 1, 1, 0, 1, 1, 1, 0],
                                       "attention.compress_ratios": ratios}))
        self.assertEqual(old["layer_types"], ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"])
        self.assertEqual(old, new)
        self.assertEqual(old["indexer_compress_ratio"], 4)

    def test_contradicting_metadata_is_refused(self):
        with self.assertRaises(SystemExit):
            self.config(self.meta(full_attention_interval=4, **{"attention.compress_ratios": [0, 0, 4, 0, 0, 0, 4, 0]}))


if __name__ == "__main__":
    unittest.main()
