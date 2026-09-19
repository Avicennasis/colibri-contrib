"""the converter's source fingerprint and machine-readable progress lines.

The real conversion needs torch, safetensors and a ~700 GB FP8 checkpoint, so the
end-to-end runs here drive main()'s --indir path over a fixture of empty stand-in
shards with convert_shard and save_file replaced: what is exercised is the skip /
resume / staleness bookkeeping around the conversion, not the quantizer itself
(that is tests/test_int3_convert.py and the oracles' job)."""
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


CONVERTER = Path(__file__).resolve().parent.parent / "tools" / "convert_fp8_to_int4.py"


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_fingerprint_test", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"numpy": types.ModuleType("numpy")}):
        spec.loader.exec_module(module)
    return module


convert = load_converter()

PARAMS = {"ebits": 4, "io_bits": 8, "xbits": 4, "group_size": 64,
          "n_layers": 78, "bits_map": {}, "proj_bits": {}}


class SourceFingerprintTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.shards = []
        for i in range(3):
            shard = self.root / f"model-{i:05d}.safetensors"
            shard.write_bytes(b"x" * (10 + i))
            os.utime(shard, (1_700_000_000, 1_700_000_000))
            self.shards.append(str(shard))

    def test_same_sources_and_params_hash_the_same(self):
        first = convert.source_fingerprint(convert.source_signatures(self.shards), PARAMS)
        second = convert.source_fingerprint(convert.source_signatures(self.shards), PARAMS)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)

    def test_a_changed_mtime_size_or_parameter_changes_the_hash(self):
        base = convert.source_fingerprint(convert.source_signatures(self.shards), PARAMS)

        os.utime(self.shards[1], (1_700_000_001, 1_700_000_001))
        touched = convert.source_fingerprint(convert.source_signatures(self.shards), PARAMS)
        self.assertNotEqual(base, touched)

        Path(self.shards[1]).write_bytes(b"y" * 999)
        os.utime(self.shards[1], (1_700_000_000, 1_700_000_000))   # size alone must be enough
        resized = convert.source_fingerprint(convert.source_signatures(self.shards), PARAMS)
        self.assertNotEqual(base, resized)

        # the quant parameters are hashed in too: same bytes, different container
        for key, value in (("ebits", 8), ("group_size", 0), ("bits_map", {"sh": 8})):
            other = dict(PARAMS, **{key: value})
            self.assertNotEqual(
                base, convert.source_fingerprint(convert.source_signatures(self.shards), other),
                f"{key}={value} must not reuse a container built with {PARAMS[key]}")

    def test_a_dropped_or_added_shard_changes_the_hash(self):
        base = convert.source_fingerprint(convert.source_signatures(self.shards), PARAMS)
        fewer = convert.source_fingerprint(convert.source_signatures(self.shards[:2]), PARAMS)
        self.assertNotEqual(base, fewer)

    def test_cache_round_trips_through_the_output_dir(self):
        out = self.root / "out"
        out.mkdir()
        sigs = convert.source_signatures(self.shards)
        self.assertEqual(convert.read_source_cache(str(out), "out-"), {})   # nothing cached yet
        convert.write_source_cache(str(out), "out-", sigs, "cafebabe0000dead", False)
        cached = convert.read_source_cache(str(out), "out-")
        self.assertEqual(cached["fingerprint"], "cafebabe0000dead")
        self.assertFalse(cached["complete"])
        self.assertEqual(cached["files"], sigs)

    def test_a_corrupt_cache_is_ignored_not_fatal(self):
        out = self.root / "out"
        out.mkdir()
        (out / ".out-source.json").write_text("{ truncated")
        self.assertEqual(convert.read_source_cache(str(out), "out-"), {})


class ProgressLineTest(unittest.TestCase):
    def test_lines_are_off_by_default_and_parseable_when_enabled(self):
        buffer = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COLI_CONVERT_PROGRESS", None)
            with redirect_stdout(buffer):
                convert._progress("shard", 3, 141)
        self.assertEqual(buffer.getvalue(), "")

        buffer = io.StringIO()
        with mock.patch.dict(os.environ, {"COLI_CONVERT_PROGRESS": "1"}):
            with redirect_stdout(buffer):
                convert._progress("shard", 3, 141)
                convert._progress("done", 141, 141)
        phase, done, total = buffer.getvalue().splitlines()[0].split()[1:]
        self.assertEqual((phase, int(done), int(total)), ("shard", 3, 141))
        self.assertEqual(buffer.getvalue().splitlines()[1], "COLICONVERT done 141 141")


class ConvertSkipOnMatchTest(unittest.TestCase):
    """The point of the fingerprint: a second run over unchanged sources converts
    nothing, and a source that MOVED is reconverted even though its output exists."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.indir = self.root / "fp8"
        self.outdir = self.root / "int4"
        self.indir.mkdir()
        for i in range(3):
            (self.indir / f"model-{i:05d}.safetensors").write_bytes(b"stand-in shard")
        (self.indir / "config.json").write_text('{"model_type": "glm"}')
        (self.indir / "tokenizer.json").write_text("{}")
        self.converted = []

    def run_convert(self, *extra, progress=False):
        """Drive main()'s --indir path; returns (stdout, shards converted this run)."""
        self.converted = []

        def fake_convert_shard(path, out_dict, *args, **kwargs):
            self.converted.append(os.path.basename(path))
            out_dict["fake.weight"] = b"q"

        def fake_save_file(tensors, path):
            Path(path).write_bytes(b"".join(tensors.values()))

        safetensors = types.ModuleType("safetensors")
        numpy_module = types.ModuleType("safetensors.numpy")
        numpy_module.save_file = fake_save_file
        safetensors.numpy = numpy_module
        argv = ["convert_fp8_to_int4.py", "--indir", str(self.indir),
                "--outdir", str(self.outdir), "--ebits", "4", "--io-bits", "4", *extra]
        env = {"COLI_CONVERT_PROGRESS": "1"} if progress else {}
        buffer = io.StringIO()
        with mock.patch.dict(sys.modules, {"safetensors": safetensors,
                                           "safetensors.numpy": numpy_module}), \
             mock.patch.dict(os.environ, env), \
             mock.patch.object(convert, "convert_shard", fake_convert_shard), \
             mock.patch.object(sys, "argv", argv), \
             redirect_stdout(buffer):
            if not progress:
                os.environ.pop("COLI_CONVERT_PROGRESS", None)
            convert.main()
        return buffer.getvalue(), list(self.converted)

    def test_second_run_on_unchanged_sources_skips_the_whole_conversion(self):
        out, converted = self.run_convert(progress=True)
        self.assertEqual(converted, ["model-00000.safetensors", "model-00001.safetensors",
                                     "model-00002.safetensors"])
        self.assertTrue((self.outdir / "out-00000.safetensors").exists())
        self.assertIn("COLICONVERT shard 1 3", out)
        self.assertIn("COLICONVERT done 3 3", out)
        cached = json.loads((self.outdir / ".out-source.json").read_text())
        self.assertTrue(cached["complete"])
        self.assertEqual(sorted(cached["files"]), ["model-00000.safetensors",
                                                   "model-00001.safetensors",
                                                   "model-00002.safetensors"])

        out, converted = self.run_convert(progress=True)
        self.assertEqual(converted, [])                    # no shard opened at all
        self.assertIn("[SKIP]", out)
        self.assertIn(cached["fingerprint"], out)
        self.assertIn("COLICONVERT uptodate 3 3", out)

    def test_an_interrupted_run_is_not_skipped(self):
        # complete=False is what an interrupt leaves behind: the sources match, but the
        # container is unfinished, so the next run must resume rather than declare victory.
        self.run_convert()
        cached = json.loads((self.outdir / ".out-source.json").read_text())
        cached["complete"] = False
        (self.outdir / ".out-source.json").write_text(json.dumps(cached))

        out, converted = self.run_convert()
        self.assertNotIn("[SKIP]", out)
        self.assertEqual(converted, [])                    # per-shard resume still holds
        self.assertIn("[RESUME] 3 shard(s) already done", out)

    def test_a_changed_source_shard_is_reconverted(self):
        self.run_convert()
        shard = self.indir / "model-00001.safetensors"
        shard.write_bytes(b"stand-in shard, repaired")     # same NAME, different bytes

        out, converted = self.run_convert()
        self.assertIn("[SOURCE] 1 source shard(s) changed", out)
        self.assertEqual(converted, ["model-00001.safetensors"])

    def test_force_reconverts_everything(self):
        self.run_convert()
        out, converted = self.run_convert("--force")
        self.assertNotIn("[SKIP]", out)
        self.assertEqual(len(converted), 3)

    def test_changed_parameters_still_refuse_to_share_an_outdir(self):
        # The #355 guard predates the fingerprint and must keep firing first: a different
        # --ebits is a different container, not a stale one.
        self.run_convert()
        out, converted = self.run_convert("--group-size", "0")
        self.assertIn("Refusing to mix conversions", out)
        self.assertEqual(converted, [])


if __name__ == "__main__":
    unittest.main()
