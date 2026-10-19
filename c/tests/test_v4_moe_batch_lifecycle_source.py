"""The batched GPU MoE's expert bank lives only while a prefill runs.

coli_v4_gpu_moe_batch_union (COLI_CUDA_MOE_BATCH=1) keeps a process-wide
expert bank, and with COLI_CUDA_MOE_DOUBLE a second bank plus a prefetch
worker that holds a store lease. A regression in releasing that state is
silent on output (the CPU union gives the same bits) and only shows up as
VRAM, disk traffic or a crash at teardown, which no fixture without the
4096-wide geometry can reach. So this pins the shape at source level:

- every exit of coli_v4_session_generate (cancel and failures included)
  releases the bank, and engine close releases it before teardown.
"""
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "deepseek_v4.c"


def find(text, needle, start=0, where="deepseek_v4.c"):
    """text.index(needle, start), except that a missing needle fails the test
    with a message naming it rather than erroring with a bare ValueError."""
    at = text.find(needle, start)
    if at < 0:
        raise AssertionError(f"{where}: no {needle!r}")
    return at


def function_body(text, signature):
    start = find(text, signature)
    end = find(text, "\n}\n", start)
    return text[start:end]


class V4MoeBatchLifecycleSourceTest(unittest.TestCase):
    def setUp(self):
        self.text = SOURCE.read_text(encoding="utf-8")

    def test_every_generate_exit_releases_the_bank(self):
        where = "coli_v4_session_generate"
        body = function_body(self.text, "int coli_v4_session_generate(ColiV4Session")
        call = find(body, "int result = session_generate_impl(", where=where)
        release = find(body, "coli_v4_gpu_moe_batch_release();", call,
                       where=f"{where}, after the session_generate_impl call")
        find(body, "return result;", release, where=f"{where}, after the release")
        self.assertEqual(body.count("return"), 1,
                         f"{where}: expected exactly one return")

    def test_engine_close_releases_before_teardown(self):
        where = "coli_v4_gpu_engine_close"
        body = function_body(self.text, "void coli_v4_gpu_engine_close(")
        release = find(body, "coli_v4_gpu_moe_batch_release();", where=where)
        for later in ("dsv4_cuda_tensor_free(", "v4_gpu_expert_mirrors_free(",
                      "dsv4_cuda_shutdown();"):
            self.assertLess(release, find(body, later, where=where),
                            f"{where} calls {later} before the release")


if __name__ == "__main__":
    unittest.main()
