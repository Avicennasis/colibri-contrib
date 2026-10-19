"""Speculative-decode batches never reach the batched GPU MoE's expert bank.

coli_v4_gpu_moe_batch_union (COLI_CUDA_MOE_BATCH=1) serves prefills from a
process-wide expert bank. Its only size gate used to be the prompt-size hint
that generate() sets once per request, so the speculative verify and replay
batches of the decode that follows (target_batch with batch > 1) took the
bank too: re-created after the prefill had released it, and refilled layer
by layer on every draft step. That is silent on output (the CPU union gives
the same bits) and only shows up as VRAM, store traffic and decode speed,
which no fixture without the 4096-wide, 256-expert geometry can reach. So
this pins the shape at source level:

- target_batch passes its use_prefill_pool intent down for the length of the
  call, and the union refuses decode batches before any bank work;
- in generate(), the prefill loop's target_batch passes 1 and every call
  after the prefill's bank release (speculative verify/replay) passes 0.
"""
import re
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


def generate_body(text):
    """The body that runs generate()'s prefill loop and decode: the public
    coli_v4_session_generate, or the impl it wraps."""
    for signature in ("static int session_generate_impl(",
                      "int coli_v4_session_generate(ColiV4Session"):
        if signature in text:
            body = function_body(text, signature)
            if "target_batch(" in body:
                return body
    raise AssertionError("generate() body not found")


def call_arguments(text, start):
    """Top-level arguments of the call whose '(' is at or after start."""
    open_at = find(text, "(", start)
    depth, current, arguments = 0, [], []
    for char in text[open_at + 1:]:
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                arguments.append("".join(current).strip())
                return arguments
            depth -= 1
        if char == "," and depth == 0:
            arguments.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    raise AssertionError("unterminated call")


class V4MoeBatchSpeculativeSourceTest(unittest.TestCase):
    def setUp(self):
        self.text = SOURCE.read_text(encoding="utf-8")

    def test_target_batch_passes_the_prefill_intent_down(self):
        where = "target_batch"
        body = function_body(self.text, "static int target_batch(ColiV4Engine")
        mark = find(body, "coli_v4_gpu_moe_batch_prefill(use_prefill_pool ? 1 : 0);",
                    where=where)
        call = find(body, "target_batch_impl(", mark,
                    where=f"{where}, after setting the prefill flag")
        find(body, "coli_v4_gpu_moe_batch_prefill(-1);", call,
             where=f"{where}, after the target_batch_impl call")

    def test_union_refuses_decode_batches_before_any_bank_work(self):
        where = "coli_v4_gpu_moe_batch_union"
        body = function_body(self.text, "int coli_v4_gpu_moe_batch_union(")
        refuse = find(body, "if (!v4_gpu_moe_batch_in_prefill) return -1;",
                      where=where)
        for bank_work in ("dsv4_cuda_expert_bank_create(",
                          "pthread_create(&v4_bank2_thread"):
            self.assertLess(refuse, find(body, bank_work, where=where),
                            f"{where} reaches {bank_work} before the refusal")

    def test_generate_prefill_passes_1_and_decode_passes_0(self):
        body = generate_body(self.text)
        release = find(body, "coli_v4_gpu_moe_batch_release();",
                       where="generate()")
        calls = [match.start() for match in
                 re.finditer(r"\btarget_batch\(", body)]
        self.assertTrue(any(at < release for at in calls),
                        "generate(): no target_batch call before the release")
        self.assertTrue(any(at > release for at in calls),
                        "generate(): no target_batch call after the release")
        for at in calls:
            arguments = call_arguments(body, at)
            self.assertEqual(len(arguments), 15, arguments)
            self.assertEqual(arguments[10], "1" if at < release else "0",
                             f"target_batch at offset {at}: {arguments}")


if __name__ == "__main__":
    unittest.main()
