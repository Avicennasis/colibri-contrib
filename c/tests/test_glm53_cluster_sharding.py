"""Token-exact parity gate for GLM-5.3: local experts vs cluster-delegated.

The GLM-5.3 counterpart of test_cluster_sharding.py (GLM-5.2). The engine
routes ONLY the routed experts to the workers: `cluster_moe_batch` shards the
layer's union of chosen experts, which ffn_layer_ex builds from the router's
per-token picks; the shared expert, KDA, the indexer and the attention run on
the coordinator in both the baseline and the delegated run, so they are not
part of the shard under test.

Three things are asserted, and each could fail on its own:

  1. the oracle: the baseline run (no workers) reproduces ref.json's
     teacher-forcing and greedy tokens at every position, which is what
     glm53_tiny_harness.py checks on the f32 fixture and which the int4 pair
     derived from it keeps (measured; the multimodal pair does not, int4
     rounding on a 128-wide random model flips one token there);
  2. the parity: the delegated run prints the same teacher_forcing, greedy
     and last_logits lines, byte for byte -- the workers compute the same
     bytes with the same kernels, so the sum is the same floats. The engine
     prints the logits of the last position only, so every prefix of the
     prompt runs too, with and without workers: prefix k's last_logits are
     position k-1's, and the parity is bit-exact at every position, not just
     the last. (A worker that nudged one row of its answer by 1e-3 passed the
     last-position check: the nudged row was never the last token's.)
  3. the delegation acted: the coordinator reports the workers it connected to
     and reads no expert from its own disk (`experts hits 0 miss 0`), while the
     baseline did read them. Without this a worker that nobody asked would
     pass the parity trivially.

Two workers, so the (eid + layer) % workers shard rule sends experts to both
and a request that reaches the wrong worker would be a different answer.

Fixture: the int4 streaming pair of the tiny text oracle, generated and not
committed, exactly as the GLM-5.3 CI job does it:

    python3 tools/make_glm53_tiny.py --output /tmp/glm53_tiny
    python3 tools/make_glm53_streaming_pair.py --fixture /tmp/glm53_tiny \\
                                               --output /tmp/glm53_tiny_stream
    GLM53_TINY_STREAM=/tmp/glm53_tiny_stream-i4 python3 -m unittest tests.test_glm53_cluster_sharding

The f32 fixture itself cannot serve here: the engine streams (and so can
delegate) experts only from the int4 container, resident f32 experts never
reach the hook. The workers and the coordinator run the SAME c/glm53 binary
with the same numeric env, so the comparison is meaningful rather than lucky.
"""

import json
import os
import re
import socket
import subprocess
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ENGINE = next((C_DIR / name for name in ("glm53", "glm53.exe")
               if (C_DIR / name).exists()), None)
FIXTURE = Path(os.environ["GLM53_TINY_STREAM"]) if os.environ.get("GLM53_TINY_STREAM") else None

COMPARED = ("teacher_forcing", "greedy", "last_logits")
WORKERS = 2

# The dense precision and the thread sizing, pinned for every process so the
# result does not depend on the shell that runs the test.
_NUMERIC_ENV = {
    "GLM53_BITS": "32",
    "COLI_NO_OMP_TUNE": "1",
    "COLI_METAL": "0",
    "COLI_VULKAN": "0",
}


def _fixture_ok(d):
    return d is not None and all((d / f).exists()
                                 for f in ("config.json", "model.safetensors", "ref.json"))


def _skip_reason():
    if ENGINE is None:
        return "glm53 is not built (run: make glm53)"
    if FIXTURE is None:
        return "GLM53_TINY_STREAM not set to the int4 pair of the tiny oracle"
    return f"{FIXTURE} is not a fixture (config.json, model.safetensors, ref.json)"


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _lines(result):
    return {line.split()[0]: line.split()[1:]
            for line in result.stdout.splitlines() if line.strip()}


def _expert_counters(result):
    match = re.search(r"^experts hits (\d+) miss (\d+) bytes (\d+)$", result.stdout, re.M)
    return tuple(int(v) for v in match.groups()) if match else None


@unittest.skipUnless(ENGINE is not None and _fixture_ok(FIXTURE), _skip_reason())
class Glm53ClusterShardingParityTest(unittest.TestCase):
    """Local CPU must equal cluster-delegated expert sharding, token-exact."""

    def _command(self, ids, greedy):
        return [str(ENGINE), "--model", str(FIXTURE), "--ids", ",".join(map(str, ids)),
                "--greedy", str(greedy), "--logits"]

    def _run(self, env, ids, greedy):
        result = subprocess.run(self._command(ids, greedy), cwd=C_DIR, env=env,
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def _start_worker(self, port):
        env = {**os.environ, **_NUMERIC_ENV,
               "SNAP": str(FIXTURE), "EXPERT_WORKER": "1", "CLUSTER_WORKER_PORT": str(port)}
        env.pop("CLUSTER_WORKERS", None)
        worker = subprocess.Popen([str(ENGINE)], cwd=C_DIR, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if worker.poll() is not None:
                stderr = worker.stderr.read() if worker.stderr else ""
                self.fail(f"cluster worker exited early: {stderr}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    return worker
            except OSError:
                time.sleep(0.05)
        worker.kill()
        self.fail("cluster worker did not start listening")

    def _stop(self, worker):
        worker.terminate()
        try:
            worker.wait(timeout=2)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        for stream in (worker.stdout, worker.stderr):
            if stream is not None:
                stream.close()

    def test_delegated_experts_match_the_oracle_token_for_token(self):
        ref = json.loads((FIXTURE / "ref.json").read_text())
        ids, greedy = ref["prompt_ids"], len(ref["greedy_new_ids"])
        common_env = {**os.environ, **_NUMERIC_ENV}
        for name in ("CLUSTER_WORKERS", "EXPERT_WORKER"):
            common_env.pop(name, None)

        ports = [_free_port() for _ in range(WORKERS)]
        cluster_env = {**common_env, "CLUSTER_WORKERS": ",".join(f"127.0.0.1:{p}" for p in ports)}
        workers = []
        per_position = []   # (position, baseline last_logits, delegated last_logits)
        try:
            for port in ports:
                workers.append(self._start_worker(port))
            baseline = self._run(common_env, ids, greedy)
            delegated = self._run(cluster_env, ids, greedy)
            for k in range(1, len(ids) + 1):
                per_position.append((k - 1,
                                     _lines(self._run(common_env, ids[:k], 0)).get("last_logits"),
                                     _lines(self._run(cluster_env, ids[:k], 0)).get("last_logits")))
        finally:
            for worker in workers:
                self._stop(worker)

        base, dele = _lines(baseline), _lines(delegated)

        # 1. the oracle, position by position, so a miss names where it is
        mismatches = [(pos, got, want) for pos, (got, want) in enumerate(
            zip([int(v) for v in base["teacher_forcing"]], ref["teacher_forcing_ids"], strict=True))
            if got != want]
        self.assertEqual(mismatches, [], f"baseline mismatched the oracle at (position, got, expected): {mismatches}")
        self.assertEqual([int(v) for v in base["greedy"]], ref["greedy_new_ids"],
                         "baseline greedy tokens differ from the oracle")

        # 2. the parity, byte for byte on every compared line, then the logits
        #    of every position through the prefixes
        for field in COMPARED:
            self.assertEqual(base.get(field), dele.get(field),
                             f"cluster delegation changed `{field}`")
        for position, local, remote in per_position:
            self.assertIsNotNone(local, f"prefix {position + 1} printed no last_logits")
            self.assertEqual(local, remote,
                             f"cluster delegation changed the logits at position {position}")

        # 3. the delegation acted
        self.assertIn(f"[CLUSTER] coordinator connected to {WORKERS} expert worker(s)",
                      delegated.stderr, "the coordinator did not report its workers")
        base_counters, dele_counters = _expert_counters(baseline), _expert_counters(delegated)
        self.assertIsNotNone(base_counters, f"no expert counters: {baseline.stdout}")
        self.assertIsNotNone(dele_counters, f"no expert counters: {delegated.stdout}")
        self.assertGreater(base_counters[1], 0, "the baseline streamed no expert: not a streaming fixture")
        self.assertEqual(dele_counters, (0, 0, 0),
                         "the coordinator read experts from its own disk while delegating")

    def test_a_coordinator_with_no_reachable_worker_refuses_to_run(self):
        """Silently computing locally would make every parity check vacuous."""
        ref = json.loads((FIXTURE / "ref.json").read_text())
        env = {**os.environ, **_NUMERIC_ENV, "CLUSTER_WORKERS": f"127.0.0.1:{_free_port()}"}
        env.pop("EXPERT_WORKER", None)
        result = subprocess.run(self._command(ref["prompt_ids"], 0), cwd=C_DIR, env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("[CLUSTER] no expert workers reachable", result.stderr)
        self.assertNotIn("teacher_forcing", result.stdout)


if __name__ == "__main__":
    unittest.main()
