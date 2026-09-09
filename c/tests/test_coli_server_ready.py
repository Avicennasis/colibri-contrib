"""`coli chat`'s private-server drain-to-ready + post-ready liveness (#50404):
surface the child's real exit reason when it dies during load, and refuse /
name half-alive servers (HTTP up, engine dead — FreeToken supervisor.py pattern,
issues #110/#123 class)."""
import os
from importlib.machinery import SourceFileLoader
import importlib.util
import time
import types
import unittest
from pathlib import Path
from unittest import mock


COLI = Path(__file__).resolve().parent.parent / "coli"


def load_coli():
    # No .py extension (it is the `coli` launcher): the loader must be explicit.
    loader = SourceFileLoader("coli_server_ready_test", str(COLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coli = load_coli()


class FakeChild:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


class ChildExitReasonTest(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "Windows has no SIGKILL; the code's 'signal 9' fallback is the correct answer there (#51279)")
    def test_signal_death_names_the_oom_killer(self):
        self.assertEqual(coli._child_exit_reason(FakeChild(-9)), "killed by SIGKILL")

    def test_exit_codes_decode(self):
        self.assertEqual(coli._child_exit_reason(FakeChild(3)), "exit code 3")
        self.assertEqual(coli._child_exit_reason(FakeChild(0)), "exited cleanly")

    def test_alive_child_says_so(self):
        self.assertIn("still alive", coli._child_exit_reason(FakeChild(None)))


class WaitServerReadyTest(unittest.TestCase):
    def _ready(self, child, health, timeout_s=1.0):
        with mock.patch.object(coli, "_probe_model_id", return_value="glm-5.2"):
            return coli.wait_server_ready(child, "http://127.0.0.1:8000",
                                          poll_s=0.01, timeout_s=timeout_s, health=health)

    def test_child_death_during_load_surfaces_the_reason(self):
        mid, why = self._ready(FakeChild(-9), lambda: None)
        self.assertIsNone(mid)
        self.assertIn("killed by SIGKILL", why)
        self.assertIn("while loading", why)

    def test_half_alive_server_is_named(self):
        # The child lives, /health answers ok -- with engine=dead. This is the
        # case the old loop could only ever reach "timed out" on.
        health = {"status": "ok", "engine": "dead", "engine_reason": "killed by SIGKILL"}
        mid, why = self._ready(FakeChild(None), lambda: health)
        self.assertIsNone(mid)
        self.assertIn("half-alive", why)
        self.assertIn("killed by SIGKILL", why)

    def test_healthy_server_returns_model_id(self):
        mid, why = self._ready(FakeChild(None), lambda: {"status": "ok"})
        self.assertEqual((mid, why), ("glm-5.2", None))

    def test_unready_server_times_out(self):
        mid, why = self._ready(FakeChild(None), lambda: {}, timeout_s=0.05)
        self.assertEqual((mid, why), (None, "timed out"))

    def test_engine_loading_is_not_half_alive(self):
        # Bind-before-load: /health answers ok with engine=loading during the
        # drain. That is a starting server, not a broken one -- keep waiting.
        states = [{"status": "ok", "engine": "loading"}, {"status": "ok"}]
        mid, why = self._ready(FakeChild(None), lambda: states.pop(0) if states else {"status": "ok"})
        self.assertEqual((mid, why), ("glm-5.2", None))


class ServerProbeTest(unittest.TestCase):
    def test_refuses_half_alive_server(self):
        health = {"status": "ok", "engine": "dead", "engine_reason": "killed by SIGKILL"}
        with mock.patch.object(coli, "_server_health", return_value=health), \
             mock.patch.object(coli, "_probe_model_id", return_value="glm-5.2") as models:
            self.assertIsNone(coli.server_probe("http://127.0.0.1:8000"))
            models.assert_not_called()

    def test_older_and_loading_servers_still_attach(self):
        # A server built before /health carried an engine field reports nothing
        # there; engine="loading" is a starting server. Both must stay attachable.
        for health in ({"status": "ok"}, {"status": "ok", "engine": "loading"}):
            with mock.patch.object(coli, "_server_health", return_value=health), \
                 mock.patch.object(coli, "_probe_model_id", return_value="glm-5.2"):
                self.assertEqual(coli.server_probe("http://127.0.0.1:8000"), "glm-5.2")


class WatchServerTest(unittest.TestCase):
    def test_names_engine_death_mid_session(self):
        child = FakeChild(None)
        health = {"status": "ok", "engine": "dead", "engine_reason": "killed by SIGKILL"}
        printed = []
        with mock.patch.object(coli, "_server_health", return_value=health), \
             mock.patch("builtins.print", lambda *a, **k: printed.extend(a)):
            stop = coli._watch_server(child, "http://127.0.0.1:8000", poll_s=0.01)
            time.sleep(0.3)           # first poll fires, names the death, returns
            stop.set()
        self.assertTrue(any("server lost" in line and "SIGKILL" in line for line in printed))

    def test_quiet_on_orderly_quit(self):
        # The caller sets the stop event before terminating the child: an
        # expected death must not be misreported (supervisor.py's
        # shutting-down discrimination).
        child = FakeChild(None)
        health = {"status": "ok", "engine": "dead", "engine_reason": "exit code 0"}
        printed = []
        with mock.patch.object(coli, "_server_health", return_value=health), \
             mock.patch("builtins.print", lambda *a, **k: printed.extend(a)):
            stop = coli._watch_server(child, "http://127.0.0.1:8000", poll_s=0.5)
            stop.set()                # orderly quit wins before the first poll
            time.sleep(0.7)
        self.assertEqual(printed, [])


if __name__ == "__main__":
    unittest.main()
