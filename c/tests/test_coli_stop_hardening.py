"""`coli stop` hardening (#50404): /proc starttime PID-reuse guard, identity-checked
group signalling, PSS via smaps_rollup. Ported from FreeToken's daemon osproc.py
(Apache-2.0) -- read_starttime / proc_pgid / signal_group / read_pss_bytes.

The reuse tests double as mutation tests: delete the starttime comparison (or the
pgid==pid assertion) and the corresponding test below fails, because the fixture
presents exactly the recycled-pid / foreign-group case the guard exists for."""
import os
from importlib.machinery import SourceFileLoader
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


COLI = Path(__file__).resolve().parent.parent / "coli"


def load_coli():
    # No .py extension (it is the `coli` launcher): the loader must be explicit.
    loader = SourceFileLoader("coli_stop_hardening_test", str(COLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coli = load_coli()

# /proc/<pid>/stat fixture: comm contains spaces AND a nested paren so a naive
# whitespace split would miscount fields. field 5 (pgrp) = 4242 == pid,
# field 22 (starttime) = 999777.
STAT = b"4242 (exe (worker) v2) S 1 4242 4242 0 -1 4194304 0 0 0 5 3 0 0 0 2 0 10 0 999777 140737488640 512 0\n"


class ProcStatParseTest(unittest.TestCase):
    def test_starttime_and_pgid_parse_past_comm_with_spaces(self):
        with mock.patch("builtins.open", mock.mock_open(read_data=STAT)):
            self.assertEqual(coli._proc_starttime(4242), 999777)
            self.assertEqual(coli._proc_pgid(4242), 4242)

    def test_vanished_pid_is_none_not_an_error(self):
        with mock.patch("builtins.open", side_effect=FileNotFoundError("/proc/9/stat")):
            self.assertIsNone(coli._proc_starttime(9))
            self.assertIsNone(coli._proc_pgid(9))
            self.assertEqual(coli._proc_pss_gb(9), 0.0)

    def test_short_stat_line_is_none(self):
        with mock.patch("builtins.open", mock.mock_open(read_data=b"1 (x) R\n")):
            self.assertIsNone(coli._proc_starttime(1))


class PidfileTest(unittest.TestCase):
    def test_parses_optional_starttime(self):
        with tempfile.TemporaryDirectory() as d:
            pf = Path(d) / "coli-serve-8000.pid"
            pf.write_text("4242 /models/glm 999777\n")
            self.assertEqual(coli._pidfile_pid(str(pf)), (4242, 999777))
            pf.write_text("4242 /models/glm\n")          # legacy: pre-hardening pidfile
            self.assertEqual(coli._pidfile_pid(str(pf)), (4242, None))
            pf.write_text("4242 /models/glm not-a-number\n")
            self.assertEqual(coli._pidfile_pid(str(pf)), (4242, None))
            self.assertEqual(coli._pidfile_pid(str(pf / "absent")), (None, None))
            pf.write_text("garbage\n")
            self.assertEqual(coli._pidfile_pid(str(pf)), (None, None))


class PidReuseGuardTest(unittest.TestCase):
    """The PID-reuse fixture: same pid, different start time = a recycled pid that
    must NOT be signalled, however alive it looks."""

    def test_starttime_mismatch_means_recycled(self):
        with mock.patch.object(coli, "_pid_alive", return_value=True), \
             mock.patch.object(coli, "_proc_starttime", return_value=111):
            self.assertFalse(coli._pid_is_same_process(4242, 999777))

    def test_matching_starttime_passes(self):
        with mock.patch.object(coli, "_pid_alive", return_value=True), \
             mock.patch.object(coli, "_proc_starttime", return_value=999777):
            self.assertTrue(coli._pid_is_same_process(4242, 999777))

    def test_legacy_pidfile_and_dead_pid_keep_old_semantics(self):
        with mock.patch.object(coli, "_pid_alive", return_value=True), \
             mock.patch.object(coli, "_proc_starttime", return_value=None):
            self.assertTrue(coli._pid_is_same_process(4242, None))   # no /proc: degrade
        with mock.patch.object(coli, "_pid_alive", return_value=False):
            self.assertFalse(coli._pid_is_same_process(4242, 999777))


@unittest.skipIf(os.name == "nt", "POSIX process groups: os.getpgrp/killpg do not exist on Windows")
class GroupSignalTest(unittest.TestCase):
    def _signal(self, pgid, own_pgrp=999, killpg_error=None, kill_error=None):
        killpg = mock.Mock(side_effect=killpg_error)
        kill = mock.Mock(side_effect=kill_error)
        with mock.patch.object(coli, "_proc_pgid", return_value=pgid), \
             mock.patch.object(coli.os, "getpgrp", return_value=own_pgrp), \
             mock.patch.object(coli.os, "killpg", killpg), \
             mock.patch.object(coli.os, "kill", kill):
            signalled = coli._stop_signal(4242, 15)
        return killpg, kill, signalled

    def test_group_leader_gets_killpg(self):
        killpg, kill, ok = self._signal(pgid=4242)
        killpg.assert_called_once_with(4242, 15)
        kill.assert_not_called()
        self.assertTrue(ok)

    def test_non_leader_gets_single_pid_kill(self):
        # A recycled pid sitting in a foreign group: pgid != pid, so never killpg.
        killpg, kill, ok = self._signal(pgid=17)
        killpg.assert_not_called()
        kill.assert_called_once_with(4242, 15)
        self.assertTrue(ok)

    def test_never_signals_colis_own_group(self):
        # Even a leader whose group IS ours (coli stop launched from the same job):
        # a killpg would take out coli itself.
        killpg, kill, ok = self._signal(pgid=4242, own_pgrp=4242)
        killpg.assert_not_called()
        kill.assert_called_once_with(4242, 15)

    def test_killpg_race_falls_back_to_pid(self):
        killpg, kill, ok = self._signal(pgid=4242, killpg_error=ProcessLookupError())
        killpg.assert_called_once()
        kill.assert_called_once_with(4242, 15)
        self.assertTrue(ok)

    def test_dead_target_reports_unsignalled(self):
        killpg, kill, ok = self._signal(pgid=17, kill_error=ProcessLookupError())
        self.assertFalse(ok)


class PssTest(unittest.TestCase):
    def test_sums_rollup_pss_kb(self):
        rollup = b"Rss:\t 131072 kB\nPss:\t 24576 kB\nSwapPss:\t 0 kB\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=rollup)):
            self.assertAlmostEqual(coli._proc_pss_gb(4242), 24576 * 1024 / 1e9)

    def test_missing_pss_line_is_zero(self):
        with mock.patch("builtins.open", mock.mock_open(read_data=b"Rss:\t 100 kB\n")):
            self.assertEqual(coli._proc_pss_gb(4242), 0.0)


if __name__ == "__main__":
    unittest.main()
