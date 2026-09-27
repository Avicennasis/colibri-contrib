"""LogRing: bounded ring + all-time `?since=` cursor, ported from
FreeToken's daemon logring.py. The cursor semantics are the contract scripts
poll with — the tests pin them: exclusive lower bound, monotonic next that
survives eviction, dropped-count derivable from (next - since - len)."""
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from logring import LogRing


class LogRingTest(unittest.TestCase):
    def test_since_is_exclusive_of_already_seen(self):
        ring = LogRing()
        ring.append("a"); ring.append("b")
        lines, next1 = ring.since(0)
        self.assertEqual([l["text"] for l in lines], ["a", "b"])
        self.assertEqual(next1, 2)
        lines, next2 = ring.since(next1)          # nothing new
        self.assertEqual((lines, next2), ([], 2))
        ring.append("c")
        lines, next3 = ring.since(next1)
        self.assertEqual([l["text"] for l in lines], ["c"])
        self.assertEqual(next3, 3)

    def test_capacity_evicts_oldest_but_seq_survives(self):
        ring = LogRing(capacity=3)
        for i in range(7):
            ring.append(f"line{i}")
        self.assertEqual(len(ring), 3)
        lines, nxt = ring.since(0)
        self.assertEqual([l["text"] for l in lines], ["line4", "line5", "line6"])
        self.assertEqual(nxt, 7)
        self.assertEqual([l["seq"] for l in lines], [4, 5, 6])   # no seq reuse

    def test_asleep_poller_can_count_what_it_missed(self):
        ring = LogRing(capacity=2)
        for i in range(10):
            ring.append(f"line{i}")
        lines, nxt = ring.since(1)                 # slept since seq 1
        dropped = nxt - 1 - len(lines)
        self.assertEqual(dropped, 7)               # seqs 1..7 fell off the ring
        self.assertEqual([l["seq"] for l in lines], [8, 9])

    def test_records_carry_kind_and_timestamp(self):
        ring = LogRing()
        rec = ring.append("GET /v1/models 200 3ms", kind="request")
        self.assertEqual((rec["kind"], rec["text"], rec["seq"]), ("request", "GET /v1/models 200 3ms", 0))
        self.assertIsInstance(rec["ts"], float)

    def test_concurrent_appends_never_reuse_a_seq(self):
        ring = LogRing(capacity=64)
        def hammer():
            for _ in range(200):
                ring.append("x", kind="request")
        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(ring), 64)
        lines, nxt = ring.since(0)
        seqs = [l["seq"] for l in lines]
        self.assertEqual(nxt, 8 * 200)
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(seqs), len(set(seqs)))     # unique == monotonic


if __name__ == "__main__":
    unittest.main()
