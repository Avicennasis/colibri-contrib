#!/usr/bin/env python3
"""logring — bounded in-memory log ring for `coli serve`.

Ported from FreeToken's daemon logring.py (Apache-2.0), minus its SSE
fan-out: colibri's gateway is a thread-per-connection http.server with no
event loop to hang a push on, so the contract here is pull-only — a script
polls GET /logs?since=N and receives the records with seq >= N plus the
next cursor to poll with. The records live in RAM only (nothing written,
nothing to rotate), the deque bounds memory by evicting the oldest line,
and the all-time sequence counter survives eviction, so a poller that
slept through a full ring still learns HOW MUCH it missed:
next - cursor - len(lines) is the dropped count.

stdlib only, importable from openai_server and coli alike.

Threading: request threads (one per HTTP connection) call append() and
since() concurrently; one internal lock guards buffer, counter and
capacity. Best-effort by design — this is a diagnostics surface, never a
dependency of the request path.
"""
import threading
import time
from collections import deque


class LogRing:
    def __init__(self, capacity=4000):
        self._buf = deque(maxlen=capacity)   # old lines evicted, seqs never reused
        self._next = 0                       # all-time monotonic cursor base
        self._lock = threading.Lock()

    def append(self, text, kind="event"):
        """Add one finalized line; returns the stored record with its `seq`."""
        rec = {"ts": time.time(), "kind": kind, "text": str(text)}
        with self._lock:
            rec["seq"] = self._next
            self._buf.append(rec)
            self._next += 1
        return rec

    def since(self, cursor):
        """(records with seq >= cursor, next cursor). `cursor` is EXCLUSIVE of
        what the caller has already seen — pass the `next` from the previous
        poll, 0 for the first one. A cursor older than the oldest surviving
        record returns the whole ring: the gap is next - cursor - len()."""
        with self._lock:
            return [r for r in self._buf if r["seq"] >= cursor], self._next

    def cursor(self):
        with self._lock:
            return self._next

    def __len__(self):
        with self._lock:
            return len(self._buf)
