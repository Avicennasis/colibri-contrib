# Local cluster mode

The coordinator keeps token generation, routing, and KV state local while
disk-backed expert workers execute routed FFNs on other Macs. A layer's routed
batch-union is sent as one persistent TCP request, so a token does not incur one
round trip per expert.

Two engines speak the wire (`COLIEX01`): GLM-5.2 (`colibri`) and GLM-5.3
(`glm53`). A coordinator talks to workers of its own family only, since the
expert math is the family's; `coli cluster worker` picks the binary from the
model's `config.json`, as every other launcher does. On GLM-5.3 the workers
serve the int4 expert container (the streamed experts); KDA, the indexer, the
attention and the shared expert stay on the coordinator, and the answer is
bit-identical to the single-machine run whatever the number of workers.

Start the optional registration service:

```bash
./coli cluster coordinator --host 0.0.0.0 --port 8765
```

On each worker, with the same converted model available locally:

```bash
./coli cluster worker --model /nvme/glm52_i4 --port 9100 \
  --coordinator http://COORDINATOR:8765 --advertise-host WORKER_IP
```

Run the coordinator with discovery, or provide `--cluster-workers
HOST:PORT,...` for a static setup:

```bash
./coli serve --model /nvme/glm52_i4 \
  --cluster-coordinator http://127.0.0.1:8765
```

The transport is disabled unless workers are configured, so the existing
single-machine path remains unchanged. Dense-layer sharding and browser/WebGPU
workers are separate follow-up seams.

The gates: `tests/test_cluster_protocol.c` and `tests/test_glm53_cluster_protocol.c`
pin the wire contract of each engine over a socketpair; `tests/test_cluster_sharding.py`
(GLM-5.2) and `tests/test_glm53_cluster_sharding.py` (GLM-5.3) run the tiny
oracle fixture with and without workers and require the same tokens, with zero
oracle mismatches on both sides.
