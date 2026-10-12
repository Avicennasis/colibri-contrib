#!/usr/bin/env python3
"""Qwen3.8: the routed experts as int4-g64, in a sidecar next to the release.

The release stores every routed expert as FP8 e4m3 with one scale per 128x128
block, 4.9 MB per expert on Qwen3.8-Flash-Next. Decode waits on those bytes:
an expert that is not in the RAM cache is read from disk, and how many fit in
the cache decides how often that happens. This converter writes the same
experts as int4 with one f32 scale per 64 inputs, 2.76 MB per expert, into
<model>/experts-int4g64/. The FP8 shards are only read: the snapshot stays
usable as it is, and qwen38 picks the sidecar up when it is there
(Q38_EXPERT_INT4=0 ignores it).

Only the routed experts change. The shared expert, attention, DeltaNet, PLE,
vision and MTP tensors are not read at all.

Numerics, so the engine and every tier read the bytes they already know:
  dequantize  e4m3 byte -> quant.h's E4M3_LUT value, times the block's
              weight_scale_inv (BF16 or F32, widened to f32), one f32 multiply:
              exactly the expanded FP8 path in qwen38_core.h. BF16/F16/F32
              experts (the tiny fixture, fused or per expert) are widened.
  quantize    tools/convert_qwen36.py's quantize_row_grouped with --ebits 4
              --gs 64: per row, groups of 64 inputs (a short last group is
              zero-padded), scale = max(absmax, 1e-12) / 7 in f32, code =
              round-half-even(w / scale) clamped to [-8, 7].
  codes       stored as v+8 in expert_ffn.h's planar layout: a row is blocks
              of 64 elements, 32 bytes each, byte k holding element k in its
              low nibble and element k+32 in its high nibble. A width that is
              not a multiple of 64 keeps its tail in pairs (byte j = element
              2j low, 2j+1 high), as idot.h's planarize_i4 leaves it. A row is
              ceil(cols/2) bytes.
  scales      f32, [rows][ceil(cols/64)].

Layout: one safetensors file per layer, layer-NNN.safetensors, and index.json.
Every expert is one contiguous record, so a cache miss is one read:

  gate codes U8 [I, H/2] | up codes U8 [I, H/2] | down codes U8 [H, I/2] |
  gate scales F32 [I, H/64] | up scales F32 [I, H/64] | down scales F32 [H, I/64]

named like the release's tensors (<prefix>.layers.L.mlp.experts.E.gate_proj.weight
and the same name + ".qs" for its scales, the .qs convention of the other
int4 containers). Records follow each other with no gap; the data region of
a file starts on a 4096-byte boundary, so on the release's geometry (record
2,764,800 bytes = 675 pages) every record is page aligned.

index.json carries the format name and version, the geometry, the group size,
the record size and the layers already written. It is rewritten (atomically)
after every layer and says "complete": true only when all of them are there;
the engine refuses anything else. A layer is written to a .part file and
renamed when it is whole, so an interrupted run resumes at the first layer
the index does not list.

--int2 writes int2_g32.h's format into <model>/experts-int2g32/ instead
(qwen38 reads it with Q38_EXPERT_INT2=1): per row, groups of 32 inputs on the
fixed levels -10, -3, 3, 10 times the group's scale, the scales stored as one
u8 per group under one f32 per row; 1.40 MB per expert, 2.25 bits per weight.
A group's scale and codes are fitted in turn (nearest level, then the least
squares scale for those codes) against the activation-weighted error when
--act-stats is given, then the u8 scales are refitted for the codes they
end up with. The record:

  gate codes U8 [I, H/4] | up codes | down codes U8 [H, I/4] |
  gate q U8 [I, H/32] | up q | down q U8 [H, I/32] |
  gate d F32 [I, 1] | up d | down d F32 [H, 1]

named <...>.weight, <...>.weight.q8 and <...>.weight.d.

Usage:
  python3 tools/convert_qwen38_experts_int4.py --model /path/to/qwen38-fp8 --plan
  python3 tools/convert_qwen38_experts_int4.py --model /path/to/qwen38-fp8 [--workers N]
  python3 tools/convert_qwen38_experts_int4.py --model /path/to/qwen38-fp8 --int2 --act-stats act.bin

Peak RAM does not grow with the model: each worker holds one expert in flight
(about 0.11 GB with numpy), and the writer at most the records of one layer
(1.4 GB) when the disk lags behind the workers. Measured on a 12-thread
i7-1355U with a synthetic snapshot of the release's expert geometry: 22 s of
compute per layer with 12 workers.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import threading
import time

import numpy as np

try:
    sys.stdout.reconfigure(errors="backslashreplace")
except AttributeError:
    pass

SIDECAR_DIR = "experts-int4g64"
FORMAT = "colibri.qwen38.experts-int4g64"
VERSION = 1
GS = 64
CODES = "planar64-v8"
SCALES = "f32-per-64"
DATA_ALIGN = 4096
PROJ = ("gate_proj", "up_proj", "down_proj")
SIDECAR_DIR2 = "experts-int2g32"
FORMAT2 = "colibri.qwen38.experts-int2g32"
GS2 = 32
CODES2 = "planar128-2bit-lv10"
SCALES2 = "u8-per-32-f32-per-row"
LEVELS2 = np.array([-10.0, -3.0, 3.0, 10.0])
FP8_BLOCK = 128

_FLOAT_DTYPES = {"BF16": 2, "F16": 2, "F32": 4}
_FP8_DTYPES = ("F8_E4M3", "F8_E4M3FN", "float8_e4m3fn")


def e4m3_table():
    """quant.h's E4M3_LUT (torch.float8_e4m3fn): bias 7, subnormals at
    exponent 0, only 0x7F/0xFF are NaN (no infinities)."""
    table = np.empty(256, np.float32)
    for byte in range(256):
        exponent, mantissa = (byte >> 3) & 0xF, byte & 7
        if exponent == 0xF and mantissa == 7:
            value = math.nan
        elif exponent == 0:
            value = mantissa / 8.0 * 2.0 ** -6
        else:
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        table[byte] = -value if byte & 0x80 else value
    return table


E4M3 = e4m3_table()


# ---------- int4-g64: quantize, pack, and their inverses ----------

def groups(cols):
    return (cols + GS - 1) // GS


def row_bytes(cols):
    return (cols + 1) // 2


def quantize_g64(w):
    """w f32 [R, C] -> (codes u8 [R, C] holding v+8, scales f32 [R, ceil(C/64)]).
    tools/convert_qwen36.py quantize_row_grouped, bits=4, in numpy."""
    rows, cols = w.shape
    ng = groups(cols)
    pad = ng * GS - cols
    padded = np.pad(w, ((0, 0), (0, pad))) if pad else w
    g = padded.reshape(rows, ng, GS)
    amax = np.abs(g).max(axis=2, keepdims=True)
    scales = np.maximum(amax, np.float32(1e-12)) / np.float32(7.0)
    codes = np.clip(np.rint(g / scales), -8, 7).astype(np.int16) + 8
    return codes.reshape(rows, ng * GS)[:, :cols].astype(np.uint8), scales.reshape(rows, ng)


ALPHAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)


def quantize_g64_weighted(w, imp, alphas=ALPHAS):
    """quantize_g64 with the scale of every group of 64 chosen among absmax/7 * alpha
    (alphas holds 1.0, so it is never worse than quantize_g64 by this measure) to
    minimise sum_i imp_i * (w_i - s * code_i)^2: imp is the mean square of each input
    channel's activations over calibration text (Q38_ACT_STATS), so an error counts as
    much as the product it spoils. Same codes and scales layout as quantize_g64."""
    rows, cols = w.shape
    ng = groups(cols)
    pad = ng * GS - cols
    padded = np.pad(w, ((0, 0), (0, pad))) if pad else w
    g = padded.reshape(rows, ng, GS)
    wimp = (np.pad(imp, (0, pad)) if pad else imp).reshape(1, ng, GS).astype(np.float32)
    base = np.maximum(np.abs(g).max(axis=2, keepdims=True), np.float32(1e-12)) / np.float32(7.0)
    best_err = best_q = best_s = None
    for a in alphas:
        s = (base * np.float32(a)).astype(np.float32)
        q = np.clip(np.rint(g / s), -8, 7)
        err = (np.square(g - q * s) * wimp).sum(axis=2, keepdims=True)
        if best_err is None:
            best_err, best_q, best_s = err, q, s
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_q = np.where(better, q, best_q)
            best_s = np.where(better, s, best_s)
    codes = (best_q.astype(np.int16) + 8).reshape(rows, ng * GS)[:, :cols].astype(np.uint8)
    return codes, best_s.reshape(rows, ng).astype(np.float32)


class ActStats:
    """The engine's Q38_ACT_STATS file: per layer and expert the rows seen and the sum
    of squares of each gate/up input channel (hidden) and down input channel (inter)."""
    def __init__(self, path):
        with open(path, "rb") as f:
            if f.read(8) != b"Q38ACT1\0":
                raise ValueError(f"{path}: not a Q38_ACT_STATS file")
            self.layers, self.experts, self.hidden, self.inter = struct.unpack("<4i", f.read(16))
        cells = self.layers * self.experts
        self.n = np.memmap(path, np.int64, "r", offset=24, shape=(cells,))
        self.sums = np.memmap(path, np.float64, "r", offset=24 + 8 * cells, shape=(cells, self.hidden + self.inter))
        self.path, self.rows = path, int(np.asarray(self.n).sum())

    def importance(self, layer, expert, k):
        """mean square per input channel for matrix k (0 gate, 1 up: hidden; 2 down: inter);
        uniform where calibration never routed the expert"""
        i = layer * self.experts + expert
        n = int(self.n[i])
        width = self.hidden if k < 2 else self.inter
        if n == 0:
            return np.ones(width, np.float32)
        part = self.sums[i, :self.hidden] if k < 2 else self.sums[i, self.hidden:]
        return (np.asarray(part) / n).astype(np.float32) + np.float32(1e-12)


def pack_planar(codes):
    """u8 codes [R, C] (0..15) -> u8 [R, ceil(C/2)]: planar blocks of 64, tail in pairs."""
    rows, cols = codes.shape
    body = cols // GS * GS
    parts = []
    if body:
        blocks = codes[:, :body].reshape(rows, body // GS, 2, GS // 2)
        parts.append((blocks[:, :, 0, :] | (blocks[:, :, 1, :] << 4)).reshape(rows, body // 2))
    if body < cols:
        tail = codes[:, body:]
        if tail.shape[1] & 1:
            tail = np.concatenate([tail, np.full((rows, 1), 8, np.uint8)], axis=1)
        parts.append(tail[:, 0::2] | (tail[:, 1::2] << 4))
    return np.ascontiguousarray(np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0])


def unpack_planar(packed, cols):
    """Inverse of pack_planar: u8 [R, ceil(C/2)] -> codes u8 [R, C]."""
    rows = packed.shape[0]
    body = cols // GS * GS
    out = np.empty((rows, cols), np.uint8)
    if body:
        blocks = packed[:, :body // 2].reshape(rows, body // GS, GS // 2)
        out[:, :body] = np.stack([blocks & 15, blocks >> 4], axis=2).reshape(rows, body)
    if body < cols:
        tail = packed[:, body // 2:]
        pairs = np.stack([tail & 15, tail >> 4], axis=2).reshape(rows, -1)
        out[:, body:] = pairs[:, :cols - body]
    return out


def quantize_int2_g32(w, imp=None, iters=6):
    """w f32 [R, C] (C % 128 == 0) -> (codes u8 [R, C] in 0..3, q u8 [R, C/32], d f32 [R]):
    w ~ d[r] * q[r, g] * LEVELS2[code]. Per group of 32 the scale and the codes are
    fitted in turn against sum_i imp_i * (w_i - s * level_i)^2 (imp uniform without
    statistics), then the scales are stored as u8 under the row's largest and refitted,
    with the codes, for what the u8 can hold."""
    rows, cols = w.shape
    ng = cols // GS2
    g = w.reshape(rows, ng, GS2).astype(np.float64)
    a = np.ones((1, ng, GS2)) if imp is None else imp.astype(np.float64).reshape(1, ng, GS2)
    edges = (LEVELS2[1:] + LEVELS2[:-1]) / 2
    tiny = 1e-30

    def refit(scale):
        code = np.searchsorted(edges, g / np.maximum(scale, tiny)[..., None])
        lv = LEVELS2[code]
        num, den = (a * g * lv).sum(axis=2), (a * lv * lv).sum(axis=2)
        return np.where(den > 0, num / np.maximum(den, tiny), scale)

    s = np.sqrt((a * g * g).sum(axis=2) / a.sum(axis=2)) / math.sqrt(float(np.mean(LEVELS2 ** 2)))
    for _ in range(iters):
        s = refit(s)
    d = np.maximum(s.max(axis=1), tiny) / 255.0
    d = d.astype(np.float32).astype(np.float64)          # the f32 the file stores
    q = np.clip(np.rint(s / d[:, None]), 1, 255)
    for _ in range(2):
        q = np.clip(np.rint(refit(d[:, None] * q) / d[:, None]), 0, 255)
    code = np.searchsorted(edges, g / np.maximum(d[:, None] * q, tiny)[..., None])
    return code.reshape(rows, cols).astype(np.uint8), q.astype(np.uint8), d.astype(np.float32)


def quantize_int2_g32_gptq(w, imp, U, iters=6, batch=128, margin=1.25):
    """quantize_int2_g32 with the error of every quantized column carried into the columns
    still to come (GPTQ's update, written from the paper): U is the upper Cholesky factor
    of H^-1 for this matrix's inputs, so quantizing column i moves the rest of its row by
    -(w_i - q_i) / U[i,i] * U[i, i+1:], which is the least-squares answer under H. Columns
    go in order; a group's scale is fitted when its first column comes up, on the weights
    as the earlier groups' errors left them. The row scale d is fixed first (the plain
    fit's, times `margin` for the room the updates may need)."""
    rows, cols = w.shape
    W = w.astype(np.float64).copy()
    a = np.ones(cols) if imp is None else imp.astype(np.float64)
    _, _, d0 = quantize_int2_g32(w, imp)
    d = (d0.astype(np.float64) * margin).astype(np.float32).astype(np.float64)
    edges = (LEVELS2[1:] + LEVELS2[:-1]) / 2
    codes = np.zeros((rows, cols), np.uint8)
    qs = np.zeros((rows, cols // GS2), np.uint8)
    tiny = 1e-30
    s = np.ones(rows)
    for b0 in range(0, cols, batch):
        b1 = min(b0 + batch, cols)
        W1 = W[:, b0:b1].copy()
        E1 = np.zeros_like(W1)
        U1 = U[b0:b1, b0:b1]
        for i in range(b1 - b0):
            col = b0 + i
            if col % GS2 == 0:
                g = W1[:, i:i + GS2]
                ga = a[col:col + GS2][None, :]
                s = np.sqrt((ga * g * g).sum(axis=1) / ga.sum()) / math.sqrt(float(np.mean(LEVELS2 ** 2)))
                for _ in range(iters):
                    lv = LEVELS2[np.searchsorted(edges, g / np.maximum(s, tiny)[:, None])]
                    num, den = (ga * g * lv).sum(axis=1), (ga * lv * lv).sum(axis=1)
                    s = np.where(den > 0, num / np.maximum(den, tiny), s)
                u = np.clip(np.rint(s / d), 1, 255)
                qs[:, col // GS2] = u
                s = d * u
            c = np.searchsorted(edges, W1[:, i] / np.maximum(s, tiny))
            codes[:, col] = c
            err = (W1[:, i] - LEVELS2[c] * s) / U1[i, i]
            W1[:, i + 1:] -= np.outer(err, U1[i, i + 1:])
            E1[:, i] = err
        W[:, b1:] -= E1 @ U[b0:b1, b1:]
    return codes, qs, d.astype(np.float32)


class CovStats:
    """The engine's Q38_ACT_COV file: per layer sum_t x x^T of the routed experts' input.
    upper_inv(layer, imp) is the upper Cholesky factor of H^-1 for one expert, with
    H = D^1/2 C D^1/2: C the layer's correlations (damped by `damp` on the diagonal, as
    GPTQ damps H), D the expert's own mean squares (Q38_ACT_STATS); the factor of C^-1 is
    computed once per layer and an expert only rescales its columns."""
    def __init__(self, path, damp=0.01):
        with open(path, "rb") as f:
            if f.read(8) != b"Q38COV1\0":
                raise ValueError(f"{path}: not a Q38_ACT_COV file")
            self.layers, self.hidden = struct.unpack("<2i", f.read(8))
            self.n = np.frombuffer(f.read(8 * self.layers), np.int64).copy()
        self.mat = np.memmap(path, np.float64, "r", offset=16 + 8 * self.layers,
                             shape=(self.layers, self.hidden, self.hidden))
        self.damp, self._layer, self._u = damp, None, None

    def _factor(self, layer):
        if self._layer != layer:
            m = np.array(self.mat[layer])
            dg = np.sqrt(np.maximum(np.diag(m), 1e-30))
            c = m / dg[:, None] / dg[None, :] + self.damp * np.eye(self.hidden)
            self._u = np.linalg.cholesky(np.linalg.inv(c)).T     # C^-1 = U^T U, U upper
            self._layer = layer
        return self._u

    def upper_inv(self, layer, imp):
        return self._factor(layer) / np.sqrt(np.maximum(imp.astype(np.float64), 1e-30))[None, :]


class Dump:
    """The engine's Q38_ACT_DUMP directory: per layer, every calibration token's routed-expert
    input with its route (K expert ids, K gates). rows(layer, expert) are the inputs that expert
    really received and the gates they were weighed with; the layer's full matrix of inputs is
    kept for the prior of experts with few rows."""
    def __init__(self, path, max_tokens=None):
        self.dir, self.max_tokens, self._layer = path, max_tokens, None

    def _load(self, layer):
        if self._layer == layer:
            return
        raw = np.fromfile(os.path.join(self.dir, f"layer-{layer:03d}.bin"), np.uint8)
        if raw[:8].tobytes() != b"Q38DMP1\0":
            raise ValueError(f"{self.dir}: layer {layer} is not a Q38_ACT_DUMP file")
        h, k = struct.unpack("<2i", raw[8:16].tobytes())
        rec = raw[16:].reshape(-1, 8 * k + 4 * h)
        if self.max_tokens:
            rec = rec[:self.max_tokens]
        self.ids = np.ascontiguousarray(rec[:, :4 * k]).view(np.int32)
        self.gates = np.ascontiguousarray(rec[:, 4 * k:8 * k]).view(np.float32)
        self.x = np.ascontiguousarray(rec[:, 8 * k:]).view(np.float32)
        self.pool = None
        self._layer = layer

    def rows(self, layer, expert):
        self._load(layer)
        hit = self.ids == expert
        rows = hit.any(axis=1)
        gate = (self.gates * hit).sum(axis=1)       # a route lists an expert once; summed all the same
        return self.x[rows].astype(np.float64), gate[rows].astype(np.float64)

    def layer_moment(self, layer):
        """sum_t x x^T / tokens over every token of the layer (the prior)."""
        self._load(layer)
        if self.pool is None:
            x = self.x.astype(np.float64)
            self.pool = x.T @ x / max(len(x), 1)
        return self.pool


def _upper_inv(h):
    """The upper Cholesky factor U of h^-1 (h^-1 = U^T U), for GPTQ."""
    return np.linalg.cholesky(np.linalg.inv(h)).T


def _hessian(x, wts, prior, alpha, damp):
    """sum_t w_t x_t x_t^T, plus alpha times the prior scaled to its trace, plus damp times
    its mean diagonal: rows are few next to the width, so the prior fills the directions
    they never visit."""
    h = (x * wts[:, None]).T @ x if len(x) else np.zeros((prior.shape[0],) * 2)
    if prior is not None:
        tp = float(np.trace(prior))
        th = float(np.trace(h))
        h = h + (alpha * th / tp if th > 0 and tp > 0 else 1.0) * prior
    h = h + damp * float(np.mean(np.diag(h))) * np.eye(h.shape[0])
    return h


def quantize_expert_int2_exact(w3, x, gates, layer_prior, alpha=0.1, damp=0.01, gate_weight=True):
    """One expert from the rows it really received: gate and up with GPTQ under
    H = sum_t g_t^2 x_t x_t^T (+ the layer's moment as a prior), then down with GPTQ under the
    SwiGLU outputs of the QUANTIZED gate and up on the same rows (what down will really see).
    w3 = (gate, up, down) f32; returns [(codes, q, d)] * 3."""
    wts = gates ** 2 if gate_weight else np.ones(len(x))
    h = _hessian(x, wts, layer_prior, alpha, damp)
    u = _upper_inv(h)
    imp = np.diag(h).astype(np.float32)
    out = [quantize_int2_g32_gptq(w3[k], imp, u) for k in range(2)]
    if len(x):
        g = x @ dequantize_int2(*out[0]).T
        hid = g / (1.0 + np.exp(-g)) * (x @ dequantize_int2(*out[1]).T)
        hd = (hid * wts[:, None]).T @ hid
        hd = hd + alpha * np.diag(np.diag(hd)) + damp * float(np.mean(np.diag(hd)) or 1.0) * np.eye(hd.shape[0])
        out.append(quantize_int2_g32_gptq(w3[2], np.diag(hd).astype(np.float32), _upper_inv(hd)))
    else:
        out.append(quantize_int2_g32(w3[2]))
    return out


def pack_int2(codes):
    """u8 codes [R, C] (0..3) -> u8 [R, C/4]: blocks of 128, byte k holding elements
    k, k+32, k+64, k+96 in bits 0-1, 2-3, 4-5, 6-7 (int2_g32.h)."""
    rows, cols = codes.shape
    c = codes.reshape(rows, cols // 128, 4, 32)
    return (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).reshape(rows, cols // 4)


def dequantize_int2(codes, q, d):
    rows, cols = codes.shape
    return (LEVELS2[codes].reshape(rows, cols // GS2, GS2) * (d[:, None] * q.astype(np.float64))[..., None]).reshape(rows, cols)


def dequantize(codes, scales):
    """codes u8 [R, C], scales f32 [R, ceil(C/64)] -> f32 [R, C]."""
    cols = codes.shape[1]
    return (codes.astype(np.float32) - 8.0) * np.repeat(scales, GS, axis=1)[:, :cols]


def rel_l2(reference, approx):
    """||reference - approx|| / ||reference||, in float64. Plain reductions, not
    np.linalg.norm: that is a BLAS call, and every worker's BLAS starting a
    thread per core made a 12-worker layer 2.4x slower (46 s against 19 s)."""
    ref = reference.astype(np.float64)
    norm = float(np.sum(np.square(ref)))
    diff = float(np.sum(np.square(ref - approx.astype(np.float64))))
    return math.sqrt(diff / norm) if norm > 0 else 0.0


# ---------- the geometry of one record ----------

class Geometry:
    def __init__(self, layers, experts, hidden, inter, bits=4):
        self.layers, self.experts, self.hidden, self.inter = layers, experts, hidden, inter
        self.bits = bits
        self.shape = ((inter, hidden), (inter, hidden), (hidden, inter))
        if bits == 2:
            if hidden % 128 or inter % 128:
                raise ValueError(f"hidden={hidden}, moe_intermediate_size={inter}: int2-g32 needs "
                                 f"widths that are multiples of 128")
            self.code_bytes = [r * c // 4 for r, c in self.shape]
            self.q_count = [r * c // GS2 for r, c in self.shape]
            self.scale_count = [r for r, c in self.shape]
            self.scale_off = sum(self.code_bytes) + sum(self.q_count)
            self.record_bytes = self.scale_off + 4 * sum(self.scale_count)
        else:
            self.code_bytes = [r * row_bytes(c) for r, c in self.shape]
            self.scale_count = [r * groups(c) for r, c in self.shape]
            self.scale_off = sum(self.code_bytes)
            self.record_bytes = self.scale_off + 4 * sum(self.scale_count)
        if self.scale_off % 4:
            raise ValueError(f"hidden={hidden}, moe_intermediate_size={inter}: the codes of an "
                             f"expert are {self.scale_off} bytes, not a multiple of 4, so its f32 "
                             f"scales could not follow them aligned; this geometry is not supported")

    def tensors(self, prefix, layer, expert):
        """(name, dtype, shape, offset in the record, bytes) in record order."""
        base = f"{prefix}.layers.{layer}.mlp.experts.{expert}"
        out, off = [], 0
        if self.bits == 2:
            for k, (r, c) in enumerate(self.shape):
                out.append((f"{base}.{PROJ[k]}.weight", "U8", [r, c // 4], off, self.code_bytes[k]))
                off += self.code_bytes[k]
            for k, (r, c) in enumerate(self.shape):
                out.append((f"{base}.{PROJ[k]}.weight.q8", "U8", [r, c // GS2], off, self.q_count[k]))
                off += self.q_count[k]
            for k, (r, c) in enumerate(self.shape):
                out.append((f"{base}.{PROJ[k]}.weight.d", "F32", [r, 1], off, 4 * r))
                off += 4 * r
            assert off == self.record_bytes
            return out
        for k, (r, c) in enumerate(self.shape):
            out.append((f"{base}.{PROJ[k]}.weight", "U8", [r, row_bytes(c)], off, self.code_bytes[k]))
            off += self.code_bytes[k]
        for k, (r, c) in enumerate(self.shape):
            n = 4 * self.scale_count[k]
            out.append((f"{base}.{PROJ[k]}.weight.qs", "F32", [r, groups(c)], off, n))
            off += n
        assert off == self.record_bytes
        return out

    def fp8_expert_bytes(self):
        return sum(r * c for r, c in self.shape)


# ---------- the release, read by offset ----------

_pread_lock = threading.Lock()


def _pread(fd, nbytes, offset):
    """os.pread where the platform has it. Windows has none (Python 3.14 included):
    there a seek and a read, under a lock, since they move the descriptor's offset."""
    pread = getattr(os, "pread", None)
    if pread is not None:
        return pread(fd, nbytes, offset)
    with _pread_lock:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.read(fd, nbytes)


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 512 << 20:
            raise ValueError(f"{path}: safetensors header of {n} bytes")
        return json.loads(f.read(n)), 8 + n


class Source:
    """The routed experts of a Qwen3.8 snapshot, located by name in its shards."""

    def __init__(self, model_dir):
        self.dir = model_dir
        with open(os.path.join(model_dir, "config.json"), "rb") as f:
            self.config_bytes = f.read()
        config = json.loads(self.config_bytes)
        text = config.get("text_config", config)
        if text.get("model_type") != "qwen4_exp_text":
            sys.exit(f"{model_dir}: text model_type {text.get('model_type')!r} is not qwen4_exp_text")
        self.geometry = Geometry(int(text["num_hidden_layers"]), int(text["num_experts"]),
                                 int(text["hidden_size"]), int(text["moe_intermediate_size"]))
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                weight_map = json.load(f)["weight_map"]
            files = sorted(set(weight_map.values()))
        else:
            weight_map = None
            files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
        if not files:
            sys.exit(f"{model_dir}: no safetensors shards")
        self.tensors = {}
        for name in files:
            path = os.path.join(model_dir, name)
            header, base = read_header(path)
            for key, meta in header.items():
                if key == "__metadata__" or (weight_map and weight_map.get(key) != name):
                    continue
                if ".mlp.experts." not in key:
                    continue
                begin, end = meta["data_offsets"]
                self.tensors[key] = (path, base + begin, end - begin, meta["dtype"], meta["shape"])
        if os.path.exists(index_path):
            prefixes = [p for p in ("model.language_model", "model")
                        if p + ".embed_tokens.weight" in weight_map]
        else:
            prefixes = []
            for name in files:
                header, _ = read_header(os.path.join(model_dir, name))
                prefixes += [p for p in ("model.language_model", "model")
                             if p + ".embed_tokens.weight" in header]
        if not prefixes:
            sys.exit(f"{model_dir}: no Qwen4-Exp text embedding; is this a Qwen3.8 snapshot?")
        self.prefix = prefixes[0]
        self._fds = {}

    def fingerprint(self):
        """What a resumed run must find unchanged: the config and the shards
        that hold routed experts (name and size)."""
        shards = sorted({t[0] for t in self.tensors.values()})
        return {"config_sha256": hashlib.sha256(self.config_bytes).hexdigest(),
                "shards": {os.path.basename(p): os.path.getsize(p) for p in shards}}

    def _read(self, path, offset, nbytes):
        fd = self._fds.get(path)
        if fd is None:
            fd = self._fds[path] = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        out = bytearray()
        while len(out) < nbytes:
            chunk = _pread(fd, nbytes - len(out), offset + len(out))
            if not chunk:
                raise IOError(f"{path}: short read at {offset + len(out)}")
            out += chunk
        return bytes(out)

    def _float(self, entry, element_offset, count):
        path, offset, nbytes, dtype, _ = entry
        size = _FLOAT_DTYPES[dtype]
        raw = self._read(path, offset + element_offset * size, count * size)
        if dtype == "BF16":
            return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        if dtype == "F16":
            return np.frombuffer(raw, np.float16).astype(np.float32)
        return np.frombuffer(raw, np.float32).copy()

    def matrix(self, layer, expert, k):
        """Expert `expert`'s projection k (gate, up, down) as the engine sees it, f32."""
        rows, cols = self.geometry.shape[k]
        name = f"{self.prefix}.layers.{layer}.mlp.experts.{expert}.{PROJ[k]}.weight"
        entry = self.tensors.get(name)
        if entry is not None:
            path, offset, nbytes, dtype, shape = entry
            if list(shape) != [rows, cols]:
                raise ValueError(f"{name}: shape {shape}, expected {[rows, cols]}")
            if dtype in _FP8_DTYPES:
                if nbytes != rows * cols:
                    raise ValueError(f"{name}: {nbytes} bytes for an e4m3 [{rows}, {cols}]")
                scale = self.tensors.get(name + "_scale_inv")
                nb_r, nb_c = -(-rows // FP8_BLOCK), -(-cols // FP8_BLOCK)
                if scale is None or list(scale[4]) != [nb_r, nb_c] or scale[3] not in _FLOAT_DTYPES:
                    raise ValueError(f"{name}_scale_inv: missing, or not a [{nb_r}, {nb_c}] float tensor")
                block = self._float(scale, 0, nb_r * nb_c).reshape(nb_r, nb_c)
                codes = np.frombuffer(self._read(path, offset, nbytes), np.uint8).reshape(rows, cols)
                expand = np.repeat(np.repeat(block, FP8_BLOCK, axis=0)[:rows], FP8_BLOCK, axis=1)[:, :cols]
                w = E4M3[codes] * expand
            elif dtype in _FLOAT_DTYPES:
                w = self._float(entry, 0, rows * cols).reshape(rows, cols)
            else:
                raise ValueError(f"{name}: unsupported expert dtype {dtype}")
        else:
            fused = "gate_up_proj" if k < 2 else "down_proj"
            fname = f"{self.prefix}.layers.{layer}.mlp.experts.{fused}"
            entry = self.tensors.get(fname)
            g = self.geometry
            want = [g.experts, 2 * g.inter, g.hidden] if k < 2 else [g.experts, g.hidden, g.inter]
            if entry is None:
                raise ValueError(f"layer {layer} expert {expert}: neither {name} nor {fname}")
            if list(entry[4]) != want or entry[3] not in _FLOAT_DTYPES:
                raise ValueError(f"{fname}: expected a float {want}, found {entry[3]} {entry[4]}")
            first = expert * want[1] * want[2] + (rows * cols if k == 1 else 0)
            w = self._float(entry, first, rows * cols).reshape(rows, cols)
        if not np.isfinite(w).all():
            raise ValueError(f"{name}: non-finite weights; refusing to quantize them")
        return w


# ---------- one expert, one layer ----------

_SOURCE = None
_ACT = None
_BITS = 4
_COV = None
_DUMP = None


def _init_worker(model_dir, act_path=None, bits=4, cov_path=None, dump_path=None):
    global _SOURCE, _ACT, _BITS, _COV, _DUMP
    _SOURCE = Source(model_dir)
    _ACT = ActStats(act_path) if act_path else None
    _BITS = bits
    _COV = CovStats(cov_path) if cov_path else None
    _DUMP = Dump(dump_path) if dump_path else None


def convert_expert_int2(layer, expert):
    codes, qs, ds, errors = [], [], [], []
    if _DUMP is not None:
        w3 = [_SOURCE.matrix(layer, expert, k) for k in range(3)]
        x, gates = _DUMP.rows(layer, expert)
        for k, (c, q, d) in enumerate(quantize_expert_int2_exact(w3, x, gates, _DUMP.layer_moment(layer))):
            errors.append(rel_l2(w3[k], dequantize_int2(c, q, d)))
            codes.append(pack_int2(c).tobytes())
            qs.append(q.tobytes())
            ds.append(d.tobytes())
        return b"".join(codes) + b"".join(qs) + b"".join(ds), errors, peak_rss_mb()
    for k in range(3):
        w = _SOURCE.matrix(layer, expert, k)
        imp = _ACT.importance(layer, expert, k) if _ACT else None
        if _COV is not None and k < 2:   # gate and up read the layer's hidden row; down an expert's own
            c, q, d = quantize_int2_g32_gptq(w, imp, _COV.upper_inv(layer, imp if imp is not None else np.ones(w.shape[1])))
        else:
            c, q, d = quantize_int2_g32(w, imp)
        errors.append(rel_l2(w, dequantize_int2(c, q, d)))
        codes.append(pack_int2(c).tobytes())
        qs.append(q.tobytes())
        ds.append(d.tobytes())
    return b"".join(codes) + b"".join(qs) + b"".join(ds), errors, peak_rss_mb()


def convert_expert(task):
    """(layer, expert) -> (the record's bytes, relative L2 error of gate, up,
    down, this process's peak RSS in MB)."""
    layer, expert = task
    if _BITS == 2:
        return convert_expert_int2(layer, expert)
    codes, scales, errors = [], [], []
    for k in range(3):
        w = _SOURCE.matrix(layer, expert, k)
        q, s = quantize_g64_weighted(w, _ACT.importance(layer, expert, k)) if _ACT else quantize_g64(w)
        errors.append(rel_l2(w, dequantize(q, s)))
        codes.append(pack_planar(q).tobytes())
        scales.append(s.astype(np.float32).tobytes())
    return b"".join(codes) + b"".join(scales), errors, peak_rss_mb()


def layer_file(layer):
    return f"layer-{layer:03d}.safetensors"


def layer_header(geometry, prefix, layer):
    """The layer file's header, padded so the data region starts on DATA_ALIGN."""
    header, meta = {}, {"format": FORMAT2 if geometry.bits == 2 else FORMAT, "version": str(VERSION), "layer": str(layer)}
    for expert in range(geometry.experts):
        base = expert * geometry.record_bytes
        for name, dtype, shape, off, n in geometry.tensors(prefix, layer, expert):
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [base + off, base + off + n]}
    header["__metadata__"] = meta
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-(8 + len(raw))) % DATA_ALIGN)
    return struct.pack("<Q", len(raw)) + raw


def layer_file_bytes(geometry, prefix, layer):
    return len(layer_header(geometry, prefix, layer)) + geometry.experts * geometry.record_bytes


def write_layer(src, out_dir, layer, pool):
    g = src.geometry
    final = os.path.join(out_dir, layer_file(layer))
    part = final + ".part"
    tasks = [(layer, e) for e in range(g.experts)]
    errors, worker_mb = [], 0.0
    with open(part, "wb") as f:
        f.write(layer_header(g, src.prefix, layer))
        results = pool.imap(convert_expert, tasks, chunksize=4) if pool else map(convert_expert, tasks)
        for record, err, rss in results:
            if len(record) != g.record_bytes:
                raise RuntimeError(f"layer {layer}: record of {len(record)} bytes, expected {g.record_bytes}")
            f.write(record)
            errors.append(err)
            worker_mb = max(worker_mb, rss)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, final)
    return np.array(errors), worker_mb


def write_index(out_dir, index):
    tmp = os.path.join(out_dir, "index.json.part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(out_dir, "index.json"))


def new_index(src):
    g = src.geometry
    if g.bits == 2:
        return {
            "format": FORMAT2, "version": VERSION, "complete": False,
            "group_size": GS2, "codes": CODES2, "scales": SCALES2,
            "layers": g.layers, "experts": g.experts,
            "hidden_size": g.hidden, "moe_intermediate_size": g.inter,
            "prefix": src.prefix, "record_bytes": g.record_bytes,
            "record": ["gate codes", "up codes", "down codes", "gate q", "up q", "down q",
                       "gate d", "up d", "down d"],
            "quantizer": "per 32 inputs: levels -10 -3 3 10, scale and codes fitted in turn (uniform "
                         "weights), scales as u8 under the row's largest and refitted",
            "files": [layer_file(i) for i in range(g.layers)],
            "done": [],
            "source": src.fingerprint(),
        }
    return {
        "format": FORMAT, "version": VERSION, "complete": False,
        "group_size": GS, "codes": CODES, "scales": SCALES,
        "layers": g.layers, "experts": g.experts,
        "hidden_size": g.hidden, "moe_intermediate_size": g.inter,
        "prefix": src.prefix, "record_bytes": g.record_bytes,
        "record": ["gate codes", "up codes", "down codes", "gate scales", "up scales", "down scales"],
        "quantizer": "per 64 inputs: scale = max(absmax, 1e-12)/7 (f32), code = rint(w/scale) "
                     "clamped to [-8,7], stored v+8 (tools/convert_qwen36.py quantize_row_grouped)",
        "files": [layer_file(i) for i in range(g.layers)],
        "done": [],
        "source": src.fingerprint(),
    }


def resume_index(out_dir, src):
    """The index of a previous run on the same snapshot, or a fresh one."""
    path = os.path.join(out_dir, "index.json")
    fresh = new_index(src)
    if not os.path.exists(path):
        return fresh
    with open(path, encoding="utf-8") as f:
        index = json.load(f)
    keys = ("format", "version", "group_size", "codes", "scales", "layers", "experts",
            "hidden_size", "moe_intermediate_size", "prefix", "record_bytes", "source")
    stale = [k for k in keys if index.get(k) != fresh[k]]
    if stale:
        sys.exit(f"{path} was written for something else ({', '.join(stale)} differ); "
                 f"remove {out_dir} to convert this snapshot from scratch")
    done = []
    for layer in index.get("done", []):
        path = os.path.join(out_dir, layer_file(layer))
        if os.path.exists(path) and os.path.getsize(path) == layer_file_bytes(src.geometry, src.prefix, layer):
            done.append(layer)
        else:
            print(f"layer {layer}: listed as done but {path} is missing or the wrong size; converting it again")
    fresh["done"] = sorted(done)
    fresh["complete"] = len(done) == src.geometry.layers
    return fresh


def parse_layers(spec, count):
    if not spec:
        return list(range(count))
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b) if b else int(a)
        if not 0 <= lo <= hi < count:
            sys.exit(f"--layers {spec}: {part} is outside 0..{count - 1}")
        out.update(range(lo, hi + 1))
    return sorted(out)


def peak_rss_mb():
    """This process's peak resident set in MB (0 where getrusage is missing)."""
    try:
        import resource
    except ImportError:
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 20) if sys.platform == "darwin" else peak / 1024


def convert(model_dir, out_dir=None, workers=1, layers=None, plan=False, act_path=None, bits=4, cov_path=None,
            dump_path=None):
    """Write (or finish) model_dir's sidecar; returns its directory."""
    src = Source(model_dir)
    if bits == 2:
        geo = src.geometry
        try:
            src.geometry = Geometry(geo.layers, geo.experts, geo.hidden, geo.inter, bits=2)
        except ValueError as exc:
            sys.exit(f"{model_dir}: {exc}")
    act = ActStats(act_path) if act_path else None
    if act and (act.layers, act.experts, act.hidden, act.inter) != (src.geometry.layers, src.geometry.experts,
                                                                  src.geometry.hidden, src.geometry.inter):
        sys.exit(f"{act_path}: statistics of another geometry")
    g = src.geometry
    out_dir = out_dir or os.path.join(model_dir, SIDECAR_DIR2 if bits == 2 else SIDECAR_DIR)
    label = "int2-g32" if bits == 2 else "int4-g64"
    total = sum(layer_file_bytes(g, src.prefix, i) for i in range(g.layers))
    fp8 = g.fp8_expert_bytes() * g.experts * g.layers
    print(f"{model_dir}: {g.layers} layers x {g.experts} experts, hidden {g.hidden}, "
          f"moe_intermediate {g.inter}, prefix {src.prefix}")
    print(f"  per expert: FP8 {g.fp8_expert_bytes():,} bytes -> {label} {g.record_bytes:,} bytes "
          f"({100 * g.record_bytes / g.fp8_expert_bytes():.1f}%)")
    print(f"  sidecar {out_dir}: {total:,} bytes ({total / 2**30:.2f} GiB) "
          f"for {fp8:,} bytes of FP8 experts")
    if plan:
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    index = resume_index(out_dir, src)
    if (cov_path or dump_path) and bits != 2:
        sys.exit("--cov and --dump are for --int2")
    if dump_path and (cov_path or act_path):
        sys.exit("--dump carries the statistics itself: no --act-stats or --cov with it")
    cov = CovStats(cov_path) if cov_path else None
    if cov and (cov.layers, cov.hidden) != (g.layers, g.hidden):
        sys.exit(f"{cov_path}: matrices of another geometry")
    if bits == 2:
        quantizer = (f"weighted: per 32 inputs levels -10 -3 3 10, scale and codes fitted in turn against the "
                     f"activation-weighted squared error (Q38_ACT_STATS {os.path.basename(act_path)}, "
                     f"{act.rows} expert rows), scales as u8 under the row's largest and refitted") if act else None
        if dump_path:
            quantizer = ("every expert fitted to the rows it received (Q38_ACT_DUMP " + os.path.basename(dump_path.rstrip("/")) +
                         "): gate and up with GPTQ under H = sum g^2 x x^T + 0.1 x the layer's moment, down with GPTQ "
                         "under the quantized gate/up's SwiGLU outputs; levels -10 -3 3 10, u8 scales per 32")
        if cov:
            quantizer = (quantizer or "uniform weights") + (
                f"; gate and up with the error carried column by column (GPTQ) under H = D^1/2 C D^1/2 "
                f"(Q38_ACT_COV {os.path.basename(cov_path)}, {int(cov.n[0])} tokens, damp {cov.damp})")
    else:
        quantizer = (f"weighted: per 64 inputs the scale among absmax/7 * {list(ALPHAS)} with the least "
                     f"activation-weighted squared error (Q38_ACT_STATS {os.path.basename(act_path)}, "
                     f"{act.rows} expert rows)") if act else None
    if quantizer and index["done"] and index.get("quantizer") != quantizer:
        sys.exit(f"{out_dir}: started with another quantizer; give a new --out")
    if quantizer:
        index["quantizer"] = quantizer
    wanted = parse_layers(layers, g.layers)
    todo = [i for i in wanted if i not in index["done"]]
    if len(wanted) > len(todo):
        print(f"  {len(wanted) - len(todo)} layer(s) already converted, skipped")
    write_index(out_dir, index)

    pool = None
    if workers > 1 and todo:
        from multiprocessing import get_context
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")   # one BLAS thread per worker: the workers are the parallelism
        pool = get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(model_dir, act_path, bits, cov_path, dump_path))
    else:
        _init_worker(model_dir, act_path, bits, cov_path, dump_path)
    worst = 0.0
    try:
        for layer in todo:
            started = time.time()
            errors, worker_mb = write_layer(src, out_dir, layer, pool)
            index["done"] = sorted(index["done"] + [layer])
            index["complete"] = len(index["done"]) == g.layers
            write_index(out_dir, index)
            worst = max(worst, float(errors.max()))
            print(f"  layer {layer:3d}: {g.experts} experts in {time.time() - started:6.1f}s  "
                  f"rel L2 gate {errors[:, 0].mean():.4f} up {errors[:, 1].mean():.4f} "
                  f"down {errors[:, 2].mean():.4f} (max {errors.max():.4f})  "
                  f"peak RSS {peak_rss_mb():.0f} MB"
                  + (f" + {workers} workers x {worker_mb:.0f} MB" if pool else ""), flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    state = "complete" if index["complete"] else f"{len(index['done'])}/{g.layers} layers"
    print(f"{out_dir}: {state}" + (f", worst expert matrix rel L2 {worst:.4f}" if todo else ""))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Qwen3.8 routed experts FP8 -> int4-g64 sidecar")
    ap.add_argument("--model", required=True, help="the Qwen3.8 snapshot (config.json + shards)")
    ap.add_argument("--out", help=f"sidecar directory (default: <model>/{SIDECAR_DIR}, where the engine looks)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="processes converting experts in parallel (default: every CPU)")
    ap.add_argument("--layers", help="only these layers, e.g. 0-3,7 (the index stays incomplete)")
    ap.add_argument("--plan", action="store_true", help="print sizes and exit without writing")
    ap.add_argument("--act-stats", help="the engine's Q38_ACT_STATS file: each group's scale chosen against "
                                        "activation-weighted error instead of absmax/7")
    ap.add_argument("--int2", action="store_true", help=f"write int2-g32 (int2_g32.h) into <model>/{SIDECAR_DIR2}/ "
                                                        "instead (qwen38: Q38_EXPERT_INT2=1)")
    ap.add_argument("--cov", help="the engine's Q38_ACT_COV file (with --int2): gate and up quantized with the "
                                  "error carried column by column under the layer's input correlations")
    ap.add_argument("--dump", help="the engine's Q38_ACT_DUMP directory (with --int2): every expert fitted with GPTQ "
                                   "to the inputs it really received, down to the SwiGLU outputs of its quantized gate/up")
    a = ap.parse_args()
    convert(a.model, a.out, a.workers, a.layers, a.plan, a.act_stats, 2 if a.int2 else 4, a.cov, a.dump)


if __name__ == "__main__":
    main()
