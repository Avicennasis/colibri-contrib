#!/usr/bin/env python3
"""Turn a GGUF model into a colibri model directory, without re-quantizing.

Two modes, both lossless (the ggml blocks are used or copied as they are):

  link  (default)  the directory holds config.json, tokenizer.json, the chat
                   template and gguf.json, which points at the GGUF where it
                   already is. No weight is copied: the engine reads the file
                   in place.
  copy             the same, plus model.colibri.gguf: every dense tensor copied
                   byte for byte, and each layer's routed experts re-laid out
                   one expert per 4 KiB-aligned slot, gate | up | down
                   contiguous, so an expert is ONE aligned read instead of
                   three scattered ones. The original metadata is copied
                   verbatim; colibri.* keys describe the new layout.

Either way the directory is an ordinary colibri model: `coli chat`, `coli
serve`, `coli web` and the planner treat it like any other.

Supported architectures: qwen35moe (Qwen3.5 / Qwen3.6 MoE -> qwen36 engine),
qwen4exp (Qwen3.8-Flash-Next -> qwen38 engine).

Usage:
  python3 tools/gguf_import.py MODEL.gguf OUT_DIR [--mode link|copy]
  (MODEL may be the first part of a split model: NAME-00001-of-000NN.gguf)
"""
import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_reader import GGUFReader, GGUFError  # noqa: E402

LAYOUT = "expert-packed-v1"
ALIGN = 4096
T_UINT32, T_STRING, T_ARRAY, T_UINT64 = 4, 8, 9, 10
GGML_I8 = 24


# ------------------------------------------------------------------ reading
def split_parts(path):
    """[path] or every part of a NAME-00001-of-000NN.gguf split, in order."""
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", str(path))
    if not m or int(m.group(2)) != 1:
        return [Path(path)]
    n = int(m.group(3))
    parts = [Path(f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]
    missing = [str(p) for p in parts if not p.is_file()]
    if missing:
        sys.exit(f"split model is incomplete, missing: {', '.join(missing)}")
    return parts


def open_model(path):
    parts = split_parts(path)
    readers = [GGUFReader(str(p)) for p in parts]
    meta = readers[0].metadata
    tensors = {}
    for i, r in enumerate(readers):
        for t in r.tensors:
            if t["name"] in tensors:
                sys.exit(f"tensor {t['name']} appears in two parts")
            tensors[t["name"]] = dict(t, part=i)
    return parts, readers, meta, tensors


def kv_section(path):
    """Byte range of the metadata key/values in one GGUF file, plus the counts.
    Used to copy the metadata verbatim, types included."""
    with open(path, "rb") as f:
        head = f.read(24)
        if head[:4] != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        version, n_t, n_kv = struct.unpack("<IQQ", head[4:24])
        pos = 24
        f.seek(pos)

        def rd(n):
            nonlocal pos
            b = f.read(n)
            if len(b) != n:
                sys.exit(f"{path}: truncated metadata")
            pos += n
            return b

        def skip_value(t):
            size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if t in size:
                rd(size[t])
            elif t == T_STRING:
                (n,) = struct.unpack("<Q", rd(8)); rd(n)
            elif t == T_ARRAY:
                et, n = struct.unpack("<IQ", rd(12))
                if et in size:
                    rd(size[et] * n)
                else:
                    for _ in range(n):
                        skip_value(et)
            else:
                sys.exit(f"{path}: unknown metadata type {t}")

        for _ in range(n_kv):
            (kl,) = struct.unpack("<Q", rd(8)); rd(kl)
            (t,) = struct.unpack("<I", rd(4))
            skip_value(t)
        end = pos
        f.seek(24)
        return f.read(end - 24), n_kv, version


# --------------------------------------------------------- architecture
def qwen35moe_config(meta, tensors, overrides=None):
    p = "qwen35moe."
    need = lambda k: meta[p + k] if (p + k) in meta else sys.exit(f"GGUF metadata lacks {p + k}")
    n_mtp = int(meta.get(p + "nextn_predict_layers", 0))
    layers = int(need("block_count")) - n_mtp
    interval = int(meta.get(p + "full_attention_interval", 4))
    head_dim = int(need("attention.key_length"))
    rope_n = int(meta.get(p + "rope.dimension_count", head_dim // 4))
    vh, kh = int(need("ssm.time_step_rank")), int(need("ssm.group_count"))
    inner = int(need("ssm.inner_size"))
    emb = tensors.get("token_embd.weight") or sys.exit("GGUF has no token_embd.weight")
    vocab = int(emb["dims"][1])
    theta = float(meta.get(p + "rope.freq_base", 10000000.0))
    cfg = {
        "model_type": "qwen3_5_moe_text",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "hidden_size": int(need("embedding_length")),
        "num_hidden_layers": layers,
        "num_attention_heads": int(need("attention.head_count")),
        "num_key_value_heads": int(need("attention.head_count_kv")),
        "head_dim": head_dim,
        "num_experts": int(need("expert_count")),
        "num_experts_per_tok": int(need("expert_used_count")),
        "moe_intermediate_size": int(need("expert_feed_forward_length")),
        "shared_expert_intermediate_size": int(meta.get(p + "expert_shared_feed_forward_length", need("expert_feed_forward_length"))),
        "vocab_size": vocab,
        "rms_norm_eps": float(meta.get(p + "attention.layer_norm_rms_epsilon", 1e-6)),
        "max_position_embeddings": int(meta.get(p + "context_length", 262144)),
        "partial_rotary_factor": rope_n / head_dim,
        "rope_parameters": {"rope_type": "default", "rope_theta": theta, "partial_rotary_factor": rope_n / head_dim},
        "linear_conv_kernel_dim": int(need("ssm.conv_kernel")),
        "linear_key_head_dim": int(need("ssm.state_size")),
        "linear_value_head_dim": inner // vh,
        "linear_num_key_heads": kh,
        "linear_num_value_heads": vh,
        "layer_types": ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(layers)],
        "full_attention_interval": interval,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
    }
    return cfg, layers


def qwen4exp_config(meta, tensors, overrides=None):
    """Qwen3.8-Flash-Next (llama.cpp's qwen4exp) as the qwen38 engine's config.json. The
    PLE table is one tensor here (per_layer_token_embd), so split_ngram_parts is 1; its
    hash constants stay in the GGUF metadata, where the engine reads them."""
    p = "qwen4exp."
    overrides = overrides or {}
    need = lambda k: meta[p + k] if (p + k) in meta else sys.exit(f"GGUF metadata lacks {p + k}")
    n_mtp = int(meta.get(p + "nextn_predict_layers", 0))
    layers = int(need("block_count")) - n_mtp
    head_dim = int(need("attention.key_length"))
    rope_n = int(meta.get(p + "rope.dimension_count", head_dim))
    vh, kh = int(need("ssm.time_step_rank")), int(need("ssm.group_count"))
    inner = int(need("ssm.inner_size"))
    emb = tensors.get("token_embd.weight") or sys.exit("GGUF has no token_embd.weight")
    ratios = [int(r) for r in meta.get(p + "attention.compress_ratios", [])][:layers]
    if p + "attention.recurrent_layers" in meta:
        recurrent = [int(r) for r in meta[p + "attention.recurrent_layers"]][:layers]
    else:   # files from before llama.cpp wrote recurrent_layers (unsloth's): every Nth layer attends
        every = int(need("full_attention_interval"))
        recurrent = [int((i + 1) % every != 0) for i in range(layers)]
    if any(ratios) and [int(not r) for r in ratios] != recurrent:
        sys.exit("GGUF metadata disagrees on the attention layers: compress_ratios vs the recurrent layers")
    ratio = int(overrides.get("indexer_compress_ratio") or max(ratios or [0]))
    if not ratio:
        sys.exit("GGUF metadata gives no indexer compress ratio (attention.compress_ratios is all 0): "
                 "pass --indexer-compress-ratio")
    attn = [i for i in range(layers) if not recurrent[i]]
    idx_dim = int(need("attention.indexer.key_length"))
    kproj = tensors.get(f"blk.{attn[0]}.indexer.k_proj.weight") if attn else None
    if kproj is None:
        sys.exit("GGUF has no indexer k_proj")
    ple_layers = list(meta.get(p + "ple.layers", []))
    ngram, per = int(need("ple.ngram_size")), int(need("ple.heads_per_ngram"))
    row = int(need("embedding_length_per_layer_input"))
    theta = float(meta.get(p + "rope.freq_base", 10000000.0))
    cfg = {
        "model_type": "qwen4_exp_text",
        "architectures": ["Qwen4ExpForCausalLM"],
        "hidden_size": int(need("embedding_length")),
        "num_hidden_layers": layers,
        "vocab_size": int(emb["dims"][1]),
        "max_position_embeddings": int(meta.get(p + "context_length", 262144)),
        "eos_token_id": int(meta.get(p + "ple.eos_token_id", meta.get("tokenizer.ggml.eos_token_id", -1))),
        "rms_norm_eps": float(meta.get(p + "attention.layer_norm_rms_epsilon", 1e-6)),
        "rope_parameters": {"rope_type": "default", "rope_theta": theta, "partial_rotary_factor": rope_n / head_dim},
        "partial_rotary_factor": rope_n / head_dim,
        "hc_count": int(need("hyper_connection.count")),
        "hc_lowrank": int(need("hyper_connection.low_rank")),
        "num_attention_heads": int(need("attention.head_count")),
        "num_key_value_heads": int(need("attention.head_count_kv")),
        "head_dim": head_dim,
        "indexer_n_heads": int(need("attention.indexer.head_count")),
        "indexer_kv_heads": int(kproj["dims"][1]) // idx_dim,
        "indexer_head_dim": idx_dim,
        "indexer_budget": int(need("attention.indexer.top_k")),
        "indexer_compress_ratio": ratio,
        "num_experts": int(need("expert_count")),
        "num_experts_per_tok": int(need("expert_used_count")),
        "moe_intermediate_size": int(need("expert_feed_forward_length")),
        "shared_expert_intermediate_size": int(meta.get(p + "expert_shared_feed_forward_length", need("expert_feed_forward_length"))),
        "norm_topk_prob": True,
        "linear_num_key_heads": kh,
        "linear_num_value_heads": vh,
        "linear_key_head_dim": int(need("ssm.state_size")),
        "linear_value_head_dim": inner // vh,
        "linear_conv_kernel_dim": int(need("ssm.conv_kernel")),
        "layer_types": ["linear_attention" if r else "full_attention" for r in recurrent],
        "ple_layer_ids": [int(i) + 1 for i in ple_layers],
        "ngram_size": ngram,
        "heads_per_ngram": per,
        "ple_conv_kernel_size": int(need("ple.conv_kernel")),
        "ple_embed_dim": row * (ngram - 1) * per,
        "split_ngram_parts": 1,
        "hidden_act": "silu",
        "output_gate_type": "sigmoid",
        "attention_bias": False,
        "tie_word_embeddings": False,
    }
    return cfg, layers


ARCHES = {"qwen35moe": qwen35moe_config, "qwen4exp": qwen4exp_config}

# unsloth ships Qwen3.8-Flash-Next's MTP head apart, in an MTP/ folder of the repository:
# shared-* borrow token_embd and output from the model (the engine always does), the
# others carry copies. Their README recommends shared-Q8_0.
MTP_PREFERENCE = ("shared-Q8_0", "Q8_0", "shared-Q4_K_M", "Q4_K_M", "shared-BF16", "BF16")


def mtp_candidates(src):
    """mtp-*.gguf next to the model: its folder, the parent, an MTP/ folder in either; best first."""
    seen, found = set(), []
    for d in (src.parent, src.parent / "MTP", src.parent.parent, src.parent.parent / "MTP"):
        for f in sorted(d.glob("mtp-*.gguf")) if d.is_dir() else ():
            if f.resolve() not in seen:
                seen.add(f.resolve()); found.append(f)
    rank = lambda f: next((i for i, tag in enumerate(MTP_PREFERENCE) if f.name.endswith(f"-{tag}.gguf")), len(MTP_PREFERENCE))
    return sorted(found, key=rank)


def mtp_head_ok(path, meta, layers):
    """'' when path is a qwen4exp MTP head for this model (one nextn layer at blk.<layers>), else why not."""
    p = "qwen4exp."
    try:
        r = GGUFReader(str(path))
    except (GGUFError, OSError) as e:
        return str(e)
    try:
        hm, names = r.metadata, {t["name"] for t in r.tensors}
        if hm.get("general.architecture") != "qwen4exp":
            return f"architecture {hm.get('general.architecture')!r}, not qwen4exp"
        if int(hm.get(p + "nextn_predict_layers", 0)) != 1:
            return "not one MTP layer (nextn_predict_layers)"
        for k in ("block_count", "embedding_length", "expert_count", "expert_feed_forward_length"):
            want = int(meta[p + k]) + (1 if k == "block_count" else 0)
            if int(hm.get(p + k, -1)) != want:
                return f"{k} {hm.get(p + k)} does not match the model ({want})"
        if f"blk.{layers}.nextn.eh_proj.weight" not in names:
            return f"no blk.{layers}.nextn.eh_proj.weight"
        return ""
    finally:
        r.close()


# ------------------------------------------------------------- tokenizer
def tokenizer_files(meta):
    """tokenizer.json in HF's shape: added tokens (token_type 3 CONTROL = special,
    4 USER_DEFINED = not special) live only in added_tokens, the rest in
    model.vocab, as in the checkpoints the GGUF was converted from."""
    toks = meta.get("tokenizer.ggml.tokens")
    if not toks:
        sys.exit("GGUF carries no tokenizer (tokenizer.ggml.tokens)")
    model = meta.get("tokenizer.ggml.model", "gpt2")
    if model != "gpt2":
        sys.exit(f"tokenizer model {model!r} is not byte-level BPE (gpt2): not supported")
    types = meta.get("tokenizer.ggml.token_type") or [1] * len(toks)
    vocab, added = {}, []
    for i, t in enumerate(toks):
        kind = int(types[i]) if i < len(types) else 1
        if kind in (3, 4):
            added.append({"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": kind == 3})
        else:
            vocab[t] = i
    merges = list(meta.get("tokenizer.ggml.merges") or [])
    tok = {"version": "1.0", "added_tokens": added,
           "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True},
           "decoder": {"type": "ByteLevel"},
           "model": {"type": "BPE", "vocab": vocab, "merges": merges}}
    ids = {k: meta.get(f"tokenizer.ggml.{k}_token_id") for k in ("bos", "eos", "padding")}
    name = lambda i: toks[int(i)] if i is not None and int(i) < len(toks) else None
    tcfg = {"chat_template": meta.get("tokenizer.chat_template"),
            "bos_token": name(ids["bos"]), "eos_token": name(ids["eos"]), "pad_token": name(ids["padding"]),
            "add_bos_token": bool(meta.get("tokenizer.ggml.add_bos_token", False))}
    eos = [int(ids["eos"])] if ids["eos"] is not None else []
    for extra in ("<|endoftext|>", "<|im_end|>"):
        if extra in toks and toks.index(extra) not in eos:
            eos.append(toks.index(extra))
    gen = {"eos_token_id": eos, "bos_token_id": ids["bos"], "pad_token_id": ids["padding"]}
    return tok, tcfg, gen


# ------------------------------------------------------------ copy mode
def write_packed(parts, readers, meta, tensors, layers, out_path):
    n_exp = None
    kinds = ("gate", "up", "down")
    per_layer = []
    for L in range(layers):
        ts = [tensors.get(f"blk.{L}.ffn_{k}_exps.weight") for k in kinds]
        if any(t is None for t in ts):
            sys.exit(f"layer {L}: fused expert tensors missing")
        e = int(ts[0]["dims"][2])
        n_exp = n_exp or e
        eb = [t["nbytes"] // e for t in ts]
        stride = (sum(eb) + ALIGN - 1) // ALIGN * ALIGN
        per_layer.append((ts, eb, stride))
    expert_names = {f"blk.{L}.ffn_{k}_exps.weight" for L in range(layers) for k in kinds}
    dense = [t for n, t in tensors.items() if n not in expert_names]

    kv, n_kv, _ = kv_section(parts[0])
    extra = []

    def kv_str(k, v):
        kb, vb = k.encode(), v.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<IQ", T_STRING, len(vb)) + vb

    def kv_arr(k, et, fmt, vals):
        kb = k.encode()
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<IIQ", T_ARRAY, et, len(vals)) + b"".join(struct.pack(fmt, v) for v in vals)

    extra.append(kv_str("colibri.layout", LAYOUT))
    extra.append(kv_str("colibri.source", os.path.basename(str(parts[0]))))
    extra.append(kv_arr("colibri.expert_types", T_UINT32, "<I", [t["ggml_type"] for ts, _, _ in per_layer for t in ts]))
    extra.append(kv_arr("colibri.expert_bytes", T_UINT64, "<Q", [b for _, eb, _ in per_layer for b in eb]))
    extra.append(kv_arr("colibri.expert_stride", T_UINT64, "<Q", [s for _, _, s in per_layer]))

    # tensor infos: dense tensors as they were, then one packed tensor per layer
    infos, offsets, off = [], [], 0
    def align(x, a): return (x + a - 1) // a * a
    for t in dense:
        off = align(off, ALIGN)
        infos.append((t["name"], t["dims"], t["ggml_type"])); offsets.append(off); off += t["nbytes"]
    for L, (ts, eb, stride) in enumerate(per_layer):
        off = align(off, ALIGN)
        infos.append((f"blk.{L}.ffn_exps_colibri", [stride, n_exp], GGML_I8)); offsets.append(off); off += stride * n_exp
    total = off

    def info_bytes():
        b = b""
        for (name, dims, typ), o in zip(infos, offsets):
            nb = name.encode()
            b += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", int(d)) for d in dims)
            b += struct.pack("<IQ", typ, o)
        return b

    # pad the header with a string key so the data section starts on a 4 KiB
    # boundary: every tensor offset is a multiple of 4 KiB, so every expert
    # slot is 4 KiB-aligned in the FILE, not only relative to the data start
    base = 24 + len(kv) + sum(len(e) for e in extra) + len(info_bytes())
    pad_key = b"colibri.pad"
    fixed = 8 + len(pad_key) + 4 + 8
    padlen = (-(base + fixed)) % ALIGN
    extra.append(kv_str("colibri.pad", " " * padlen))
    header = b"GGUF" + struct.pack("<IQQ", 3, len(infos), n_kv + len(extra)) + kv + b"".join(extra) + info_bytes()
    assert len(header) % ALIGN == 0, len(header)

    tmp = Path(str(out_path) + ".tmp")
    with open(tmp, "wb") as out:
        out.write(header)
        data0 = len(header)
        fds = [open(p, "rb") for p in parts]
        def copy(part, src, n, dst):
            out.seek(dst); f = fds[part]; f.seek(src)
            while n:
                chunk = f.read(min(n, 64 << 20))
                if not chunk:
                    sys.exit("source GGUF ended early")
                out.write(chunk); n -= len(chunk)
        done = 0
        for t, o in zip(dense, offsets):
            copy(t["part"], t["abs_offset"], t["nbytes"], data0 + o)
        for L, (ts, eb, stride) in enumerate(per_layer):
            o = offsets[len(dense) + L]
            for e in range(n_exp):
                slot = data0 + o + e * stride; inner = 0
                for t, b in zip(ts, eb):
                    copy(t["part"], t["abs_offset"] + e * b, b, slot + inner); inner += b
            done += 1
            print(f"\r  experts repacked: layer {done}/{layers}", end="", flush=True)
        out.truncate(data0 + total)
        for f in fds: f.close()
    print()
    os.replace(tmp, out_path)
    return data0 + total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--mode", choices=("link", "copy"), default="link")
    ap.add_argument("--mtp", type=Path, default=None,
                    help="qwen4exp: the MTP head's GGUF (unsloth's mtp-*.gguf); unset, one next to the model "
                         "is used (its folder, the parent, an MTP/ folder in either; shared-Q8_0 first)")
    ap.add_argument("--no-mtp", action="store_true", help="qwen4exp: no MTP head, plain decoding")
    ap.add_argument("--indexer-compress-ratio", type=int, default=0,
                    help="qwen4exp: when the GGUF's attention.compress_ratios are all 0 (a converter "
                         "that read the layers as qwen_sparse_attention)")
    args = ap.parse_args()
    src = args.gguf.resolve()
    parts, readers, meta, tensors = open_model(src)
    arch = meta.get("general.architecture")
    if arch not in ARCHES:
        sys.exit(f"architecture {arch!r} is not supported yet (supported: {', '.join(ARCHES)})")
    cfg, layers = ARCHES[arch](meta, tensors, {"indexer_compress_ratio": args.indexer_compress_ratio})
    mtp_file, mtp_where = None, None
    if arch == "qwen4exp" and not args.no_mtp:
        if f"blk.{layers}.nextn.eh_proj.weight" in tensors:
            mtp_where = "in the model's GGUF"
        else:
            for cand in ([args.mtp] if args.mtp else mtp_candidates(src)):
                why = mtp_head_ok(cand, meta, layers)
                if not why:
                    mtp_file = cand.resolve(); mtp_where = str(mtp_file); break
                if args.mtp:
                    sys.exit(f"--mtp {cand}: {why}")
                print(f"  {cand.name}: not this model's MTP head ({why})")
        if mtp_where:   # the head drafts; the engine reads the rest of it from the GGUF
            cfg["mtp_num_hidden_layers"] = 1
            cfg["mtp"] = {"layer_types": ["full_attention"], "rope_theta": cfg["rope_parameters"]["rope_theta"]}
            cfg["mtp_use_dedicated_embeddings"] = False
    tok, tcfg, gen = tokenizer_files(meta)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg["colibri_gguf"] = {"arch": arch, "mode": args.mode, "source": str(src)}
    for name, obj in (("config.json", cfg), ("tokenizer.json", tok), ("tokenizer_config.json", tcfg),
                      ("generation_config.json", gen)):
        (args.out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1 if name != "tokenizer.json" else None))
    target = src
    if args.mode == "copy":
        target = args.out / "model.colibri.gguf"
        size = write_packed(parts, readers, meta, tensors, layers, target)
        print(f"  wrote {target} ({size / 1e9:.2f} GB, experts packed one per {ALIGN}-byte-aligned slot)")
    link = {"format": "colibri-gguf-v1", "gguf": str(target) if args.mode == "link" else target.name,
            "mode": args.mode, "arch": arch, "parts": len(parts),
            "size": sum(p.stat().st_size for p in parts) if args.mode == "link" else target.stat().st_size}
    if mtp_file:
        link["mtp"] = str(mtp_file)
    (args.out / "gguf.json").write_text(json.dumps(link, indent=1))
    for r in readers:
        r.close()
    print(f"colibri model directory ready: {args.out}  ({arch}, {layers} layers, mode {args.mode})")
    if arch == "qwen4exp":
        print(f"  MTP head: {mtp_where or ('off (--no-mtp)' if args.no_mtp else 'none found: decoding without drafts (--mtp names one)')}")


if __name__ == "__main__":
    try:
        main()
    except GGUFError as e:
        sys.exit(f"GGUF: {e}")
