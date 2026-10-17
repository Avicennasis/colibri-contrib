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
                   verbatim; colibri.* keys describe the new layout. Every
                   architecture with routed experts but qwen4exp (whose engine
                   reads its GGUF in place only).

Either way the directory is an ordinary colibri model: `coli chat`, `coli
serve`, `coli web` and the planner treat it like any other.

Supported architectures: qwen35moe (Qwen3.5 / Qwen3.6 MoE -> qwen36 engine), qwen35 (their
dense models -> qwen36 engine), qwen3moe (Qwen3 MoE, Qwen3-Coder -> qwen36 engine), qwen3 (the dense
Qwen3 -> qwen36 engine), qwen4exp (Qwen3.8-Flash-Next -> qwen38 engine), deepseek4 (DeepSeek V4
Flash -> deepseek_v4 engine), inkling (Inkling -> inkling engine), glm5-next (GLM-5.3-Flash ->
glm53 engine), glm-dsa (GLM-5.2 / GLM-5.3 -> colibri engine), mimo2 (MiMo-V2.6 -> mimo engine).

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


def open_model(path, header_only=False):
    parts = split_parts(path)
    readers = [GGUFReader(str(p), header_only=header_only) for p in parts]
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


def qwen35_config(meta, tensors, overrides=None):
    """A dense model of the Qwen3.5 family (llama.cpp's qwen35: Qwen3.5 0.8B to 27B, Qwen3.8 27B)
    as the qwen36 engine's config.json: the qwen35moe layout with an MLP instead of experts,
    and the embedding tied to the head when the file has no output.weight."""
    p = "qwen35."
    need = lambda k: meta[p + k] if (p + k) in meta else sys.exit(f"GGUF metadata lacks {p + k}")
    moe_like = {(("qwen35moe." + k[len(p):]) if k.startswith(p) else k): v for k, v in meta.items()}
    moe_like.setdefault("qwen35moe.expert_count", 1)
    moe_like.setdefault("qwen35moe.expert_used_count", 1)
    moe_like.setdefault("qwen35moe.expert_feed_forward_length", int(need("feed_forward_length")))
    cfg, layers = qwen35moe_config(moe_like, tensors, overrides)
    for k in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size"):
        cfg.pop(k, None)
    cfg.update({"model_type": "qwen3_5_text", "architectures": ["Qwen3_5ForCausalLM"],
                "intermediate_size": int(need("feed_forward_length")),
                "tie_word_embeddings": "output.weight" not in tensors})
    return cfg, layers


def qwen3moe_config(meta, tensors, overrides=None):
    """Qwen3 MoE (llama.cpp's qwen3moe: Qwen3-30B-A3B, Qwen3-Coder) as the qwen36 engine's
    config.json: every layer attention, no shared expert, no output gate, full rotary."""
    p = "qwen3moe."
    need = lambda k: meta[p + k] if (p + k) in meta else sys.exit(f"GGUF metadata lacks {p + k}")
    layers = int(need("block_count"))
    # written when the config names head_dim; else the width split over the heads
    head_dim = int(meta.get(p + "attention.key_length", int(need("embedding_length")) // int(need("attention.head_count"))))
    emb = tensors.get("token_embd.weight") or sys.exit("GGUF has no token_embd.weight")
    cfg = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": int(need("embedding_length")),
        "num_hidden_layers": layers,
        "num_attention_heads": int(need("attention.head_count")),
        "num_key_value_heads": int(need("attention.head_count_kv")),
        "head_dim": head_dim,
        "num_experts": int(need("expert_count")),
        "num_experts_per_tok": int(need("expert_used_count")),
        "moe_intermediate_size": int(need("expert_feed_forward_length")),
        "intermediate_size": int(meta.get(p + "feed_forward_length", need("expert_feed_forward_length"))),
        "norm_topk_prob": True,
        "vocab_size": int(emb["dims"][1]),
        "rms_norm_eps": float(meta.get(p + "attention.layer_norm_rms_epsilon", 1e-6)),
        "max_position_embeddings": int(meta.get(p + "context_length", 262144)),
        "rope_theta": float(meta.get(p + "rope.freq_base", 1000000.0)),
        "layer_types": ["full_attention"] * layers,
        "tie_word_embeddings": "output.weight" not in tensors,
        "hidden_act": "silu",
    }
    return cfg, layers


def qwen3_config(meta, tensors, overrides=None):
    """The dense Qwen3 (llama.cpp's qwen3: Qwen3 0.6B to 32B) as the qwen36 engine's config.json:
    qwen3moe's layers with one MLP of feed_forward_length each."""
    p = "qwen3."
    need = lambda k: meta[p + k] if (p + k) in meta else sys.exit(f"GGUF metadata lacks {p + k}")
    moe_like = {(("qwen3moe." + k[len(p):]) if k.startswith(p) else k): v for k, v in meta.items()}
    moe_like.setdefault("qwen3moe.expert_count", 1)
    moe_like.setdefault("qwen3moe.expert_used_count", 1)
    moe_like.setdefault("qwen3moe.expert_feed_forward_length", int(need("feed_forward_length")))
    cfg, layers = qwen3moe_config(moe_like, tensors, overrides)
    for k in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "norm_topk_prob"):
        cfg.pop(k, None)
    cfg.update({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                "intermediate_size": int(need("feed_forward_length"))})
    return cfg, layers


def deepseek4_config(meta, tensors, overrides=None):
    """DeepSeek V4 (llama.cpp's deepseek4: V4 Flash, Flash 0731, Pro) as the deepseek_v4 engine's
    config.json, the keys of the original checkpoint's (conversion/deepseek.py writes each from
    one of them). The engine reads the tensors through st_gguf.h under their checkpoint names;
    the GGUF of the model carries no MTP head (llama.cpp writes it to its own mtp-*.gguf)."""
    p = "deepseek4."
    need = lambda k: meta[p + k] if (p + k) in meta else sys.exit(f"GGUF metadata lacks {p + k}")
    # block_count counts the MTP blocks only where the file holds them (llama.cpp's mtp files;
    # antirez's main files say nextn_predict_layers 1 and keep the head apart): the layers are
    # the blocks with attention
    layers = sum(1 for b in range(int(need("block_count"))) if f"blk.{b}.attn_q_a.weight" in tensors)
    gating = int(meta.get(p + "expert_gating_func", 4))
    if gating != 4:
        sys.exit(f"{p}expert_gating_func {gating}: the engine routes with sqrt(softplus) (4) only")
    emb = tensors.get("token_embd.weight") or sys.exit("GGUF has no token_embd.weight")
    clamp = meta.get(p + "swiglu_clamp_exp") or [10.0]
    ratios = [int(r) for r in need("attention.compress_ratios")]
    if len(ratios) < layers:
        sys.exit(f"{p}attention.compress_ratios has {len(ratios)} entries for {layers} layers")
    cfg = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "torch_dtype": "bfloat16",
        "expert_dtype": "fp4",
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "hidden_size": int(need("embedding_length")),
        "num_hidden_layers": layers,
        "num_attention_heads": int(need("attention.head_count")),
        "num_key_value_heads": int(meta.get(p + "attention.head_count_kv", 1)),
        "head_dim": int(need("attention.key_length")),
        "q_lora_rank": int(need("attention.q_lora_rank")),
        "qk_rope_head_dim": int(need("rope.dimension_count")),
        "o_groups": int(need("attention.output_group_count")),
        "o_lora_rank": int(need("attention.output_lora_rank")),
        "sliding_window": int(need("attention.sliding_window")),
        "index_n_heads": int(need("attention.indexer.head_count")),
        "index_head_dim": int(need("attention.indexer.key_length")),
        "index_topk": int(need("attention.indexer.top_k")),
        "n_routed_experts": int(need("expert_count")),
        "num_experts_per_tok": int(need("expert_used_count")),
        "n_shared_experts": int(meta.get(p + "expert_shared_count", 1)),
        "moe_intermediate_size": int(need("expert_feed_forward_length")),
        "num_hash_layers": int(meta.get(p + "hash_layer_count", 0)),
        "num_nextn_predict_layers": 0,
        "hc_mult": int(need("hyper_connection.count")),
        "hc_sinkhorn_iters": int(need("hyper_connection.sinkhorn_iterations")),
        "vocab_size": int(emb["dims"][1]),
        "max_position_embeddings": int(meta.get(p + "context_length", 1048576)),
        "rms_norm_eps": float(meta.get(p + "attention.layer_norm_rms_epsilon", 1e-6)),
        "hc_eps": float(meta.get(p + "hyper_connection.epsilon", 1e-6)),
        "routed_scaling_factor": float(meta.get(p + "expert_weights_scale", 1.0)),
        "norm_topk_prob": bool(meta.get(p + "expert_weights_norm", True)),
        "swiglu_limit": float(clamp[0]),
        "rope_theta": float(meta.get(p + "rope.freq_base", 10000.0)),
        "compress_rope_theta": float(need("attention.compress_rope_freq_base")),
        "compress_ratios": ratios,
        "rope_scaling": {
            "type": meta.get(p + "rope.scaling.type", "yarn"),
            "factor": float(meta.get(p + "rope.scaling.factor", 1.0)),
            "original_max_position_embeddings": int(meta.get(p + "rope.scaling.original_context_length",
                                                             meta.get(p + "context_length", 65536))),
            "beta_fast": int(meta.get(p + "rope.scaling.yarn_beta_fast", 32)),
            "beta_slow": int(meta.get(p + "rope.scaling.yarn_beta_slow", 1)),
        },
        "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "scale_fmt": "ue8m0",
                                "quant_method": "fp8", "weight_block_size": [128, 128]},
        "tie_word_embeddings": False,
        "bos_token_id": int(meta.get("tokenizer.ggml.bos_token_id", 0)),
        "eos_token_id": int(meta.get("tokenizer.ggml.eos_token_id", 1)),
    }
    return cfg, layers


def inkling_config(meta, tensors, overrides=None):
    """Inkling (llama.cpp's inkling: Inkling-Small and the full model) as the inkling engine's
    config.json, in the shape of transformers' port (inkling_text), whose tensor names
    st_gguf.h gives the GGUF's. Local (sliding) and global layers differ in their KV heads:
    the GGUF's per-layer head_count_kv with its sliding_window_pattern gives both."""
    p = "inkling."
    need = lambda k: meta[p + k] if (p + k) in meta else sys.exit(f"GGUF metadata lacks {p + k}")
    layers = int(need("block_count"))
    local = [bool(x) for x in need("attention.sliding_window_pattern")][:layers]
    kv = need("attention.head_count_kv")
    kv = [int(x) for x in kv][:layers] if isinstance(kv, (list, tuple)) else [int(kv)] * layers
    kv_local = next((kv[i] for i in range(layers) if local[i]), kv[0])
    kv_global = next((kv[i] for i in range(layers) if not local[i]), kv[0])
    heads, head_dim = int(need("attention.head_count")), int(need("attention.key_length"))
    dense = int(meta.get(p + "dense_block_count", 0))
    emb = tensors.get("token_embd.weight") or sys.exit("GGUF has no token_embd.weight")
    vocab = int(meta.get(p + "vocab_size", emb["dims"][1]))
    eos = int(meta.get("tokenizer.ggml.eos_token_id", 200006))
    cfg = {
        "architectures": ["InklingForCausalLM"],
        "model_type": "inkling",   # the checkpoint's, which the family registry knows
        "hidden_size": int(need("embedding_length")),
        "num_hidden_layers": layers,
        "intermediate_size": int(need("feed_forward_length")),
        "moe_intermediate_size": int(need("expert_feed_forward_length")),
        "num_attention_heads": heads, "swa_num_attention_heads": heads,
        "num_key_value_heads": kv_global, "swa_num_key_value_heads": kv_local,
        "head_dim": head_dim, "swa_head_dim": head_dim,
        "sliding_window_size": int(need("attention.sliding_window")),
        "local_layer_ids": [i for i in range(layers) if local[i]],
        "d_rel": int(need("d_rel")), "rel_extent": int(need("rel_extent")),
        "conv_kernel_size": int(need("shortconv_kernel")), "sconv_kernel_size": int(need("shortconv_kernel")),
        "dense_mlp_idx": dense, "mlp_layer_types": ["dense"] * dense + ["sparse"] * (layers - dense),
        "n_routed_experts": int(need("expert_count")), "n_shared_experts": int(meta.get(p + "expert_shared_count", 0)),
        "num_experts_per_tok": int(need("expert_used_count")),
        "route_scale": float(meta.get(p + "expert_weights_scale", 1.0)),
        "shared_expert_sink": True,
        "logits_mup_width_multiplier": float(need("logit_scale_denom")),
        "log_scaling_n_floor": int(meta.get(p + "log_scaling_n_floor", 0)),
        "log_scaling_alpha": float(meta.get(p + "log_scaling_alpha", 0.0)),
        "rms_norm_eps": float(meta.get(p + "attention.layer_norm_rms_epsilon", 1e-6)),
        "vocab_size": vocab,
        "unpadded_vocab_size": int(meta.get(p + "unpadded_vocab_size", vocab)),
        "max_position_embeddings": int(meta.get(p + "context_length", 1048576)),
        "hidden_act": "silu",
        "eos_token_id": eos, "bos_token_id": eos,
    }
    return cfg, layers


def glm5next_config(meta, tensors, overrides=None):
    """GLM-5.3-Flash (llama.cpp's glm5-next) as the glm53 engine's config.json, the shape of
    zai-org's (Glm5NextForConditionalGeneration, its text_config). head_count_kv marks the MLA
    layers (1) among the KDA ones (0); the MTP block counts in block_count only where the file
    holds it."""
    p = "glm5-next."
    need = lambda k: meta[p + k] if (p + k) in meta else sys.exit(f"GGUF metadata lacks {p + k}")
    blocks = int(need("block_count"))
    nextn = int(meta.get(p + "nextn_predict_layers", 0))
    has_mtp = nextn > 0 and f"blk.{blocks - 1}.nextn.eh_proj.weight" in tensors
    layers = blocks - nextn if has_mtp else blocks
    kv = need("attention.head_count_kv")
    kinds = ["deepseek_sparse_attention" if int(x) else "linear_attention" for x in kv][:layers]
    dense = int(meta.get(p + "leading_dense_block_count", 0))
    heads = int(need("attention.head_count"))
    emb = tensors.get("token_embd.weight") or sys.exit("GGUF has no token_embd.weight")
    itypes = meta.get(p + "attention.indexer.types")
    clamp = meta.get(p + "swiglu_clamp_exp") or [10.0]
    text = {
        "model_type": "glm5_next_text",
        "vocab_size": int(meta.get(p + "vocab_size", emb["dims"][1])),
        "hidden_size": int(need("embedding_length")),
        "intermediate_size": int(need("feed_forward_length")),
        "moe_intermediate_size": int(need("expert_feed_forward_length")),
        "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": heads,
        "n_shared_experts": int(meta.get(p + "expert_shared_count", 1)), "n_routed_experts": int(need("expert_count")),
        "num_experts_per_tok": int(need("expert_used_count")),
        "kv_lora_rank": int(need("attention.kv_lora_rank")), "q_lora_rank": int(need("attention.q_lora_rank")),
        "qk_rope_head_dim": int(meta.get(p + "rope.dimension_count", 0)),
        "qk_nope_head_dim": int(need("attention.key_length_mla")), "v_head_dim": int(need("attention.value_length_mla")),
        "max_position_embeddings": int(meta.get(p + "context_length", 1048576)),
        "layer_types": kinds,
        "mlp_layer_types": ["dense"] * dense + ["sparse"] * (layers - dense), "first_k_dense_replace": dense,
        "index_topk": int(need("attention.indexer.top_k")), "index_kpool": int(meta.get(p + "attention.indexer.kpool", 1)),
        "index_head_dim": int(need("attention.indexer.key_length")), "index_n_heads": int(need("attention.indexer.head_count")),
        "index_kpool_always_select_tail": bool(meta.get(p + "attention.indexer.kpool_select_tail", True)),
        "linear_head_dim": int(need("kda.head_dim")), "linear_num_heads": heads,
        "linear_conv_kernel_dim": int(need("ssm.conv_kernel")),
        "linear_attn_config": {"num_heads": heads, "head_dim": int(need("kda.head_dim")),
                               "short_conv_kernel_size": int(need("ssm.conv_kernel")),
                               "gate_lower_bound": float(meta.get(p + "kda.gate_lower_bound", -5.0)),
                               "kda_layers": [i for i in range(layers) if kinds[i] == "linear_attention"]},
        "hc_mult": int(need("hyper_connection.count")), "hc_sinkhorn_iters": int(need("hyper_connection.sinkhorn_iterations")),
        "hc_eps": float(meta.get(p + "hyper_connection.epsilon", 1e-6)),
        "rms_norm_eps": float(meta.get(p + "attention.layer_norm_rms_epsilon", 1e-5)),
        "routed_scaling_factor": float(meta.get(p + "expert_weights_scale", 1.0)),
        "norm_topk_prob": bool(meta.get(p + "expert_weights_norm", True)),
        "swiglu_limit": float(clamp[0]),
        "num_nextn_predict_layers": nextn if has_mtp else 0,
        "mla_use_nope": True, "scoring_func": "sigmoid", "tie_word_embeddings": False,
        "bos_token_id": meta.get("tokenizer.ggml.bos_token_id"), "eos_token_id": meta.get("tokenizer.ggml.eos_token_id"),
    }
    if itypes is not None:
        text["indexer_types"] = ["full" if t else "shared" for t in itypes][:layers]
    return {"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next", "text_config": text}, layers


def glmdsa_config(meta, tensors, overrides=None):
    """GLM-5.2 / GLM-5.3 (llama.cpp's glm-dsa) as the colibri engine's config.json, the shape
    of zai-org's (GlmMoeDsaForCausalLM). The NextN block counts in block_count only where the
    file holds it; a layer whose indexer weights are in the file indexes on its own ("full"),
    one without reuses the previous selection ("shared")."""
    p = "glm-dsa."
    need = lambda k: meta[p + k] if (p + k) in meta else sys.exit(f"GGUF metadata lacks {p + k}")
    blocks = int(need("block_count"))
    nextn = int(meta.get(p + "nextn_predict_layers", 0))
    has_mtp = nextn > 0 and f"blk.{blocks - 1}.nextn.eh_proj.weight" in tensors
    layers = blocks - nextn if has_mtp else blocks
    heads = int(need("attention.head_count"))
    rope = int(need("rope.dimension_count"))
    emb = tensors.get("token_embd.weight") or sys.exit("GGUF has no token_embd.weight")
    if int(meta.get(p + "expert_group_count", 1)) != 1:
        sys.exit("glm-dsa: expert groups other than 1 are not this model")
    cfg = {
        "architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa",
        "vocab_size": int(meta.get(p + "vocab_size", emb["dims"][1])),
        "hidden_size": int(need("embedding_length")),
        "intermediate_size": int(need("feed_forward_length")),
        "moe_intermediate_size": int(need("expert_feed_forward_length")),
        "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": heads,
        "n_routed_experts": int(need("expert_count")), "num_experts_per_tok": int(need("expert_used_count")),
        "n_shared_experts": int(meta.get(p + "expert_shared_count", 1)),
        "first_k_dense_replace": int(meta.get(p + "leading_dense_block_count", 0)),
        "q_lora_rank": int(need("attention.q_lora_rank")), "kv_lora_rank": int(need("attention.kv_lora_rank")),
        "qk_rope_head_dim": rope, "qk_nope_head_dim": int(need("attention.key_length_mla")) - rope,
        "v_head_dim": int(need("attention.value_length_mla")),
        "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc", "scoring_func": "sigmoid",
        "norm_topk_prob": bool(meta.get(p + "expert_weights_norm", True)),
        "routed_scaling_factor": float(meta.get(p + "expert_weights_scale", 1.0)),
        "rms_norm_eps": float(meta.get(p + "attention.layer_norm_rms_epsilon", 1e-5)),
        "rope_parameters": {"rope_type": "default", "rope_theta": float(meta.get(p + "rope.freq_base", 10000.0))},
        "max_position_embeddings": int(meta.get(p + "context_length", 202752)),
        "index_topk": int(need("attention.indexer.top_k")), "index_n_heads": int(need("attention.indexer.head_count")),
        "index_head_dim": int(need("attention.indexer.key_length")),
        "indexer_types": ["full" if f"blk.{i}.indexer.attn_q_b.weight" in tensors else "shared" for i in range(layers)],
        "num_nextn_predict_layers": nextn if has_mtp else 0,
        "tie_word_embeddings": "output.weight" not in tensors,
        "bos_token_id": meta.get("tokenizer.ggml.bos_token_id"), "eos_token_id": meta.get("tokenizer.ggml.eos_token_id"),
    }
    return cfg, layers


def mimo2_config(meta, tensors, overrides=None):
    """MiMo-V2 (llama.cpp's mimo2) as the mimo engine's config.json, the shape of Xiaomi's
    (MiMoV2FlashForCausalLM): the layer kinds from the sliding-window pattern, the MoE layers
    from the files' expert tensors, each kind's KV heads and attention sinks from its layers.
    The converter writes qkv_proj as one [Q|K|V] (qkv_shards 1); the NextN blocks past the
    model's layers are not this engine's."""
    p = "mimo2."
    need = lambda k: meta[p + k] if (p + k) in meta else sys.exit(f"GGUF metadata lacks {p + k}")
    layers = int(need("block_count")) - int(meta.get(p + "nextn_predict_layers", 0))
    pattern = [int(x) for x in need("attention.sliding_window_pattern")][:layers]
    kv = need("attention.head_count_kv")
    kv = [int(x) for x in kv] if isinstance(kv, list) else [int(kv)] * layers
    moe = [int(f"blk.{i}.ffn_gate_exps.weight" in tensors) for i in range(layers)]
    head_dim, rope = int(need("attention.key_length")), int(need("rope.dimension_count"))
    kinds = {k: [i for i in range(layers) if pattern[i] == k] for k in (0, 1)}
    if not kinds[0] or not kinds[1]:
        sys.exit("mimo2: the GGUF has no full-attention or no sliding-window layer")
    kv_of = {k: {kv[i] for i in kinds[k]} for k in (0, 1)}
    if len(kv_of[0]) != 1 or len(kv_of[1]) != 1:
        sys.exit("mimo2: a layer kind with two KV head counts")
    emb = tensors.get("token_embd.weight") or sys.exit("GGUF has no token_embd.weight")
    sink = {k: f"blk.{kinds[k][0]}.attn_sinks.weight" in tensors for k in (0, 1)}
    heads = int(need("attention.head_count"))
    cfg = {
        "architectures": ["MiMoV2FlashForCausalLM"], "model_type": "mimo_v2",
        "vocab_size": int(meta.get(p + "vocab_size", emb["dims"][1])),
        "hidden_size": int(need("embedding_length")), "intermediate_size": int(need("feed_forward_length")),
        "num_hidden_layers": layers, "hybrid_layer_pattern": pattern, "moe_layer_freq": moe,
        "num_attention_heads": heads, "num_key_value_heads": kv_of[0].pop(),
        "head_dim": head_dim, "v_head_dim": int(need("attention.value_length")),
        "swa_num_attention_heads": heads, "swa_num_key_value_heads": kv_of[1].pop(),
        "swa_head_dim": head_dim, "swa_v_head_dim": int(need("attention.value_length")),
        "sliding_window": int(need("attention.sliding_window")),
        "add_full_attention_sink_bias": sink[0], "add_swa_attention_sink_bias": sink[1],
        "attention_value_scale": float(meta.get(p + "attention.value_scale", 1.0)),
        # int(head_dim * factor) is the rotary width the engine takes
        "partial_rotary_factor": (rope + 0.5) / head_dim,
        "rope_theta": float(meta.get(p + "rope.freq_base", 10000.0)),
        "swa_rope_theta": float(meta.get(p + "rope.freq_base_swa", meta.get(p + "rope.freq_base", 10000.0))),
        "n_routed_experts": int(need("expert_count")), "num_experts_per_tok": int(need("expert_used_count")),
        "moe_intermediate_size": int(need("expert_feed_forward_length")),
        "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc", "scoring_func": "sigmoid",
        "norm_topk_prob": bool(meta.get(p + "expert_weights_norm", True)),
        "routed_scaling_factor": float(meta.get(p + "expert_weights_scale", 1.0)),
        "layernorm_epsilon": float(meta.get(p + "attention.layer_norm_rms_epsilon", 1e-6)),
        "max_position_embeddings": int(meta.get(p + "context_length", 262144)),
        "qkv_shards": 1, "tie_word_embeddings": "output.weight" not in tensors,
        "bos_token_id": meta.get("tokenizer.ggml.bos_token_id"), "eos_token_id": meta.get("tokenizer.ggml.eos_token_id"),
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


COPY_UNREAD = {"qwen4exp"}   # their engine does not read the colibri container (copy mode)
ARCHES = {"qwen35moe": qwen35moe_config, "qwen35": qwen35_config, "qwen3moe": qwen3moe_config, "qwen3": qwen3_config, "deepseek4": deepseek4_config, "inkling": inkling_config, "glm5-next": glm5next_config, "glm-dsa": glmdsa_config, "mimo2": mimo2_config, "qwen4exp": qwen4exp_config}

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
# The pre-tokenizer of a GGUF tokenizer, by llama.cpp's name for it (tokenizer.ggml.pre ->
# llama-vocab.cpp's regex_exprs), in the shape the checkpoint's tokenizer.json gives it.
DEEPSEEK_SPLITS = ["\\p{N}{1,3}", "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
                   "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|"
                   " ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"]
# o200k (GPT-4o's), as Inkling's tokenizer.json has it; llama.cpp's "inkling" pre-type is the
# same split written with lookaheads
O200K_SPLITS = ["[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
                "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"]
# cl100k's, as GLM-5.3-Flash's tokenizer.json has it (llama.cpp's "glm4")
CL100K_SPLITS = ["(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
                 "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"]
PRE_SPLITS = {"glm4": CL100K_SPLITS, "deepseek-v3": DEEPSEEK_SPLITS, "joyai-llm": DEEPSEEK_SPLITS,
              "hunyuan-dense": DEEPSEEK_SPLITS, "hy_v4": DEEPSEEK_SPLITS,
              "inkling": O200K_SPLITS, "gpt-4o": O200K_SPLITS}
# BPE options of the tokenizer.json a family ships (the GGUF does not carry them)
PRE_MODEL = {"inkling": {"ignore_merges": True}, "gpt-4o": {"ignore_merges": True}, "glm4": {"ignore_merges": True}}


def pre_tokenizer(pre):
    splits = PRE_SPLITS.get(pre or "")
    if not splits:   # the default: ByteLevel's own (GPT-2) regex
        return {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True}
    return {"type": "Sequence", "pretokenizers":
            [{"type": "Split", "pattern": {"Regex": r}, "behavior": "Isolated", "invert": False} for r in splits] +
            [{"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]}


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
    kinds = [int(types[i]) if i < len(types) else 1 for i in range(len(toks))]
    # the added tokens past the BPE vocabulary; one inside it (DeepSeek's <｜begin▁of▁sentence｜>)
    # is in model.vocab as well, as the checkpoint has it
    tail = len(toks)
    while tail > 0 and kinds[tail - 1] in (3, 4):
        tail -= 1
    vocab, added = {}, []
    for i, t in enumerate(toks):
        if kinds[i] in (3, 4):
            added.append({"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": kinds[i] == 3})
        if kinds[i] not in (3, 4) or i < tail:
            vocab[t] = i
    merges = list(meta.get("tokenizer.ggml.merges") or [])
    tok = {"version": "1.0", "added_tokens": added,
           "pre_tokenizer": pre_tokenizer(meta.get("tokenizer.ggml.pre")),
           "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
           "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                     "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
                     **PRE_MODEL.get(meta.get("tokenizer.ggml.pre") or "", {}),
                     "vocab": vocab, "merges": merges}}
    ids = {k: meta.get(f"tokenizer.ggml.{k}_token_id") for k in ("bos", "eos", "padding")}
    name = lambda i: toks[int(i)] if i is not None and int(i) < len(toks) else None
    tcfg = {"chat_template": meta.get("tokenizer.chat_template"),
            "bos_token": name(ids["bos"]), "eos_token": name(ids["eos"]), "pad_token": name(ids["padding"]),
            "add_bos_token": bool(meta.get("tokenizer.ggml.add_bos_token", False))}
    eos = [int(ids["eos"])] if ids["eos"] is not None else []
    for k in ("eot", "eom"):   # end of turn, end of message (GLM's <|user|>, <|observation|>)
        i = meta.get(f"tokenizer.ggml.{k}_token_id")
        if i is not None and int(i) not in eos:
            eos.append(int(i))
    for extra in ("<|endoftext|>", "<|im_end|>"):
        if extra in toks and toks.index(extra) not in eos:
            eos.append(toks.index(extra))
    gen = {"eos_token_id": eos, "bos_token_id": ids["bos"], "pad_token_id": ids["padding"]}
    return tok, tcfg, gen


# ------------------------------------------------------------ copy mode
def write_packed(parts, readers, meta, tensors, layers, out_path):
    n_exp = None
    kinds = ("gate", "up", "down")
    per_layer = []   # (tensors, bytes per expert, stride) of a layer with routed experts, else None
    for L in range(layers):
        ts = [tensors.get(f"blk.{L}.ffn_{k}_exps.weight") for k in kinds]
        if all(t is None for t in ts):   # a dense layer: nothing to pack
            per_layer.append(None)
            continue
        if any(t is None for t in ts):
            sys.exit(f"layer {L}: fused expert tensors missing")
        e = int(ts[0]["dims"][2])
        if n_exp and e != n_exp:
            sys.exit(f"layer {L}: {e} experts, the layers before {n_exp}")
        n_exp = n_exp or e
        eb = [t["nbytes"] // e for t in ts]
        stride = (sum(eb) + ALIGN - 1) // ALIGN * ALIGN
        per_layer.append((ts, eb, stride))
    if not n_exp:
        sys.exit("--mode copy: this model has no routed experts to pack; --mode link reads it in place")
    expert_names = {f"blk.{L}.ffn_{k}_exps.weight" for L, pl in enumerate(per_layer) if pl for k in kinds}
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
    # per layer: the three matrices' types, bytes and [cols, rows]; a dense layer's are 0
    none = ([None] * 3, [0] * 3, 0)
    extra.append(kv_arr("colibri.expert_types", T_UINT32, "<I", [t["ggml_type"] if t else 0 for ts, _, _ in (pl or none for pl in per_layer) for t in ts]))
    extra.append(kv_arr("colibri.expert_bytes", T_UINT64, "<Q", [b for _, eb, _ in (pl or none for pl in per_layer) for b in eb]))
    extra.append(kv_arr("colibri.expert_stride", T_UINT64, "<Q", [s for _, _, s in (pl or none for pl in per_layer)]))
    extra.append(kv_arr("colibri.expert_dims", T_UINT64, "<Q", [int(d) for ts, _, _ in (pl or none for pl in per_layer)
                                                                for t in ts for d in ((t["dims"][0], t["dims"][1]) if t else (0, 0))]))

    # tensor infos: dense tensors as they were, then one packed tensor per layer
    infos, offsets, off = [], [], 0
    def align(x, a): return (x + a - 1) // a * a
    for t in dense:
        off = align(off, ALIGN)
        infos.append((t["name"], t["dims"], t["ggml_type"])); offsets.append(off); off += t["nbytes"]
    packed = [(L, pl) for L, pl in enumerate(per_layer) if pl]
    for L, (ts, eb, stride) in packed:
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
        for j, (L, (ts, eb, stride)) in enumerate(packed):
            o = offsets[len(dense) + j]
            for e in range(n_exp):
                slot = data0 + o + e * stride; inner = 0
                for t, b in zip(ts, eb):
                    copy(t["part"], t["abs_offset"] + e * b, b, slot + inner); inner += b
            done += 1
            print(f"\r  experts repacked: layer {done}/{len(packed)}", end="", flush=True)
        out.truncate(data0 + total)
        for f in fds: f.close()
    print()
    os.replace(tmp, out_path)
    return data0 + total


def model_name(meta):
    """What the file calls the model, {"name": "Qwen3-0.6B", "size": "0.6B"} from general.name
    (else basename and size label), {} without either: the banner and the API id of a model
    no colibri variant names (family_registry.display_for, default_model_id)."""
    name = meta.get("general.name")
    if not isinstance(name, str) or not name.strip():
        name = "-".join(str(meta[k]) for k in ("general.basename", "general.size_label") if meta.get(k))
    if not name.strip():
        return {}
    out = {"name": name.strip()}
    if isinstance(meta.get("general.size_label"), str):
        out["size"] = meta["general.size_label"]
    return out


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
    # GGUF_CHECK=1: the file is a header fetched by tools/gguf_header.py, the folder is for the
    # engine's check of it (no tensor is read)
    parts, readers, meta, tensors = open_model(src, header_only=bool(os.environ.get("GGUF_CHECK")))
    arch = meta.get("general.architecture")
    if arch not in ARCHES:
        sys.exit(f"architecture {arch!r} is not supported yet (supported: {', '.join(ARCHES)})")
    if args.mode == "copy" and arch in COPY_UNREAD:
        sys.exit(f"--mode copy: the {arch} engine reads its GGUF in place only; --mode link (lossless, no copy) is the way")
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
    cfg["colibri_gguf"] = {"arch": arch, "mode": args.mode, "source": str(src), **model_name(meta)}
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
