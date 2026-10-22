#!/usr/bin/env python3
"""A tiny Flint (RecursiveSI/flint, model_type "flint") with random weights, in the
checkpoint's own layout, and the greedy continuation of a prompt as a ref.json.

Flint is a Qwen3-MoE decoder with five changes (the model card and the release's
model definition, sglang/flint_sglang/model.py):

1. a learned sink per attention head (self_attn.softmax_offset): one more logit in
   the softmax's denominator, with no value;
2. sliding-window attention on the layers config.layer_types names
   "sliding_attention": a row sees itself and the sliding_window - 1 rows before it;
3. no RoPE on the layers in config.no_rope_layer_indices;
4. a dense SwiGLU MLP (intermediate_size) on config.mlp_only_layers, the routed
   experts everywhere else;
5. sigmoid routing: the top-k of sigmoid(router logits), their sigmoids renormalised
   to sum to one, plus an ungated shared expert (mlp.shared_expert).

transformers ships no Flint model, so the reference here is written from that
definition, in float32: FlintRef below. The fixture has no tokenizer; --vocab 128
keeps every id an ASCII byte for the serve harnesses, as make_qwen36_tiny.py does.

    python3 tools/make_flint_tiny.py --out flint_tiny --emit-ref flint_tiny/ref_full.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError as exc:   # pragma: no cover
    sys.exit(f"Missing deps: {exc}. Run: pip install torch safetensors")


def tiny_config(vocab=128, hidden=64, layers=7, heads=8, kv_heads=2, head_dim=16, experts=16,
                topk=4, moe_inter=32, shared_inter=32, dense_inter=96, window=5):
    # the real pattern at toy depth: two dense full-attention layers, then sliding
    # layers with a full one every fourth, the full ones without RoPE
    types = ["full_attention", "full_attention"]
    for i in range(2, layers):
        types.append("full_attention" if (i - 1) % 4 == 0 or i == layers - 1 else "sliding_attention")
    nope = [i for i, t in enumerate(types) if t == "full_attention"]
    return {
        "architectures": ["FlintForCausalLM"], "model_type": "flint",
        "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 1,
        "decoder_sparse_step": 1, "eos_token_id": [vocab - 1], "head_dim": head_dim,
        "hidden_act": "silu", "hidden_size": hidden, "intermediate_size": dense_inter,
        "layer_types": types, "max_position_embeddings": 4096, "mlp_only_layers": [0, 1],
        "moe_intermediate_size": moe_inter, "norm_topk_prob": True,
        "num_attention_heads": heads, "num_experts": experts, "num_experts_per_tok": topk,
        "num_hidden_layers": layers, "num_key_value_heads": kv_heads,
        "no_rope_layer_indices": nope, "pad_token_id": 0, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "router_score_function": "sigmoid",
        "shared_expert_intermediate_size": shared_inter, "sliding_window": window,
        "softmax_type": "learnable", "tie_word_embeddings": False, "torch_dtype": "float32",
        "use_sliding_window": True, "vocab_size": vocab,
    }


def random_weights(cfg, seed):
    g = torch.Generator().manual_seed(seed)

    def r(*shape, std=0.15):
        return (torch.randn(*shape, generator=g) * std).float()

    def norm(n):   # around 1, so a container that read them zero-centred could not pass
        return (1.0 + 0.2 * torch.randn(n, generator=g)).float()

    D, V, H, KV, hd = cfg["hidden_size"], cfg["vocab_size"], cfg["num_attention_heads"], \
        cfg["num_key_value_heads"], cfg["head_dim"]
    w = {"model.embed_tokens.weight": r(V, D, std=1.0), "model.norm.weight": norm(D),
         "lm_head.weight": r(V, D, std=0.3)}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = norm(D)
        w[p + "post_attention_layernorm.weight"] = norm(D)
        w[p + "self_attn.q_proj.weight"] = r(H * hd, D)
        w[p + "self_attn.k_proj.weight"] = r(KV * hd, D)
        w[p + "self_attn.v_proj.weight"] = r(KV * hd, D)
        w[p + "self_attn.o_proj.weight"] = r(D, H * hd)
        w[p + "self_attn.q_norm.weight"] = norm(hd)
        w[p + "self_attn.k_norm.weight"] = norm(hd)
        w[p + "self_attn.softmax_offset"] = r(H, std=1.0)
        if i in cfg["mlp_only_layers"]:
            F = cfg["intermediate_size"]
            w[p + "mlp.gate_proj.weight"] = r(F, D)
            w[p + "mlp.up_proj.weight"] = r(F, D)
            w[p + "mlp.down_proj.weight"] = r(D, F)
            continue
        E, F, SF = cfg["num_experts"], cfg["moe_intermediate_size"], cfg["shared_expert_intermediate_size"]
        w[p + "mlp.gate.weight"] = r(E, D, std=0.5)
        for e in range(E):
            w[p + f"mlp.experts.{e}.gate_proj.weight"] = r(F, D)
            w[p + f"mlp.experts.{e}.up_proj.weight"] = r(F, D)
            w[p + f"mlp.experts.{e}.down_proj.weight"] = r(D, F)
        w[p + "mlp.shared_expert.gate_proj.weight"] = r(SF, D)
        w[p + "mlp.shared_expert.up_proj.weight"] = r(SF, D)
        w[p + "mlp.shared_expert.down_proj.weight"] = r(D, SF)
    return w


class FlintRef:
    """Flint's forward over a whole sequence, float32, no cache (the fixture is tiny)."""

    def __init__(self, cfg, w):
        self.c, self.w = cfg, w

    def rms(self, x, weight):
        eps = self.c["rms_norm_eps"]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    def rope(self, x, pos):   # x [T, heads, hd], rotate_half over the whole head
        hd = x.shape[-1]
        inv = 1.0 / (self.c["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
        ang = pos[:, None].double() * inv[None, :]
        cos = torch.cat([ang.cos(), ang.cos()], -1).float()[:, None, :]
        sin = torch.cat([ang.sin(), ang.sin()], -1).float()[:, None, :]
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def attention(self, i, h):
        c, w, p = self.c, self.w, f"model.layers.{i}.self_attn."
        T = h.shape[0]
        H, KV, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        q = (h @ w[p + "q_proj.weight"].T).view(T, H, hd)
        k = (h @ w[p + "k_proj.weight"].T).view(T, KV, hd)
        v = (h @ w[p + "v_proj.weight"].T).view(T, KV, hd)
        q, k = self.rms(q, w[p + "q_norm.weight"]), self.rms(k, w[p + "k_norm.weight"])
        if i not in c["no_rope_layer_indices"]:
            pos = torch.arange(T)
            q, k = self.rope(q, pos), self.rope(k, pos)
        sliding = c["layer_types"][i] == "sliding_attention"
        t = torch.arange(T)
        allowed = t[None, :] <= t[:, None]
        if sliding:
            allowed &= t[None, :] > t[:, None] - c["sliding_window"]
        sink = w[p + "softmax_offset"].float()
        out = torch.empty(T, H, hd)
        rep = H // KV
        for hh in range(H):
            s = (q[:, hh, :] @ k[:, hh // rep, :].T) / math.sqrt(hd)
            s = s.masked_fill(~allowed, float("-inf"))
            m = torch.maximum(s.max(-1, keepdim=True).values, sink[hh].expand(T, 1))
            e = torch.exp(s - m)
            den = e.sum(-1, keepdim=True) + torch.exp(sink[hh] - m)
            out[:, hh, :] = (e / den) @ v[:, hh // rep, :]
        return out.reshape(T, H * hd) @ w[p + "o_proj.weight"].T

    def swiglu(self, x, pre):
        w = self.w
        g, u = x @ w[pre + "gate_proj.weight"].T, x @ w[pre + "up_proj.weight"].T
        return (torch.nn.functional.silu(g) * u) @ w[pre + "down_proj.weight"].T

    def mlp(self, i, h):
        c, w, p = self.c, self.w, f"model.layers.{i}.mlp."
        if i in c["mlp_only_layers"]:
            return self.swiglu(h, p)
        scores = torch.sigmoid(h @ w[p + "gate.weight"].T)
        K = c["num_experts_per_tok"]
        top = torch.topk(scores, K, dim=-1).indices
        wts = scores.gather(1, top)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
        out = torch.zeros_like(h)
        for r in range(h.shape[0]):
            for j in range(K):
                e = int(top[r, j])
                out[r] += wts[r, j] * self.swiglu(h[r:r + 1], p + f"experts.{e}.")[0]
        return out + self.swiglu(h, p + "shared_expert.")

    def logits(self, ids):
        w, c = self.w, self.c
        x = w["model.embed_tokens.weight"][torch.tensor(ids)]
        for i in range(c["num_hidden_layers"]):
            x = x + self.attention(i, self.rms(x, w[f"model.layers.{i}.input_layernorm.weight"]))
            x = x + self.mlp(i, self.rms(x, w[f"model.layers.{i}.post_attention_layernorm.weight"]))
        return self.rms(x, w["model.norm.weight"]) @ w["lm_head.weight"].T

    def generate(self, prompt, n):
        """The greedy continuation and the smallest margin between the first and the
        second logit over its steps."""
        ids, gap = list(prompt), float("inf")
        for _ in range(n):
            top = self.logits(ids)[-1].topk(2)
            gap = min(gap, float(top.values[0] - top.values[1]))
            ids.append(int(top.indices[0]))
        return ids, gap


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--emit-ref", default=None, help="write the greedy continuation here")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--prompt-ids", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--vocab", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-gap", type=float, default=0.1,
                    help="draw seeds from --seed on until every greedy step's first logit beats the "
                         "second by this much: the container's int8 experts move the logits by a few "
                         "thousandths, and a near tie would fail the gate for no fault")
    args = ap.parse_args()
    cfg = tiny_config(vocab=args.vocab, hidden=args.hidden, window=args.window)
    prompt = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
    for seed in range(args.seed, args.seed + 200):
        w = random_weights(cfg, seed)
        if not args.emit_ref:
            break
        with torch.no_grad():
            full, gap = FlintRef(cfg, w).generate(prompt, args.max_new)
        if gap >= args.min_gap:
            break
    else:
        sys.exit(f"no seed in {args.seed}..{args.seed + 199} gives a margin of {args.min_gap}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    save_file({k: v.contiguous() for k, v in w.items()}, str(out / "model.safetensors"),
              metadata={"format": "pt"})
    print(f"tiny Flint -> {out} ({sum(v.numel() for v in w.values()) / 1e6:.2f}M parameters)")
    if args.emit_ref:
        Path(args.emit_ref).parent.mkdir(parents=True, exist_ok=True)
        Path(args.emit_ref).write_text(json.dumps(
            {"prompt_ids": prompt, "full_ids": full, "mode": "full", "model": "flint_tiny"}, indent=2))
        print(f"ref.json -> {args.emit_ref} (seed {seed}, smallest margin {gap:.3f})\n"
              f"  prompt_ids={prompt}\n  full_ids ={full}")


if __name__ == "__main__":
    main()
