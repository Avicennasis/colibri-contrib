/* st_gguf.h -- a GGUF model as a safetensors index (st.h's shards).
 *
 * An engine that reads its checkpoint through st.h finds every tensor by the name the
 * original checkpoint gives it. For a folder tools/gguf_import.py wrote (gguf.json names the
 * GGUF), st_init_gguf opens the GGUF in place and lists its tensors under those names: the
 * inverse of llama.cpp's converter for the architecture (st_gguf_maps below). A fused expert
 * tensor [experts][rows][cols] becomes one entry per expert, each its contiguous slice of the
 * file, as the checkpoint has them. Shapes are row-major as safetensors gives them (ggml's ne
 * reversed). Dtypes: F32, F16, BF16, I32, I8 keep their st.h codes; a block-quantized tensor
 * (Q8_0, Q4_K, IQ2_XS, MXFP4, ...) gets ST_GGML_BASE + its ggml type, which the float readers
 * refuse and the engine computes with ggml_quants.h. No byte is copied or converted.
 *
 * Opt-in: an engine defines ST_GGUF before including st.h; without it st.h is unchanged.
 * A GGUF tensor without a counterpart in the map keeps its GGUF name. */
#ifndef ST_GGUF_H
#define ST_GGUF_H

#include "gguf.h"

#define ST_GGML_BASE 32
static inline int st_is_ggml(int dtype) { return dtype >= ST_GGML_BASE; }
static inline int st_ggml_type(int dtype) { return dtype - ST_GGML_BASE; }

typedef struct { const char *gg, *hf; } st_gg_name;
typedef struct {
    const char *arch;
    const char *layer_fmt;          /* the checkpoint's layer prefix, "%d" = the layer */
    const st_gg_name *globals;      /* whole names */
    const st_gg_name *layer;        /* after "blk.N." -> after the layer prefix; a "%d" in hf = per expert */
} st_gg_map;

/* DeepSeek V4 (deepseek4): conversion/deepseek.py, DeepseekV4Model._map_dsv4_tensor_name. */
static const st_gg_name st_gg_dsv4_glob[] = {
    {"token_embd.weight", "embed.weight"}, {"output_norm.weight", "norm.weight"},
    {"output.weight", "head.weight"}, {"output_hc_fn.weight", "hc_head_fn"},
    {"output_hc_base.weight", "hc_head_base"}, {"output_hc_scale.weight", "hc_head_scale"},
    {NULL, NULL}};
static const st_gg_name st_gg_dsv4_layer[] = {
    {"hc_attn_fn.weight", "hc_attn_fn"}, {"hc_attn_base.weight", "hc_attn_base"},
    {"hc_attn_scale.weight", "hc_attn_scale"}, {"hc_ffn_fn.weight", "hc_ffn_fn"},
    {"hc_ffn_base.weight", "hc_ffn_base"}, {"hc_ffn_scale.weight", "hc_ffn_scale"},
    {"attn_sinks.weight", "attn.attn_sink"},
    {"attn_q_a.weight", "attn.wq_a.weight"}, {"attn_q_b.weight", "attn.wq_b.weight"},
    {"attn_q_a_norm.weight", "attn.q_norm.weight"},
    {"attn_kv.weight", "attn.wkv.weight"}, {"attn_kv_a_norm.weight", "attn.kv_norm.weight"},
    {"attn_output_a.weight", "attn.wo_a.weight"}, {"attn_output_b.weight", "attn.wo_b.weight"},
    {"attn_compressor_ape.weight", "attn.compressor.ape"},
    {"attn_compressor_kv.weight", "attn.compressor.wkv.weight"},
    {"attn_compressor_gate.weight", "attn.compressor.wgate.weight"},
    {"attn_compressor_norm.weight", "attn.compressor.norm.weight"},
    {"indexer.attn_q_b.weight", "attn.indexer.wq_b.weight"},
    {"indexer.proj.weight", "attn.indexer.weights_proj.weight"},
    {"indexer_compressor_ape.weight", "attn.indexer.compressor.ape"},
    {"indexer_compressor_kv.weight", "attn.indexer.compressor.wkv.weight"},
    {"indexer_compressor_gate.weight", "attn.indexer.compressor.wgate.weight"},
    {"indexer_compressor_norm.weight", "attn.indexer.compressor.norm.weight"},
    {"attn_norm.weight", "attn_norm.weight"}, {"ffn_norm.weight", "ffn_norm.weight"},
    {"ffn_gate_inp.weight", "ffn.gate.weight"}, {"exp_probs_b.bias", "ffn.gate.bias"},
    {"ffn_gate_tid2eid.weight", "ffn.gate.tid2eid"},
    {"ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight"},
    {"ffn_down_shexp.weight", "ffn.shared_experts.w2.weight"},
    {"ffn_up_shexp.weight", "ffn.shared_experts.w3.weight"},
    {"ffn_gate_exps.weight", "ffn.experts.%d.w1.weight"},
    {"ffn_down_exps.weight", "ffn.experts.%d.w2.weight"},
    {"ffn_up_exps.weight", "ffn.experts.%d.w3.weight"},
    {NULL, NULL}};

static const st_gg_map st_gguf_maps[] = {
    {"deepseek4", "layers.%d.", st_gg_dsv4_glob, st_gg_dsv4_layer},
};

static const st_gg_map *st_gguf_map(const char *arch) {
    for (size_t i = 0; arch && i < sizeof st_gguf_maps / sizeof *st_gguf_maps; i++)
        if (!strcmp(st_gguf_maps[i].arch, arch)) return &st_gguf_maps[i];
    return NULL;
}

/* The GGUF metadata of the model st_init_gguf opened (architecture, tokenizer, ...): the
 * engine reads its config from here or from the config.json the importer wrote. */
static gguf_ctx st_gguf_meta;
static int st_gguf_active;

static int st_gguf_dtype(int type) {
    switch (type) {
        case GQ_F32: return 2;
        case GQ_F16: return 1;
        case GQ_BF16: return 0;
        default: break;
    }
    if (type == 26) return 7;   /* I32 (the hash router's table) */
    if (type == 24) return 3;   /* I8 */
    if (type == 27) return 6;   /* I64 */
    return ST_GGML_BASE + type;
}

static st_tensor *st_gguf_add(shards *S, const char *name) {
    if (S->n == S->cap) {
        S->cap *= 2;
        st_tensor *t = (st_tensor *)realloc(S->t, (size_t)S->cap * sizeof(st_tensor));
        if (!t) coli_load_fail(COLI_LOAD_FAIL_NOMEM, "out of memory indexing the GGUF");
        S->t = t;
    }
    st_tensor *t = &S->t[S->n++];
    memset(t, 0, sizeof *t);
    t->name = strdup(name);
    return t;
}

/* 1 when snap_dir is a gguf_import folder and S now indexes its GGUF, 0 when it is not one.
 * A broken GGUF or one of an architecture without a map is fatal (the folder says GGUF). */
static int st_init_gguf(shards *S, const char *snap_dir) {
    char gj[4096]; snprintf(gj, sizeof gj, "%s/gguf.json", snap_dir);
    FILE *f = fopen(gj, "rb");
    if (!f) return 0;
    char buf[16384]; size_t nb = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[nb] = 0;
    char *arena = NULL; jval *root = json_parse(buf, &arena);
    jval *gv = root ? json_get(root, "gguf") : NULL;
    const char *g = gv && gv->t == J_STR ? gv->str : NULL;
    if (!g || !*g) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "%s: no \"gguf\" path", gj);
    char path[4096];
    if (g[0] == '/') snprintf(path, sizeof path, "%s", g); else snprintf(path, sizeof path, "%s/%s", snap_dir, g);
    free(arena);
    if (getenv("GGUF_CHECK")) gguf_header_only = 1;
    gguf_ctx *G = &st_gguf_meta;
    if (gguf_open(G, path)) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF: %s", G->err);
    const char *arch = gguf_get_str(G, "general.architecture");
    const st_gg_map *map = st_gguf_map(arch);
    if (!map) coli_load_fail(COLI_LOAD_FAIL_UNSUPPORTED, "%s: GGUF architecture '%s' has no tensor map in st_gguf.h", path, arch ? arch : "?");
    int fds[ST_MAX_SHARDS];
    if (G->n_files > ST_MAX_SHARDS) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "%s: %d parts", path, G->n_files);
    for (int p = 0; p < G->n_files; p++) fds[p] = st_open_fd(S, G->paths[p]);
    S->cap = G->n_tensors + 16; S->t = (st_tensor *)calloc((size_t)S->cap, sizeof(st_tensor));
    if (!S->t) coli_load_fail(COLI_LOAD_FAIL_NOMEM, "out of memory indexing the GGUF");
    int experts = 0, renamed = 0, kept = 0;
    char nm[512];
    for (int i = 0; i < G->n_tensors; i++) {
        const gguf_tensor *t = &G->t[i];
        const char *hf = NULL; int L = -1, off = 0;
        for (const st_gg_name *e = map->globals; e->gg && !hf; e++) if (!strcmp(t->name, e->gg)) hf = e->hf;
        if (!hf && sscanf(t->name, "blk.%d.%n", &L, &off) == 1 && L >= 0 && off)
            for (const st_gg_name *e = map->layer; e->gg && !hf; e++) if (!strcmp(t->name + off, e->gg)) hf = e->hf;
        const int per_expert = hf && strstr(hf, "%d") != NULL;
        const int dims = per_expert ? 2 : t->n_dims;
        const int E = per_expert ? (int)t->ne[2] : 1;
        if (per_expert && (t->n_dims != 3 || t->nbytes % E)) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF %s: not [experts][rows][cols]", t->name);
        const int64_t slice = t->nbytes / E;
        for (int x = 0; x < E; x++) {
            if (!hf) snprintf(nm, sizeof nm, "%s", t->name);
            else {
                char rest[256];
                if (per_expert) snprintf(rest, sizeof rest, hf, x); else snprintf(rest, sizeof rest, "%s", hf);
                if (L >= 0) { char pre[64]; snprintf(pre, sizeof pre, map->layer_fmt, L); snprintf(nm, sizeof nm, "%s%s", pre, rest); }
                else snprintf(nm, sizeof nm, "%s", rest);
            }
            st_tensor *s = st_gguf_add(S, nm);
            s->fd = fds[t->file]; s->off = (int64_t)t->off + slice * x; s->nbytes = slice;
            s->dtype = st_gguf_dtype(t->type);
            s->rank = dims; s->numel = 1;
            for (int d = 0; d < dims; d++) { s->shape[d] = t->ne[dims - 1 - d]; s->numel *= s->shape[d]; }
        }
        if (per_expert) experts++; else if (hf) renamed++; else kept++;
    }
    S->hcap = 1; while (S->hcap < S->n * 2) S->hcap <<= 1;
    S->hidx = (int *)malloc((size_t)S->hcap * sizeof(int));
    if (!S->hidx) coli_load_fail(COLI_LOAD_FAIL_NOMEM, "out of memory indexing the GGUF");
    for (int i = 0; i < S->hcap; i++) S->hidx[i] = -1;
    for (int i = 0; i < S->n; i++) {
        if (st_find(S, S->t[i].name)) coli_load_fail(COLI_LOAD_FAIL_FORMAT, "GGUF: two tensors map to %s", S->t[i].name);
        uint64_t h = st_hash(S->t[i].name) & (uint64_t)(S->hcap - 1);
        while (S->hidx[h] >= 0) h = (h + 1) & (uint64_t)(S->hcap - 1);
        S->hidx[h] = i;
    }
    st_gguf_active = 1;
    fprintf(stderr, "[gguf] %s: %s, %d part(s), %d tensors under the checkpoint's names (%d renamed, %d fused expert "
                    "tensors split per expert, %d kept as named), read in place\n",
            path, arch, G->n_files, S->n, renamed, experts, kept);
    return 1;
}

#endif /* ST_GGUF_H */
