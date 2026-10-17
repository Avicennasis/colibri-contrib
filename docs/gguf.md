---
title: "GGUF models"
description: "Run a GGUF you already have, read in place: no conversion, no second quantization."
---

# GGUF models

colibri runs the GGUF files people already have on disk. The ggml blocks are computed as the file holds them, from IQ1_S to Q8_0 and MXFP4: nothing is converted or quantized a second time.

```bash
./coli import ~/models/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf   # first part of a split model
./coli chat --model ~/models/GLM-5.3-UD-IQ1_S.colibri         # or coli serve / coli web
```

`coli import` writes a model directory next to the GGUF, named after it, or wherever `--out` says. The directory holds `config.json`, the tokenizer, the chat template and `gguf.json`, which points at the file. From there `coli chat`, `coli serve`, `coli web`, `coli plan` and `coli doctor` treat it like any other model.

## Supported architectures

| GGUF architecture (`general.architecture`) | Models | Engine |
|---|---|---|
| `qwen35moe` | Qwen3.5 / Qwen3.6 MoE | qwen36 |
| `qwen35` | Qwen3.5 / Qwen3.8 dense | qwen36 |
| `qwen3moe` | Qwen3 MoE, Qwen3-Coder | qwen36 |
| `qwen3` | Qwen3 dense | qwen36 |
| `qwen4exp` | Qwen3.8-Flash-Next (and its MTP head) | qwen38 |
| `deepseek4` | DeepSeek V4 Flash | deepseek_v4 |
| `inkling` | Inkling | inkling |
| `glm5-next` | GLM-5.3-Flash | glm53 |
| `glm-dsa` | GLM-5.2, GLM-5.3 | colibri |
| `mimo2` | MiMo-V2.6 | mimo |

A GGUF of another architecture is refused by `coli import` with the list above.

## Two modes

- **`--mode link`** (the default): the engine reads the GGUF where it is. Nothing is copied.
- **`--mode copy`**: also writes `model.colibri.gguf`, with every dense tensor byte for byte and each layer's routed experts re-laid out one expert per 4 KiB-aligned slot (gate, up and down contiguous), so that an expert is one aligned read. It works for every architecture with routed experts but `qwen4exp`, whose engine reads in place only.

Both are lossless: the same blocks, the same tokens.

## What the engines do with the blocks

- **The CPU interleaves rows by 16.** Matrices whose type has the form (Q4_K, Q5_K, Q6_K, Q3_K, IQ4_XS, Q8_0, Q4_0, IQ4_NL) are held with their rows interleaved by 16, which speeds up prompts and speculative verification. Experts are packed when they are read, dense matrices once at load. Each engine's switch keeps the file's rows, as `GQ_EXACT=1` does:

  | Engine | Switch |
  |---|---|
  | qwen36 | `Q36_ILV16=0` |
  | qwen38 | `Q38_ILV16=0` |
  | deepseek_v4 | `COLI_V4_ROWS16=0` |
  | inkling | `INK_ILV16=0` |
  | glm53 | `G53_ILV16=0` |
  | colibri | `GLM_ILV16=0` |

- **The Vulkan expert tier** (`COLI_VULKAN=1`) takes a GGUF's experts as the file has them. Each layer keeps its own types, since published files mix them.
- **MXFP4 experts** (DeepSeek V4, MiMo) are laid out as their checkpoints' FP4, so they run on those engines' own kernels and device forms.
- **GLM's MLA**: `k_b` / `v_b` (llama.cpp splits `kv_b_proj`) stay the file's blocks.
- **Dense matrices stay on the CPU.** The dense chains and CUDA take none of a GGUF's ggml matrices; the routed experts are where a device helps most.

## Checking a GGUF before downloading it

`tools/gguf_header.py` fetches only the header of a model on the Hugging Face Hub (a few MB). With `GGUF_CHECK=1` the importer writes the model directory from that header alone, so you see the config it would get. The qwen36 and deepseek_v4 engines then check the folder against every tensor they would load, without reading one:

```bash
python3 tools/gguf_header.py unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf /tmp/hdr
GGUF_CHECK=1 python3 tools/gguf_import.py /tmp/hdr/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf /tmp/hdr/imp
GGUF_CHECK=1 SNAP=/tmp/hdr/imp ./qwen36
```
