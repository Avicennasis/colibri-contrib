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
| `deepseek41` | DeepSeek V4.1 Flash (antirez's format) | deepseek_v41 |
| `inkling` | Inkling | inkling |
| `glm5-next` | GLM-5.3-Flash | glm53 |
| `glm-dsa` | GLM-5.2, GLM-5.3 | colibri |
| `mimo2` | MiMo-V2.6 | mimo |
| `olmoe` | OLMoE | olmoe |

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
  | deepseek_v41 | `V41_ILV16=0` |
  | inkling | `INK_ILV16=0` |
  | glm53 | `G53_ILV16=0` |
  | colibri | `GLM_ILV16=0` |
  | olmoe | `OLM_ILV16=0` (dense matrices; its experts keep the file's rows) |

- **The Vulkan expert tier** (`COLI_VULKAN=1`) takes a GGUF's experts as the file has them. Each layer keeps its own types, since published files mix them.
- **MXFP4 experts** (DeepSeek V4 and V4.1, MiMo) are laid out as their checkpoints' FP4, so they run on those engines' own kernels and device forms.
- **OLMoE's experts keep the file's rows**: they fit in RAM, and interleaving one at its read cost more than the 16-row kernels gave back on its matrices. llama.cpp's `olmoe` files carry no `norm_topk_prob`; the import writes false, as the official checkpoints have it and as llama.cpp computes them.
- **DeepSeek V4.1's engram tables** are read a row at a time, as from the checkpoint: a `deepseek41` file holds each row's e4m3 values and their scales together. These files carry no DSpark head, so nothing drafts.
- **GLM's MLA**: `k_b` / `v_b` (llama.cpp splits `kv_b_proj`) stay the file's blocks.
- **The Vulkan dense chain** (qwen36, qwen38) takes a GGUF's dense matrices as the file has them too. Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_S, IQ3_S, IQ4_NL and IQ4_XS run on a GEMV with one decoder per type (`shaders/chain_gemv_bd.comp`), and the other types on the general block decoder. The routed experts run inside the same frame when the tier holds every expert of a layer. A decode step or a verify of up to 16 rows runs there; a prompt's longer chunks go through the CPU and the expert tier.
- **The CUDA backend takes a GGUF's blocks too** (`cuda_bd.cuh`): every ggml block type of 32 weights or more, each row as the file has it. A decode step and a speculative verify (up to 8 rows, sharing each weight load, each row with a decode step's bits) run in the CPU's integer form on Q8_0, Q2_K to Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and Q2_0 (the CPU's codes, scales and integer sums: the two differ only in the order of the float sums), the other types in floats. A prompt chunk goes through the f16 tensor-core tiles (`COLI_CUDA_BD_TILES`).
- **qwen36's CUDA chain runs a GGUF whole on the card**: its experts in the CUDA tier as the file has them (gate/up and down types by layer, one arena a device: `QT_GGUF_ARENA`), computed in the chain's frame (without the chain, through the tier's expert groups, the misses on the CPU), and its dense matrices, lm_head and DeltaNet projections as their blocks. Qwen3.6-35B-A3B on an RTX PRO 4000 (24 GB), every expert on the card:

  | | CUDA chain | Vulkan chain |
  |---|---|---|
  | UD-IQ2_XXS: decode, prefill of 467 tokens | 118 tok/s, 1356 tok/s | 100 tok/s, 1600 tok/s |
  | UD-IQ2_XXS: a code edit of 400 tokens (prompt lookup) | 2.38 s | 3.3 s |
  | UD-Q4_K_M: decode, prefill | 96 tok/s, 1388 tok/s | 75 tok/s, 1450 tok/s |
  | UD-Q4_K_M: the code edit | 2.67 s | 3.1-3.7 s |

  The greedy text is the CPU's (the same integers). Before, a CUDA run of a GGUF computed every routed expert on the CPU (UD-IQ2_XXS: 13.7 tok/s of decode).
- **The other engines' dense matrices stay on the CPU** on CUDA, and their CUDA tiers take no ggml block types yet.
- **A GGUF those engines' CUDA path does not take runs on Vulkan on an NVIDIA card** (qwen38, colibri for GLM-5.2, whose CUDA path keeps a GGUF's matrices and experts on the CPU, deepseek_v4, whose CUDA tier mirrors native FP4 experts only, and qwen36 for a file whose expert types the CUDA chain's frame lacks). A CUDA run of such a GGUF computed every routed expert on the CPU (Qwen3.8 IQ2_S/Q2_0 on an RTX PRO 4000: 28 tok/s of prompt, 9 of decode). A build with both backends switches such a run to Vulkan on the same card when CUDA was asked for and `COLI_VULKAN` was not set (114-118 and 46-49 tok/s); `COLI_VULKAN=0` keeps CUDA. `coli setup` builds these engines' CUDA engine with Vulkan too where the Vulkan toolchain is there (on Linux; the Windows CUDA package has both).

## Checking a GGUF before downloading it

`tools/gguf_header.py` fetches only the header of a model on the Hugging Face Hub (a few MB). With `GGUF_CHECK=1` the importer writes the model directory from that header alone, so you see the config it would get. The qwen36, deepseek_v4 and deepseek_v41 engines then check the folder against every tensor they would load, without reading one:

```bash
python3 tools/gguf_header.py unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf /tmp/hdr
GGUF_CHECK=1 python3 tools/gguf_import.py /tmp/hdr/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf /tmp/hdr/imp
GGUF_CHECK=1 SNAP=/tmp/hdr/imp ./qwen36
```
