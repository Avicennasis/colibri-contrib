# Changelog

All notable changes to this deployment fork of `colibri` will be documented in
this file. Upstream's own history lives at
[JustVugg/colibri](https://github.com/JustVugg/colibri); this changelog covers
only the `avic-deploy` branch's divergence from the pinned upstream base.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Bahushruth split-shard converter support (`e7d8535`, base `bad64d1`):
  `_hf_headers()` gated-HF auth, cross-shard `_scale_inv` resolution, and
  `--mtp --indir` / `--indexer --indir` local-extraction branches in
  `c/tools/convert_fp8_to_int4.py`; combined patch at
  `c/tools/colibri-bahushruth.patch` + synthetic cross-shard unit test.
- Repo plumbing: `.gitignore` `.stargazer-*` exclusion, git-standards core kit
  (2026-07-22).
