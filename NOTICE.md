# NOTICE

This repository is a **private deployment fork** of
[JustVugg/colibri](https://github.com/JustVugg/colibri) (Apache License 2.0 —
see `LICENSE`, which is upstream's and unmodified).

- Upstream base: commit `bad64d1b06cbda80dbdfb1f8f502370d0407bcbb`
- Fork branch: `avic-deploy` — carries the Bahushruth split-shard converter
  patches (see `CHANGELOG.md` and `c/tools/colibri-bahushruth.patch`)
- Purpose: runs GLM-5.2-abliterated (int4, CPU-only) on dev — runbooks at
  wiki.simmons.systems `services/ai-automation/colibri` /
  `operations/colibri-install`
- Community files (`CONTRIBUTING.md`, `.github/ISSUE_TEMPLATE/`,
  `pull_request_template.md`) are upstream's and are intentionally preserved.
- Never push to the upstream remote from this fork.
