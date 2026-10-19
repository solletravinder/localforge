# Changelog

## 0.3.0 — 2026-10-19

- Added a desktop Models manager with installed/missing status, download progress, cancellation,
  model picking, local model registration, and removal/unregistration.
- The main model picker now reflects the real catalog and local registrations instead of a fixed
  two-model list.
- Added `lforge models list --json` for reliable desktop integration.
- Model switching and file removal are blocked while the server is running; downloads and local
  registration remain available.
- Progress tracking and automatic catalog refresh after download, registration, or removal.

## 0.2.0 — 2026-10-19

- Expanded the desktop Settings dialog into three tabs: Generation, llama.cpp, and KV cache.
- Added editable max output tokens, temperature, top-p, thinking mode, batch, ubatch, CPU
  threads, parallel slots, flash attention, warmup, and K/V cache types.
- Added validation for argument ranges, ubatch/batch ordering, and quantized KV without flash
  attention.
- Added a matching windowed `LocalForge.exe` release asset.
- Version reporting now comes from a single package version.

## 0.1.0 — 2026-10-19

- Initial public release.
- Qwen3.5-4B abliterated at 131k context, 70/30 CPU/iGPU split, Q8 K / Q4 V cache.
- Gemma 4 E2B uncensored CPU fallback and SD1.5 image generation.
- OpenAI-compatible streaming API, native tool calls, Jan integration, LiteRT adapter.
- Windowed LocalForge desktop control panel.
