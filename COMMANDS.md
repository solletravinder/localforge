# LocalForge — desktop app and commands

**Installed path:** run `LocalForge-Setup-*.exe` from the latest release. It bundles Python,
dependencies, `llama-server.exe`, and the Qwen3 0.6B sanity model. No terminal or Python installation
is needed.

**Project path:** double-click `LocalForge.exe` or the LocalForge shortcut on the desktop.
Use the window to start/stop the server, open Jan, open the API docs, and watch logs and RAM.
**Models** lists, downloads, picks, registers, and removes models; downloads show progress and can
be cancelled. **Settings** edits generation defaults (max tokens, temperature, top-p, thinking),
llama.cpp runtime arguments (context, GPU layers, batch/ubatch, threads, parallel slots, flash
attention, warmup), and K/V cache types. Settings are saved on **Save** and apply after Stop and
Start. The model is unloaded when the window closes.

Keep `LocalForge.exe` inside the `local-forge` project folder; it uses the existing `.venv` and
models. Jan connects to `http://127.0.0.1:8090/v1` with model `qwen3.5-4b-abliterated` and
context `131000`.

The commands below remain available for advanced work. Run them from this folder.

**Ready now:** uncensored Gemma 4 E2B chat/vision, Qwen3.5-4B chat/vision, and SD1.5 image
generation. **Remaining:** LFM and the better photoreal image model install with `run.bat setup`.

## One-time setup

```bat
run.bat setup
```

Downloads the two remaining text models and the photoreal image model (~9.5 GB total). It resumes
if interrupted. If bandwidth is limited, use `run.bat setup --text-only` or
`run.bat setup --image-only`.

Check everything:

```bat
run.bat doctor
run.bat models list
```

## Chat

Streaming is the default. Gemma 4 E2B uncensored is ready now.

```bat
run.bat chat
run.bat chat --model qwen3.5-4b-abliterated --no-thinking
run.bat chat --model lfm2.5-2.6b-abliterated
run.bat chat --image "C:\path\to\photo.jpg"
```

In chat: `/help`, `/model <id>`, `/image <path>`, `/system <text>`, `/reset`, `/quit`.

## Local API

```bat
run.bat serve
```

OpenAI base URL: `http://127.0.0.1:8090/v1`
Docs: `http://127.0.0.1:8090/docs`

Streaming request:

```bash
curl -N http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-e2b-uncensored-ov","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

For Qwen without thinking, add `"chat_template_kwargs":{"enable_thinking":false}`.

## Jan AI tools / web search

```bat
run.bat serve --preload
```

- Base URL: `http://127.0.0.1:8090/v1`
- API key: `local` (any non-empty string)
- Model for 131k context and tools: `qwen3.5-4b-abliterated`; set Jan's context to `131000`.
- Gemma 4 E2B remains the short-context CPU fallback.
- Enable web search/tools in Jan.

Restart the server after updates. Tool calls now come back as proper OpenAI `tool_calls`, and tool
results are fed back for the final answer. Gemma's arguments are normalized to Jan's Exa schema
(`query`, not `queries`). If Jan still shows raw `call:web_search{...}` or `<|tool|>` text, the
server is an old process on port 8090 or Jan is not sending `tools`.

Gemma now runs on **CPU** by default. That is intentional: on 16 GB, the GPU-shared-memory model
plus Jan/Qoder can OOM. CPU measured 18.7 tok/s decode here. Image generation also uses CPU now,
so it is slower; stop the chat server before attempting a GPU image run.

## Qwen CPU / iGPU split

Current setting is **70/30**: 10 of 32 layers on the iGPU, ubatch 1024, one server slot. Measured
with a 2048-token prompt:

```text
CPU only:            51.0 tok/s prefill, 8.84 tok/s decode
70/30 (10 layers):   57.3 tok/s prefill, 7.13 tok/s decode
Full iGPU (99):      72.1 tok/s prefill, 7.34 tok/s decode
```

The 70/30 split is the usable middle when full offload will not fit. It gives +12% prefill over
CPU and keeps decode close to full-GPU. Full iGPU is faster at prefill but needs ~4.4 GB shared GPU
memory; use it only if you can spare it.

Tune the number of iGPU layers with `llama_ngl` (10 ≈ 30%):

```bat
run.bat config set llama_ngl 10
```

Restart the server after changing it. `0` is CPU-only, `99` is full iGPU.

### 131k context

The server is configured for **131,072 tokens** (your 131,000-token Jan limit plus headroom):

```bat
run.bat config set llama_ctx 131072
run.bat config set llama_ubatch 1024
run.bat config set llama_parallel 1
run.bat config set llama_flash_attn on
run.bat config set llama_cache_k q8_0
run.bat config set llama_cache_v q4_0
```

Qwen3.5-4B has 32 layers, but only every fourth layer is full attention, so the KV cache for
131k is about 2 GB with Q8 K / Q4 V rather than tens of GB. This configuration loads in ~5 s,
leaves ~3 GB of physical RAM free on this laptop, and answered a normal request successfully.
It is verified, not theoretical.

The catch is cold prompt processing, not memory. Measured on this machine:

```text
 4,218-token prompt, 131k server:  46.3 tok/s prefill (91 s, cold)
 ~8,400-token prompt, partly cached: 34.9 tok/s prefill
```

A completely cold 131k prompt is therefore **tens of minutes at best and realistically 1–2 hours**.
That is the 4B-on-UHD limit; no llama.cpp flag removes it. The server reuses the cached prompt
prefix, so follow-up turns only process new tokens and are fast. Do not restart the server in the
middle of a long thread unless you can wait for the cold pass again.

If RAM is tighter than expected, use Q8 K / Q4 V as above, keep `llama_parallel` at 1, and close
Jan/Qoder before the first large prompt. If quality matters more than headroom, set
`llama_cache_v q8_0` (about 0.6 GB more RAM).

## Images

An SD1.5 image model is configured and working now. `run.bat setup` will replace it with the
better photoreal Realistic Vision V6.0 B1 model when the download completes. Defaults are
512×768, 28 steps, guidance 6.5, plus a built-in photoreal negative prompt.

```bat
run.bat image "editorial portrait of an adult, soft window light, 85mm lens, natural skin texture"
```

Tune only if needed:

```bat
run.bat image "your prompt" --width 512 --height 768 --steps 30 --guidance 6
```

For SDXL or another checkpoint:

```bat
run.bat image "your prompt" --model-path "D:\path\to\openvino-model" --width 832 --height 1216
```

## Download / remove models

```bat
run.bat setup --text-only
run.bat setup --image-only
run.bat models download qwen3.5-4b-abliterated
run.bat models download lfm2.5-2.6b-abliterated
run.bat models remove <id>
```

## Stop

Close the API window, or press `Ctrl+C` in the chat.
