# pi-vision

Universal image understanding for **text-only models** in [pi](https://pi.dev). When your main model can't see images (DeepSeek, Kimi, local GGUF), pi-vision describes them with a lightweight local vision model so you never miss a screenshot, diagram, or photo.

## How it works

- Watches every provider request
- If the active model supports images → does nothing (passthrough)
- If it's text-only → sends images to a local vision model (MLX or Ollama), replaces them with text descriptions
- Descriptions are **cached by image hash** so re-sending history never re-captions

## Quick start

```bash
# 1. Install the extension
pi install git:github.com/dvictor357/vision-pi

# 2. Run setup for guided configuration
/vision setup

# 3. Choose and start a backend:
#    Option A — MLX (Apple Silicon, fast)
pip install mlx-vlm
mlx_vlm.server --model mlx-community/Qwen2-VL-2B-Instruct-4bit --port 8081

#    Option B — Ollama (cross-platform)
ollama pull moondream

# 4. Test connectivity
/vision test

# 5. Enable if not already on
/vision on
```

## Requirements

- **pi** `>=0.79`
- One of the backends below (or any OpenAI-compatible vision server)

## Backends

| Backend | Best for | Setup command |
|---------|----------|---------------|
| **MLX** (Apple Silicon) | Fast inference on M-series | `pip install mlx-vlm` then `mlx_vlm.server --model <model> --port 8081` |
| **Ollama** | Cross-platform, CPU-friendly | `ollama pull <model>` |
| **Custom** | Any OpenAI-compatible server | `register` via `/vision backend register` |

### Presets

Three quality tiers (bigger = better but slower):

| Preset | MLX Model | Ollama Model | Use for |
|--------|-----------|--------------|---------|
| `light` (default) | Qwen2-VL-2B-Instruct-4bit | moondream | Fast, basic images |
| `balanced` | Qwen2.5-VL-3B-Instruct-4bit | qwen2.5vl:3b | OCR, UI screenshots |
| `capable` | Qwen2.5-VL-7B-Instruct-4bit | qwen2.5vl:7b | Complex diagrams, charts |

Switch backends: `/vision backend ollama` or `/vision backend mlx`  
Switch presets: `/vision preset balanced`

## Configuration

### Settings file

Config lives in `~/.pi/agent/settings.json` under a `"vision"` key. pi preserves unknown top-level keys, so this block survives pi's own writes.

```json
{
  "vision": {
    "backend": "ollama",
    "preset": "balanced",
    "maxEdge": 1024,
    "timeoutMs": 120000,
    "maxImagesPerTurn": 6,
    "captionConcurrency": 2,
    "cacheMaxEntries": 500,
    "negativeCacheTTLMs": 300000
  }
}
```

### Environment variable overrides

Variables take highest priority — useful for CI, testing, or temporary overrides.

| Variable | Effect |
|----------|--------|
| `PI_VISION_BACKEND` | Backend name (`mlx`, `ollama`, or custom) |
| `PI_VISION_MODEL` | Pin a specific model id |
| `PI_VISION_BASE_URL` | Custom server URL |
| `PI_VISION_API_KEY` | API key for remote endpoints |
| `PI_VISION_CACHE_PASSPHRASE` | Passphrase for AES-256-GCM encrypted disk cache |
| `PI_VISION_DISABLED=1` | Turn off entirely |
| `PI_VISION_SETTINGS_PATH` | Custom settings.json path (testing) |
| `PI_VISION_CACHE_PATH` | Custom cache file path (testing) |

## Commands

| Command | Does |
|---------|------|
| `/vision` or `/vision status` | Show status, cache stats, connection diagnostics |
| `/vision on` / `off` | Enable / disable |
| `/vision backend <name>` | Switch to a built-in or custom backend |
| `/vision backend register <name> <url> [apiKey]` | Register a custom backend |
| `/vision backend unregister <name>` | Remove a custom backend |
| `/vision backend list` | List all registered backends |
| `/vision preset <light\|balanced\|capable>` | Switch quality tier |
| `/vision model <id>` | Pin a specific model |
| `/vision model-ref <backend\|provider/model>` | Use pi model registry for captioner resolution |
| `/vision test` | Send a test image to verify the backend |
| `/vision doctor` | Full diagnostics: config, connection, cache, session health |
| `/vision clear` | Clear in-memory cache and reset disk cache |
| `/vision purge` | Securely wipe the disk cache (overwrite then delete) |
| `/vision setup` | Print setup instructions (first-run shows guided quick-start) |

## Extensibility

### Custom backends

Register any OpenAI-compatible vision server as a backend:

```bash
/vision backend register my-vm http://my-server:8080/v1 my-api-key
```

This persists the backend in settings.json. It is available immediately for `/vision backend my-vm` and listed in `/vision backend list`.

To remove a custom backend:

```bash
/vision backend unregister my-vm
```

### Using pi model registry as captioner

Instead of the built-in backends, you can use any model in pi's ModelRegistry as the captioner — for example, an OpenAI or Anthropic multimodal model. This resolves API keys and headers through pi's own auth system.

```bash
/vision model-ref openai/gpt-4o
```

Set it back to the built-in backend system with:

```bash
/vision model-ref backend
```

### Follow-up vision analysis

After pi-vision describes an image, you can ask the LLM to inspect it further via the `vision_analyze_image` tool. The tool automatically appears in the tool list when the extension is loaded. It requires the `ref` from the `[image-description ref="..."]` tag.

## Security & Privacy

### API Keys & Secrets
- API keys are **never leaked** in notifications, status output, or extension events.
- All URLs in logs/notifications are sanitized (credentials stripped).
- Config sanitization removes `apiKey` and `cachePassphrase` before event emission.

### Encrypted Cache
When `PI_VISION_CACHE_PASSPHRASE` is set, the on-disk caption cache is encrypted with **AES-256-GCM**. The key is derived from the passphrase via PBKDF2 (100,000 iterations, SHA-512). The cache file format becomes:

```json
{"encrypted": true, "data": "<base64 of salt+iv+authTag+ciphertext>"}
```

Without the correct passphrase, the cache is unreadable.

### Cache Privacy
- **`/vision clear`** — Clears the in-memory cache and resets the disk cache to empty (preserving encryption if enabled).
- **`/vision purge`** — Securely overwrites the cache file on disk before deleting it, then clears all in-memory state. Use this when you want to ensure no cached image data remains.
- Caching can be disabled entirely by setting `"cache": false` in settings.

### Image Data & EXIF
- Images are downscaled before captioning, which **implicitly strips EXIF/metadata** as a side effect of re-encoding.
- An explicit `stripImageMetadata()` utility is available for JPEG/TIFF sources.
- **Image data is never uploaded to external servers** other than the configured backend (local or remote).

### Remote Endpoints
- pi-vision **warns** if the backend uses plain HTTP and is not localhost.
- Consider using HTTPS for remote backends to protect image data in transit.
- The `/vision doctor` command checks endpoint security.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `/vision test` fails with connection refused | Backend server not running | Start the server (see Quick start) |
| `(vision model returned an empty description)` | Model responded but with no text | Try a different preset or model |
| `Image previously failed; negative cache hit` | Previous caption attempt failed and is in cooldown | Wait 5 minutes or run `/vision clear` |
| `Maximum image-per-turn budget reached` | Too many images in one message | Increase `maxImagesPerTurn` or reduce image count |
| Remote HTTP warning | Backend URL uses plain HTTP to a remote host | Use HTTPS or switch to localhost |
| Captions are low quality | Default preset uses smallest model | Switch to `/vision preset balanced` or `capable` |
| Cache not persisting between sessions | Cache file is corrupted or passphrase changed | Run `/vision clear` to reset |

## Development

### Setup

```bash
git clone https://github.com/dvictor357/vision-pi.git
cd vision-pi
npm install
```

### Scripts

| Command | Purpose |
|---------|---------|
| `npm test` | Run all tests (node:test + tsx) |
| `npm run typecheck` | Type-check with `tsc --noEmit` |
| `npm run pack` | Dry-run `npm pack` to verify package contents |

### Project structure

```
extensions/vision.ts     — pi extension entrypoint
src/
  orchestrator.ts        — event wiring, context rewriting, default export
  config.ts              — config load/save, env override chain
  backends.ts            — backend definitions, VisionBackendRegistry
  cache.ts               — LRU cache, negative cache, in-flight dedup, encryption
  vision-model.ts        — model descriptor, captionImage, concurrency helper
  commands.ts            — /vision command handler
  follow-up-tool.ts      — vision_analyze_image tool
  image-store.ts         — bounded per-session image store
  privacy.ts             — sanitization, localhost check, EXIF stripping
test/
  vision.test.ts         — all tests
```

### Testing conventions

- All modules are tested independently with mock pi APIs.
- Use `describe`/`it` from `node:test`.
- Config I/O tests use `PI_VISION_SETTINGS_PATH` and `PI_VISION_CACHE_PATH` env vars to avoid clobbering real settings.
- Concurrency and LRU eviction behavior are tested with perf-style benchmarks.

## License

MIT
