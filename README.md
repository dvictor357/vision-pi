# pi-vision

Universal image understanding for **text-only models** in [pi](https://pi.dev). When your main model can't see images (DeepSeek, Kimi, local GGUF), pi-vision describes them with a lightweight local vision model so you never miss a screenshot, diagram, or photo.

## How it works

- Watches every provider request
- If the active model supports images → does nothing (passthrough)
- If it's text-only → sends images to a local vision model (MLX or Ollama), replaces them with text descriptions
- Descriptions are **cached by image hash** so re-sending history never re-captions

## Install

```bash
pi install git:github.com/YOUR_USERNAME/vision-pi
```

After install, enable and configure:

```bash
/vision setup
/vision test
```

## Backends

| Backend | Best for | Setup |
|---------|----------|-------|
| **MLX** (default) | Apple Silicon | `pip install mlx-vlm` → `mlx_vlm.server --model <model> --port 8081` |
| **Ollama** | Cross-platform, CPU-friendly | `ollama pull <model>` |

Switch backends: `/vision backend ollama` or `/vision backend mlx`

## Presets

Three quality tiers (bigger = better but slower):

| Preset | MLX Model | Ollama Model | Use for |
|--------|-----------|--------------|---------|
| `light` (default) | Qwen2-VL-2B-Instruct-4bit | moondream | Fast, basic images |
| `balanced` | Qwen2.5-VL-3B-Instruct-4bit | qwen2.5vl:3b | OCR, UI screenshots |
| `capable` | Qwen2.5-VL-7B-Instruct-4bit | qwen2.5vl:7b | Complex diagrams, charts |

Switch presets: `/vision preset balanced`

## Commands

| Command | Does |
|---------|------|
| `/vision` | Show current status |
| `/vision on` / `off` | Enable / disable |
| `/vision backend <mlx\|ollama>` | Switch vision backend |
| `/vision preset <light\|balanced\|capable>` | Switch quality tier |
| `/vision model <id>` | Pin a specific model |
| `/vision test` | Send a test image to verify the backend |
| `/vision clear` | Clear the caption cache |
| `/vision setup` | Print setup instructions |

## Configuration

Config lives in `~/.pi/agent/settings.json` under a `"vision"` key. Example:

```json
{
  "vision": {
    "backend": "ollama",
    "preset": "balanced",
    "maxEdge": 1024,
    "timeoutMs": 120000
  }
}
```

Environment variable overrides (highest priority):

- `PI_VISION_BACKEND` — `mlx` or `ollama`
- `PI_VISION_MODEL` — any model id
- `PI_VISION_BASE_URL` — custom server URL
- `PI_VISION_API_KEY` — API key (if needed)
- `PI_VISION_DISABLED=1` — turn off entirely

## Requirements

- **pi** `>=0.79`
- One of:
  - **mlx-vlm** (`pip install mlx-vlm`) — macOS with Apple Silicon
  - **Ollama** (`brew install ollama` or https://ollama.com) — any platform

## License

MIT
