# Changelog

## 1.0.0 (2026-06-30)

- Initial release
- MLX and Ollama backends for local image captioning
- Smart caching with LRU eviction, negative cache with TTL, in-flight dedup
- Encrypted disk cache (AES-256-GCM via PBKDF2)
- Backend extensibility with VisionBackendRegistry and `registeredBackends`
- Model registry integration via `modelRef` configuration
- Follow-up `vision_analyze_image` tool for LLM image queries
- Image store with bounded FIFO eviction and TTL
- Privacy: URL sanitization, remote HTTP warning, EXIF stripping, sanitized events
- Diagnostics: `/vision doctor`, `/vision test`, cache stats
- Commands: `/vision`, `/vision on|off`, `/vision backend`, `/vision preset`, `/vision model`, `/vision model-ref`, `/vision clear`, `/vision purge`, `/vision setup`, `/vision test`, `/vision doctor`
