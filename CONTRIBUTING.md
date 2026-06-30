# Contributing to pi-vision

## Development setup

```bash
git clone https://github.com/dvictor357/vision-pi.git
cd vision-pi
npm install
```

## Scripts

| Command | Purpose |
|---------|---------|
| `npm test` | Run all tests |
| `npm run typecheck` | Type-check with `tsc --noEmit` |
| `npm run pack` | Dry-run `npm pack` to check package contents |

## Code conventions

- **TypeScript** strict mode, NodeNext module resolution.
- **No runtime dependencies** — the package relies entirely on peer dependencies (`@earendil-works/pi-*`).
- **Tabs** for indentation.
- **No semicolons** in JS/DTS output (current style uses them; be consistent with surrounding code).
- **Conventional commits** for commit messages.

## Project structure

```
extensions/vision.ts     — pi extension entrypoint (re-exports orchestrator)
src/                     — implementation modules
  orchestrator.ts        — extension event wiring, default export
  config.ts              — config load/save with env override chain
  backends.ts            — vision backend definitions and registry
  cache.ts               — LRU caption cache, negative cache, in-flight dedup, encryption
  vision-model.ts        — build model descriptor, caption images, bounded concurrency
  commands.ts            — /vision command handler
  follow-up-tool.ts      — vision_analyze_image tool registration
  image-store.ts         — per-session image store for follow-up queries
  privacy.ts             — URL sanitization, localhost check, HTTP warnings, EXIF stripping
test/vision.test.ts      — all tests
```

## Testing guidelines

- All modules are tested independently with mocks for pi APIs.
- Use `describe`/`it` from `node:test`.
- Avoid file I/O in unit tests; use env var overrides (`PI_VISION_SETTINGS_PATH`, `PI_VISION_CACHE_PATH`) when config I/O is unavoidable.
- Concurrency and LRU eviction behavior are tested with perf-style benchmarks.

## Adding a backend

1. Add a new entry to the `BACKENDS` record in `src/backends.ts`, or use `/vision backend register <name> <baseUrl> [apiKey]` at runtime.
2. Each backend needs a `setup()` function that returns installation instructions.
3. Add the backend name to `BUILTIN_BACKEND_NAMES` if built-in.

## Releasing

1. Update version in `package.json` and `CHANGELOG.md`.
2. Run `npm run typecheck && npm test`.
3. Run `npm pack --dry-run` to verify package contents.
4. Tag the release (`git tag vX.Y.Z`) and push.
