/**
 * config.ts — Vision extension configuration: load, save, defaults.
 *
 * Loading precedence: PI_VISION_* env vars > settings.json `vision` block
 * > backend/preset defaults.
 *
 * Settings are stored in pi's own settings.json (~/.pi/agent/settings.json)
 * under a `vision` key. pi preserves unknown top-level keys when it rewrites
 * settings, so this block survives pi's own writes.
 */

import type { Api } from "@earendil-works/pi-ai";
import type { BackendDef, BackendName, PresetName, PersistedCustomBackend } from "./backends.js";
import {
	BACKENDS,
	DEFAULT_BACKEND,
	DEFAULT_PROMPT,
	isBackend,
	isPreset,
	visionBackendRegistry,
	BUILTIN_BACKEND_NAMES,
	loadCustomBackends,
	PRESET_NAMES_SET,
} from "./backends.js";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// ── Types ──────────────────────────────────────────────────────────────────

export type { BackendName, PresetName, BackendDef };

export interface VisionConfig {
	enabled: boolean;
	/** Which local server to talk to (sets default baseUrl / apiKey / presets). */
	backend: string;
	/** Resolved from backend default unless explicitly overridden. */
	baseUrl: string;
	apiKey: string;
	model: string;
	api: Api;
	/** Longest edge (px) to downscale images to before captioning. */
	maxEdge: number;
	/** Max encoded size (bytes) after downscale. */
	maxBytes: number;
	/** Hard timeout per image (ms). */
	timeoutMs: number;
	/** Max NEW (uncapped) images to caption in a single turn. */
	maxImagesPerTurn: number;
	/** Output token budget for each caption. */
	maxTokens: number;
	contextWindow: number;
	/** Persist captions to disk across sessions. */
	cache: boolean;
	/** System prompt that steers the captioner. */
	prompt: string;
	/**
	 * How to resolve the captioner model:
	 * - `"backend"` (default): use the configured `backend` + `preset` system.
	 * - `"<provider>/<modelId>"` (e.g. `"openai/gpt-4o"`): look up the model
	 *   in pi's ModelRegistry and resolve auth through pi.
	 */
	modelRef: string;
	/**
	 * Custom backends persisted in settings, keyed by name.
	 * Loaded into VisionBackendRegistry at startup.
	 */
	registeredBackends: Record<string, PersistedCustomBackend>;
	/** Max concurrent captioning requests per turn. */
	captionConcurrency: number;
	/** Max entries in the LRU caption cache. */
	cacheMaxEntries: number;
	/** TTL for negative (failure) cache entries in milliseconds. */
	negativeCacheTTLMs: number;
	/**
	 * Optional passphrase for AES-256-GCM encrypted disk cache.
	 * When set, the cache file on disk is encrypted. The passphrase itself
	 * is never persisted — set via PI_VISION_CACHE_PASSPHRASE env var or
	 * settings.json (though storing a passphrase in settings.json is
	 * discouraged; prefer the env var).
	 */
	cachePassphrase?: string;
}

// ── Paths ───────────────────────────────────────────────────────────────────

export const AGENT_DIR = join(homedir(), ".pi", "agent");
// Config lives in pi's own settings.json under a `vision` key. pi preserves
// unknown top-level keys when it rewrites settings, so this block survives.
// Override via PI_VISION_SETTINGS_PATH env var (useful in tests).
export const SETTINGS_PATH = process.env.PI_VISION_SETTINGS_PATH ?? join(AGENT_DIR, "settings.json");
export const SETTINGS_KEY = "vision";
// Override via PI_VISION_CACHE_PATH env var (useful in tests).
export const CACHE_PATH = process.env.PI_VISION_CACHE_PATH ?? join(AGENT_DIR, "tmp", "vision-cache.json");

// ── Settings I/O ───────────────────────────────────────────────────────────

/** Read the whole settings.json object (or {} if missing/malformed). */
export function readSettings(): Record<string, any> {
	try {
		if (existsSync(SETTINGS_PATH)) {
			const parsed = JSON.parse(readFileSync(SETTINGS_PATH, "utf8"));
			if (parsed && typeof parsed === "object") return parsed;
		}
	} catch {
		/* malformed settings → treat as empty */
	}
	return {};
}

/** Read just the `vision` settings block. */
export function readVisionSettings(): Record<string, any> {
	const block = readSettings()[SETTINGS_KEY];
	return block && typeof block === "object" ? block : {};
}

// ── Tunables that are independent of the chosen backend ────────────────────

export function baseDefaults() {
	return {
		enabled: true,
		api: "openai-completions" as Api,
		maxEdge: 1024,
		maxBytes: 1_400_000,
		timeoutMs: 120_000,
		maxImagesPerTurn: 6,
		maxTokens: 768,
		contextWindow: 8192,
		cache: true,
		prompt: DEFAULT_PROMPT,
		modelRef: "backend",
		captionConcurrency: 2,
		cacheMaxEntries: 500,
		negativeCacheTTLMs: 300_000,
		cachePassphrase: undefined,
	};
}

// ── Config loading ─────────────────────────────────────────────────────────

export function loadConfig(): VisionConfig {
	// Settings come from settings.json → `vision` block.
	const raw = readVisionSettings();

	// Load custom backends into the registry from persisted settings.
	const registeredBackends: Record<string, PersistedCustomBackend> =
		raw.registeredBackends && typeof raw.registeredBackends === "object"
			? raw.registeredBackends
			: {};
	loadCustomBackends(registeredBackends);

	// 1. Backend (settings → env → default).
	let backend: string = DEFAULT_BACKEND;
	if (typeof raw.backend === "string" && isBackend(raw.backend)) backend = raw.backend;
	const envBackend = process.env.PI_VISION_BACKEND;
	if (envBackend && isBackend(envBackend)) backend = envBackend;
	const def: BackendDef | undefined = visionBackendRegistry.get(backend) ?? BACKENDS[DEFAULT_BACKEND];

	// 2. Preset (settings → default light) chooses the model within the backend.
	let preset: PresetName = "light";
	if (typeof raw.preset === "string" && isPreset(raw.preset)) preset = raw.preset;

	const cfg: VisionConfig = {
		...baseDefaults(),
		backend,
		baseUrl: def.baseUrl,
		apiKey: def.apiKey,
		model: def.presets[preset] ?? def.presets[def.defaultPreset] ?? Object.values(def.presets)[0] ?? "",
		registeredBackends,
	};

	// 3. Explicit per-key overrides from settings (e.g. model, baseUrl, timeoutMs).
	//    `vision.model` is the direct way to pin any model id, regardless of preset.
	for (const k of Object.keys(cfg) as (keyof VisionConfig)[]) {
		if (k === "backend" || k === "registeredBackends") continue; // already resolved
		if (raw[k] !== undefined && raw[k] !== null) (cfg as any)[k] = raw[k];
	}

	// 4. Env overrides (highest priority).
	if (process.env.PI_VISION_BASE_URL) cfg.baseUrl = process.env.PI_VISION_BASE_URL;
	if (process.env.PI_VISION_API_KEY) cfg.apiKey = process.env.PI_VISION_API_KEY;
	if (process.env.PI_VISION_MODEL) cfg.model = process.env.PI_VISION_MODEL;
	if (process.env.PI_VISION_DISABLED === "1" || process.env.PI_VISION_DISABLED === "true") cfg.enabled = false;
	if (process.env.PI_VISION_CACHE_PASSPHRASE) cfg.cachePassphrase = process.env.PI_VISION_CACHE_PASSPHRASE;

	return cfg;
}

// ── Config saving ──────────────────────────────────────────────────────────

/** Persist into settings.json → `vision` block. Keys set to `null` in the patch
 *  are removed (so the value falls back to the backend/preset default on next
 *  load). All other settings.json keys are preserved. */
export function saveConfig(patch: Record<string, unknown>): void {
	const settings = readSettings();
	const block: Record<string, unknown> =
		settings[SETTINGS_KEY] && typeof settings[SETTINGS_KEY] === "object" ? settings[SETTINGS_KEY] : {};
	for (const [k, v] of Object.entries(patch)) {
		if (v === null) delete block[k];
		else block[k] = v;
	}
	settings[SETTINGS_KEY] = block;
	mkdirSync(dirname(SETTINGS_PATH), { recursive: true });
	writeFileSync(SETTINGS_PATH, `${JSON.stringify(settings, null, 2)}\n`, "utf8");
}
