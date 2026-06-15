/**
 * vision.ts — universal image understanding for pi
 *
 * Gives EVERY primary model the ability to "see" images, even text-only ones
 * (deepseek, kimi, local GGUF coding models, …).
 *
 * How it works
 * ------------
 * On every provider request (the `context` event) we inspect the *active* model.
 *   • If it already accepts images (model.input includes "image") → do nothing.
 *   • If it's text-only → every image in the conversation (user-attached images
 *     AND images returned by tools, e.g. screenshots) is sent to a dedicated,
 *     lightweight VISION model. Its description replaces the image as plain text,
 *     so the text-only model receives a faithful textual view of the picture.
 *
 * Designed for low-performance machines:
 *   • Captions are cached by image hash (memory + disk) so re-sending history
 *     each turn never re-captions — you pay once per unique image.
 *   • Images are downscaled before captioning (cheap on CPU).
 *   • Captioning has a hard timeout and degrades gracefully: if the vision
 *     server is down, the agent still runs (it just gets a "could not analyze"
 *     note instead of stalling).
 *
 * Backends (pick with `/vision backend <name>` or PI_VISION_BACKEND):
 *   • mlx (default) — Apple Silicon, via the `mlx-vlm` server (localhost:8081/v1).
 *       Setup:  pip install mlx-vlm
 *               mlx_vlm.server --model <model> --port 8081
 *       NOTE: vision requires `mlx_vlm.server` (mlx-vlm). `mlx_lm.server` is
 *       text-only and cannot process images.
 *   • ollama — http://localhost:11434/v1, cross-platform, CPU-friendly.
 *       Setup:  ollama pull <model>
 *
 * Each backend has three presets — bump up as the machine allows:
 *   /vision preset light       (default; smallest, fastest)
 *   /vision preset balanced    (better OCR / UI / screenshots)
 *   /vision preset capable      (best local screenshot & diagram understanding)
 *
 * CONFIGURATION lives in pi's settings.json (~/.pi/agent/settings.json) under a
 * "vision" key — pi preserves it across its own writes. Example:
 *     "vision": { "backend": "mlx", "model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit" }
 * Precedence: PI_VISION_* env vars > settings.json `vision` block > defaults.
 * `vision.model` pins any model id directly (overrides the preset).
 *
 * Commands:  /vision  · /vision on|off · /vision backend <ollama|mlx>
 *            /vision preset <name> · /vision model <id>
 *            /vision test · /vision clear · /vision setup
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { Api, Context, ImageContent, Model, TextContent } from "@earendil-works/pi-ai";
import { complete } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { resizeImage } from "@earendil-works/pi-coding-agent";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// ── Config ────────────────────────────────────────────────────────────────

type BackendName = "ollama" | "mlx";
type PresetName = "light" | "balanced" | "capable";

interface VisionConfig {
	enabled: boolean;
	/** Which local server to talk to (sets default baseUrl / apiKey / presets). */
	backend: BackendName;
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
	/** Max NEW (uncached) images to caption in a single turn. */
	maxImagesPerTurn: number;
	/** Output token budget for each caption. */
	maxTokens: number;
	contextWindow: number;
	/** Persist captions to disk across sessions. */
	cache: boolean;
	/** System prompt that steers the captioner. */
	prompt: string;
}

const AGENT_DIR = join(homedir(), ".pi", "agent");
// Config lives in pi's own settings.json under a `vision` key. pi preserves
// unknown top-level keys when it rewrites settings, so this block survives.
const SETTINGS_PATH = join(AGENT_DIR, "settings.json");
const SETTINGS_KEY = "vision";
const CACHE_PATH = join(AGENT_DIR, "tmp", "vision-cache.json");

/** Read the whole settings.json object (or {} if missing/malformed). */
function readSettings(): Record<string, any> {
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
function readVisionSettings(): Record<string, any> {
	const block = readSettings()[SETTINGS_KEY];
	return block && typeof block === "object" ? block : {};
}

interface BackendDef {
	label: string;
	baseUrl: string;
	apiKey: string;
	/** Per-backend preset → model id. Bump up as the machine allows. */
	presets: Record<PresetName, string>;
	/** Setup instructions, given the resolved model id. */
	setup: (model: string) => string[];
}

const BACKENDS: Record<BackendName, BackendDef> = {
	// Cross-platform, easiest, CPU-friendly. Models are pulled by `ollama pull`.
	ollama: {
		label: "Ollama",
		baseUrl: "http://localhost:11434/v1",
		apiKey: "ollama", // local servers ignore the key; any non-empty value works
		presets: {
			light: "moondream", // ~1.7GB, CPU-friendly captioning/VQA
			balanced: "qwen2.5vl:3b", // strong OCR / charts / UI screenshots
			capable: "qwen2.5vl:7b", // best local screenshot & diagram understanding
		},
		setup: (model) => [
			"Quick start (Ollama):",
			"  1. Install Ollama → https://ollama.com",
			`  2. ollama pull ${model}`,
			"  3. /vision test",
		],
	},
	// Apple Silicon, via the mlx-vlm server (OpenAI-compatible). Fast on M-series.
	// The model is chosen when you launch the server (--model). We default to a
	// non-8080 port so it doesn't clash with a local llama.cpp coding server.
	mlx: {
		label: "MLX (Apple Silicon)",
		baseUrl: "http://localhost:8081/v1",
		apiKey: "mlx",
		presets: {
			light: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
			balanced: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
			capable: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
		},
		setup: (model) => [
			"Quick start (MLX — Apple Silicon):",
			"  1. pip install mlx-vlm",
			`  2. mlx_vlm.server --model ${model} --port 8081`,
			"     NOTE: vision needs mlx_vlm.server (mlx-vlm package).",
			"           mlx_lm.server is text-only and cannot read images.",
			"  3. /vision test",
		],
	},
};

const PRESET_NAMES: PresetName[] = ["light", "balanced", "capable"];

function isBackend(s: string): s is BackendName {
	return s === "ollama" || s === "mlx";
}
function isPreset(s: string): s is PresetName {
	return s === "light" || s === "balanced" || s === "capable";
}

const DEFAULT_PROMPT =
	"You are the eyes of a coding agent that CANNOT see images. " +
	"Describe the given image so the agent can fully understand and act on it. " +
	"Transcribe ALL visible text verbatim (code, errors, labels, UI text, numbers). " +
	"If it is a UI/screenshot: describe layout, components, state, and any errors. " +
	"If it is a diagram/chart: describe structure, relationships, axes, and values. " +
	"Be precise, complete, and factual. Do not speculate beyond what is visible.";

/** Tunables that are independent of the chosen backend. */
function baseDefaults() {
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
	};
}

/** Default backend when nothing is configured. */
const DEFAULT_BACKEND: BackendName = "mlx";

let configCache: VisionConfig | null = null;

function loadConfig(): VisionConfig {
	if (configCache) return configCache;

	// Settings come from settings.json → `vision` block.
	const raw = readVisionSettings();

	// 1. Backend (settings → env → default).
	let backend: BackendName = DEFAULT_BACKEND;
	if (typeof raw.backend === "string" && isBackend(raw.backend)) backend = raw.backend;
	const envBackend = process.env.PI_VISION_BACKEND;
	if (envBackend && isBackend(envBackend)) backend = envBackend;
	const def = BACKENDS[backend];

	// 2. Preset (settings → default light) chooses the model within the backend.
	let preset: PresetName = "light";
	if (typeof raw.preset === "string" && isPreset(raw.preset)) preset = raw.preset;

	const cfg: VisionConfig = {
		...baseDefaults(),
		backend,
		baseUrl: def.baseUrl,
		apiKey: def.apiKey,
		model: def.presets[preset],
	};

	// 3. Explicit per-key overrides from settings (e.g. model, baseUrl, timeoutMs).
	//    `vision.model` is the direct way to pin any model id, regardless of preset.
	for (const k of Object.keys(cfg) as (keyof VisionConfig)[]) {
		if (k === "backend") continue; // already resolved
		if (raw[k] !== undefined && raw[k] !== null) (cfg as any)[k] = raw[k];
	}

	// 4. Env overrides (highest priority).
	if (process.env.PI_VISION_BASE_URL) cfg.baseUrl = process.env.PI_VISION_BASE_URL;
	if (process.env.PI_VISION_API_KEY) cfg.apiKey = process.env.PI_VISION_API_KEY;
	if (process.env.PI_VISION_MODEL) cfg.model = process.env.PI_VISION_MODEL;
	if (process.env.PI_VISION_DISABLED === "1" || process.env.PI_VISION_DISABLED === "true") cfg.enabled = false;

	configCache = cfg;
	return cfg;
}

/** Persist into settings.json → `vision` block. Keys set to `null` in the patch
 *  are removed (so the value falls back to the backend/preset default on next
 *  load). All other settings.json keys are preserved. */
function saveConfig(patch: Record<string, unknown>): void {
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
	configCache = null; // force reload
}

// ── Caption cache (memory + disk) ───────────────────────────────────────────

let memCache: Map<string, string> | null = null;

function loadCache(): Map<string, string> {
	if (memCache) return memCache;
	memCache = new Map();
	try {
		if (existsSync(CACHE_PATH)) {
			const obj = JSON.parse(readFileSync(CACHE_PATH, "utf8"));
			for (const [k, v] of Object.entries(obj)) if (typeof v === "string") memCache.set(k, v);
		}
	} catch {
		/* ignore corrupt cache */
	}
	return memCache;
}

let saveTimer: NodeJS.Timeout | null = null;
function persistCacheSoon(): void {
	if (saveTimer) return;
	saveTimer = setTimeout(() => {
		saveTimer = null;
		try {
			const cache = loadCache();
			// Keep the cache bounded — drop oldest insertions beyond 500 entries.
			const entries = [...cache.entries()];
			const trimmed = entries.slice(-500);
			mkdirSync(dirname(CACHE_PATH), { recursive: true });
			writeFileSync(CACHE_PATH, JSON.stringify(Object.fromEntries(trimmed)), "utf8");
		} catch {
			/* best-effort */
		}
	}, 1500);
	saveTimer.unref?.();
}

function cacheKey(cfg: VisionConfig, data: string): string {
	return createHash("sha256")
		.update(cfg.model)
		.update(" ")
		.update(cfg.prompt)
		.update(" ")
		.update(data)
		.digest("hex")
		.slice(0, 24);
}

// ── Vision model invocation ──────────────────────────────────────────────────

function buildVisionModel(cfg: VisionConfig): Model<Api> {
	return {
		id: cfg.model,
		name: `vision:${cfg.model}`,
		api: cfg.api,
		provider: "pi-vision",
		baseUrl: cfg.baseUrl,
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: cfg.contextWindow,
		maxTokens: cfg.maxTokens,
	};
}

function extractText(msg: { content: unknown }): string {
	const content = msg.content;
	if (typeof content === "string") return content.trim();
	if (!Array.isArray(content)) return "";
	return content
		.filter((c: any) => c?.type === "text" && typeof c.text === "string")
		.map((c: any) => c.text)
		.join("\n")
		.trim();
}

/** Run the vision model on one image, returning a description. */
async function captionImage(
	img: ImageContent,
	cfg: VisionConfig,
	model: Model<Api>,
	parentSignal: AbortSignal | undefined,
): Promise<string> {
	// Downscale to keep CPU captioning fast and within context.
	let data = img.data;
	let mimeType = img.mimeType || "image/png";
	try {
		const bytes = Buffer.from(img.data, "base64");
		const resized = await resizeImage(bytes, mimeType, {
			maxWidth: cfg.maxEdge,
			maxHeight: cfg.maxEdge,
			maxBytes: cfg.maxBytes,
		});
		if (resized) {
			data = resized.data;
			mimeType = resized.mimeType;
		}
	} catch {
		/* resize is best-effort; fall back to original bytes */
	}

	const context: Context = {
		systemPrompt: cfg.prompt,
		messages: [
			{
				role: "user",
				content: [
					{ type: "text", text: "Describe this image in full, actionable detail." },
					{ type: "image", data, mimeType },
				],
				timestamp: Date.now(),
			},
		],
	};

	// Own timeout, also chained to the agent's abort.
	const ctrl = new AbortController();
	const onParentAbort = () => ctrl.abort();
	parentSignal?.addEventListener("abort", onParentAbort, { once: true });
	const timer = setTimeout(() => ctrl.abort(), cfg.timeoutMs);
	timer.unref?.();
	try {
		const res = await complete(model, context, {
			apiKey: cfg.apiKey,
			temperature: 0.2,
			maxTokens: cfg.maxTokens,
			timeoutMs: cfg.timeoutMs,
			maxRetries: 0,
			signal: ctrl.signal,
		});
		const text = extractText(res as any);
		return text || "(vision model returned an empty description)";
	} finally {
		clearTimeout(timer);
		parentSignal?.removeEventListener("abort", onParentAbort);
	}
}

function wrapCaption(cfg: VisionConfig, text: string): string {
	return (
		`[image-description] The active model cannot see images, so ${cfg.model} described this image:\n` +
		`${text}\n[/image-description]`
	);
}

// ── Extension ────────────────────────────────────────────────────────────────

const ROLES_WITH_IMAGES = new Set(["user", "toolResult"]);

function hasImagePart(msg: AgentMessage): boolean {
	const content = (msg as any).content;
	return Array.isArray(content) && content.some((c: any) => c?.type === "image");
}

export default function (pi: ExtensionAPI) {
	let lastTurnStats = { captioned: 0, cached: 0, failed: 0 };

	pi.on("context", async (event, ctx): Promise<{ messages?: AgentMessage[] } | void> => {
		const cfg = loadConfig();
		if (!cfg.enabled) return;

		const model = ctx.model;
		if (!model) return;
		// Active model can already see — leave images untouched.
		if (Array.isArray(model.input) && model.input.includes("image")) return;

		const messages = event.messages as AgentMessage[];
		if (!messages.some(hasImagePart)) return;

		const vModel = buildVisionModel(cfg);
		const cache = loadCache();
		const stats = { captioned: 0, cached: 0, failed: 0 };
		let newThisTurn = 0;
		let touched = false;

		const out: AgentMessage[] = [];
		for (const msg of messages) {
			if (!ROLES_WITH_IMAGES.has((msg as any).role) || !hasImagePart(msg)) {
				out.push(msg);
				continue;
			}
			const content = (msg as any).content as (TextContent | ImageContent)[];
			const newContent: (TextContent | ImageContent)[] = [];
			for (const part of content) {
				if (part.type !== "image") {
					newContent.push(part);
					continue;
				}
				const key = cacheKey(cfg, part.data);
				const hit = cache.get(key);
				if (hit) {
					newContent.push({ type: "text", text: wrapCaption(cfg, hit) });
					stats.cached++;
					touched = true;
					continue;
				}
				if (newThisTurn >= cfg.maxImagesPerTurn) {
					// Budget exhausted this turn — keep the image as a note; it gets
					// captioned on the next turn (cache fills in incrementally).
					newContent.push({
						type: "text",
						text: `[image-description] (not yet analyzed — image-per-turn budget of ${cfg.maxImagesPerTurn} reached; will be described next turn) [/image-description]`,
					});
					touched = true;
					continue;
				}
				newThisTurn++;
				try {
					const text = await captionImage(part, cfg, vModel, ctx.signal);
					cache.set(key, text);
					persistCacheSoon();
					newContent.push({ type: "text", text: wrapCaption(cfg, text) });
					stats.captioned++;
				} catch (e) {
					const reason = e instanceof Error ? e.message : String(e);
					newContent.push({
						type: "text",
						text: `[image-description] Image could not be analyzed (${reason}). Vision backend: ${cfg.baseUrl} (${cfg.model}). Run /vision test to diagnose. [/image-description]`,
					});
					stats.failed++;
				}
				touched = true;
			}
			out.push({ ...(msg as any), content: newContent });
		}

		lastTurnStats = stats;
		if (touched) {
			renderStatus(ctx, cfg, stats);
			return { messages: out };
		}
	});

	pi.on("model_select", async (_e, ctx) => renderStatus(ctx, loadConfig(), lastTurnStats));
	pi.on("session_start", async (_e, ctx) => renderStatus(ctx, loadConfig(), lastTurnStats));

	function renderStatus(ctx: ExtensionContext, cfg: VisionConfig, stats: typeof lastTurnStats) {
		const model = ctx.model;
		const textOnly = !!model && Array.isArray(model.input) && !model.input.includes("image");
		// Only show the badge when we're actually doing work: a text-only primary
		// model with vision enabled.
		if (!cfg.enabled || !textOnly) {
			ctx.ui.setStatus?.("vision", "");
			return;
		}
		const theme = (ctx.ui as any).theme;
		const n = stats.captioned + stats.cached;
		const label = `👁 ${cfg.model}${n ? ` (${n})` : ""}`;
		ctx.ui.setStatus?.("vision", theme?.fg ? theme.fg("dim", label) : label);
	}

	// ── /vision command ────────────────────────────────────────────────────────

	pi.registerCommand("vision", {
		description: "Image understanding for text-only models — backend, preset, test",
		handler: async (args, ctx) => {
			const [sub, ...rest] = args.trim().split(/\s+/);
			const arg = rest.join(" ").trim();
			const cfg = loadConfig();

			switch (sub) {
				case "":
				case "status": {
					const model = ctx.model;
					const textOnly = !!model && Array.isArray(model.input) && !model.input.includes("image");
					const multimodal = !!model && Array.isArray(model.input) && model.input.includes("image");
					const lines = [
						`vision: ${cfg.enabled ? "on" : "off"}`,
						`backend: ${BACKENDS[cfg.backend].label} (${cfg.baseUrl})`,
						`model: ${cfg.model}`,
						`active model: ${model?.id ?? "?"} → ${
							multimodal ? "multimodal (vision passthrough)" : textOnly ? "text-only (vision ACTIVE)" : "unknown"
						}`,
						`cache: ${loadCache().size} image(s)`,
						`last turn: ${lastTurnStats.captioned} captioned · ${lastTurnStats.cached} cached · ${lastTurnStats.failed} failed`,
					];
					ctx.ui.notify(lines.join("\n"), "info");
					return;
				}
				case "on":
					saveConfig({ enabled: true });
					ctx.ui.notify("Vision enabled.", "info");
					renderStatus(ctx, loadConfig(), lastTurnStats);
					return;
				case "off":
					saveConfig({ enabled: false });
					ctx.ui.notify("Vision disabled.", "info");
					renderStatus(ctx, loadConfig(), lastTurnStats);
					return;
				case "backend": {
					if (!isBackend(arg)) {
						ctx.ui.notify(`Usage: /vision backend <${Object.keys(BACKENDS).join("|")}>`, "error");
						return;
					}
					// Switching backend clears any pinned endpoint/model so the new
					// backend's defaults (and current preset) take effect cleanly.
					saveConfig({ backend: arg, baseUrl: null, apiKey: null, model: null });
					const next = loadConfig();
					ctx.ui.notify(
						[`Vision backend → ${BACKENDS[arg].label}`, "", ...BACKENDS[arg].setup(next.model)].join("\n"),
						"info",
					);
					renderStatus(ctx, next, lastTurnStats);
					return;
				}
				case "preset": {
					if (!isPreset(arg)) {
						ctx.ui.notify(`Usage: /vision preset <${PRESET_NAMES.join("|")}>`, "error");
						return;
					}
					// Clear any pinned model so the preset resolves within the backend.
					saveConfig({ preset: arg, model: null });
					const next = loadConfig();
					ctx.ui.notify(
						[`Vision preset → ${arg} (${next.model})`, "", ...BACKENDS[next.backend].setup(next.model)].join("\n"),
						"info",
					);
					renderStatus(ctx, next, lastTurnStats);
					return;
				}
				case "model": {
					if (!arg) {
						ctx.ui.notify("Usage: /vision model <model-id>", "error");
						return;
					}
					saveConfig({ model: arg });
					ctx.ui.notify(`Vision model → ${arg}`, "info");
					renderStatus(ctx, loadConfig(), lastTurnStats);
					return;
				}
				case "clear":
					memCache = new Map();
					try {
						if (existsSync(CACHE_PATH)) writeFileSync(CACHE_PATH, "{}", "utf8");
					} catch {
						/* ignore */
					}
					ctx.ui.notify("Vision caption cache cleared.", "info");
					return;
				case "setup": {
					// Seed an explicit `vision` block in settings.json (only fields not
					// already set), so the user has something to edit. Backend/preset
					// driven — no pinned endpoint/model — so the commands stay clean.
					if (!readVisionSettings().backend) {
						saveConfig({
							enabled: true,
							backend: cfg.backend,
							preset: "light",
							maxEdge: cfg.maxEdge,
							timeoutMs: cfg.timeoutMs,
							maxImagesPerTurn: cfg.maxImagesPerTurn,
						});
					}
					ctx.ui.notify(
						[
							`Config: ${SETTINGS_PATH} → "vision" block`,
							`Backend: ${BACKENDS[cfg.backend].label}`,
							"",
							...BACKENDS[cfg.backend].setup(cfg.model),
							"",
							`Switch backend:  /vision backend <${Object.keys(BACKENDS).join("|")}>`,
							"Pin a model:     /vision model <id>   (or edit settings.json → vision.model)",
							"Better quality:  /vision preset balanced",
						].join("\n"),
						"info",
					);
					return;
				}
				case "test": {
					ctx.ui.notify(`Testing ${cfg.model} at ${cfg.baseUrl} …`, "info");
					// 8×8 red PNG — just checks connectivity + that the model answers.
					const testPng =
						"iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAFklEQVR42mP8z8BQz0AEYBxVSF+FAP8FA/2v3T0lAAAAAElFTkSuQmCC";
					const start = Date.now();
					try {
						const text = await captionImage(
							{ type: "image", data: testPng, mimeType: "image/png" },
							cfg,
							buildVisionModel(cfg),
							ctx.signal,
						);
						const ms = Date.now() - start;
						ctx.ui.notify(`✅ Vision OK in ${ms}ms.\n${cfg.model} said: ${text.slice(0, 200)}`, "info");
					} catch (e) {
						const reason = e instanceof Error ? e.message : String(e);
						ctx.ui.notify(
							[
								`❌ Vision failed: ${reason}`,
								`Backend: ${BACKENDS[cfg.backend].label} — ${cfg.baseUrl} (${cfg.model})`,
								"Is the server running and the model available?",
								...BACKENDS[cfg.backend].setup(cfg.model),
							].join("\n"),
							"error",
						);
					}
					return;
				}
				default:
					ctx.ui.notify(
						"Usage: /vision [status|on|off|backend <ollama|mlx>|preset <name>|model <id>|test|clear|setup]",
						"error",
					);
			}
		},
	});
}
