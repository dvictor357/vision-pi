/**
 * orchestrator.ts — Extension event wiring, status notifications, and the
 * default export consumed by pi.
 *
 * Features:
 * - Fast no-op paths for disabled / no-model / multimodal / no-image turns.
 * - Pre-captioning via `input` and `tool_result` events.
 * - In-flight de-duplication, negative cache, bounded concurrency.
 * - LRU caption cache with dirty tracking and atomic writes.
 * - Image store with stable ref IDs for follow-up vision analysis.
 * - `vision_analyze_image` tool registered for LLM follow-up queries.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { ImageContent, TextContent, Model, Api } from "@earendil-works/pi-ai";
import { loadConfig } from "./config.js";
import { visionBackendRegistry } from "./backends.js";
import {
	loadCache,
	cacheKey,
	cacheGet,
	cacheSet,
	isNegCached,
	markNegCached,
	getInFlight,
	setInFlight,
	persistSoon,
	flushCacheSync,
} from "./cache.js";
import {
	buildVisionModel,
	captionImage,
	wrapCaption,
	shortVisionModel,
	mapWithConcurrency,
} from "./vision-model.js";
import { registerVisionCommand } from "./commands.js";
import { ImageStore } from "./image-store.js";
import { registerFollowUpTool } from "./follow-up-tool.js";
import type { VisionStats } from "./commands.js";
import { sanitizeConfig, warnRemoteHttp } from "./privacy.js";

// ── Constants ──────────────────────────────────────────────────────────────

/** Short ref prefix for image references in caption wrappers. */
const REF_PREFIX = "img_";
/** How many hex chars of the cache key to use as the ref suffix. */
const REF_LENGTH = 8;

// ── Roles that may carry images ────────────────────────────────────────────

const ROLES_WITH_IMAGES = new Set(["user", "toolResult"]);

function hasImagePart(content: unknown): boolean {
	return Array.isArray(content) && content.some((c: any) => c?.type === "image");
}

/** Extract ImageContent parts from a message content array. */
function getImageParts(content: unknown): ImageContent[] {
	if (!Array.isArray(content)) return [];
	return content.filter((c: any): c is ImageContent => c?.type === "image");
}

/** Derive a short, stable ref ID from a cache key. */
function keyToRef(key: string): string {
	return REF_PREFIX + key.slice(0, REF_LENGTH);
}

// ── Resolve captioner from pi model registry ───────────────────────────────

async function resolveModelFromRegistry(
	cfg: ReturnType<typeof loadConfig>,
	modelRegistry: ExtensionContext["modelRegistry"] | undefined,
): Promise<{ model: Model<Api>; apiKey: string } | undefined> {
	if (cfg.modelRef === "backend" || !modelRegistry) return undefined;

	const slashIdx = cfg.modelRef.indexOf("/");
	if (slashIdx <= 0 || slashIdx >= cfg.modelRef.length - 1) return undefined;

	const provider = cfg.modelRef.slice(0, slashIdx);
	const modelId = cfg.modelRef.slice(slashIdx + 1);
	const registryModel = modelRegistry.find(provider, modelId);
	if (!registryModel) return undefined;

	const auth = await modelRegistry.getApiKeyAndHeaders(registryModel);
	if (!auth.ok) return undefined;

	return { model: registryModel, apiKey: auth.apiKey ?? "" };
}

// ── Capture captioner model + apiKey for the current turn ─────────────────

async function resolveCaptioner(
	cfg: ReturnType<typeof loadConfig>,
	ctx: ExtensionContext,
): Promise<{ model: Model<Api>; apiKey: string }> {
	const fromRegistry = await resolveModelFromRegistry(cfg, ctx.modelRegistry);
	if (fromRegistry) return fromRegistry;
	return { model: buildVisionModel(cfg), apiKey: cfg.apiKey };
}

// ── Caption one image (with dedup + negative cache) ────────────────────────

async function captionOne(
	key: string,
	img: ImageContent,
	cfg: ReturnType<typeof loadConfig>,
	vModel: Model<Api>,
	apiKey: string,
	signal: AbortSignal | undefined,
): Promise<string> {
	if (isNegCached(key)) throw new Error("Image previously failed; negative cache hit");

	const existing = getInFlight(key);
	if (existing) return existing;

	// Start captioning. On success, store in cache. On failure, setInFlight
	// marks the negative cache with the configured TTL.
	const promise = captionImage(img, cfg, vModel, signal, apiKey).then((text) => {
		cacheSet(key, text);
		persistSoon();
		return text;
	});
	setInFlight(key, promise);

	return promise;
}

// ── Store image and wrap with ref ──────────────────────────────────────────

/**
 * Store an image in the session image store and return a wrapped caption
 * that includes the stable ref ID.
 */
function storeAndWrap(
	imageStore: ImageStore,
	cfg: ReturnType<typeof loadConfig>,
	key: string,
	data: string,
	mimeType: string,
	text: string,
): string {
	const ref = keyToRef(key);
	imageStore.set(ref, data, mimeType);
	return wrapCaption(cfg, text, ref);
}

// ── Extension entry-point ──────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
	// Initialize caption cache from config (loads disk state, applies limits).
	const initCfg = loadConfig();
	loadCache(initCfg);

	let lastTurnStats: VisionStats = { captioned: 0, cached: 0, failed: 0 };

	// Per-session image store for follow-up queries.
	// Created at session start and cleared on shutdown.
	let imageStore: ImageStore = new ImageStore();

	// ── Pre-caption: input event ──────────────────────────────────────────

	pi.on("input", async (event, ctx) => {
		const cfg = loadConfig();
		if (!cfg.enabled) return;

		const images = event.images;
		if (!images || images.length === 0) return;

		const model = ctx.model;
		if (model && Array.isArray(model.input) && model.input.includes("image")) return;

		const resolved = await resolveCaptioner(cfg, ctx);

		for (const img of images) {
			const key = cacheKey(cfg, img.data);
			if (cacheGet(key) !== undefined) continue;
			if (isNegCached(key)) continue;
			if (getInFlight(key)) continue;

			const promise = captionImage(img, cfg, resolved.model, ctx.signal, resolved.apiKey)
				.then((text) => {
					cacheSet(key, text);
					persistSoon();
					return text;
				})
				.catch(() => {
					markNegCached(key, cfg.negativeCacheTTLMs);
					return "";
				});
			setInFlight(key, promise);
		}
	});

	// ── Pre-caption: tool_result event ────────────────────────────────────

	pi.on("tool_result", async (event, ctx) => {
		const cfg = loadConfig();
		if (!cfg.enabled) return;

		const images = getImageParts(event.content);
		if (images.length === 0) return;

		const model = ctx.model;
		if (model && Array.isArray(model.input) && model.input.includes("image")) return;

		const resolved = await resolveCaptioner(cfg, ctx);

		for (const img of images) {
			const key = cacheKey(cfg, img.data);
			if (cacheGet(key) !== undefined) continue;
			if (isNegCached(key)) continue;
			if (getInFlight(key)) continue;

			const promise = captionImage(img, cfg, resolved.model, ctx.signal, resolved.apiKey)
				.then((text) => {
					cacheSet(key, text);
					persistSoon();
					return text;
				})
				.catch(() => {
					markNegCached(key, cfg.negativeCacheTTLMs);
					return "";
				});
			setInFlight(key, promise);
		}
	});

	// ── Context event: caption images before text-only models see them ────

	pi.on("context", async (event, ctx): Promise<{ messages?: AgentMessage[] } | void> => {
		const cfg = loadConfig();

		// ── Fast no-op paths ──────────────────────────────────────────────
		if (!cfg.enabled) return;
		const model = ctx.model;
		if (!model) return;
		if (Array.isArray(model.input) && model.input.includes("image")) return;

		const messages = event.messages as AgentMessage[];
		if (!messages.some((m) => hasImagePart((m as any).content))) return;

		// ── Resolve captioner ──────────────────────────────────────────────
		const resolved = await resolveCaptioner(cfg, ctx);
		const stats = { captioned: 0, cached: 0, failed: 0 };
		let newThisTurn = 0;
		let touched = false;

		type ImageJob = {
			msgIdx: number;
			contentIdx: number;
			part: ImageContent;
			key: string;
		};

		const jobs: ImageJob[] = [];
		const out: AgentMessage[] = [];

		for (const msg of messages) {
			const role = (msg as any).role;
			const content = (msg as any).content as (TextContent | ImageContent)[];

			if (!ROLES_WITH_IMAGES.has(role) || !hasImagePart(content)) {
				out.push(msg);
				continue;
			}

			const newContent: (TextContent | ImageContent)[] = [];
			for (const part of content) {
				if (part.type !== "image") {
					newContent.push(part);
					continue;
				}
				const key = cacheKey(cfg, part.data);
				const hit = cacheGet(key);
				if (hit !== undefined) {
					// Cache hit — wrap with stored image for follow-up
					const wrapped = storeAndWrap(imageStore, cfg, key, part.data, part.mimeType || "image/png", hit);
					newContent.push({ type: "text", text: wrapped });
					stats.cached++;
					touched = true;
					continue;
				}
				if (isNegCached(key)) {
					newContent.push({
						type: "text",
						text: `[image-description] (not available — previously failed; will retry next session) [/image-description]`,
					});
					touched = true;
					continue;
				}
				if (newThisTurn >= cfg.maxImagesPerTurn) {
					newContent.push({
						type: "text",
						text: `[image-description] (not yet analyzed — image-per-turn budget of ${cfg.maxImagesPerTurn} reached; will be described next turn) [/image-description]`,
					});
					touched = true;
					continue;
				}
				// Need to caption
				jobs.push({ msgIdx: out.length, contentIdx: newContent.length, part, key });
				newThisTurn++;
				newContent.push(part); // placeholder
			}
			out.push({ ...(msg as any), content: newContent });
		}

		// ── Phase 2: caption uncached images with bounded concurrency ─────
		if (jobs.length > 0) {
			const captionResults = await mapWithConcurrency(
				jobs,
				cfg.captionConcurrency,
				async (job) => {
					try {
						const text = await captionOne(job.key, job.part, cfg, resolved.model, resolved.apiKey, ctx.signal);
						return { index: jobs.indexOf(job), text, error: undefined };
					} catch (e) {
						return {
							index: jobs.indexOf(job),
							text: undefined,
							error: e instanceof Error ? e.message : String(e),
						};
					}
				},
			);

			for (const result of captionResults) {
				if (result.error) {
					const job = jobs[result.index];
					const msgContent = (out[job.msgIdx] as any).content as (TextContent | ImageContent)[];
					msgContent[job.contentIdx] = {
						type: "text",
						text: `[image-description] Image could not be analyzed (${result.error}). Vision captioner: ${resolved.model.id}. Run /vision test to diagnose. [/image-description]`,
					};
					stats.failed++;
				} else {
					const job = jobs[result.index];
					const msgContent = (out[job.msgIdx] as any).content as (TextContent | ImageContent)[];
					const wrapped = storeAndWrap(imageStore, cfg, job.key, job.part.data, job.part.mimeType || "image/png", result.text!);
					msgContent[job.contentIdx] = { type: "text", text: wrapped };
					stats.captioned++;
				}
				touched = true;
			}
		}

		lastTurnStats = stats;
		if (touched) {
			notifyVisionChanged(ctx);
			return { messages: out };
		}
	});

	// ── Status notifications ─────────────────────────────────────────────

	function notifyVisionChanged(ctx?: {
		ui: { setStatus: (key: string, text: string | undefined) => void };
	}) {
		const cfg = loadConfig();
		const safeCfg = sanitizeConfig(cfg);
		pi.events.emit("vision_status", { cfg: safeCfg, stats: lastTurnStats });

		if (ctx?.ui) {
			if (!cfg.enabled) {
				ctx.ui.setStatus("vision", "vision off");
			} else {
				const modelShort = shortVisionModel(cfg.model);
				const suffix =
					lastTurnStats.captioned > 0 || lastTurnStats.cached > 0 || lastTurnStats.failed > 0
						? ` 📷${lastTurnStats.captioned}:${lastTurnStats.cached}:${lastTurnStats.failed}`
						: "";
				ctx.ui.setStatus("vision", `vision:${modelShort}${suffix}`);
			}
		}
	}

	// ── Session lifecycle ─────────────────────────────────────────────────

	pi.on("session_start", async (_e, _ctx) => {
		imageStore = new ImageStore();
		notifyVisionChanged(_ctx);

		// Remote HTTP warning
		const cfg = loadConfig();
		const httpWarn = warnRemoteHttp(cfg.baseUrl);
		if (httpWarn && _ctx?.ui) {
			_ctx.ui.notify(httpWarn, "warning");
		}

		// First-run guidance: if no vision settings block exists, remind user
		// to set up. We check by re-reading and looking for a backend key.
		if (cfg.enabled && !cfg.backend) {
			_ctx?.ui?.notify(
				"👋 Welcome to pi-vision! Run /vision setup to configure a vision backend, then /vision test to verify.",
				"info",
			);
		}
	});

	pi.on("model_select", async (_e, _ctx) => notifyVisionChanged(_ctx));

	pi.on("session_shutdown", async (_e, _ctx) => {
		flushCacheSync();
		imageStore.clear();
	});

	// ── Tool registration ────────────────────────────────────────────────
	// NOTE: pass a getter for imageStore, not the direct reference, so that
	// session_start rebinding (imageStore = new ImageStore()) is reflected
	// inside the tool closure.
	registerFollowUpTool(pi, () => imageStore, (ctx) => resolveCaptioner(loadConfig(), ctx), () => loadConfig());

	// ── Command registration ─────────────────────────────────────────────

	registerVisionCommand(pi, () => lastTurnStats, notifyVisionChanged);
}
