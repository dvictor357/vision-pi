/**
 * vision-model.ts — Build the pi model object, invoke the vision API,
 * and format caption output.
 *
 * Pure-ish logic: no file I/O, no mutable state. The one side-effect is
 * calling `complete()` (the pi API) and `resizeImage()` (the pi SDK).
 *
 * Supports resolution via pi's ModelRegistry (for using any pi-configured
 * multimodal model as the captioner, with auth resolved through pi).
 */

import type { Api, Context, ImageContent, Model } from "@earendil-works/pi-ai";
import { complete } from "@earendil-works/pi-ai/compat";
import { resizeImage } from "@earendil-works/pi-coding-agent";
import type { VisionConfig } from "./config.js";

// ── Re-export types ────────────────────────────────────────────────────────

export type { Model };

// ── Bounded concurrency helper ─────────────────────────────────────────────

/**
 * Process an array of items with bounded concurrency while preserving
 * positional order in the result.
 *
 * @param items - Items to process.
 * @param concurrency - Max concurrent calls to `fn`.
 * @param fn - Async function to apply to each item (receives item and index).
 * @returns Results in the same order as `items`.
 */
export async function mapWithConcurrency<T, R>(
	items: T[],
	concurrency: number,
	fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
	if (items.length === 0) return [];
	const limit = Math.max(1, Math.min(concurrency, items.length));
	const results = new Array<R>(items.length);
	let next = 0;

	async function worker(): Promise<void> {
		while (next < items.length) {
			const i = next++;
			results[i] = await fn(items[i], i);
		}
	}

	const workers = Array.from({ length: limit }, () => worker());
	await Promise.all(workers);
	return results;
}

// ── Build the model descriptor ─────────────────────────────────────────────

export function buildVisionModel(cfg: VisionConfig): Model<Api> {
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

// ── Extract text from a completion response ────────────────────────────────

export function extractText(msg: { content: unknown }): string {
	const content = msg.content;
	if (typeof content === "string") return content.trim();
	if (!Array.isArray(content)) return "";
	return content
		.filter((c: any) => c?.type === "text" && typeof c.text === "string")
		.map((c: any) => c.text)
		.join("\n")
		.trim();
}

// ── Caption a single image ─────────────────────────────────────────────────

/** Run the vision model on one image, returning a description. */
export async function captionImage(
	img: ImageContent,
	cfg: VisionConfig,
	model: Model<Api>,
	parentSignal: AbortSignal | undefined,
	apiKeyOverride?: string,
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
			apiKey: apiKeyOverride ?? cfg.apiKey,
			temperature: 0.2,
			maxTokens: cfg.maxTokens,
			timeoutMs: cfg.timeoutMs,
			maxRetries: 1,
			signal: ctrl.signal,
		});
		const text = extractText(res as any);
		return text || "(vision model returned an empty description)";
	} finally {
		clearTimeout(timer);
		parentSignal?.removeEventListener("abort", onParentAbort);
	}
}

// ── Caption wrapping ───────────────────────────────────────────────────────

/**
 * Wrap a caption so the text-only model sees it as a description block.
 * When a `ref` is provided, it's included in the opening tag so the LLM can
 * reference the image in follow-up queries via `vision_analyze_image`.
 */
export function wrapCaption(cfg: VisionConfig, text: string, ref?: string): string {
	const refAttr = ref ? ` ref="${ref}"` : "";
	return (
		`[image-description${refAttr}] The active model cannot see images, so ${cfg.model} described this image:\n` +
		`${text}\n[/image-description]`
	);
}

// ── Model name shortening ───────────────────────────────────────────────────

/** Shorten a vision model id for compact status bar display. */
export function shortVisionModel(id: string): string {
	// Strip provider prefix (e.g. "mlx-community/")
	let s = id.includes("/") ? id.split("/").pop()! : id;
	// Strip trailing bit-width / quant suffixes
	s = s.replace(/-4bit$|-8bit$|-int4$|-int8$|-GPTQ$/i, "");
	// Strip trailing date/version hashes
	s = s.replace(/-[0-9a-f]{8,}$/, "").replace(/-\d{8,}$/, "");
	// For very long names, keep most meaningful segments
	if (s.length > 28) {
		const parts = s.split("-");
		const meaningful = parts.filter(p => !/^\d{4,}$/.test(p) && !/^[0-9a-f]{6,}$/i.test(p));
		s = meaningful.slice(0, 3).join("-");
	}
	return s;
}
