/**
 * follow-up-tool.ts — Registers the `vision_analyze_image` tool that lets the
 * LLM ask targeted follow-up questions about a previously captioned image.
 *
 * The image must have been stored by the orchestrator's context handler with a
 * stable ref ID. The tool re-sends the stored image plus the question to the
 * configured captioner and returns the focused answer.
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { Model, Api } from "@earendil-works/pi-ai";
import { complete } from "@earendil-works/pi-ai/compat";
import { resizeImage } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import type { VisionConfig } from "./config.js";
import type { ImageStore } from "./image-store.js";

/**
 * Register the `vision_analyze_image` tool.
 *
 * @param pi - Extension API for tool registration.
 * @param getImageStore - Getter returning the active image store (per-session).
 *   Must be a getter (not a direct reference) so that session_start rebinding
 *   is reflected in the tool's closure.
 * @param resolveCaptioner - Function to resolve the captioner model + apiKey given context.
 * @param getCfg - Function to get the current config.
 */
export function registerFollowUpTool(
	pi: ExtensionAPI,
	getImageStore: () => ImageStore,
	resolveCaptioner: (ctx: ExtensionContext) => Promise<{ model: Model<Api>; apiKey: string }>,
	getCfg: () => VisionConfig,
): void {
	pi.registerTool({
		name: "vision_analyze_image",
		label: "Analyze Image",
		description:
			"Ask a targeted follow-up question about a specific image that was previously described. " +
			"Provide the exact `ref` from the [image-description] block and your question. " +
			"Use this to inspect details, read small text, check colors, or verify UI state.",
		promptSnippet:
			"- **vision_analyze_image**: ask a follow-up question about an image previously described in this session (use the `ref` from the [image-description ref=\"...\"] tag)",
		promptGuidelines: [
			"When you need more detail about an image than the initial description provided, use `vision_analyze_image` with the image's `ref` from the [image-description] block.",
			"The tool re-examines the original image, so you can ask about things that may have been missed or need closer inspection.",
		],
		parameters: Type.Object({
			ref: Type.String({
				description: "The ref ID from the [image-description ref=\"…\"] tag of the image to analyze.",
			}),
			question: Type.String({
				description:
					"Your specific, targeted question about this image. " +
					"Examples: 'What color is the submit button?', 'Read the error code in the red banner.', 'What is the value next to Total?', 'Describe the layout of the settings panel.'",
			}),
		}),
		execute: async (toolCallId, params, signal, _onUpdate, ctx) => {
			const { ref, question } = params;
			const cfg = getCfg();

			// 1. Look up the stored image (always resolved from getter so session_start
			//    rebinding of the store is reflected in the tool closure)
			const store = getImageStore();
			const stored = store.get(ref);
			if (!stored) {
				return {
					content: [
						{
							type: "text" as const,
							text:
								`[vision_analyze_image] Image ref "${ref}" not found. ` +
								`The image may have expired from the session store or was never captioned. ` +
								`Only images that were described with [image-description ref="..."] in this session can be queried. ` +
								`[/vision_analyze_image]`,
						},
					],
					details: null as any,
				};
			}

			// 2. Resolve captioner
			const { model: vModel, apiKey } = await resolveCaptioner(ctx);

			// 3. Downscale the stored image
			let data = stored.data;
			let mimeType = stored.mimeType;
			try {
				const bytes = Buffer.from(stored.data, "base64");
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
				/* best-effort resize; fall back to original */
			}

			// 4. Build a context with the question as the user message
			const context = {
				systemPrompt: "You are analyzing a specific aspect of an image. Answer concisely and factually.",
				messages: [
					{
						role: "user" as const,
						content: [
							{ type: "text" as const, text: question },
							{ type: "image" as const, data, mimeType },
						],
						timestamp: Date.now(),
					},
				],
			};

			// 5. Call the captioner
			const ctrl = new AbortController();
			const onParentAbort = () => ctrl.abort();
			signal?.addEventListener("abort", onParentAbort, { once: true });
			const timer = setTimeout(() => ctrl.abort(), cfg.timeoutMs);
			timer.unref?.();

			try {
				const res = await complete(vModel, context, {
					apiKey,
					temperature: 0.2,
					maxTokens: cfg.maxTokens,
					timeoutMs: cfg.timeoutMs,
					maxRetries: 1,
					signal: ctrl.signal,
				});

				const content = (res as any).content;
				let answer = "";
				if (typeof content === "string") {
					answer = content.trim();
				} else if (Array.isArray(content)) {
					answer = content
						.filter((c: any) => c?.type === "text" && typeof c.text === "string")
						.map((c: any) => c.text)
						.join("\n")
						.trim();
				}
				const result = answer || "(vision model returned an empty answer)";

				return {
					content: [
						{
							type: "text" as const,
							text: `[vision_analyze_image ref="${ref}"]\nQ: ${question}\nA: ${result}\n[/vision_analyze_image]`,
						},
					],
					details: null as any,
				};
			} catch (e) {
				const reason = e instanceof Error ? e.message : String(e);
				return {
					content: [
						{
							type: "text" as const,
							text:
								`[vision_analyze_image ref="${ref}"]\nQ: ${question}\n` +
								`Error: vision analysis failed — ${reason}\n` +
								`Captioner: ${vModel.id}. Run /vision test to diagnose.\n[/vision_analyze_image]`,
						},
					],
					details: { error: reason } as any,
				};
			} finally {
				clearTimeout(timer);
				signal?.removeEventListener("abort", onParentAbort);
			}
		},
	});
}
