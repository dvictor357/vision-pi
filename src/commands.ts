/**
 * commands.ts — The `/vision` command handler.
 *
 * Extracted from the monolith so it can be tested independently. Relies on
 * injected callbacks for mutable state access (stats, notifications) to avoid
 * circular dependencies with orchestrator.ts.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { loadConfig, saveConfig, readVisionSettings, SETTINGS_PATH } from "./config.js";
import {
	BACKENDS,
	PRESET_NAMES,
	DEFAULT_BACKEND,
	isBackend,
	isPreset,
	BUILTIN_BACKEND_NAMES,
	visionBackendRegistry,
} from "./backends.js";
import { loadCache, getCacheSize, getCacheStats, clearCache, purgeCache } from "./cache.js";
import { buildVisionModel, captionImage } from "./vision-model.js";
import { sanitizeUrl, isLocalhost, warnRemoteHttp } from "./privacy.js";

// ── Types ──────────────────────────────────────────────────────────────────

export interface VisionStats {
	captioned: number;
	cached: number;
	failed: number;
}

// ── Command registration ───────────────────────────────────────────────────

export function registerVisionCommand(
	pi: ExtensionAPI,
	getStats: () => VisionStats,
	onChanged: (ctx?: { ui: { setStatus: (key: string, text: string | undefined) => void } }) => void,
): void {
	pi.registerCommand("vision", {
		description: "Image understanding for text-only models — backend, preset, model-ref, test, doctor",
		handler: async (args, ctx) => {
			const [sub, ...rest] = args.trim().split(/\s+/);
			const arg = rest.join(" ").trim();
			const cfg = loadConfig();

			switch (sub) {
				case "":
				case "status": {
					const model = ctx.model;
					const inputTypes = model && Array.isArray(model.input) ? model.input : [];
					const textOnly = inputTypes.length > 0 && !inputTypes.includes("image");
					const multimodal = inputTypes.includes("image");
					const stats = getStats();
					const cacheInfo = getCacheStats();
					const def = visionBackendRegistry.get(cfg.backend);
					const httpWarn = warnRemoteHttp(cfg.baseUrl);
					const usingHttps = cfg.baseUrl.startsWith("https://");
					const isLocal = isLocalhost(cfg.baseUrl);

					const lines: string[] = [
						`vision: ${cfg.enabled ? "✅ on" : "⛔ off"}`,
						`captioner: ${cfg.modelRef === "backend" ? `backend (${cfg.backend})` : cfg.modelRef}`,
						`model: ${cfg.model}`,
						`backend: ${def?.label ?? cfg.backend}`,
						`endpoint: ${sanitizeUrl(cfg.baseUrl)} ${!isLocal && !usingHttps ? "⚠️  HTTP" : isLocal ? "🔒 local" : "🔒 HTTPS"}`,
						``,
						`active model: ${model?.id ?? "?"} → ${
							multimodal ? "multimodal (vision passthrough)" : textOnly ? "text-only (vision ACTIVE)" : "unknown"
						}`,
						``,
						`caption cache: ${cacheInfo.size}/${cacheInfo.maxEntries} entries ${cacheInfo.encrypted ? "🔒 encrypted" : ""}`,
						`negative cache: ${cacheInfo.negCacheSize} entries`,
						`in-flight: ${cacheInfo.inFlightCount} · dirty: ${cacheInfo.dirtyCount}`,
						``,
						`last turn: ${stats.captioned} captioned · ${stats.cached} cached · ${stats.failed} failed`,
					];
					if (httpWarn) lines.push("", httpWarn);
					ctx.ui.notify(lines.join("\n"), "info");
					return;
				}
				case "on":
					saveConfig({ enabled: true });
					ctx.ui.notify("Vision enabled.", "info");
					onChanged(ctx);
					return;
				case "off":
					saveConfig({ enabled: false });
					ctx.ui.notify("Vision disabled.", "info");
					onChanged(ctx);
					return;
				case "backend": {
					if (!arg) {
						ctx.ui.notify(
							`Usage: /vision backend <${visionBackendRegistry.names().join("|")}> or /vision backend register <name> <baseUrl> [apiKey]`,
							"error",
						);
						return;
					}
					const [backendSub, ...backendRest] = arg.split(/\s+/);
					if (backendSub === "register") {
						const name = backendRest[0];
						const baseUrl = backendRest[1];
						const apiKey = backendRest[2] ?? "";
						if (!name || !baseUrl) {
							ctx.ui.notify("Usage: /vision backend register <name> <baseUrl> [apiKey]", "error");
							return;
						}
						const httpWarn = warnRemoteHttp(baseUrl);
						visionBackendRegistry.register(name, {
							label: name,
							baseUrl,
							apiKey,
							presets: { light: "gpt-4o" },
							defaultPreset: "light",
							isCustom: true,
							setup: (model) => [
								`Custom backend "${name}":`,
								`  Server: ${sanitizeUrl(baseUrl)}`,
								`  Model: ${model}`,
								"  Ensure your server is running and reachable.",
								"  /vision test",
							],
						});
						// Persist the custom backend
						const existing = readVisionSettings().registeredBackends ?? {};
						saveConfig({
							registeredBackends: {
								...existing,
								[name]: { baseUrl, apiKey: apiKey || undefined },
							},
						});
						const msg = `Registered custom backend "${name}" → ${sanitizeUrl(baseUrl)}` + (httpWarn ? `\n${httpWarn}` : "");
						ctx.ui.notify(msg, httpWarn ? "warning" : "info");
						onChanged(ctx);
						return;
					}
					if (backendSub === "unregister") {
						const name = backendRest[0];
						if (!name) {
							ctx.ui.notify("Usage: /vision backend unregister <name>", "error");
							return;
						}
						const removed = visionBackendRegistry.unregister(name);
						if (!removed) {
							ctx.ui.notify(
								`Cannot unregister "${name}": not found or is built-in (${[...BUILTIN_BACKEND_NAMES].join(", ")}).`,
								"error",
							);
							return;
						}
						const existing = readVisionSettings().registeredBackends ?? {};
						const updated = { ...existing };
						delete updated[name];
						saveConfig({ registeredBackends: updated });
						ctx.ui.notify(`Unregistered custom backend "${name}".`, "info");
						onChanged(ctx);
						return;
					}
					if (backendSub === "list") {
						const backends = visionBackendRegistry.list();
						const lines = backends.map(
							(b) =>
								`  ${b.name ?? b.label}${b.isCustom ? " (custom)" : " (built-in)"} — ${sanitizeUrl(b.baseUrl)}` +
								(b.presets ? ` [presets: ${Object.keys(b.presets).join(", ")}]` : ""),
						);
						ctx.ui.notify(
							[`Registered backends (${backends.length}):`, ...lines].join("\n"),
							"info",
						);
						return;
					}
					// Switch to a backend by name
					const backendName = arg;
					if (!isBackend(backendName)) {
						ctx.ui.notify(
							`Unknown backend "${backendName}". Use /vision backend list to see available backends.`,
							"error",
						);
						return;
					}
					const def = visionBackendRegistry.get(backendName);
					if (!def) {
						ctx.ui.notify(`Backend "${backendName}" not found in registry.`, "error");
						return;
					}
					// Switching backend clears any pinned endpoint/model so the new
					// backend's defaults (and current preset) take effect cleanly.
					saveConfig({ backend: backendName, baseUrl: null, apiKey: null, model: null });
					const next = loadConfig();
					ctx.ui.notify(
						[`Vision backend → ${def.label}`, "", ...def.setup(next.model)].join("\n"),
						"info",
					);
					onChanged(ctx);
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
					const def = visionBackendRegistry.get(next.backend);
					ctx.ui.notify(
						[
							`Vision preset → ${arg} (${next.model})`,
							"",
							...(def?.setup(next.model) ?? []),
						].join("\n"),
						"info",
					);
					onChanged(ctx);
					return;
				}
				case "model": {
					if (!arg) {
						ctx.ui.notify("Usage: /vision model <model-id>", "error");
						return;
					}
					saveConfig({ model: arg });
					ctx.ui.notify(`Vision model → ${arg}`, "info");
					onChanged(ctx);
					return;
				}
				case "model-ref": {
					if (!arg) {
						ctx.ui.notify(
							"Usage: /vision model-ref <backend|provider/model-id>\n" +
							'  "backend"  — use configured backend + preset (default)\n' +
							'  "openai/gpt-4o" — use model from pi registry',
							"error",
						);
						return;
					}
					if (arg === "backend") {
						saveConfig({ modelRef: null });
						ctx.ui.notify("Vision captioner → backend (configured preset).", "info");
					} else {
						// Validate the reference looks like provider/model
						if (!arg.includes("/")) {
							ctx.ui.notify(
								'model-ref must be "backend" or "provider/model-id" (e.g. "openai/gpt-4o")',
								"error",
							);
							return;
						}
						saveConfig({ modelRef: arg });
						ctx.ui.notify(
							`Vision captioner → ${arg} (from pi model registry). Run /vision test to verify.`,
							"info",
						);
					}
					onChanged(ctx);
					return;
				}
				case "clear":
					clearCache();
					ctx.ui.notify("Vision caption cache cleared.", "info");
					return;
				case "purge":
					purgeCache();
					ctx.ui.notify(
						"Vision cache purged (securely overwritten and deleted from disk).",
						"info",
					);
					return;
				case "setup": {
					const isFirstRun = !readVisionSettings()?.backend;
					if (isFirstRun) {
						saveConfig({
							enabled: true,
							backend: cfg.backend,
							preset: "light",
							maxEdge: cfg.maxEdge,
							timeoutMs: cfg.timeoutMs,
							maxImagesPerTurn: cfg.maxImagesPerTurn,
						});
					}
					const def = visionBackendRegistry.get(cfg.backend);
					const httpWarn = warnRemoteHttp(cfg.baseUrl);
					const lines: string[] = [];
					if (isFirstRun) {
						lines.push(
							"👋 First run — welcome to pi-vision!",
							"",
							"Quick start:",
							"  1. Install a vision backend then start it:",
						);
					} else {
						lines.push(`Config: ${SETTINGS_PATH} → "vision" block`);
						lines.push(`Backend: ${def?.label ?? cfg.backend}`);
						lines.push("");
					}
					if (def?.setup) lines.push(...def.setup(cfg.model));
					lines.push("");
					if (isFirstRun) {
						lines.push("Then run:", "  /vision test", "");
					}
					lines.push(
						`Switch backend:  /vision backend <${visionBackendRegistry.names().join("|")}>`,
						"Pin a model:     /vision model <id>  (or edit settings.json → vision.model)",
						"Better quality:  /vision preset balanced",
						"Model registry:  /vision model-ref <provider/model-id>  (e.g. openai/gpt-4o)",
					);
					if (httpWarn) lines.push("", httpWarn);
					ctx.ui.notify(lines.join("\n"), "info");
					return;
				}
				case "test": {
					const safeUrl = sanitizeUrl(cfg.baseUrl);
					ctx.ui.notify(
						`Testing ${cfg.model} at ${safeUrl}${cfg.apiKey ? " (with API key)" : " (no API key)"} …`,
						"info",
					);
					// 8×8 red PNG — just checks connectivity + that the model answers.
					const testPng =
						"iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAlElEQVR4nO3QMREAMBDDsPAn/YWhoR60+7zb7mfTAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA9DiOHSbdjxEgAAAABJRU5ErkJggg==";
					const start = Date.now();
					try {
						const text = await captionImage(
							{ type: "image", data: testPng, mimeType: "image/png" },
							cfg,
							buildVisionModel(cfg),
							ctx.signal,
						);
						const ms = Date.now() - start;
						const wordCount = text.split(/\s+/).filter(Boolean).length;
						ctx.ui.notify(
							[
								`✅ Vision OK`,
								`  response time: ${ms}ms`,
								`  words: ${wordCount}`,
								`  model response (first 200 chars): ${text.slice(0, 200)}`,
							].join("\n"),
							"info",
						);
					} catch (e) {
						const reason = e instanceof Error ? e.message : String(e);
						const def = visionBackendRegistry.get(cfg.backend);
						const setupLines = def?.setup(cfg.model) ?? [];
						// Security: sanitize error messages that may contain URLs
						const safeReason = sanitizeUrl(reason);
						ctx.ui.notify(
							[
								`❌ Vision failed`,
								`  reason: ${safeReason}`,
								`  backend: ${def?.label ?? cfg.backend} — ${safeUrl}`,
								`  model: ${cfg.model}`,
								"",
								"Troubleshooting:",
								"  - Is the server running?",
								"  - Is the model pulled/loaded?",
								"  - Is the endpoint URL correct?",
								...setupLines,
							].join("\n"),
							"error",
						);
					}
					return;
				}
				case "doctor": {
					// Comprehensive diagnostics
					const model = ctx.model;
					const stats = getStats();
					const cacheInfo = getCacheStats();
					const def = visionBackendRegistry.get(cfg.backend);
					const inputTypes = model && Array.isArray(model.input) ? model.input : [];
					const diagnostics: string[] = [
						"🩺 Vision Doctor — Full Diagnostics",
						"",
						"── Config ──",
						`vision enabled: ${cfg.enabled ? "yes" : "no"}`,
						`backend: ${cfg.backend} (${def?.label ?? "unknown"})`,
						`endpoint: ${sanitizeUrl(cfg.baseUrl)}`,
						`model: ${cfg.model}`,
						`captioner mode: ${cfg.modelRef === "backend" ? "backend+preset" : cfg.modelRef}`,
						`api key set: ${cfg.apiKey ? "yes" : "no"}`,
						"",
						"── Pi Model ──",
						`active model: ${model?.id ?? "none"}`,
						`supports images: ${inputTypes.includes("image") ? "yes (passthrough)" : "no (vision active)"}`,
						"",
						"── Connection Check ──",
					];

					// Endpoint security check
					const httpWarn = warnRemoteHttp(cfg.baseUrl);
					if (httpWarn) {
						diagnostics.push(`⚠️  ${httpWarn}`);
					} else if (isLocalhost(cfg.baseUrl)) {
						diagnostics.push("✅ endpoint is localhost (secure)");
					} else {
						diagnostics.push("✅ endpoint uses HTTPS");
					}

					// Backend label
					if (def) {
						diagnostics.push(`✅ backend "${cfg.backend}" is registered`);
					} else {
						diagnostics.push(`❌ backend "${cfg.backend}" NOT registered`);
					}

					diagnostics.push(
						"",
						"── Cache ──",
						`caption cache: ${cacheInfo.size}/${cacheInfo.maxEntries} entries`,
						`encrypted: ${cacheInfo.encrypted ? "yes 🔒" : "no"}`,
						`negative cache: ${cacheInfo.negCacheSize} entries`,
						`in-flight: ${cacheInfo.inFlightCount}`,
						`dirty (unpersisted): ${cacheInfo.dirtyCount}`,
						"",
						"── Session ──",
						`images stored: ${stats.captioned} captioned · ${stats.cached} cached · ${stats.failed} failed (last turn)`,
						`encrypted cache passphrase: ${cfg.cachePassphrase ? "set 🔒" : "not set"}`,
					);

					// Add setup guidance if useful
					if (!cfg.enabled) {
						diagnostics.push("", "💡 Vision is disabled. Run /vision on or /vision setup to enable.");
					} else if (!def) {
						diagnostics.push(
							"",
							"💡 No backend configured. Run /vision setup to get started.",
						);
					} else if (inputTypes.includes("image")) {
						diagnostics.push(
							"",
							"ℹ️  Active model already supports images — vision is in passthrough mode.",
						);
					}

					ctx.ui.notify(diagnostics.join("\n"), "info");
					return;
				}
				default:
					ctx.ui.notify(
						"Usage: /vision [status|on|off|backend|preset|model|model-ref|test|clear|purge|setup|doctor]",
						"error",
					);
			}
		},
	});
}
