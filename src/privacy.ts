/**
 * privacy.ts — Privacy/security utilities: URL sanitization, localhost checks,
 * remote HTTP warnings, config sanitization for safe event emission.
 *
 * All functions are pure — no side effects, no file I/O.
 */

import type { VisionConfig } from "./config.js";

// ── URL sanitization ───────────────────────────────────────────────────────

/**
 * Strip credentials (user:pass@) and api_key query parameters from a URL
 * for safe display in logs, notifications, and events.
 */
export function sanitizeUrl(url: string): string {
	try {
		const parsed = new URL(url);
		if (parsed.username || parsed.password) {
			parsed.username = "***";
			parsed.password = "***";
		}
		// Strip common API key query parameters
		parsed.searchParams.delete("api_key");
		parsed.searchParams.delete("apikey");
		parsed.searchParams.delete("api-key");
		return parsed.toString();
	} catch {
		// Malformed URL — best-effort redaction
		return url.replace(/\/\/[^@]+@/, "//***@");
	}
}

// ── Localhost check ────────────────────────────────────────────────────────

/**
 * Return true if the URL hostname resolves to the local machine.
 * Covers: localhost, 127.0.0.1, [::1], *.local, *.localhost.
 */
export function isLocalhost(url: string): boolean {
	try {
		const host = new URL(url).hostname;
		return (
			host === "localhost" ||
			host === "127.0.0.1" ||
			host === "::1" ||
			host === "[::1]" ||
			host.endsWith(".local") ||
			host.endsWith(".localhost")
		);
	} catch {
		return false;
	}
}

// ── Remote HTTP warning ────────────────────────────────────────────────────

/**
 * Check whether a baseUrl is potentially unsafe (remote + plain HTTP).
 * Returns a human-readable warning string or null if the URL is safe.
 */
export function warnRemoteHttp(url: string): string | null {
	try {
		const parsed = new URL(url);
		if (isLocalhost(url)) return null;
		if (parsed.protocol === "http:") {
			return (
				`⚠️  Remote endpoint uses plain HTTP: ${sanitizeUrl(url)}\n` +
				"   Data sent to this server will NOT be encrypted in transit.\n" +
				"   Consider using HTTPS or a local server for sensitive images."
			);
		}
		return null;
	} catch {
		return null;
	}
}

// ── Config sanitization for events/notifications ───────────────────────────

/**
 * Return a safe copy of the config for event emission and notifications.
 * Secrets (apiKey, cachePassphrase) are stripped entirely.
 * baseUrl is sanitized (credentials removed).
 */
export function sanitizeConfig(
	cfg: VisionConfig,
): Record<string, unknown> {
	const safe: Record<string, unknown> = {};
	const skip = new Set(["apiKey", "cachePassphrase"]);
	for (const [k, v] of Object.entries(cfg)) {
		if (skip.has(k)) continue;
		if (k === "baseUrl") {
			safe[k] = sanitizeUrl(String(v));
		} else {
			safe[k] = v;
		}
	}
	return safe;
}

// ── EXIF / metadata stripping ──────────────────────────────────────────────

/**
 * MIME types that may carry EXIF or other sensitive metadata.
 */
const METADATA_FORMATS = new Set([
	"image/jpeg",
	"image/jpg",
	"image/tiff",
	"image/tif",
]);

/**
 * Attempt to strip EXIF/metadata from an image by re-encoding it as PNG.
 * Uses the pi SDK's resizeImage as a re-encoding pass.
 *
 * If re-encoding fails (e.g. unsupported format, missing pi SDK), the
 * original data is returned unchanged — this is best-effort.
 *
 * NOTE: the regular caption path ALREADY calls resizeImage for downscaling,
 * which implicitly strips metadata as a side effect. This function is an
 * explicit safety layer for code paths that skip resize (e.g. follow-up
 * tool re-uses stored images, which were already resized during captioning).
 */
export async function stripImageMetadata(
	data: string,
	mimeType: string,
): Promise<{ data: string; mimeType: string }> {
	const safeMime = mimeType.toLowerCase();
	if (!METADATA_FORMATS.has(safeMime)) {
		// PNG, WebP, GIF, BMP — no EXIF to strip
		return { data, mimeType };
	}

	// Re-encode as PNG (strips all JPEG/TIFF metadata)
	try {
		const { resizeImage } = await import("@earendil-works/pi-coding-agent");
		const bytes = Buffer.from(data, "base64");
		const resized = await resizeImage(bytes, mimeType, {
			maxWidth: 4096,
			maxHeight: 4096,
			maxBytes: 10_000_000, // generous — we're just re-encoding
		});
		if (resized) {
			return { data: resized.data, mimeType: resized.mimeType };
		}
	} catch {
		// best-effort; fall through to original
	}
	return { data, mimeType };
}
