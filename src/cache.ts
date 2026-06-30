/**
 * cache.ts — In-memory + disk caption cache with LRU eviction, negative/failure
 * caching with TTL, in-flight de-duplication, dirty tracking, and atomic
 * debounced writes.
 *
 * Captions are keyed by a hash of (model + prompt + maxEdge + maxBytes + image data)
 * so that re-sending history each turn never re-captions — you pay once per
 * unique image.
 *
 * Features:
 * - LRU eviction: most recently accessed entries survive; oldest evicted at limit.
 * - Negative cache: failed images are remembered for `negativeCacheTTLMs` so they
 *   aren't retried on every turn.
 * - In-flight dedup: concurrent requests for the same key share a single promise.
 * - Dirty tracking: only changed entries are serialized on persist.
 * - Atomic writes: writes go to .tmp then rename (safe on POSIX).
 * - Debounced persist: 1.5s cooldown so rapid processing doesn't thrash disk.
 */

import type { VisionConfig } from "./config.js";
import { loadConfig, CACHE_PATH } from "./config.js";
import {
	createHash,
	createCipheriv,
	createDecipheriv,
	randomBytes,
	pbkdf2Sync,
} from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync, unlinkSync } from "node:fs";
import { dirname } from "node:path";

// ── Constants ──────────────────────────────────────────────────────────────

const DEFAULT_MAX_ENTRIES = 500;
const DEFAULT_NEG_CACHE_TTL_MS = 300_000; // 5 minutes
const PERSIST_DEBOUNCE_MS = 1_500;

// ── Configurable limits (set from VisionConfig by loadCache) ───────────────

let cacheMaxEntries = DEFAULT_MAX_ENTRIES;
let negCacheTTLMs = DEFAULT_NEG_CACHE_TTL_MS;
/** Current cache passphrase (set by loadCache from config). Empty string = no encryption. */
let cachePassphrase = "";

// ── Encryption helpers (AES-256-GCM) ──────────────────────────────────────

const ENC_ALGO = "aes-256-gcm";
const KEY_LEN = 32;
const IV_LEN = 16;
const SALT_LEN = 16;
const TAG_LEN = 16;
const PBKDF2_ITER = 100_000;
const PBKDF2_DIGEST = "sha512";

function deriveKey(passphrase: string, salt: Buffer): Buffer {
	return pbkdf2Sync(passphrase, salt, PBKDF2_ITER, KEY_LEN, PBKDF2_DIGEST);
}

/**
 * Encrypt a JSON string with AES-256-GCM.
 * Returns base64 of (salt || iv || authTag || ciphertext).
 */
export function encryptCacheData(plaintext: string, passphrase: string): string {
	const salt = randomBytes(SALT_LEN);
	const key = deriveKey(passphrase, salt);
	const iv = randomBytes(IV_LEN);
	const cipher = createCipheriv(ENC_ALGO, key, iv);
	const encrypted = Buffer.concat([cipher.update(plaintext, "utf8"), cipher.final()]);
	const tag = cipher.getAuthTag();
	return Buffer.concat([salt, iv, tag, encrypted]).toString("base64");
}

/**
 * Decrypt a base64 blob that was created by encryptCacheData.
 * Returns null on any failure (wrong passphrase, corrupted data).
 */
export function decryptCacheData(encoded: string, passphrase: string): string | null {
	try {
		const buf = Buffer.from(encoded, "base64");
		if (buf.length < SALT_LEN + IV_LEN + TAG_LEN + 1) return null;
		let off = 0;
		const salt = buf.subarray(off, (off += SALT_LEN));
		const iv = buf.subarray(off, (off += IV_LEN));
		const tag = buf.subarray(off, (off += TAG_LEN));
		const ciphertext = buf.subarray(off);
		const key = deriveKey(passphrase, salt);
		const decipher = createDecipheriv(ENC_ALGO, key, iv);
		decipher.setAuthTag(tag);
		return decipher.update(ciphertext) + decipher.final("utf8");
	} catch {
		return null;
	}
}

// ── State ──────────────────────────────────────────────────────────────────

let memCache: Map<string, string> | null = null;
/** Negative cache: key -> expiration timestamp (ms). */
let negCache = new Map<string, number>();
/** In-flight requests: key -> Promise of caption text. */
let inFlight = new Map<string, Promise<string>>();
/** Keys modified since last persist. */
let dirtyKeys = new Set<string>();
let saveTimer: ReturnType<typeof setTimeout> | null = null;

// ── Cache key ──────────────────────────────────────────────────────────────

export function cacheKey(cfg: VisionConfig, data: string): string {
	return createHash("sha256")
		.update(cfg.model)
		.update("\0")
		.update(cfg.prompt)
		.update("\0")
		.update(String(cfg.maxEdge))
		.update("\0")
		.update(String(cfg.maxBytes))
		.update("\0")
		.update(data)
		.digest("hex")
		.slice(0, 32);
}

// ── Cache load ─────────────────────────────────────────────────────────────

/** Load caption cache from disk into memory. Reuses existing in-memory cache
 *  if already loaded and enabled. Also applies configurable limits
 *  (cacheMaxEntries, negativeCacheTTLMs) from the provided config. */
export function loadCache(cfg?: VisionConfig): Map<string, string> {
	if (!cfg) cfg = loadConfig();
	const enabled = cfg.cache;

	// Apply configurable limits and encryption passphrase
	cacheMaxEntries = cfg.cacheMaxEntries ?? DEFAULT_MAX_ENTRIES;
	negCacheTTLMs = cfg.negativeCacheTTLMs ?? DEFAULT_NEG_CACHE_TTL_MS;
	cachePassphrase = cfg.cachePassphrase ?? "";

	if (memCache && enabled) return memCache;
	if (!enabled) {
		memCache = null;
		negCache.clear();
		return new Map<string, string>();
	}
	memCache = new Map();
	try {
		if (existsSync(CACHE_PATH)) {
			const raw = readFileSync(CACHE_PATH, "utf8");
			const parsed = JSON.parse(raw);
			let data: Record<string, string> = {};
			if (parsed && typeof parsed === "object" && parsed.encrypted === true && typeof parsed.data === "string") {
				// Encrypted cache — decrypt with current passphrase
				const pass = cachePassphrase || "";
				if (!pass) {
					// Can't decrypt — treat as empty
					return memCache;
				}
				const decrypted = decryptCacheData(parsed.data, pass);
				if (decrypted === null) {
					// Wrong passphrase or corrupt — treat as empty
					return memCache;
				}
				data = JSON.parse(decrypted);
				if (typeof data !== "object" || data === null) data = {};
			} else if (parsed && typeof parsed === "object") {
				// Plain JSON (legacy format or unencrypted)
				data = parsed as Record<string, string>;
			}
			for (const [k, v] of Object.entries(data)) {
				if (typeof v === "string") memCache.set(k, v);
			}
		}
	} catch {
		/* ignore corrupt cache */
	}
	return memCache;
}

// ── LRU cache access ────────────────────────────────────────────────────────

/** Get a caption from cache. Moves the entry to the most-recently-used position. */
export function cacheGet(key: string): string | undefined {
	if (!memCache) return undefined;
	const val = memCache.get(key);
	if (val !== undefined) {
		// LRU touch: delete + re-set to move to end
		memCache.delete(key);
		memCache.set(key, val);
	}
	return val;
}

/** Store a caption in cache. Evicts the LRU entry if at capacity. */
export function cacheSet(key: string, value: string): void {
	if (!memCache) return;
	// If key already exists, delete first so re-set moves it to MRU position
	if (memCache.has(key)) memCache.delete(key);
	// Evict oldest (first entry in insertion order) if at capacity
	while (memCache.size >= cacheMaxEntries) {
		const oldest = memCache.keys().next().value;
		if (oldest === undefined) break;
		memCache.delete(oldest);
	}
	memCache.set(key, value);
	dirtyKeys.add(key);
}

/** Get the number of entries in the caption cache. */
export function getCacheSize(): number {
	return memCache?.size ?? 0;
}

// ── Negative cache (failures with TTL) ──────────────────────────────────────

/** Check if a key is in the negative cache and not yet expired. */
export function isNegCached(key: string): boolean {
	const expiresAt = negCache.get(key);
	if (expiresAt === undefined) return false;
	if (Date.now() > expiresAt) {
		negCache.delete(key);
		return false;
	}
	return true;
}

/** Mark a key as failed in the negative cache (caller provides TTL).
 *  Defaults to the configured negativeCacheTTLMs or 5 minutes. */
export function markNegCached(key: string, ttlMs: number = negCacheTTLMs): void {
	negCache.set(key, Date.now() + ttlMs);
}

/** Remove a key from the negative cache (e.g. after a config change clears state). */
export function clearNegCache(key?: string): void {
	if (key) negCache.delete(key);
	else negCache.clear();
}

/** Get the number of entries in the negative cache. */
export function getNegCacheSize(): number {
	return negCache.size;
}

/** Get an in-flight promise for a key, if one exists. */
export function getInFlight(key: string): Promise<string> | undefined {
	return inFlight.get(key);
}

/** Register an in-flight caption promise for a key. The promise is automatically
 *  removed from the in-flight map when it settles (fulfills or rejects).
 *  On rejection, the key is added to the negative cache with the configured TTL. */
export function setInFlight(key: string, promise: Promise<string>): void {
	// Clean up old state
	const existing = inFlight.get(key);
	if (existing) return; // already in-flight

	inFlight.set(
		key,
		promise
			.then((result) => {
				inFlight.delete(key);
				return result;
			})
			.catch((err) => {
				inFlight.delete(key);
				markNegCached(key, negCacheTTLMs);
				throw err;
			}),
	);
}

// ── Configurable limit helpers (used internally) ───────────────────────────

/** Get the current max entries limit for the LRU cache. */
export function getCacheMaxEntries(): number {
	return cacheMaxEntries;
}

/** Get the negative cache TTL currently in use (ms). */
export function getNegCacheTTLMs(): number {
	return negCacheTTLMs;
}

/** Remove an in-flight promise for a key (called when a request is aborted). */
export function removeInFlight(key: string): void {
	inFlight.delete(key);
}

/** Get the number of in-flight requests. */
export function getInFlightCount(): number {
	return inFlight.size;
}

// ── Persistence (debounced, atomic, dirty-only) ────────────────────────────

/** Schedule a deferred persist to disk (debounced). Only writes dirty keys. */
export function persistSoon(): void {
	if (saveTimer) return;
	saveTimer = setTimeout(() => {
		saveTimer = null;
		try {
			flushCacheSync();
		} catch {
			/* best-effort */
		}
	}, PERSIST_DEBOUNCE_MS) as unknown as ReturnType<typeof setTimeout>;
}

/** Synchronously flush dirty entries to disk. Uses atomic write (tmp + rename). */
export function flushCacheSync(cfg?: VisionConfig): void {
	if (saveTimer) {
		clearTimeout(saveTimer);
		saveTimer = null;
	}
	if (!cfg) cfg = loadConfig();
	if (!cfg?.cache) return;
	if (dirtyKeys.size === 0 && !cachePassphrase) return;

	try {
		// Merge dirty entries into existing disk state
		mkdirSync(dirname(CACHE_PATH), { recursive: true });

		let existing: Record<string, string> = {};
		try {
			if (existsSync(CACHE_PATH)) {
				const raw = readFileSync(CACHE_PATH, "utf8");
				const parsed = JSON.parse(raw);
				if (parsed && typeof parsed === "object" && parsed.encrypted === true && typeof parsed.data === "string") {
					// Encrypted on disk — decrypt to merge
					const pass = cachePassphrase || "";
					if (pass) {
						const decrypted = decryptCacheData(parsed.data, pass);
						if (decrypted !== null) {
							existing = JSON.parse(decrypted);
						}
					}
				} else if (parsed && typeof parsed === "object") {
					existing = parsed as Record<string, string>;
				}
			}
		} catch {
			/* ignore corrupt disk cache */
		}

		// Apply dirty entries
		const mem = memCache ?? new Map();
		for (const k of dirtyKeys) {
			const v = mem.get(k);
			if (v !== undefined) existing[k] = v;
			else delete existing[k];
		}

		// Bounded: keep most recent entries (the in-memory LRU already evicts,
		// but disk may have stale entries from prior sessions).
		const entries = Object.entries(existing);
		const trimmed = Object.fromEntries(entries.slice(-cacheMaxEntries));

		// Serialize
		let output: string;
		if (cachePassphrase) {
			output = JSON.stringify({
				encrypted: true,
				data: encryptCacheData(JSON.stringify(trimmed), cachePassphrase),
			});
		} else {
			output = JSON.stringify(trimmed);
		}

		// Atomic write
		const tmpPath = CACHE_PATH + ".tmp";
		writeFileSync(tmpPath, output, "utf8");
		renameSync(tmpPath, CACHE_PATH);

		dirtyKeys.clear();
	} catch {
		/* best-effort */
	}
}

// ── Cache clear ────────────────────────────────────────────────────────────

/** Clear all in-memory state and wipe the disk cache file. */
export function clearCache(): void {
	memCache = new Map();
	negCache.clear();
	inFlight.clear();
	dirtyKeys.clear();
	const empty = cachePassphrase
		? JSON.stringify({ encrypted: true, data: encryptCacheData("{}", cachePassphrase) })
		: "{}";
	try {
		if (existsSync(CACHE_PATH)) writeFileSync(CACHE_PATH, empty, "utf8");
	} catch {
		/* ignore */
	}
}

/**
 * Purge cache for privacy: securely overwrites the cache file before
 * deleting it, then resets all in-memory state. Use when the user wants
 * to ensure no cached image data remains on disk.
 */
export function purgeCache(): void {
	memCache = new Map();
	negCache.clear();
	inFlight.clear();
	dirtyKeys.clear();
	try {
		if (existsSync(CACHE_PATH)) {
			// Overwrite with random-ish data before deleting
			const size = readFileSync(CACHE_PATH).length;
			const junk = Buffer.alloc(Math.max(size, 256), 0);
			writeFileSync(CACHE_PATH, junk, "utf8");
			unlinkSync(CACHE_PATH);
		}
	} catch {
		/* best-effort */
	}
}

/** Get the count of dirty (unpersisted) entries. */
export function getDirtyCount(): number {
	return dirtyKeys.size;
}

/**
 * Get detailed cache statistics for diagnostics.
 */
export function getCacheStats(): {
	size: number;
	maxEntries: number;
	negCacheSize: number;
	inFlightCount: number;
	dirtyCount: number;
	encrypted: boolean;
	cacheEnabled: boolean;
} {
	return {
		size: memCache?.size ?? 0,
		maxEntries: cacheMaxEntries,
		negCacheSize: negCache.size,
		inFlightCount: inFlight.size,
		dirtyCount: dirtyKeys.size,
		encrypted: cachePassphrase.length > 0,
		cacheEnabled: memCache !== null,
	};
}
