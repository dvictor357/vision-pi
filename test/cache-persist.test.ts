/**
 * cache-persist.test.ts — Cache persistence: atomic writes, encrypted
 * file round-trip, clear/purge disk behavior.
 *
 * NOTE: PI_VISION_CACHE_PATH and PI_VISION_SETTINGS_PATH are captured at
 * import time in config.ts, so we use a SINGLE shared path per describe
 * block rather than per-test random paths.
 */

import assert from "node:assert";
import { describe, it, beforeEach, afterEach } from "node:test";
import { existsSync, readFileSync, writeFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

// ── Helpers ────────────────────────────────────────────────────────────────

function randomHex(): string {
	return Math.random().toString(16).slice(2, 10);
}

const SHARED_CACHE_PATH = join(tmpdir(), `vision-test-cache-${randomHex()}.json`);
const SHARED_SETTINGS_PATH = join(tmpdir(), `vision-test-settings-${randomHex()}.json`);

// Set paths BEFORE any module imports so the top-level CACHE_PATH/SETTINGS_PATH
// capture these values.
process.env.PI_VISION_CACHE_PATH = SHARED_CACHE_PATH;
process.env.PI_VISION_SETTINGS_PATH = SHARED_SETTINGS_PATH;

/**
 * Reset module-level cache state by re-loading cache with a fresh config.
 * This forces cachePassphrase and other module vars to be updated.
 */
async function resetCacheState(passphrase?: string) {
	const { loadCache } = await import("../src/cache.js");
	const { loadConfig } = await import("../src/config.js");
	if (passphrase) {
		process.env.PI_VISION_CACHE_PASSPHRASE = passphrase;
	} else {
		delete process.env.PI_VISION_CACHE_PASSPHRASE;
	}
	const config = loadConfig();
	loadCache(config);
}

// ── Cache persistence tests ────────────────────────────────────────────────

describe("cache persistence — atomic writes", () => {
	afterEach(async () => {
		// Reset passphrase so clearCache writes plain {} not encrypted
		await resetCacheState();
		const { clearCache } = await import("../src/cache.js");
		clearCache();
		try { unlinkSync(SHARED_CACHE_PATH); } catch { /* ok */ }
		try { unlinkSync(SHARED_CACHE_PATH + ".tmp"); } catch { /* ok */ }
	});

	it("flushCacheSync persists cache to disk as JSON", async () => {
		const { loadCache, cacheSet, flushCacheSync } = await import("../src/cache.js");
		const { loadConfig } = await import("../src/config.js");
		const config = loadConfig();
		loadCache(config);
		cacheSet("k1", "caption-one");
		cacheSet("k2", "caption-two");
		flushCacheSync(config);

		assert.ok(existsSync(SHARED_CACHE_PATH), "cache file should exist after flush");
		const raw = readFileSync(SHARED_CACHE_PATH, "utf8");
		const data = JSON.parse(raw);
		assert.strictEqual(data.k1, "caption-one");
		assert.strictEqual(data.k2, "caption-two");
	});

	it("flushCacheSync with encryption produces encrypted cache file", async () => {
		await resetCacheState("test-encryption-key");
		const { loadCache, cacheSet, flushCacheSync } = await import("../src/cache.js");
		const { loadConfig } = await import("../src/config.js");
		const config = loadConfig();
		loadCache(config);
		cacheSet("secret-key", "secret-caption");
		flushCacheSync(config);

		assert.ok(existsSync(SHARED_CACHE_PATH), "cache file should exist");
		const raw = readFileSync(SHARED_CACHE_PATH, "utf8");
		const parsed = JSON.parse(raw);
		assert.strictEqual(parsed.encrypted, true, "cache file should be marked encrypted");
		assert.ok(typeof parsed.data === "string", "cache data should be a base64 string");
		assert.ok(!parsed.data.includes("secret-caption"), "data should not contain plaintext");
	});

	it("loadCache reads encrypted cache with correct passphrase", async () => {
		// Write encrypted cache
		await resetCacheState("test-key");
		const { loadCache, cacheSet, flushCacheSync, clearCache } = await import("../src/cache.js");
		const { loadConfig } = await import("../src/config.js");
		const config = loadConfig();
		clearCache();
		loadCache(config);
		cacheSet("k1", "persistent-caption");
		flushCacheSync(config); // writes encrypted file

		// Force re-read from disk by temporarily disabling cache
		config.cache = false;
		loadCache(config); // sets memCache = null

		// Re-enable with same passphrase
		config.cache = true;
		const result = loadCache(config);
		assert.strictEqual(result.get("k1"), "persistent-caption", "should decrypt and retrieve caption");
	});

	it("loadCache with wrong passphrase returns empty cache", async () => {
		// Write with passphrase
		await resetCacheState("correct-key");
		const { loadCache, cacheSet, flushCacheSync, clearCache } = await import("../src/cache.js");
		const { loadConfig } = await import("../src/config.js");
		const config = loadConfig();
		clearCache();
		loadCache(config);
		cacheSet("k1", "secret");
		flushCacheSync(config);

		// Force re-read by disabling cache first, which sets memCache = null
		config.cache = false;
		loadCache(config);

		// Now reload with wrong passphrase
		await resetCacheState("wrong-key");
		const config2 = loadConfig();
		config2.cache = true;
		const result = loadCache(config2);
		assert.strictEqual(result.size, 0, "wrong passphrase should yield empty cache");
	});

	it("clearCache resets disk cache to empty", async () => {
		// Ensure no passphrase is set
		await resetCacheState();
		const { loadCache, cacheSet, flushCacheSync, clearCache } = await import("../src/cache.js");
		const { loadConfig, CACHE_PATH } = await import("../src/config.js");
		const config = loadConfig();
		loadCache(config);
		cacheSet("k1", "v1");
		flushCacheSync(config);

		clearCache();
		assert.ok(existsSync(CACHE_PATH), "cache file should still exist after clear");
		const raw = readFileSync(CACHE_PATH, "utf8");
		assert.strictEqual(raw, "{}", "cleared cache should be empty JSON object");
	});

	it("clearCache preserves encryption if passphrase is set", async () => {
		await resetCacheState("test-key");
		const { loadCache, cacheSet, flushCacheSync, clearCache, decryptCacheData } = await import("../src/cache.js");
		const { loadConfig, CACHE_PATH } = await import("../src/config.js");
		const config = loadConfig();
		loadCache(config);
		cacheSet("k1", "v1");
		flushCacheSync(config);

		clearCache();
		const raw = readFileSync(CACHE_PATH, "utf8");
		const parsed = JSON.parse(raw);
		assert.strictEqual(parsed.encrypted, true, "should remain encrypted after clear");
		const decrypted = decryptCacheData(parsed.data, "test-key");
		assert.strictEqual(decrypted, "{}", "decrypted cleared data should be empty object");
	});

	it("purgeCache removes the cache file from disk", async () => {
		await resetCacheState();
		const { loadCache, cacheSet, flushCacheSync, purgeCache } = await import("../src/cache.js");
		const { loadConfig, CACHE_PATH } = await import("../src/config.js");
		const config = loadConfig();
		loadCache(config);
		cacheSet("k1", "v1");
		flushCacheSync(config);
		assert.ok(existsSync(CACHE_PATH), "cache should exist before purge");

		purgeCache();
		assert.ok(!existsSync(CACHE_PATH), "cache should be deleted after purge");
	});

	it("dirty tracking clears after flush", async () => {
		await resetCacheState();
		const { loadCache, cacheSet, flushCacheSync, getDirtyCount } = await import("../src/cache.js");
		const { loadConfig } = await import("../src/config.js");
		const config = loadConfig();
		loadCache(config);
		cacheSet("k1", "v1");
		assert.strictEqual(getDirtyCount(), 1, "one dirty entry");

		flushCacheSync(config);
		assert.strictEqual(getDirtyCount(), 0, "dirty cleared after flush");
	});

	it("load+flush round-trip preserves all entries", async () => {
		await resetCacheState();
		const { loadCache, cacheSet, flushCacheSync } = await import("../src/cache.js");
		const { loadConfig } = await import("../src/config.js");
		const config = loadConfig();
		loadCache(config);
		for (let i = 0; i < 10; i++) cacheSet(`k${i}`, `caption${i}`);
		flushCacheSync(config);

		// Force re-read from disk by cycling cache
		config.cache = false;
		loadCache(config); // sets memCache = null
		config.cache = true;
		const result = loadCache(config);
		assert.strictEqual(result.size, 10, "all 10 entries should survive");
		for (let i = 0; i < 10; i++) {
			assert.strictEqual(result.get(`k${i}`), `caption${i}`, `k${i} should survive round-trip`);
		}
	});
});

// ── Cache with disabled caching ────────────────────────────────────────────

describe("cache — disabled caching", () => {
	beforeEach(async () => {
		process.env.PI_VISION_CACHE_PATH = SHARED_CACHE_PATH;
		process.env.PI_VISION_SETTINGS_PATH = SHARED_SETTINGS_PATH;
	});

	afterEach(async () => {
		await resetCacheState();
		const { clearCache } = await import("../src/cache.js");
		clearCache();
		try { unlinkSync(SHARED_CACHE_PATH); } catch { /* ok */ }
	});

	it("loadCache with cache:false returns empty and doesn't read disk", async () => {
		const { loadCache, clearCache } = await import("../src/cache.js");
		const { loadConfig } = await import("../src/config.js");

		// Write some data to disk first
		writeFileSync(SHARED_CACHE_PATH, JSON.stringify({ existing: "data" }), "utf8");

		process.env.PI_VISION_DISABLED = "1";
		const config = loadConfig();
		config.cache = false;
		clearCache();
		const result = loadCache(config);
		assert.strictEqual(result.size, 0, "disabled cache should be empty");
		assert.ok(!result.has("existing"), "should not load disk data");
	});
});
