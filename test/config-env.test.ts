/**
 * config-env.test.ts — Config loading with env var overrides,
 * settings persistence, and edge cases.
 *
 * NOTE: SETTINGS_PATH and CACHE_PATH are captured at module import time in
 * config.ts. We set shared paths at the module level before any import.
 */

import assert from "node:assert";
import { describe, it, beforeEach, afterEach } from "node:test";
import { existsSync, mkdirSync, writeFileSync, unlinkSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

// ── Helpers ────────────────────────────────────────────────────────────────

function setEnv(key: string, val: string | undefined) {
	if (val === undefined) delete process.env[key];
	else process.env[key] = val;
}

function randomHex(): string {
	return Math.random().toString(16).slice(2, 10);
}

interface EnvSnapshot {
	PI_VISION_BACKEND?: string;
	PI_VISION_MODEL?: string;
	PI_VISION_BASE_URL?: string;
	PI_VISION_API_KEY?: string;
	PI_VISION_DISABLED?: string;
	PI_VISION_CACHE_PASSPHRASE?: string;
	PI_VISION_SETTINGS_PATH?: string;
	PI_VISION_CACHE_PATH?: string;
}

function saveEnv(): EnvSnapshot {
	return {
		PI_VISION_BACKEND: process.env.PI_VISION_BACKEND,
		PI_VISION_MODEL: process.env.PI_VISION_MODEL,
		PI_VISION_BASE_URL: process.env.PI_VISION_BASE_URL,
		PI_VISION_API_KEY: process.env.PI_VISION_API_KEY,
		PI_VISION_DISABLED: process.env.PI_VISION_DISABLED,
		PI_VISION_CACHE_PASSPHRASE: process.env.PI_VISION_CACHE_PASSPHRASE,
		PI_VISION_SETTINGS_PATH: process.env.PI_VISION_SETTINGS_PATH,
		PI_VISION_CACHE_PATH: process.env.PI_VISION_CACHE_PATH,
	};
}

function restoreEnv(snapshot: EnvSnapshot): void {
	setEnv("PI_VISION_BACKEND", snapshot.PI_VISION_BACKEND);
	setEnv("PI_VISION_MODEL", snapshot.PI_VISION_MODEL);
	setEnv("PI_VISION_BASE_URL", snapshot.PI_VISION_BASE_URL);
	setEnv("PI_VISION_API_KEY", snapshot.PI_VISION_API_KEY);
	setEnv("PI_VISION_DISABLED", snapshot.PI_VISION_DISABLED);
	setEnv("PI_VISION_CACHE_PASSPHRASE", snapshot.PI_VISION_CACHE_PASSPHRASE);
	setEnv("PI_VISION_SETTINGS_PATH", snapshot.PI_VISION_SETTINGS_PATH);
	setEnv("PI_VISION_CACHE_PATH", snapshot.PI_VISION_CACHE_PATH);
}

// SHARED paths — SETTINGS_PATH and CACHE_PATH are captured at module import time,
// so all tests in this file share the same paths.
const SHARED_SETTINGS_PATH = join(tmpdir(), `vision-test-settings-${randomHex()}.json`);
const SHARED_CACHE_PATH = join(tmpdir(), `vision-test-cache-${randomHex()}.json`);

// ── Config env override tests ──────────────────────────────────────────────

describe("config — env overrides", () => {
	let envSnapshot: EnvSnapshot;

	beforeEach(() => {
		envSnapshot = saveEnv();
		setEnv("PI_VISION_SETTINGS_PATH", SHARED_SETTINGS_PATH);
		setEnv("PI_VISION_CACHE_PATH", SHARED_CACHE_PATH);
	});

	afterEach(() => {
		restoreEnv(envSnapshot);
		try { unlinkSync(SHARED_SETTINGS_PATH); } catch { /* ok */ }
	});

	it("loadConfig respects PI_VISION_BACKEND env var", async () => {
		setEnv("PI_VISION_BACKEND", "ollama");
		const { loadConfig } = await import("../src/config.js");
		const cfg = loadConfig();
		assert.strictEqual(cfg.backend, "ollama");
	});

	it("loadConfig respects PI_VISION_MODEL env var", async () => {
		setEnv("PI_VISION_MODEL", "custom-model-v1");
		const { loadConfig } = await import("../src/config.js");
		const cfg = loadConfig();
		assert.strictEqual(cfg.model, "custom-model-v1");
	});

	it("loadConfig respects PI_VISION_BASE_URL env var", async () => {
		setEnv("PI_VISION_BASE_URL", "http://example.com:8080/v1");
		const { loadConfig } = await import("../src/config.js");
		const cfg = loadConfig();
		assert.strictEqual(cfg.baseUrl, "http://example.com:8080/v1");
	});

	it("loadConfig respects PI_VISION_API_KEY env var", async () => {
		setEnv("PI_VISION_API_KEY", "sk-test-key-12345");
		const { loadConfig } = await import("../src/config.js");
		const cfg = loadConfig();
		assert.strictEqual(cfg.apiKey, "sk-test-key-12345");
	});

	it("loadConfig respects PI_VISION_DISABLED=1", async () => {
		setEnv("PI_VISION_DISABLED", "1");
		const { loadConfig } = await import("../src/config.js");
		const cfg = loadConfig();
		assert.strictEqual(cfg.enabled, false);
	});

	it("loadConfig respects PI_VISION_DISABLED=true", async () => {
		setEnv("PI_VISION_DISABLED", "true");
		const { loadConfig } = await import("../src/config.js");
		const cfg = loadConfig();
		assert.strictEqual(cfg.enabled, false);
	});

	it("loadConfig respects PI_VISION_CACHE_PASSPHRASE env var", async () => {
		setEnv("PI_VISION_CACHE_PASSPHRASE", "my-secret-passphrase");
		const { loadConfig } = await import("../src/config.js");
		const cfg = loadConfig();
		assert.strictEqual(cfg.cachePassphrase, "my-secret-passphrase");
	});

	it("env var overrides settings.json", async () => {
		// Write settings first
		writeFileSync(
			SHARED_SETTINGS_PATH,
			JSON.stringify({ vision: { backend: "ollama", model: "from-settings" } }, null, 2),
			"utf8",
		);
		setEnv("PI_VISION_MODEL", "from-env"); // Env wins over settings

		const { loadConfig } = await import("../src/config.js");
		const cfg = loadConfig();
		assert.strictEqual(cfg.backend, "ollama"); // from settings (not overridden by env)
		assert.strictEqual(cfg.model, "from-env"); // env wins
	});

	it("invalid backend in env falls back to default", async () => {
		setEnv("PI_VISION_BACKEND", "nonexistent-backend");
		const { loadConfig } = await import("../src/config.js");
		const cfg = loadConfig();
		// isBackend("nonexistent-backend") is false, so it falls back to DEFAULT_BACKEND (mlx)
		assert.strictEqual(cfg.backend, "mlx");
	});
});

// ── Config settings persistence ────────────────────────────────────────────

describe("config — settings persistence", () => {
	let envSnapshot: EnvSnapshot;

	beforeEach(() => {
		envSnapshot = saveEnv();
		setEnv("PI_VISION_SETTINGS_PATH", SHARED_SETTINGS_PATH);
		setEnv("PI_VISION_CACHE_PATH", SHARED_CACHE_PATH);
	});

	afterEach(() => {
		restoreEnv(envSnapshot);
		try { unlinkSync(SHARED_SETTINGS_PATH); } catch { /* ok */ }
	});

	it("saveConfig writes to settings.json", async () => {
		const { saveConfig, loadConfig } = await import("../src/config.js");
		saveConfig({ backend: "ollama", preset: "balanced" });
		const cfg = loadConfig();
		assert.strictEqual(cfg.backend, "ollama");
	});

	it("saveConfig preserves other settings keys", async () => {
		writeFileSync(
			SHARED_SETTINGS_PATH,
			JSON.stringify({ otherKey: "should-survive", vision: {} }, null, 2),
			"utf8",
		);
		const { saveConfig } = await import("../src/config.js");
		saveConfig({ backend: "ollama" });
		const raw = readFileSync(SHARED_SETTINGS_PATH, "utf8");
		const parsed = JSON.parse(raw);
		assert.strictEqual(parsed.otherKey, "should-survive");
	});

	it("saveConfig with null deletes the key", async () => {
		const { saveConfig, loadConfig } = await import("../src/config.js");
		saveConfig({ backend: "ollama", model: "temp-model" });
		// Now delete model
		saveConfig({ model: null });
		const cfg = loadConfig();
		// model should fall back to default (from preset light → moondream for ollama)
		assert.notStrictEqual(cfg.model, "temp-model");
	});

	it("readVisionSettings returns empty object when no vision block", async () => {
		const { readVisionSettings } = await import("../src/config.js");
		writeFileSync(SHARED_SETTINGS_PATH, JSON.stringify({}, null, 2), "utf8");
		const result = readVisionSettings();
		assert.deepStrictEqual(result, {});
	});

	it("readVisionSettings returns vision block when present", async () => {
		const { readVisionSettings } = await import("../src/config.js");
		writeFileSync(
			SHARED_SETTINGS_PATH,
			JSON.stringify({ vision: { backend: "mlx" } }, null, 2),
			"utf8",
		);
		const result = readVisionSettings();
		assert.deepStrictEqual(result, { backend: "mlx" });
	});

	it("loadConfig falls back to defaults for missing keys", async () => {
		const { loadConfig } = await import("../src/config.js");
		writeFileSync(SHARED_SETTINGS_PATH, JSON.stringify({ vision: {} }, null, 2), "utf8");
		const cfg = loadConfig();
		assert.strictEqual(cfg.enabled, true);
		assert.strictEqual(cfg.maxEdge, 1024);
		assert.strictEqual(cfg.timeoutMs, 120_000);
	});

	it("registeredBackends persists through save/load cycle", async () => {
		const { saveConfig, loadConfig, readVisionSettings } = await import("../src/config.js");
		saveConfig({
			registeredBackends: {
				"my-custom": { baseUrl: "http://my-server:8080/v1", apiKey: "my-key" },
			},
		});
		const raw = readVisionSettings();
		assert.ok(raw.registeredBackends);
		assert.strictEqual(raw.registeredBackends["my-custom"].baseUrl, "http://my-server:8080/v1");

		const cfg = loadConfig();
		assert.strictEqual(cfg.registeredBackends["my-custom"]?.baseUrl, "http://my-server:8080/v1");
	});
});

// ── Config interaction with backend registry ──────────────────────────────

describe("config — backend registry interaction", () => {
	let envSnapshot: EnvSnapshot;

	beforeEach(() => {
		envSnapshot = saveEnv();
		setEnv("PI_VISION_SETTINGS_PATH", SHARED_SETTINGS_PATH);
		setEnv("PI_VISION_CACHE_PATH", SHARED_CACHE_PATH);
	});

	afterEach(() => {
		restoreEnv(envSnapshot);
		try { unlinkSync(SHARED_SETTINGS_PATH); } catch { /* ok */ }

		// Clean up test backends from the singleton registry
		import("../src/backends.js").then(({ visionBackendRegistry }) => {
			visionBackendRegistry.unregister("test-custom-registry");
		});
	});

	it("loadConfig registers custom backends from settings", async () => {
		const { loadConfig } = await import("../src/config.js");
		const { visionBackendRegistry, isBackend } = await import("../src/backends.js");

		// Write settings with registered backends
		writeFileSync(
			SHARED_SETTINGS_PATH,
			JSON.stringify(
				{
					vision: {
						registeredBackends: {
							"test-custom-registry": { baseUrl: "http://custom:8080/v1", apiKey: "test-key" },
						},
					},
				},
				null,
				2,
			),
			"utf8",
		);

		loadConfig();
		assert.ok(isBackend("test-custom-registry"), "custom backend should be in registry after loadConfig");
		const def = visionBackendRegistry.get("test-custom-registry");
		assert.strictEqual(def?.baseUrl, "http://custom:8080/v1");
		assert.strictEqual(def?.apiKey, "test-key");
		assert.ok(def?.isCustom);
	});
});
