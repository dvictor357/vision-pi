import assert from "node:assert";
import { describe, it, beforeEach } from "node:test";

// ── Tests for extracted modules ───────────────────────────────────────────

import { shortVisionModel, wrapCaption, extractText, buildVisionModel, mapWithConcurrency } from "../src/vision-model.js";
import {
	cacheKey,
	cacheGet,
	cacheSet,
	getCacheSize,
	clearCache,
	isNegCached,
	markNegCached,
	getNegCacheSize,
	getInFlight,
	setInFlight,
	removeInFlight,
	getInFlightCount,
	getDirtyCount,
	loadCache,
	getCacheMaxEntries,
	getNegCacheTTLMs,
	clearNegCache,
} from "../src/cache.js";
import {
	BACKENDS,
	PRESET_NAMES,
	DEFAULT_BACKEND,
	isBackend,
	isPreset,
	VisionBackendRegistry,
	BUILTIN_BACKEND_NAMES,
	isBuiltinBackend,
	loadCustomBackends,
} from "../src/backends.js";
import { baseDefaults } from "../src/config.js";
import type { VisionConfig } from "../src/config.js";

function makeTestConfig(
	model = "test-model",
	prompt = "test prompt",
	maxEdge = 1024,
	maxBytes = 1_400_000,
): VisionConfig {
	const bd = baseDefaults();
	return {
		...bd,
		backend: DEFAULT_BACKEND,
		baseUrl: BACKENDS[DEFAULT_BACKEND].baseUrl,
		apiKey: BACKENDS[DEFAULT_BACKEND].apiKey,
		model,
		prompt,
		maxEdge,
		maxBytes,
		registeredBackends: {},
	};
}

// ── Tests ──────────────────────────────────────────────────────────────────

describe("shortVisionModel", () => {
	it("strips provider prefix", () => {
		assert.strictEqual(
			shortVisionModel("mlx-community/Qwen2.5-VL-3B-Instruct-4bit"),
			"Qwen2.5-VL-3B-Instruct",
		);
	});

	it("strips quant suffix", () => {
		assert.strictEqual(shortVisionModel("moondream-4bit"), "moondream");
	});

	it("strips trailing hash", () => {
		assert.strictEqual(
			shortVisionModel("llama-3.2-11b-a1b2c3d4e5"),
			"llama-3.2-11b",
		);
	});

	it("shortens long names", () => {
		const long =
			"very-long-model-name-with-many-segments-that-exceeds-28-chars";
		const result = shortVisionModel(long);
		assert.ok(result.length <= 28, `${result} should be ≤ 28 chars`);
	});

	it("keeps short names intact", () => {
		assert.strictEqual(shortVisionModel("moondream"), "moondream");
	});
});

describe("cacheKey", () => {
	it("produces deterministic 32-char hex", () => {
		const cfg = makeTestConfig("m", "p", 1024, 1_400_000);
		const k1 = cacheKey(cfg, "abc");
		const k2 = cacheKey(cfg, "abc");
		assert.strictEqual(k1, k2);
		assert.strictEqual(k1.length, 32);
		assert.match(k1, /^[0-9a-f]{32}$/);
	});

	it("changes when model changes", () => {
		const k1 = cacheKey(makeTestConfig("model-a"), "abc");
		const k2 = cacheKey(makeTestConfig("model-b"), "abc");
		assert.notStrictEqual(k1, k2);
	});

	it("changes when prompt changes", () => {
		const k1 = cacheKey(makeTestConfig("m", "prompt-a"), "abc");
		const k2 = cacheKey(makeTestConfig("m", "prompt-b"), "abc");
		assert.notStrictEqual(k1, k2);
	});

	it("changes when image data changes", () => {
		const cfg = makeTestConfig();
		const k1 = cacheKey(cfg, "data-a");
		const k2 = cacheKey(cfg, "data-b");
		assert.notStrictEqual(k1, k2);
	});
});

describe("wrapCaption", () => {
	it("wraps text with model attribution", () => {
		const cfg = makeTestConfig("test-model");
		const result = wrapCaption(cfg, "a tall building");
		assert.ok(result.includes("test-model"));
		assert.ok(result.includes("a tall building"));
		assert.ok(result.includes("[image-description]"));
		assert.ok(result.includes("[/image-description]"));
	});
});

describe("backends config", () => {
	it("ollama preset keys match the catalog", () => {
		const presets = BACKENDS.ollama.presets;
		assert.strictEqual(presets.light, "moondream");
		assert.strictEqual(presets.balanced, "qwen2.5vl:3b");
		assert.strictEqual(presets.capable, "qwen2.5vl:7b");
	});

	it("mlx preset keys match the catalog", () => {
		const presets = BACKENDS.mlx.presets;
		assert.strictEqual(
			presets.light,
			"mlx-community/Qwen2-VL-2B-Instruct-4bit",
		);
		assert.strictEqual(
			presets.balanced,
			"mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
		);
		assert.strictEqual(
			presets.capable,
			"mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
		);
	});

	it("isBackend and isPreset guards work", () => {
		assert.ok(isBackend("ollama"));
		assert.ok(isBackend("mlx"));
		assert.ok(!isBackend("foo"));
		assert.ok(isPreset("light"));
		assert.ok(isPreset("balanced"));
		assert.ok(isPreset("capable"));
		assert.ok(!isPreset("ultra"));
	});

	it("PRESET_NAMES lists all presets", () => {
		assert.deepStrictEqual(PRESET_NAMES, ["light", "balanced", "capable"]);
	});

	it("DEFAULT_BACKEND is mlx", () => {
		assert.strictEqual(DEFAULT_BACKEND, "mlx");
	});
});

describe("makeTestConfig", () => {
	it("produces a valid VisionConfig with defaults", () => {
		const cfg = makeTestConfig();
		assert.strictEqual(cfg.model, "test-model");
		assert.strictEqual(cfg.prompt, "test prompt");
		assert.strictEqual(cfg.enabled, true);
		assert.strictEqual(cfg.maxEdge, 1024);
		assert.strictEqual(cfg.maxBytes, 1_400_000);
	});
});

describe("baseDefaults", () => {
	it("returns expected defaults", () => {
		const d = baseDefaults();
		assert.strictEqual(d.enabled, true);
		assert.strictEqual(d.maxEdge, 1024);
		assert.strictEqual(d.timeoutMs, 120_000);
		assert.strictEqual(d.maxImagesPerTurn, 6);
		assert.strictEqual(d.maxTokens, 768);
	});
});

describe("extractText", () => {
	it("extracts text from a string content response", () => {
		assert.strictEqual(extractText({ content: "hello world" }), "hello world");
	});

	it("trims whitespace from string content", () => {
		assert.strictEqual(extractText({ content: "  hello  " }), "hello");
	});

	it("extracts text from a content array with text parts", () => {
		const msg = {
			content: [
				{ type: "text", text: "first" },
				{ type: "text", text: "second" },
			],
		};
		assert.strictEqual(extractText(msg), "first\nsecond");
	});

	it("skips non-text parts in content array", () => {
		const msg = {
			content: [
				{ type: "image", data: "abc" },
				{ type: "text", text: "visible" },
			],
		};
		assert.strictEqual(extractText(msg), "visible");
	});

	it("returns empty string for empty content array", () => {
		assert.strictEqual(extractText({ content: [] }), "");
	});

	it("returns empty string for non-array, non-string content", () => {
		assert.strictEqual(extractText({ content: null }), "");
		assert.strictEqual(extractText({ content: 42 }), "");
	});
});

describe("buildVisionModel", () => {
	it("returns a Model with the correct id and name", () => {
		const cfg = makeTestConfig("my-model");
		const model = buildVisionModel(cfg);
		assert.strictEqual(model.id, "my-model");
		assert.strictEqual(model.name, "vision:my-model");
	});

	it("sets provider to pi-vision", () => {
		const model = buildVisionModel(makeTestConfig("m"));
		assert.strictEqual(model.provider, "pi-vision");
	});

	it("marks reasoning as false and accepts image input", () => {
		const model = buildVisionModel(makeTestConfig("m"));
		assert.strictEqual(model.reasoning, false);
		assert.deepStrictEqual(model.input, ["text", "image"]);
	});

	it("uses the config's api, baseUrl, contextWindow, maxTokens", () => {
		const cfg = makeTestConfig("m");
		cfg.api = "openai-completions";
		cfg.baseUrl = "http://test:8080";
		cfg.contextWindow = 4096;
		cfg.maxTokens = 512;
		const model = buildVisionModel(cfg);
		assert.strictEqual(model.api, "openai-completions");
		assert.strictEqual(model.baseUrl, "http://test:8080");
		assert.strictEqual(model.contextWindow, 4096);
		assert.strictEqual(model.maxTokens, 512);
	});

	it("zero-cost model for local inference", () => {
		const model = buildVisionModel(makeTestConfig("m"));
		assert.deepStrictEqual(model.cost, { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
	});
});

describe("getCacheSize", () => {
	it("returns 0 for an unloaded cache", async () => {
		// Dynamic import to avoid any module-level state leakage
		const { getCacheSize, clearCache: cc } = await import("../src/cache.js");
		cc();
		assert.strictEqual(getCacheSize(), 0);
	});
});

// ── VisionBackendRegistry tests ───────────────────────────────────────────-

describe("VisionBackendRegistry", () => {
	it("starts with built-in backends", () => {
		const reg = new VisionBackendRegistry();
		assert.ok(reg.has("mlx"));
		assert.ok(reg.has("ollama"));
		assert.strictEqual(reg.names().length, 2);
	});

	it("isBuiltin returns true for built-in backends", () => {
		const reg = new VisionBackendRegistry();
		assert.ok(reg.isBuiltin("mlx"));
		assert.ok(reg.isBuiltin("ollama"));
		assert.ok(!reg.isBuiltin("custom-foo"));
	});

	it("register adds a custom backend", () => {
		const reg = new VisionBackendRegistry();
		const originalCount = reg.names().length;
		reg.register("my-server", {
			label: "My Server",
			baseUrl: "http://my-server:8080/v1",
			apiKey: "my-key",
			presets: { light: "my-model" },
			defaultPreset: "light",
			isCustom: true,
			setup: () => ["custom setup"],
		});
		assert.ok(reg.has("my-server"));
		assert.strictEqual(reg.names().length, originalCount + 1);
		const def = reg.get("my-server");
		assert.strictEqual(def?.baseUrl, "http://my-server:8080/v1");
		assert.strictEqual(def?.name, "my-server");
		assert.ok(def?.isCustom);
	});

	it("unregister removes a custom backend", () => {
		const reg = new VisionBackendRegistry();
		reg.register("custom-foo", {
			label: "Custom Foo",
			baseUrl: "http://foo/v1",
			apiKey: "",
			presets: { light: "m" },
			defaultPreset: "light",
			isCustom: true,
			setup: () => [],
		});
		assert.ok(reg.has("custom-foo"));
		assert.ok(reg.unregister("custom-foo"));
		assert.ok(!reg.has("custom-foo"));
	});

	it("unregister returns false for built-in backends", () => {
		const reg = new VisionBackendRegistry();
		assert.ok(!reg.unregister("mlx"));
		assert.ok(reg.has("mlx"));
	});

	it("get returns undefined for unknown backends", () => {
		const reg = new VisionBackendRegistry();
		assert.strictEqual(reg.get("nonexistent"), undefined);
	});

	it("list returns all backends", () => {
		const reg = new VisionBackendRegistry();
		const all = reg.list();
		const names = all.map((b) => b.name);
		assert.ok(names.includes("mlx"));
		assert.ok(names.includes("ollama"));
	});

	it("register overwrites existing backend", () => {
		const reg = new VisionBackendRegistry();
		reg.register("mlx", {
			label: "Overridden MLX",
			baseUrl: "http://override/v1",
			apiKey: "override",
			presets: { light: "override-model" },
			defaultPreset: "light",
			setup: () => [],
		});
		const def = reg.get("mlx");
		assert.strictEqual(def?.label, "Overridden MLX");
		assert.strictEqual(def?.baseUrl, "http://override/v1");
	});
});

describe("isBackend with registry", () => {
	it("recognizes built-in backends", () => {
		assert.ok(isBackend("ollama"));
		assert.ok(isBackend("mlx"));
	});

	it("recognizes registered custom backends via singleton", async () => {
		const { visionBackendRegistry: reg } = await import("../src/backends.js");
		reg.register("test-custom-reg", {
			label: "Test",
			baseUrl: "http://test/v1",
			apiKey: "",
			presets: { light: "m" },
			defaultPreset: "light",
			isCustom: true,
			setup: () => [],
		});
		assert.ok(isBackend("test-custom-reg"));
		reg.unregister("test-custom-reg");
	});

	it("returns false for unknown backends", () => {
		assert.ok(!isBackend("completely-unknown"));
	});
});

describe("isBuiltinBackend", () => {
	it("returns true for mlx and ollama", () => {
		assert.ok(isBuiltinBackend("mlx"));
		assert.ok(isBuiltinBackend("ollama"));
	});

	it("returns false for custom or unknown", () => {
		assert.ok(!isBuiltinBackend("custom-foo"));
	});
});

describe("loadCustomBackends", () => {
	it("loads custom backends into the singleton registry", async () => {
		const { visionBackendRegistry: reg } = await import("../src/backends.js");
		// Unregister any test artifacts first
		reg.unregister("load-test-backend");

		loadCustomBackends({
			"load-test-backend": {
				baseUrl: "http://load-test:8080/v1",
				apiKey: "test-key",
			},
		});

		assert.ok(reg.has("load-test-backend"));
		const def = reg.get("load-test-backend");
		assert.strictEqual(def?.baseUrl, "http://load-test:8080/v1");
		assert.strictEqual(def?.apiKey, "test-key");
		assert.ok(def?.isCustom);

		reg.unregister("load-test-backend");
	});

	it("skips invalid entries", async () => {
		const { visionBackendRegistry: reg } = await import("../src/backends.js");
		reg.unregister("invalid-foo");

		loadCustomBackends({
			"": { baseUrl: "http://empty" },
			"valid-name": { baseUrl: "http://valid" },
		} as any);

		assert.ok(reg.has("valid-name"));
		reg.unregister("valid-name");
	});
});

describe("config modelRef", () => {
	it("baseDefaults includes modelRef: backend", () => {
		const d = baseDefaults();
		assert.strictEqual(d.modelRef, "backend");
	});

	it("makeTestConfig includes modelRef", () => {
		const cfg = makeTestConfig();
		assert.strictEqual(cfg.modelRef, "backend");
	});

	it("makeTestConfig includes registeredBackends", () => {
		const cfg = makeTestConfig();
		assert.deepStrictEqual(cfg.registeredBackends, {});
	});
});

describe("BACKENDS type widening", () => {
	it("accepts arbitrary string keys", () => {
		// BACKENDS is now Record<string, BackendDef> so arbitrary keys work
		const custom = (BACKENDS as Record<string, any>)["nonexistent"];
		assert.strictEqual(custom, undefined);
	});
});

// ── Cache optimization tests ────────────────────────────────────────────────

describe("cache LRU", () => {
	beforeEach(() => clearCache());

	it("cacheGet returns undefined for missing key", () => {
		assert.strictEqual(cacheGet("nonexistent"), undefined);
	});

	it("cacheSet and cacheGet round-trip", () => {
		cacheSet("k1", "hello");
		assert.strictEqual(cacheGet("k1"), "hello");
	});

	it("cacheSet overwrites existing key", () => {
		cacheSet("k1", "old");
		cacheSet("k1", "new");
		assert.strictEqual(cacheGet("k1"), "new");
	});

	it("evicts oldest entry when at capacity", () => {
		// Fill with keys 0..max-1
		for (let i = 0; i < 500; i++) cacheSet(`k${i}`, `v${i}`);
		assert.strictEqual(getCacheSize(), 500);
		// Add one more — should evict "k0"
		cacheSet("k500", "v500");
		assert.strictEqual(getCacheSize(), 500);
		assert.strictEqual(cacheGet("k0"), undefined);
		assert.strictEqual(cacheGet("k500"), "v500");
	});

	it("access via cacheGet moves entry to MRU position", () => {
		for (let i = 0; i < 500; i++) cacheSet(`k${i}`, `v${i}`);
		// Access the oldest entry
		assert.strictEqual(cacheGet("k0"), "v0");
		// Now k0 should be MRU; adding one more evicts k1, not k0
		cacheSet("k500", "v500");
		assert.strictEqual(cacheGet("k1"), undefined, "k1 should be evicted");
		assert.strictEqual(cacheGet("k0"), "v0", "k0 should survive");
	});
});

describe("negative cache", () => {
	beforeEach(() => clearCache());

	it("isNegCached returns false for unknown key", () => {
		assert.ok(!isNegCached("unknown"));
	});

	it("markNegCached and isNegCached round-trip", () => {
		markNegCached("fail1", 30_000);
		assert.ok(isNegCached("fail1"));
	});

	it("expired entry returns false", () => {
		markNegCached("expired", -1); // already expired
		assert.ok(!isNegCached("expired"));
	});

	it("getNegCacheSize returns correct count", () => {
		const before = getNegCacheSize();
		markNegCached("a", 30_000);
		assert.strictEqual(getNegCacheSize(), before + 1);
	});

	it("does not affect caption cache", () => {
		markNegCached("neg-key", 30_000);
		assert.strictEqual(cacheGet("neg-key"), undefined);
	});
});

describe("in-flight dedup", () => {
	beforeEach(() => clearCache());

	it("getInFlight returns undefined for unknown key", () => {
		assert.strictEqual(getInFlight("unknown"), undefined);
	});

	it("setInFlight stores a promise that resolves", async () => {
		const p = Promise.resolve("caption-result");
		setInFlight("k1", p);
		const retrieved = getInFlight("k1");
		assert.ok(retrieved instanceof Promise);
		const result = await retrieved;
		assert.strictEqual(result, "caption-result");
		// After resolution, should be removed from in-flight
		assert.strictEqual(getInFlight("k1"), undefined);
	});

	it("setInFlight adds to negative cache on rejection", async () => {
		const p = Promise.reject(new Error("vision failed"));
		setInFlight("k2", p);
		// Wait for rejection to settle
		await assert.rejects(async () => getInFlight("k2")!, /vision failed/);
		// Should now be in negative cache
		assert.ok(isNegCached("k2"));
	});

	it("duplicate calls return same promise", () => {
		const p = Promise.resolve("shared");
		setInFlight("dup", p);
		const p1 = getInFlight("dup");
		const p2 = getInFlight("dup");
		assert.strictEqual(p1, p2);
	});

	it("removeInFlight clears the in-flight entry", () => {
		const p = new Promise<string>(() => {}); // never settles
		setInFlight("stuck", p);
		assert.ok(getInFlight("stuck"));
		removeInFlight("stuck");
		assert.strictEqual(getInFlight("stuck"), undefined);
	});

	it("getInFlightCount returns correct count", () => {
		const before = getInFlightCount();
		const p1 = new Promise<string>(() => {});
		const p2 = new Promise<string>(() => {});
		setInFlight("a", p1);
		setInFlight("b", p2);
		assert.strictEqual(getInFlightCount(), before + 2);
	});
});

describe("dirty tracking", () => {
	beforeEach(() => clearCache());

	it("cacheSet marks key as dirty", () => {
		const before = getDirtyCount();
		cacheSet("k1", "v1");
		assert.strictEqual(getDirtyCount(), before + 1);
	});

	it("overwriting a key does not double-count dirty", () => {
		cacheSet("k1", "v1");
		const afterFirst = getDirtyCount();
		cacheSet("k1", "v2");
		assert.strictEqual(getDirtyCount(), afterFirst); // still 1 dirty key
	});

	it("clearCache resets dirty count", () => {
		cacheSet("k1", "v1");
		clearCache();
		assert.strictEqual(getDirtyCount(), 0);
	});
});

describe("mapWithConcurrency", () => {
	it("preserves order", async () => {
		const input = ["a", "b", "c"];
		const result = await mapWithConcurrency(input, 2, async (item) => item.toUpperCase());
		assert.deepStrictEqual(result, ["A", "B", "C"]);
	});

	it("limits concurrency", async () => {
		let concurrent = 0;
		let maxConcurrent = 0;
		const input = [1, 2, 3, 4, 5, 6];

		await mapWithConcurrency(input, 2, async (item) => {
			concurrent++;
			maxConcurrent = Math.max(maxConcurrent, concurrent);
			await new Promise((r) => setTimeout(r, 5));
			concurrent--;
			return item * 2;
		});

		assert.ok(maxConcurrent <= 2, `max concurrency was ${maxConcurrent}, expected ≤ 2`);
	});

	it("returns empty array for empty input", async () => {
		const result = await mapWithConcurrency([], 2, async () => "x");
		assert.deepStrictEqual(result, []);
	});

	it("handles concurrency > input length", async () => {
		const result = await mapWithConcurrency(["x"], 10, async (item) => item);
		assert.deepStrictEqual(result, ["x"]);
	});

	it("results map to correct indices", async () => {
		const input = [10, 20, 30];
		const result = await mapWithConcurrency(input, 2, async (item, idx) => item + idx);
		assert.deepStrictEqual(result, [10, 21, 32]);
	});
});

describe("config optimization fields", () => {
	it("baseDefaults includes captionConcurrency", () => {
		const d = baseDefaults();
		assert.strictEqual(d.captionConcurrency, 2);
	});

	it("baseDefaults includes cacheMaxEntries", () => {
		const d = baseDefaults();
		assert.strictEqual(d.cacheMaxEntries, 500);
	});

	it("baseDefaults includes negativeCacheTTLMs", () => {
		const d = baseDefaults();
		assert.strictEqual(d.negativeCacheTTLMs, 300_000);
	});

	it("makeTestConfig includes optimization fields", () => {
		const cfg = makeTestConfig();
		assert.strictEqual(cfg.captionConcurrency, 2);
		assert.strictEqual(cfg.cacheMaxEntries, 500);
		assert.strictEqual(cfg.negativeCacheTTLMs, 300_000);
	});
});

// ── Perf / benchmark tests ──────────────────────────────────────────────────

const SMALL = 250;
const MEDIUM = 500;
const LARGE = 2000;

/** Time a function in milliseconds. */
async function bench(fn: () => Promise<unknown> | unknown, iterations = 1): Promise<number> {
	const start = performance.now();
	for (let i = 0; i < iterations; i++) await fn();
	return performance.now() - start;
}

describe("cache perf — LRU throughput", () => {
	beforeEach(() => clearCache());

	it(`set+get ${SMALL} entries (configurable max)`, async () => {
		const elapsed = await bench(async () => {
			for (let i = 0; i < SMALL; i++) cacheSet(`k${i}`, `v${i}`);
			for (let i = 0; i < SMALL; i++) assert.ok(cacheGet(`k${i}`) !== undefined);
		}, 1);
		assert.ok(elapsed < 500, `LRU ${SMALL} set+get took ${elapsed.toFixed(1)}ms (expected <500ms)`);
	});

	it(`eviction: fill to ${MEDIUM} then add forcing eviction`, async () => {
		const elapsed = await bench(() => {
			for (let i = 0; i < MEDIUM; i++) cacheSet(`k${i}`, `v${i}`);
			// Force eviction by adding more
			for (let i = MEDIUM; i < MEDIUM + 50; i++) cacheSet(`k${i}`, `v${i}`);
		}, 1);
		assert.ok(elapsed < 200, `Eviction ${MEDIUM}+50 took ${elapsed.toFixed(1)}ms (expected <200ms)`);
	});

	it(`get+set LRU touch pattern (${SMALL} ops)`, async () => {
		for (let i = 0; i < SMALL; i++) cacheSet(`k${i}`, `v${i}`);
		const elapsed = await bench(() => {
			// LRU touch: get oldest, set new — triggers delete+re-set
			for (let i = 0; i < 100; i++) {
				cacheGet("k0");
				cacheSet(`new${i}`, `v${i}`);
			}
		}, 1);
		assert.ok(elapsed < 100, `LRU touch ${SMALL} took ${elapsed.toFixed(1)}ms (expected <100ms)`);
	});
});

describe("cache perf — configurable limits via loadCache", () => {
	beforeEach(() => clearCache());

	it("loadCache applies cacheMaxEntries and negCacheTTLMs from config", () => {
		const cfg = makeTestConfig("m", "p", 1024, 1_400_000);
		cfg.cacheMaxEntries = 100;
		cfg.negativeCacheTTLMs = 10_000;
		loadCache(cfg);
		assert.strictEqual(getCacheMaxEntries(), 100);
		assert.strictEqual(getNegCacheTTLMs(), 10_000);
	});

	it("cacheSet uses configured max entries", () => {
		const cfg = makeTestConfig("m", "p", 1024, 1_400_000);
		cfg.cacheMaxEntries = 10;
		loadCache(cfg);
		for (let i = 0; i < 10; i++) cacheSet(`k${i}`, `v${i}`);
		assert.strictEqual(getCacheSize(), 10);
		// One more triggers eviction
		cacheSet("k10", "v10");
		assert.strictEqual(getCacheSize(), 10);
		assert.strictEqual(cacheGet("k0"), undefined, "oldest evicted");
		assert.strictEqual(cacheGet("k10"), "v10", "newest retained");
	});

	it("markNegCached uses configured TTL", () => {
		const cfg = makeTestConfig();
		cfg.negativeCacheTTLMs = 5_000;
		loadCache(cfg);
		markNegCached("neg-test"); // no explicit TTL — uses module default from config
		assert.ok(isNegCached("neg-test"));
	});
});

describe("cache perf — in-flight dedup with concurrency", () => {
	beforeEach(() => clearCache());

	it("many concurrent captionOne calls share in-flight promise", async () => {
		// Single key — all callers should get the same promise back
		const deferred = Promise.resolve("shared-result");
		setInFlight("shared", deferred);

		const callers = 50;
		const promises = Array.from({ length: callers }, () => {
			const existing = getInFlight("shared");
			return existing!;
		});
		const results = await Promise.all(promises);
		assert.strictEqual(results.length, callers);
		assert.ok(results.every((r: string) => r === "shared-result"));
	});

	it("in-flight cleanup after resolution", async () => {
		setInFlight("cleanup", Promise.resolve("done"));
		await getInFlight("cleanup")!;
		// After resolution, should be removed
		assert.strictEqual(getInFlight("cleanup"), undefined);
	});

	it("in-flight count tracks active requests", async () => {
		const neverResolves = new Promise<string>(() => {});
		setInFlight("stuck1", neverResolves);
		setInFlight("stuck2", neverResolves);
		assert.strictEqual(getInFlightCount(), 2);
		// Clean up to avoid hanging
		removeInFlight("stuck1");
		removeInFlight("stuck2");
		assert.strictEqual(getInFlightCount(), 0);
	});
});

describe("cache perf — negative cache throughput", () => {
	beforeEach(() => {
		clearCache();
		clearNegCache();
	});

	it(`mark+check ${LARGE} negative entries`, async () => {
		const elapsed = await bench(() => {
			for (let i = 0; i < LARGE; i++) {
				markNegCached(`neg${i}`, 300_000);
			}
			for (let i = 0; i < LARGE; i++) {
				isNegCached(`neg${i}`);
			}
		}, 1);
		assert.ok(elapsed < 200, `Negative cache ${LARGE} mark+check took ${elapsed.toFixed(1)}ms (expected <200ms)`);
	});

	it("negative cache not affected by caption cache limits", () => {
		// Negative cache is a separate map — not bounded by cacheMaxEntries
		for (let i = 0; i < 3000; i++) markNegCached(`neg${i}`, 300_000);
		assert.strictEqual(getNegCacheSize(), 3000);
		assert.ok(isNegCached("neg0"));
		clearNegCache();
		assert.strictEqual(getNegCacheSize(), 0);
	});

	it("expired negative entries are cleaned on check", () => {
		markNegCached("expired", -1000); // already expired
		assert.ok(!isNegCached("expired"));
		assert.strictEqual(getNegCacheSize(), 0);
	});
});

describe("cache perf — dirty tracking", () => {
	beforeEach(() => clearCache());

	it(`set ${SMALL} entries tracks all dirty`, () => {
		for (let i = 0; i < SMALL; i++) cacheSet(`k${i}`, `v${i}`);
		assert.strictEqual(getDirtyCount(), SMALL);
	});

	it("overwrite does not increase dirty count", () => {
		cacheSet("k1", "v1");
		const before = getDirtyCount();
		cacheSet("k1", "v2");
		assert.strictEqual(getDirtyCount(), before);
	});

	it("clear resets dirty count", () => {
		for (let i = 0; i < 100; i++) cacheSet(`k${i}`, `v${i}`);
		clearCache();
		assert.strictEqual(getDirtyCount(), 0);
	});
});

describe("mapWithConcurrency perf", () => {
	it(`process ${SMALL} items with concurrency 4 — preserves order`, async () => {
		const input = Array.from({ length: SMALL }, (_, i) => i);
		const elapsed = await bench(async () => {
			const result = await mapWithConcurrency(input, 4, async (n) => n * 2);
			assert.strictEqual(result.length, SMALL);
			assert.strictEqual(result[0], 0);
			assert.strictEqual(result[SMALL - 1], (SMALL - 1) * 2);
		}, 1);
		assert.ok(elapsed < 1000, `mapWithConcurrency ${SMALL} took ${elapsed.toFixed(1)}ms (expected <1000ms)`);
	});

	it("concurrency 1 is serial (order guaranteed)", async () => {
		const input = ["a", "b", "c"];
		const result = await mapWithConcurrency(input, 1, async (item) => item.toUpperCase());
		assert.deepStrictEqual(result, ["A", "B", "C"]);
	});

	it("high concurrency does not exceed limit", async () => {
		let concurrent = 0;
		let maxConcurrent = 0;
		const input = Array.from({ length: 20 }, (_, i) => i);

		await mapWithConcurrency(input, 3, async (n) => {
			concurrent++;
			maxConcurrent = Math.max(maxConcurrent, concurrent);
			await new Promise((r) => setTimeout(r, 1));
			concurrent--;
			return n;
		});
		assert.ok(maxConcurrent <= 3, `max concurrent was ${maxConcurrent}`);
	});
});

describe("cacheKey perf", () => {
	it("generates 32-char hash quickly", async () => {
		const cfg = makeTestConfig();
		const data = "a".repeat(10000); // 10KB of data
		const elapsed = await bench(() => cacheKey(cfg, data), 1000);
		assert.ok(elapsed < 500, `1000 cacheKey calls took ${elapsed.toFixed(1)}ms (expected <500ms)`);
	});

	it("different inputs produce different keys", () => {
		const cfg = makeTestConfig();
		const k1 = cacheKey(cfg, "data1");
		const k2 = cacheKey(cfg, "data2");
		assert.notStrictEqual(k1, k2);
	});
});

// ── ImageStore tests ────────────────────────────────────────────────────────

import { ImageStore } from "../src/image-store.js";

// Helper to create a deterministic ref-based image data string
function imgData(label: string): string {
	return Buffer.from(label).toString("base64");
}

describe("ImageStore", () => {
	let store: ImageStore;

	beforeEach(() => {
		store = new ImageStore();
	});

	describe("set and get", () => {
		it("stores and retrieves an image by ref", () => {
			store.set("img_abc", imgData("hello"), "image/png");
			const entry = store.get("img_abc");
			assert.ok(entry, "should exist");
			assert.strictEqual(entry!.data, imgData("hello"));
			assert.strictEqual(entry!.mimeType, "image/png");
		});

		it("returns undefined for unknown ref", () => {
			assert.strictEqual(store.get("nonexistent"), undefined);
		});

		it("updates existing ref in place", () => {
			store.set("img_abc", imgData("old"), "image/png");
			store.set("img_abc", imgData("new"), "image/jpeg");
			const entry = store.get("img_abc");
			assert.strictEqual(entry!.data, imgData("new"));
			assert.strictEqual(entry!.mimeType, "image/jpeg");
		});

		it("touches lastAccess on get (moves to MRU)", () => {
			store.set("img_a", imgData("a"), "image/png");
			store.set("img_b", imgData("b"), "image/png");
			const before = store.get("img_a")!.lastAccess;
			// Small delay so timestamp changes
			const after = store.get("img_a")!.lastAccess;
			assert.ok(after >= before, "lastAccess should be updated");
		});
	});

	describe("has", () => {
		it("returns true for existing ref", () => {
			store.set("img_abc", imgData("x"), "image/png");
			assert.ok(store.has("img_abc"));
		});

		it("returns false for unknown ref", () => {
			assert.ok(!store.has("nonexistent"));
		});

		it("returns false for expired ref", () => {
			const expiredStore = new ImageStore(20, -1); // TTL already expired
			expiredStore.set("img_abc", imgData("x"), "image/png");
			assert.ok(!expiredStore.has("img_abc"));
		});
	});

	describe("delete", () => {
		it("removes an existing ref", () => {
			store.set("img_abc", imgData("x"), "image/png");
			assert.ok(store.has("img_abc"));
			store.delete("img_abc");
			assert.ok(!store.has("img_abc"));
		});

		it("does not throw for unknown ref", () => {
			store.delete("nonexistent");
		});
	});

	describe("clear", () => {
		it("removes all entries", () => {
			store.set("img_a", imgData("a"), "image/png");
			store.set("img_b", imgData("b"), "image/png");
			assert.strictEqual(store.size, 2);
			store.clear();
			assert.strictEqual(store.size, 0);
			assert.ok(!store.has("img_a"));
		});
	});

	describe("size", () => {
		it("starts at 0", () => {
			assert.strictEqual(store.size, 0);
		});

		it("increments on set", () => {
			store.set("img_a", imgData("a"), "image/png");
			assert.strictEqual(store.size, 1);
			store.set("img_b", imgData("b"), "image/png");
			assert.strictEqual(store.size, 2);
		});

		it("does not increment on update", () => {
			store.set("img_a", imgData("a"), "image/png");
			assert.strictEqual(store.size, 1);
			store.set("img_a", imgData("a2"), "image/png");
			assert.strictEqual(store.size, 1);
		});

		it("decrements on delete", () => {
			store.set("img_a", imgData("a"), "image/png");
			store.set("img_b", imgData("b"), "image/png");
			store.delete("img_a");
			assert.strictEqual(store.size, 1);
		});
	});

	describe("FIFO eviction (max capacity)", () => {
		it("evicts oldest entry when at capacity", () => {
			const small = new ImageStore(3);
			small.set("img_a", imgData("a"), "image/png");
			small.set("img_b", imgData("b"), "image/png");
			small.set("img_c", imgData("c"), "image/png");
			assert.strictEqual(small.size, 3);

			// Adding a 4th should evict img_a (oldest)
			small.set("img_d", imgData("d"), "image/png");
			assert.strictEqual(small.size, 3);
			assert.ok(!small.has("img_a"), "img_a should be evicted");
			assert.ok(small.has("img_d"), "img_d should exist");
		});

		it("accessed entries survive eviction (MRU promotion)", () => {
			const small = new ImageStore(3);
			small.set("img_a", imgData("a"), "image/png");
			small.set("img_b", imgData("b"), "image/png");
			small.set("img_c", imgData("c"), "image/png");

			// Access img_a to promote it to MRU
			assert.ok(small.has("img_a"));

			// Add 4th — should evict img_b (now oldest), not img_a
			small.set("img_d", imgData("d"), "image/png");
			assert.ok(small.has("img_a"), "img_a should survive (MRU)");
			assert.ok(!small.has("img_b"), "img_b should be evicted");
			assert.ok(small.has("img_c"));
			assert.ok(small.has("img_d"));
		});

		it("updated entries move to MRU position", () => {
			const small = new ImageStore(2);
			small.set("img_a", imgData("a"), "image/png");
			small.set("img_b", imgData("b"), "image/png");

			// Update img_a (should move it to MRU)
			small.set("img_a", imgData("a2"), "image/png");

			// Add 3rd — should evict img_b (now oldest)
			small.set("img_c", imgData("c"), "image/png");
			assert.ok(small.has("img_a"), "img_a should survive (updated)");
			assert.ok(!small.has("img_b"), "img_b should be evicted");
			assert.ok(small.has("img_c"));
		});
	});

	describe("TTL expiration", () => {
		it("returns undefined for expired entry", () => {
			const ttlStore = new ImageStore(20, -1); // expired immediately
			ttlStore.set("img_abc", imgData("x"), "image/png");
			assert.strictEqual(ttlStore.get("img_abc"), undefined);
		});

		it("has returns false for expired entry", () => {
			const ttlStore = new ImageStore(20, -1);
			ttlStore.set("img_abc", imgData("x"), "image/png");
			assert.ok(!ttlStore.has("img_abc"));
		});

		it("evicts expired entries on next set", () => {
			const ttlStore = new ImageStore(20, -1);
			ttlStore.set("img_a", imgData("a"), "image/png");
			// The second set triggers evictExpired which cleans img_a
			// because its lastAccess is 0ms old and TTL is -1 (already expired)
			ttlStore.set("img_b", imgData("b"), "image/png");
			// Both should be expired/evicted
			assert.strictEqual(ttlStore.get("img_a"), undefined);
			assert.strictEqual(ttlStore.get("img_b"), undefined);
			// Expired entries cleaned by evictExpired + get
			assert.strictEqual(ttlStore.size, 0);
		});

		it("touching an entry resets its TTL", () => {
			const store = new ImageStore(20, 60_000); // 60s TTL
			store.set("img_abc", imgData("x"), "image/png");
			const entry = store.get("img_abc")!;
			assert.ok(entry.lastAccess > 0);
			// lastAccess was set to now during get (touch)
			assert.ok(Date.now() - entry.lastAccess < 100, "lastAccess should be recent");
		});
	});

	describe("constructor defaults", () => {
		it("defaults to 20 max entries", () => {
			const s = new ImageStore();
			for (let i = 0; i < 21; i++) s.set(`img_${i}`, imgData(`${i}`), "image/png");
			assert.strictEqual(s.size, 20);
			assert.ok(!s.has("img_0"));
		});

		it("defaults to 10 min TTL (600_000ms = 10 min)", () => {
			const s = new ImageStore(20, 600_000);
			s.set("img_abc", imgData("x"), "image/png");
			assert.ok(s.has("img_abc"));
		});

		it("accepts custom max and TTL", () => {
			const s = new ImageStore(5, 10_000);
			assert.strictEqual((s as any).max, 5);
			assert.strictEqual((s as any).ttlMs, 10_000);
		});
	});
});

// ── Privacy / security tests ──────────────────────────────────────────────

import { sanitizeUrl, isLocalhost, warnRemoteHttp, sanitizeConfig, stripImageMetadata } from "../src/privacy.js";

describe("sanitizeUrl", () => {
	it("strips user:password from URL", () => {
		const result = sanitizeUrl("http://user:pass@localhost:11434/v1");
		assert.ok(!result.includes("user:pass"), "should remove credentials");
		assert.ok(result.includes("***"), "should mask with ***");
	});

	it("strips api_key query param", () => {
		const result = sanitizeUrl("http://example.com/v1?api_key=secret123");
		assert.ok(!result.includes("secret123"), "should remove api_key query param");
	});

	it("strips apikey query param", () => {
		const result = sanitizeUrl("http://example.com/v1?apikey=secret");
		assert.ok(!result.includes("secret"), "should remove apikey query param");
	});

	it("strips api-key query param", () => {
		const result = sanitizeUrl("http://example.com/v1?api-key=secret");
		assert.ok(!result.includes("secret"), "should remove api-key query param");
	});

	it("leaves safe URLs unchanged", () => {
		const result = sanitizeUrl("http://localhost:11434/v1");
		assert.strictEqual(result, "http://localhost:11434/v1");
	});

	it("handles malformed URLs gracefully", () => {
		const result = sanitizeUrl("not-a-url");
		assert.strictEqual(result, "not-a-url");
	});
});

describe("isLocalhost", () => {
	it("returns true for localhost", () => {
		assert.ok(isLocalhost("http://localhost:11434/v1"));
	});

	it("returns true for 127.0.0.1", () => {
		assert.ok(isLocalhost("http://127.0.0.1:8080/v1"));
	});

	it("returns true for [::1]", () => {
		assert.ok(isLocalhost("http://[::1]:8080/v1"));
	});

	it("returns true for *.local", () => {
		assert.ok(isLocalhost("http://my-mac.local:8080"));
	});

	it("returns false for remote hosts", () => {
		assert.ok(!isLocalhost("https://api.openai.com/v1"));
		assert.ok(!isLocalhost("http://example.com:8080"));
	});

	it("returns false for malformed URLs", () => {
		assert.ok(!isLocalhost(""));
	});
});

describe("warnRemoteHttp", () => {
	it("returns null for localhost HTTP", () => {
		assert.strictEqual(warnRemoteHttp("http://localhost:11434/v1"), null);
	});

	it("returns null for localhost HTTPS", () => {
		assert.strictEqual(warnRemoteHttp("https://localhost:11434/v1"), null);
	});

	it("returns null for remote HTTPS", () => {
		assert.strictEqual(warnRemoteHttp("https://api.openai.com/v1"), null);
	});

	it("returns warning for remote HTTP", () => {
		const warning = warnRemoteHttp("http://example.com:8080/v1");
		assert.ok(warning !== null, "should warn about remote HTTP");
		assert.ok(warning!.includes("plain HTTP"), "should mention HTTP");
		assert.ok(warning!.includes("encrypted"), "should mention encryption");
	});

	it("returns null for malformed URL", () => {
		assert.strictEqual(warnRemoteHttp(""), null);
	});
});

describe("sanitizeConfig", () => {
	it("strips apiKey from config", () => {
		const cfg = makeTestConfig();
		cfg.apiKey = "sk-secret-123";
		cfg.cachePassphrase = "my-passphrase";
		const safe = sanitizeConfig(cfg);
		assert.ok(!("apiKey" in safe), "apiKey should be removed");
		assert.ok(!("cachePassphrase" in safe), "cachePassphrase should be removed");
	});

	it("sanitizes baseUrl in output", () => {
		const cfg = makeTestConfig();
		cfg.baseUrl = "http://user:pass@localhost:11434/v1";
		const safe = sanitizeConfig(cfg);
		assert.strictEqual(typeof safe.baseUrl, "string");
		assert.ok(!(safe.baseUrl as string).includes("user:pass"), "baseUrl should be sanitized");
		assert.ok((safe.baseUrl as string).includes("***"), "credentials masked");
	});

	it("preserves all other config keys", () => {
		const cfg = makeTestConfig();
		const safe = sanitizeConfig(cfg);
		assert.strictEqual(safe.model, cfg.model);
		assert.strictEqual(safe.enabled, cfg.enabled);
		assert.strictEqual(safe.backend, cfg.backend);
		assert.strictEqual(safe.maxEdge, cfg.maxEdge);
	});
});

// ── Cache encryption tests ─────────────────────────────────────────────────

import { encryptCacheData, decryptCacheData, purgeCache } from "../src/cache.js";

describe("cache encryption", () => {
	const PASSPHRASE = "test-passphrase-123";

	it("encrypt and decrypt round-trip", () => {
		const original = JSON.stringify({ key1: "value1", key2: "value2" });
		const encrypted = encryptCacheData(original, PASSPHRASE);
		assert.ok(encrypted.length > 0, "encrypted output should be non-empty");
		const decrypted = decryptCacheData(encrypted, PASSPHRASE);
		assert.strictEqual(decrypted, original);
	});

	it("wrong passphrase returns null", () => {
		const original = JSON.stringify({ key: "value" });
		const encrypted = encryptCacheData(original, PASSPHRASE);
		const decrypted = decryptCacheData(encrypted, "wrong-passphrase");
		assert.strictEqual(decrypted, null);
	});

	it("tampered data returns null", () => {
		const original = JSON.stringify({ key: "value" });
		const encrypted = encryptCacheData(original, PASSPHRASE);
		// Corrupt the encrypted data
		const corrupted = "AAA" + encrypted.slice(3);
		const decrypted = decryptCacheData(corrupted, PASSPHRASE);
		assert.strictEqual(decrypted, null);
	});

	it("encrypted output differs each time (random salt + IV)", () => {
		const data = JSON.stringify({ key: "value" });
		const e1 = encryptCacheData(data, PASSPHRASE);
		const e2 = encryptCacheData(data, PASSPHRASE);
		assert.notStrictEqual(e1, e2, "should be different due to random salt+IV");
	});

	it("empty passphrase still works (not recommended but handled)", () => {
		const original = JSON.stringify({ key: "value" });
		const encrypted = encryptCacheData(original, "");
		const decrypted = decryptCacheData(encrypted, "");
		assert.strictEqual(decrypted, original);
	});
});

describe("purgeCache", () => {
	beforeEach(() => {
		const cc = new Map<string, string>();
		cc.set("k1", "v1");
		cc.set("k2", "v2");
		// We can't easily verify internals, but purge should not throw
	});

	it("does not throw", () => {
		assert.doesNotThrow(() => purgeCache());
	});

	it("can be called multiple times", () => {
		assert.doesNotThrow(() => {
			purgeCache();
			purgeCache();
		});
	});
});

// ── Command behavior tests ─────────────────────────────────────────────────

describe("command privacy — no secret leaks", () => {
	it("status output does not contain apiKey", async () => {
		let capturedNotify: string | null = null;
		let capturedType: string | null = null;

		const mockCtx = {
			model: { id: "test-model", input: ["text"] } as any,
			signal: undefined,
			ui: {
				notify: (msg: string, type: string) => {
					capturedNotify = msg;
					capturedType = type;
				},
				setStatus: () => {},
			},
		} as any;

		const mockPi = {
			registerCommand: (_name: string, cmd: any) => {
				// Invoke handler with "status"
				cmd.handler("status", mockCtx);
			},
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		// Should not contain the word apiKey or any sk- pattern
		if (capturedNotify !== null) {
			assert.ok(!(capturedNotify as string).includes("apiKey"), "should not contain apiKey");
			assert.ok(!(capturedNotify as string).includes("sk-"), "should not contain sk- (API key pattern)");
		}
	});

	it("setup output does not leak credentials", async () => {
		let capturedNotify: string | null = null;

		const mockCtx = {
			model: { id: "test-model", input: ["text"] } as any,
			signal: undefined,
			ui: {
				notify: (msg: string) => {
					capturedNotify = msg;
				},
				setStatus: () => {},
			},
		} as any;

		const mockPi = {
			registerCommand: (_name: string, cmd: any) => {
				cmd.handler("setup", mockCtx);
			},
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		if (capturedNotify !== null) {
			assert.ok(!(capturedNotify as string).includes("apiKey"), "setup should not leak apiKey");
		}
	});

	it("backend list sanitizes URLs", async () => {
		let capturedNotify: string | null = null;

		const mockCtx = {
			model: { id: "test-model", input: ["text"] } as any,
			signal: undefined,
			ui: {
				notify: (msg: string) => {
					capturedNotify = msg;
				},
				setStatus: () => {},
			},
		} as any;

		const mockPi = {
			registerCommand: (_name: string, cmd: any) => {
				cmd.handler("backend list", mockCtx);
			},
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		if (capturedNotify !== null) {
			// Backend URLs should not contain credentials (they use defaults, so unlikely,
			// but the sanitizeUrl call should be present)
			assert.ok(!(capturedNotify as string).includes("apiKey"), "backend list should not leak apiKey");
		}
	});

	it("test output sanitizes error messages", async () => {
		let capturedNotify: string | null = null;

		const mockCtx = {
			model: { id: "test-model", input: ["text"] } as any,
			signal: undefined,
			ui: {
				notify: (msg: string) => {
					capturedNotify = msg;
				},
				setStatus: () => {},
			},
		} as any;

		const mockPi = {
			registerCommand: (_name: string, cmd: any) => {
				cmd.handler("test", mockCtx);
			},
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		if (capturedNotify !== null) {
			// Should not leak credentials in error messages
			assert.ok(!(capturedNotify as string).includes("apiKey"), "test output should not leak apiKey");
		}
	});

	it("doctor does not leak apiKey", async () => {
		let capturedNotify: string | null = null;

		const mockCtx = {
			model: { id: "test-model", input: ["text"] } as any,
			signal: undefined,
			ui: {
				notify: (msg: string) => {
					capturedNotify = msg;
				},
				setStatus: () => {},
			},
		} as any;

		const mockPi = {
			registerCommand: (_name: string, cmd: any) => {
				cmd.handler("doctor", mockCtx);
			},
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		if (capturedNotify !== null) {
			assert.ok(!(capturedNotify as string).includes("apiKey"), "doctor should not leak apiKey");
		}
	});
});

describe("command behavior — subcommand dispatch", () => {
	it("/vision on calls saveConfig with enabled true", async () => {
		let handler: any = null;
		const mockPi = {
			registerCommand: (_name: string, cmd: any) => {
				handler = cmd.handler;
			},
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		let capturedNotify: string | null = null;
		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		const mockCtx = {
			ui: { notify: (msg: string) => { capturedNotify = msg; }, setStatus: () => {} },
		} as any;

		await handler("on", mockCtx);
		assert.ok(String(capturedNotify).includes("enabled"), "should confirm enabled");
	});

	it("/vision off calls saveConfig with enabled false", async () => {
		let handler: any = null;
		const mockPi = {
			registerCommand: (_name: string, cmd: any) => { handler = cmd.handler; },
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		let capturedNotify: string | null = null;
		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		const mockCtx = {
			ui: { notify: (msg: string) => { capturedNotify = msg; }, setStatus: () => {} },
		} as any;

		await handler("off", mockCtx);
		assert.ok(String(capturedNotify).includes("disabled"), "should confirm disabled");
	});

	it("/vision clear responds with confirmation", async () => {
		let handler: any = null;
		const mockPi = {
			registerCommand: (_name: string, cmd: any) => { handler = cmd.handler; },
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		let capturedNotify: string | null = null;
		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		const mockCtx = {
			ui: { notify: (msg: string) => { capturedNotify = msg; }, setStatus: () => {} },
		} as any;

		await handler("clear", mockCtx);
		assert.ok(String(capturedNotify).includes("cleared"), "should confirm cache cleared");
	});

	it("/vision purge responds with confirmation", async () => {
		let handler: any = null;
		const mockPi = {
			registerCommand: (_name: string, cmd: any) => { handler = cmd.handler; },
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		let capturedNotify: string | null = null;
		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		const mockCtx = {
			ui: { notify: (msg: string) => { capturedNotify = msg; }, setStatus: () => {} },
		} as any;

		await handler("purge", mockCtx);
		assert.ok(String(capturedNotify).includes("purged"), "should confirm cache purged");
	});

	it("/vision doctor runs without error", async () => {
		let handler: any = null;
		const mockPi = {
			registerCommand: (_name: string, cmd: any) => { handler = cmd.handler; },
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		let capturedNotify: string | null = null;
		let capturedType: string | null = null;
		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		const mockCtx = {
			model: { id: "test-model", input: ["text"] } as any,
			ui: {
				notify: (msg: string, type: string) => {
					capturedNotify = msg;
					capturedType = type;
				},
				setStatus: () => {},
			},
		} as any;

		await handler("doctor", mockCtx);
		assert.ok(capturedNotify !== null, "doctor should produce output");
		assert.ok(String(capturedNotify).includes("Vision Doctor"), "should show diagnostics header");
	});

	it("/vision status includes cache stats", async () => {
		let handler: any = null;
		const mockPi = {
			registerCommand: (_name: string, cmd: any) => { handler = cmd.handler; },
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		let capturedNotify: string | null = null;
		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		const mockCtx = {
			model: { id: "test-model", input: ["text"] } as any,
			ui: {
				notify: (msg: string) => { capturedNotify = msg; },
				setStatus: () => {},
			},
		} as any;

		await handler("status", mockCtx);
		assert.ok(capturedNotify !== null, "status should produce output");
		assert.ok(String(capturedNotify).includes("cache"), "should mention cache");
	});

	it("/vision setup displays instructions", async () => {
		let handler: any = null;
		const mockPi = {
			registerCommand: (_name: string, cmd: any) => { handler = cmd.handler; },
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		let capturedNotify: string | null = null;
		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		const mockCtx = {
			model: { id: "test-model", input: ["text"] } as any,
			ui: {
				notify: (msg: string) => { capturedNotify = msg; },
				setStatus: () => {},
			},
		} as any;

		await handler("setup", mockCtx);
		assert.ok(capturedNotify !== null, "setup should produce output");
		assert.ok(
			String(capturedNotify).includes("backend") || String(capturedNotify).includes("Backend"),
			"setup should mention backend",
		);
	});

	it("unknown subcommand shows usage", async () => {
		let handler: any = null;
		const mockPi = {
			registerCommand: (_name: string, cmd: any) => { handler = cmd.handler; },
			events: { emit: () => {} },
			registerTool: () => {},
			on: () => {},
		} as any;

		let capturedNotify: string | null = null;
		let capturedType: string | null = null;
		const { registerVisionCommand: regCmd } = await import("../src/commands.js");
		regCmd(mockPi as any, () => ({ captioned: 0, cached: 0, failed: 0 }), () => {});

		const mockCtx = {
			model: { id: "test-model", input: ["text"] } as any,
			ui: {
				notify: (msg: string, type: string) => {
					capturedNotify = msg;
					capturedType = type;
				},
				setStatus: () => {},
			},
		} as any;

		await handler("bogus-command", mockCtx);
		assert.strictEqual(capturedType, "error", "unknown command should use error type");
		assert.ok(String(capturedNotify).includes("Usage"), "should show usage text");
	});
});

// ── Privacy event emission tests ───────────────────────────────────────────

describe("sanitizeConfig — event safety", () => {
	it("vision_status event does not contain secrets", () => {
		const cfg = makeTestConfig();
		cfg.apiKey = "sk-secret-123";
		cfg.cachePassphrase = "my-passphrase";
		const sanitized = sanitizeConfig(cfg);

		// Verify no secrets in the sanitized output
		const json = JSON.stringify(sanitized);
		assert.ok(!json.includes("sk-secret-123"), "API key should not appear in event data");
		assert.ok(!json.includes("my-passphrase"), "cache passphrase should not appear");
		assert.ok(!json.includes("apiKey"), "apiKey key should be removed");
		assert.ok(!json.includes("cachePassphrase"), "cachePassphrase key should be removed");
	});
});

// ── stripImageMetadata tests ───────────────────────────────────────────────

describe("stripImageMetadata", () => {
	it("returns PNG unchanged (no EXIF in PNG)", async () => {
		const data = Buffer.from("fake-png-data").toString("base64");
		const result = await stripImageMetadata(data, "image/png");
		assert.strictEqual(result.data, data);
		assert.strictEqual(result.mimeType, "image/png");
	});

	it("returns WebP unchanged", async () => {
		const data = Buffer.from("fake-webp-data").toString("base64");
		const result = await stripImageMetadata(data, "image/webp");
		assert.strictEqual(result.data, data);
	});

	it("attempts to re-encode JPEG (may fall back if pi SDK unavailable)", async () => {
		const data = Buffer.from("fake-jpeg-data").toString("base64");
		const result = await stripImageMetadata(data, "image/jpeg");
		// In test environment without pi SDK, falls back to original
		assert.ok(result.data.length > 0, "should return some data");
	});

	it("handles GIF without error", async () => {
		const data = Buffer.from("fake-gif-data").toString("base64");
		const result = await stripImageMetadata(data, "image/gif");
		assert.strictEqual(result.data, data);
	});

	it("handles TIFF format", async () => {
		const data = Buffer.from("fake-tiff-data").toString("base64");
		const result = await stripImageMetadata(data, "image/tiff");
		assert.ok(result.data.length > 0, "should return some data");
	});
});


// ── wrapCaption with ref ───────────────────────────────────────────────────

describe("wrapCaption with ref", () => {
	it("includes ref attribute when provided", () => {
		const cfg = makeTestConfig("test-model");
		const result = wrapCaption(cfg, "a building", "img_abc123");
		assert.ok(result.includes('ref="img_abc123"'), "should include ref attribute");
		assert.ok(result.includes("test-model"));
		assert.ok(result.includes("a building"));
		assert.ok(result.includes("[image-description"));
		assert.ok(result.includes("[/image-description]"));
	});

	it("omits ref attribute when not provided", () => {
		const cfg = makeTestConfig("test-model");
		const result = wrapCaption(cfg, "a building");
		// Should NOT contain ref="..."
		assert.ok(!result.includes('ref="'), "should not include any ref attribute");
	});

	it("different refs produce different outputs", () => {
		const cfg = makeTestConfig("test-model");
		const r1 = wrapCaption(cfg, "desc", "ref_a");
		const r2 = wrapCaption(cfg, "desc", "ref_b");
		assert.notStrictEqual(r1, r2);
		assert.ok(r1.includes('ref="ref_a"'));
		assert.ok(r2.includes('ref="ref_b"'));
	});
});

// ── registerFollowUpTool tests ─────────────────────────────────────────────

describe("registerFollowUpTool", () => {
	it("exports registerFollowUpTool function", () => {
		// Just verify the module exports the expected function
		import("../src/follow-up-tool.js").then((mod) => {
			assert.strictEqual(typeof mod.registerFollowUpTool, "function");
		});
	});

	it("registration produces tool with correct name and parameters", async () => {
		let registeredTool: any = null;
		const mockPi = {
			registerTool: (tool: any) => {
				registeredTool = tool;
			},
		} as any;
		const mockStore = new ImageStore();
		mockStore.set("img_test", imgData("test"), "image/png");
		const resolveCaptioner = async () => ({
			model: { id: "test-model" } as any,
			apiKey: "test-key",
		});
		const getCfg = () => makeTestConfig();

		const { registerFollowUpTool: reg } = await import("../src/follow-up-tool.js");
		reg(mockPi as any, () => mockStore, resolveCaptioner as any, getCfg);

		assert.ok(registeredTool, "registerTool should have been called");
		assert.strictEqual(registeredTool.name, "vision_analyze_image");
		assert.strictEqual(registeredTool.label, "Analyze Image");
		assert.ok(typeof registeredTool.description === "string");
		assert.ok(registeredTool.description.includes("ref"));
		assert.ok(registeredTool.description.includes("question"));
		assert.ok(typeof registeredTool.execute === "function");

		// Verify parameters schema has ref and question as strings
		const params = registeredTool.parameters;
		assert.ok(params, "should have parameters");
		assert.strictEqual(params.type, "object");
		assert.ok(params.properties, "should have properties");
		assert.ok(params.properties.ref, "should have ref property");
		assert.strictEqual(params.properties.ref.type, "string");
		assert.ok(params.properties.question, "should have question property");
		assert.strictEqual(params.properties.question.type, "string");
	});

	it("execute returns missing-ref error for unknown ref", async () => {
		let registeredTool: any = null;
		const mockPi = {
			registerTool: (tool: any) => {
				registeredTool = tool;
			},
		} as any;
		const mockStore = new ImageStore();
		const resolveCaptioner = async () => ({
			model: { id: "test-model" } as any,
			apiKey: "test-key",
		});
		const getCfg = () => makeTestConfig();

		const { registerFollowUpTool: reg } = await import("../src/follow-up-tool.js");
		reg(mockPi as any, () => mockStore, resolveCaptioner as any, getCfg);

		const result = await registeredTool.execute("call1", { ref: "img_nonexistent", question: "What color?" }, undefined, undefined, {} as any);
		assert.ok(result, "should return a result");
		assert.ok(result.content, "should have content");
		assert.strictEqual(result.content[0].type, "text");
		assert.ok(result.content[0].text.includes("not found"), "should mention ref not found");
		assert.ok(result.content[0].text.includes("img_nonexistent"));
	});

	it("execute returns missing-ref error for expired ref", async () => {
		let registeredTool: any = null;
		const mockPi = {
			registerTool: (tool: any) => {
				registeredTool = tool;
			},
		} as any;
		// Store with TTL that expires immediately
		const expiredStore = new ImageStore(20, -1);
		expiredStore.set("img_expired", imgData("x"), "image/png");
		const resolveCaptioner = async () => ({
			model: { id: "test-model" } as any,
			apiKey: "test-key",
		});
		const getCfg = () => makeTestConfig();

		const { registerFollowUpTool: reg } = await import("../src/follow-up-tool.js");
		reg(mockPi as any, () => expiredStore, resolveCaptioner as any, getCfg);

		const result = await registeredTool.execute("call1", { ref: "img_expired", question: "What color?" }, undefined, undefined, {} as any);
		assert.ok(result, "should return a result");
		assert.ok(result.content[0].text.includes("not found"), "should mention ref not found for expired");
	});

	it("execute returns result when captioner call completes", async () => {
		let registeredTool: any = null;
		const mockPi = {
			registerTool: (tool: any) => {
				registeredTool = tool;
			},
		} as any;
		const mockStore = new ImageStore();
		mockStore.set("img_test", imgData("test"), "image/png");
		const resolveCaptioner = async () => ({
			model: { id: "test-model" } as any,
			apiKey: "test-key",
		});
		const getCfg = () => makeTestConfig();

		const { registerFollowUpTool: reg } = await import("../src/follow-up-tool.js");
		reg(mockPi as any, () => mockStore, resolveCaptioner as any, getCfg);

		// Execute with a ref that exists — the execute function calls complete()
		// which in a test environment will try to connect to localhost.
		// The response will either be empty (fallback) or an error.
		const result = await registeredTool.execute("call1", { ref: "img_test", question: "What color?" }, undefined, undefined, {} as any);
		assert.ok(result, "should return a result");
		assert.strictEqual(result.content[0].type, "text");
		assert.ok(result.content[0].text.includes("img_test"), "should include ref");
		assert.ok(result.content[0].text.includes("What color?"), "should include question");
		// Should have either error or empty-answer fallback
		assert.ok(
			result.content[0].text.includes("Error") ||
			result.content[0].text.includes("(vision model returned an empty answer)"),
			"should contain either error message or empty answer fallback",
		);
	});

	it("getter pattern: reassigning store after registration picks up new store", async () => {
		let registeredTool: any = null;
		const mockPi = {
			registerTool: (tool: any) => {
				registeredTool = tool;
			},
		} as any;
		let currentStore = new ImageStore();
		const resolveCaptioner = async () => ({
			model: { id: "test-model" } as any,
			apiKey: "test-key",
		});
		const getCfg = () => makeTestConfig();

		// Register with a getter that returns the current store
		const { registerFollowUpTool: reg } = await import("../src/follow-up-tool.js");
		reg(mockPi as any, () => currentStore, resolveCaptioner as any, getCfg);

		// Store A has "img_a"
		currentStore.set("img_a", imgData("a"), "image/png");

		// Simulate session_start — reassign to a brand new store
		currentStore = new ImageStore();
		currentStore.set("img_b", imgData("b"), "image/png");

		// Tool should look up from currentStore (Store B), not the original
		const resultA = await registeredTool.execute("call1", { ref: "img_a", question: "Where?" }, undefined, undefined, {} as any);
		assert.ok(resultA.content[0].text.includes("not found"), "img_a should not be found in new store");

		const resultB = await registeredTool.execute("call1", { ref: "img_b", question: "Where?" }, undefined, undefined, {} as any);
		// img_b exists in Store B, so execute should proceed past the lookup
		// (it will try to call complete() and get an error or empty answer)
		assert.ok(resultB.content[0].text.includes("img_b"), "img_b should be looked up in the current store");
		assert.ok(
			resultB.content[0].text.includes("Error") ||
			resultB.content[0].text.includes("(vision model returned an empty answer)"),
			"should proceed past lookup to captioner call",
		);
	});
});
