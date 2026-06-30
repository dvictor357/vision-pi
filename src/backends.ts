/**
 * backends.ts — Vision backend definitions (MLX, Ollama), preset catalog,
 * and the VisionBackendRegistry for built-in + custom OpenAI-compatible backends.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export type BackendName = string;
export type PresetName = "light" | "balanced" | "capable";

export interface BackendDef {
	/** Backend identifier (matches the key in the registry). */
	name?: string;
	label: string;
	baseUrl: string;
	apiKey: string;
	/** Per-backend preset → model id. Bump up as the machine allows. */
	presets: Record<string, string>;
	/** Default preset to use when switching to this backend. */
	defaultPreset: PresetName;
	/** Whether this is a user-registered custom backend. */
	isCustom?: boolean;
	/** Setup instructions, given the resolved model id. */
	setup: (model: string) => string[];
}

// ── Backend catalog ─────────────────────────────────────────────────────────

export const BACKENDS: Record<string, BackendDef> = {
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
		defaultPreset: "light",
		setup: (model) => [
			"Quick start (Ollama):",
			"  1. Install Ollama → https://ollama.com",
			`  2. ollama pull ${model}`,
			"  3. /vision test",
		],
	},
	// Apple Silicon, via the mlx-vlm server (OpenAI-compatible). Fast on M-series.
	mlx: {
		label: "MLX (Apple Silicon)",
		baseUrl: "http://localhost:8081/v1",
		apiKey: "mlx",
		presets: {
			light: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
			balanced: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
			capable: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
		},
		defaultPreset: "light",
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

export const PRESET_NAMES: PresetName[] = ["light", "balanced", "capable"];

/** Default preset names as a Set for fast lookup. */
export const PRESET_NAMES_SET = new Set<string>(["light", "balanced", "capable"]);

/** Default backend when nothing is configured. */
export const DEFAULT_BACKEND = "mlx";

/** System prompt that steers the captioner. */
export const DEFAULT_PROMPT =
	"You are the eyes of a coding agent that CANNOT see images. " +
	"Describe the given image so the agent can fully understand and act on it. " +
	"Transcribe ALL visible text verbatim (code, errors, labels, UI text, numbers). " +
	"If it is a UI/screenshot: describe layout, components, state, and any errors. " +
	"If it is a diagram/chart: describe structure, relationships, axes, and values. " +
	"Be precise, complete, and factual. Do not speculate beyond what is visible.";

// ── Type guards ────────────────────────────────────────────────────────────

export const BUILTIN_BACKEND_NAMES = new Set(["ollama", "mlx"]);

export function isBuiltinBackend(s: string): boolean {
	return BUILTIN_BACKEND_NAMES.has(s);
}

/** Check if a string is a known backend name (built-in or registered). */
export function isBackend(s: string): boolean {
	return BUILTIN_BACKEND_NAMES.has(s) || visionBackendRegistry.has(s);
}

export function isPreset(s: string): s is PresetName {
	return s === "light" || s === "balanced" || s === "capable";
}

// ── VisionBackendRegistry ─────────────────────────────────────────────────

/**
 * Registry for vision backends. Holds built-in MLX/Ollama and any
 * user-registered custom OpenAI-compatible backends.
 *
 * The singleton `visionBackendRegistry` is used by config and commands.
 * Create a fresh instance for testing.
 */
export class VisionBackendRegistry {
	private backends = new Map<string, BackendDef>();

	constructor() {
		this.registerBuiltins();
	}

	private registerBuiltins(): void {
		for (const [name, def] of Object.entries(BACKENDS)) {
			this.backends.set(name, { ...def, name, isCustom: false });
		}
	}

	/** Register a backend. Overwrites if name already exists. */
	register(name: string, def: BackendDef): void {
		this.backends.set(name, { ...def, name });
	}

	/** Look up a registered backend by name. */
	get(name: string): BackendDef | undefined {
		return this.backends.get(name);
	}

	/** Check if a backend name is registered. */
	has(name: string): boolean {
		return this.backends.has(name);
	}

	/** Remove a custom backend by name. Returns false if built-in or not found. */
	unregister(name: string): boolean {
		if (BUILTIN_BACKEND_NAMES.has(name)) return false;
		return this.backends.delete(name);
	}

	/** List all registered backends. */
	list(): BackendDef[] {
		return [...this.backends.values()];
	}

	/** Get all backend names. */
	names(): string[] {
		return [...this.backends.keys()];
	}

	/** Check if a backend is built-in. */
	isBuiltin(name: string): boolean {
		return BUILTIN_BACKEND_NAMES.has(name);
	}
}

/** Singleton registry shared across the extension. */
export const visionBackendRegistry = new VisionBackendRegistry();

/**
 * Interface for persisted custom backend definitions (as stored in settings.json).
 */
export interface PersistedCustomBackend {
	baseUrl: string;
	apiKey?: string;
	presets?: Record<string, string>;
}

/**
 * Load custom backends from persisted config into the registry.
 * Call this at startup with the custom backends stashed in settings.
 */
export function loadCustomBackends(
	registeredBackends: Record<string, PersistedCustomBackend>,
): void {
	for (const [name, def] of Object.entries(registeredBackends)) {
		if (!name || typeof name !== "string") continue;
		visionBackendRegistry.register(name, {
			label: name,
			baseUrl: def.baseUrl,
			apiKey: def.apiKey ?? "",
			presets: def.presets ?? { light: "gpt-4o" },
			defaultPreset: "light",
			isCustom: true,
			setup: (model) => [
				`Custom backend "${name}":`,
				`  Server: ${def.baseUrl}`,
				`  Model: ${model}`,
				"  Ensure your server is running and reachable.",
				"  /vision test",
			],
		});
	}
}
