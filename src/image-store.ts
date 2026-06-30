/**
 * image-store.ts — Bounded per-session image store for follow-up vision queries.
 *
 * Stores image data keyed by a short ref ID (derived from the cache key).
 * The store is bounded to a configurable max entries (FIFO eviction).
 * Entries older than a TTL are evicted lazily on access.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface StoredImage {
	/** Base64-encoded image data. */
	data: string;
	/** MIME type (e.g. "image/png", "image/jpeg"). */
	mimeType: string;
	/** Timestamp when this entry was last accessed or stored. */
	lastAccess: number;
}

// ── ImageStore ─────────────────────────────────────────────────────────────

const DEFAULT_MAX_IMAGES = 20;
const DEFAULT_TTL_MS = 600_000; // 10 minutes

export class ImageStore {
	private entries = new Map<string, StoredImage>();
	private readonly max: number;
	private readonly ttlMs: number;

	constructor(max = DEFAULT_MAX_IMAGES, ttlMs = DEFAULT_TTL_MS) {
		this.max = max;
		this.ttlMs = ttlMs;
	}

	/**
	 * Store an image under the given ref. If the store is at capacity,
	 * the oldest entry (first inserted) is evicted.
	 */
	set(ref: string, data: string, mimeType: string): void {
		const now = Date.now();

		// If ref already exists, update in place and move to MRU
		if (this.entries.has(ref)) {
			this.entries.delete(ref);
			this.entries.set(ref, { data, mimeType, lastAccess: now });
			return;
		}

		// Evict expired entries before checking capacity
		this.evictExpired();

		// Evict oldest entry if at capacity
		while (this.entries.size >= this.max) {
			const oldest = this.entries.keys().next().value;
			if (oldest === undefined) break;
			this.entries.delete(oldest);
		}

		this.entries.set(ref, { data, mimeType, lastAccess: now });
	}

	/**
	 * Look up a stored image by ref. Returns undefined if the ref is unknown
	 * or its TTL has expired.
	 */
	get(ref: string): StoredImage | undefined {
		const entry = this.entries.get(ref);
		if (!entry) return undefined;

		// Check TTL
		if (Date.now() - entry.lastAccess > this.ttlMs) {
			this.entries.delete(ref);
			return undefined;
		}

		// Touch: move to MRU
		entry.lastAccess = Date.now();
		this.entries.delete(ref);
		this.entries.set(ref, entry);
		return entry;
	}

	/** Check if a ref exists and is not expired. */
	has(ref: string): boolean {
		return this.get(ref) !== undefined;
	}

	/** Remove a specific entry. */
	delete(ref: string): void {
		this.entries.delete(ref);
	}

	/** Clear all stored images. */
	clear(): void {
		this.entries.clear();
	}

	/** Current number of stored images. */
	get size(): number {
		return this.entries.size;
	}

	/** Remove entries whose TTL has expired. */
	private evictExpired(): void {
		const now = Date.now();
		for (const [ref, entry] of this.entries) {
			if (now - entry.lastAccess > this.ttlMs) {
				this.entries.delete(ref);
			}
		}
	}
}
