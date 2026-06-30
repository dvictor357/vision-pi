/**
 * vision.ts — pi extension entrypoint for universal image understanding.
 *
 * This file IS the documented pi entrypoint (package.json → extensions/vision.ts).
 * Implementation lives in ../src/* modules.
 *
 * See ../src/orchestrator.ts for the default export that wires everything.
 *
 * Commands:  /vision  · /vision on|off · /vision backend <ollama|mlx>
 *            /vision preset <name> · /vision model <id>
 *            /vision test · /vision clear · /vision setup
 */

export { default } from "../src/orchestrator.js";
