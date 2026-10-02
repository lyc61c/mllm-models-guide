# MLLM expanded tutorial — independent review notes

Review status: preliminary content and figure review complete; final Markdown/HTML render review waits for the explicit final-file freeze.

## Figure provenance and integrity

- Inspected all 15 files listed in `docs/tutorials/reconstructed-figure-sources.json`.
- Every reconstructed PNG exists and its SHA-256 matches the recorded value.
- The generated figures consistently identify themselves as reconstructions rather than official paper figures.
- Closed-model internals are generally kept in gray dashed boxes and are not populated with invented layer or parameter counts.
- The GPT-6 Astra / GPT-6.1 Sol figure correctly keeps external tools outside the model, routes screenshot observations to visual input, and presents the two disclosed training elements in parallel rather than as a fabricated fixed stage sequence.

## Items to verify in the frozen tutorial

1. The Qwen3.6 figure labels tool feedback as “text or screenshots” and returns it to a combined `Text / reasoning history / tool results` box that feeds token embeddings. The Qwen3.8 figure similarly returns “Screenshots or results” to a text/history box. Their frozen captions or surrounding prose should state that screenshots remain visual observations and the return arrow is a system-level simplification; it must not imply screenshots are converted to text embeddings.
2. The Gemini 3.8 Flash figure labels a `Reasoning / Plan` box as “internal chain of thought” and maps effort to a reasoning-token budget. The frozen prose should retain the current limitation that this box is a behavioral abstraction, not a disclosed separate neural module or a claim that hidden reasoning is exposed.
3. The Gemini 3.1 Pro figure draws tool results both toward the model and toward multimodal context. The frozen caption should make clear that these are two views of the same feedback concept, not two independently disclosed internal routes.
4. The Kimi K3 QAT shorthand must be scoped in the frozen caption or prose to the disclosed quantized routed-expert weights/activations; shared experts and other components must not be implied to use the same low precision.
5. For every reconstructed figure in the frozen Markdown, verify that the caption says it is reconstructed/non-official and that any gray-box uncertainty described in the figure remains explicit in the text.

## Preliminary content assessment

- `closed_models.md` maintains a clear boundary between public interface/system behavior and undisclosed neural internals. It repeatedly avoids inferring encoders, expert counts, loss recipes, freeze schedules, or reward details from benchmark behavior.
- `omni_models.md` is technically detailed enough to explain structures, training stages, and evaluation conditions rather than listing product features. It distinguishes HTTP text/tool services from realtime speech branches and distinguishes paper-original figures from reconstructions.
- No preliminary blocking factual issue was found. Final verdict remains pending because the canonical Markdown and generated HTML have not yet been frozen and hash-checked.
