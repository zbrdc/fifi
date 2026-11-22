# Shared architecture TODO

Scope: GGUF serving only. Share decisions that must agree across serving and diagnostics. Keep model-specific execution order, numerical paths, and device ownership with their families.

- [x] **Share `gemma4` build eligibility.** The family name is recognized on every build, but the starter only serves Nemotron Bolt E4B on CUDA. Startup rejects a build without CUDA before reading sampling defaults. `check` reports it as unsupported for both explicitly named and GGUF-detected entries.
- [x] **Share Bolt checkpoint identity between boot and preflight.** Simplified startup already calls preflight. The check distinguishes a compatible identity from tensor validation, memory fit, and qualification.
- [x] **Use configured Qwen GGUF options in support-bundle preflight.** Use the loader's settings resolver for context, KV, and content-pin syntax. The loader still verifies the digest and decides memory admission.
- [x] **Test scheduler invariants across single and batched workers.** Check that final-prefill cancellation publishes no tokens and a memory denial restarts with the request's own saved prefix.
- [x] **Make Qwen CUDA diagnostic file selection match startup.** Accept an explicit `.gguf` without inspecting unrelated weights. The weight check names the selected checkpoint, and estimates use its byte count, including split shards.

See [model support](reference/models.md) and [diagnostic commands](how-to/diagnostics.md).
