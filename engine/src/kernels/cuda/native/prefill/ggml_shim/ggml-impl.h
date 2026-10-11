// Fifi shadow header for the vendored llama.cpp MMQ sources (engine/vendor/ggml-cuda-mmq; decision 0049, CPF-V2).
// This is glue, not a patched vendor file: it stands in for ggml/src/ggml-impl.h.
// Text marked "derived" is copied from that file at llama.cpp acecd56032ddc34bada14a2d978f110d9c987095 (MIT, Copyright (c) 2023-2026
// The ggml authors; engine/vendor/ggml-cuda-mmq/LICENSE has the notice).
//
// Nothing in the kept sources needs a definition from ggml-impl.h. The file exists because common.cuh includes it.
#pragma once
