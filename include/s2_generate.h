#pragma once
// s2_generate.h — Autoregressive generation loop
//
// Port of generate() from ggml_pure.py.
// Combines Slow-AR prefill + step-by-step generation + Fast-AR decode.

#include "s2_model.h"
#include "s2_sampler.h"
#include "s2_tokenizer.h"
#include "s2_prompt.h"

#include <cstdint>
#include <vector>
#include <functional>

namespace s2 {

struct GenerateParams {
    int32_t max_new_tokens          = 512;
    float   temperature             = 0.7f;
    float   top_p                   = 0.7f;
    int32_t top_k                   = 30;
    int32_t min_tokens_before_end   = 64;
    int32_t n_threads               = 4;
    bool    verbose                 = true;
    // Silence runway: frames forced as the first generation outputs instead
    // of sampling, frame-major (t * num_codebooks + cb), codebook-space.
    // Filled by the pipeline from canonical silence codes so the first
    // phoneme never lands on frame zero — the Peter checkpoint otherwise
    // clips weak onsets (/h/) stochastically. Empty = disabled.
    std::vector<int32_t> seed_frames;
    // KV cache reservation floor, in tokens beyond the prompt. The pipeline
    // sizes the KV cache for max(max_new_tokens, kv_reserve_tokens) so a
    // warm-up call can reserve the full serving ceiling while generating
    // only a handful of frames (boot in seconds, not minutes). 0 = off.
    int32_t kv_reserve_tokens = 0;
};

// Generate VQ codes autoregressively.
// Returns flattened (num_codebooks, T_generated) codes in row-major order.
struct GenerateResult {
    std::vector<int32_t> codes;
    int32_t num_codebooks = 0;
    int32_t n_frames      = 0;
};

GenerateResult generate(
    SlowARModel & model,
    const TokenizerConfig & config,
    const PromptTensor & prompt,
    const GenerateParams & params
);

} // namespace s2
