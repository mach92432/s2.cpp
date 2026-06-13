#include "../include/s2_generate.h"
#include <chrono>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>

namespace s2 {

GenerateResult generate(
    SlowARModel & model,
    const TokenizerConfig & config,
    const PromptTensor & prompt,
    const GenerateParams & params
) {
    GenerateResult out;
    out.num_codebooks = model.hparams().num_codebooks;
    if (out.num_codebooks <= 0) out.num_codebooks = 1;

    const int32_t vocab_size   = model.hparams().vocab_size;
    const int32_t sem_begin    = model.hparams().semantic_begin_id;
    const int32_t sem_end      = model.hparams().semantic_end_id;
    const int32_t codebook_size = model.hparams().codebook_size;
    const int32_t im_end_id    = config.im_end_id;
    const int32_t num_cb       = out.num_codebooks;

    // Build semantic mask: -inf everywhere except [sem_begin, sem_end] and im_end
    std::vector<float> sem_mask(vocab_size, -std::numeric_limits<float>::infinity());
    for (int32_t i = sem_begin; i <= sem_end && i < vocab_size; ++i) {
        sem_mask[i] = 0.0f;
    }
    if (im_end_id >= 0 && im_end_id < vocab_size) {
        sem_mask[im_end_id] = 0.0f;
    }

    // Prefill
    // eval_cached expects time-major: flat_tokens[t * (num_cb+1) + cb]
    // but prompt.data is codebook-major: data[cb * cols + t]
    // Transpose (num_cb+1, T) → (T, num_cb+1)
    const int32_t rows = prompt.rows;  // num_codebooks + 1
    const int32_t cols = prompt.cols;  // T
    std::vector<int32_t> prompt_tm(static_cast<size_t>(rows) * cols);
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t c = 0; c < cols; ++c) {
            prompt_tm[static_cast<size_t>(c) * rows + r] = prompt.data[static_cast<size_t>(r) * cols + c];
        }
    }

    StepResult state;
    if (params.verbose) {
        std::cout << "[Generate] Prefilling " << prompt.cols << " tokens..." << std::endl;
    }
    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t_pf0 = clk::now();
    if (!model.prefill(prompt_tm, prompt.cols, params.n_threads, state)) {
        std::cerr << "[Generate] Prefill failed." << std::endl;
        return out;
    }
    const double ms_prefill = ms_since(t_pf0, clk::now());
    // Per-stage accumulators for the AR loop profile
    double ms_step = 0.0, ms_fast = 0.0, ms_sample = 0.0;
    auto t_loop0 = clk::now();

    // Apply semantic mask to initial logits
    auto apply_mask_and_sample = [&](const std::vector<float> & logits,
                                     bool block_im_end) -> int32_t {
        std::vector<float> biased(vocab_size);
        for (int32_t i = 0; i < vocab_size; ++i) {
            biased[i] = logits[i] + sem_mask[i];
        }
        if (block_im_end && im_end_id >= 0 && im_end_id < vocab_size) {
            biased[im_end_id] = -std::numeric_limits<float>::infinity();
        }
        SamplerParams sparams;
        sparams.temperature = params.temperature;
        sparams.top_p       = params.top_p;
        sparams.top_k       = params.top_k;
        // Pass im_end_id so it is always eligible for sampling when not blocked,
        // regardless of GPU numerical precision (fixes NVIDIA/NV_coopmat2 EOS dropout).
        const int32_t force_id = block_im_end ? -1 : im_end_id;
        return sample_token(biased.data(), vocab_size, sparams, force_id);
    };

    // Sample first main_token
    bool block_end = (params.min_tokens_before_end > 0);
    int32_t main_token = apply_mask_and_sample(state.logits, block_end);

    // Pre-allocate codes array
    out.codes.resize(static_cast<size_t>(num_cb) * params.max_new_tokens, 0);
    out.n_frames = 0;

    std::vector<float> fast_logits;

    SamplerParams sparams;
    sparams.temperature = params.temperature;
    sparams.top_p       = params.top_p;
    sparams.top_k       = params.top_k;

    // RAS state
    std::vector<int32_t> ras_window;
    const int32_t ras_window_size = 10;
    const float ras_high_temp     = 1.0f;
    const float ras_high_top_p    = 0.9f;

    if (params.verbose) {
        std::cout << "[Generate] Generating (max " << params.max_new_tokens << " tokens)..." << std::endl;
    }

    const int32_t n_seed = params.seed_frames.empty()
        ? 0
        : (int32_t)(params.seed_frames.size() / num_cb);
    if (params.verbose && n_seed > 0) {
        std::cout << "[Generate] Silence runway: seeding " << n_seed
                  << " frames; semantic codes:";
        for (int32_t t = 0; t < n_seed; ++t) {
            std::cout << " " << params.seed_frames[(size_t)t * num_cb];
        }
        std::cout << std::endl;
    }

    int32_t step = 0;
    while (main_token != im_end_id && step < params.max_new_tokens) {
        // Silence runway: force the first n_seed frames from known-silence
        // codes instead of the sampled token, so the utterance opens from
        // quiet and the first phoneme survives. The sampled token (and RAS)
        // resume at step == n_seed with the KV state conditioned on silence.
        const bool seeded = step < n_seed;
        if (seeded) {
            main_token = params.seed_frames[(size_t)step * num_cb] + sem_begin;
        }

        // RAS check
        if (!seeded && !ras_window.empty() &&
            std::find(ras_window.begin(), ras_window.end(), main_token) != ras_window.end() &&
            main_token >= sem_begin && main_token <= sem_end)
        {
            // Resample with high temperature
            std::vector<float> biased(vocab_size);
            for (int32_t i = 0; i < vocab_size; ++i) {
                biased[i] = state.logits[i] + sem_mask[i];
            }
            if (step < params.min_tokens_before_end && im_end_id >= 0 && im_end_id < vocab_size) {
                biased[im_end_id] = -std::numeric_limits<float>::infinity();
            }
            SamplerParams ras_sparams;
            ras_sparams.temperature = ras_high_temp;
            ras_sparams.top_p       = ras_high_top_p;
            ras_sparams.top_k       = params.top_k;
            const int32_t ras_force_id = (step < params.min_tokens_before_end) ? -1 : im_end_id;
            main_token = sample_token(biased.data(), vocab_size, ras_sparams, ras_force_id);
        }

        // Update RAS window (seeded silence frames stay out of it — they are
        // intentional repeats, and counting them would trigger a high-temp
        // RAS resample on the first naturally-sampled frame after the runway,
        // right at the onset we are protecting)
        if (!seeded) {
            ras_window.push_back(main_token);
            if ((int32_t)ras_window.size() > ras_window_size) {
                ras_window.erase(ras_window.begin());
            }
        }

        // sem_code: convert from vocabulary space to codebook space
        int32_t sem_code = main_token - sem_begin;
        if (sem_code < 0)           sem_code = 0;
        if (sem_code >= codebook_size) sem_code = codebook_size - 1;

        // Build codebook prefix for fast decoder
        // codebooks_cb[0] = sem_code (codebook-space index)
        std::vector<int32_t> codebooks_cb;
        codebooks_cb.reserve(num_cb);
        codebooks_cb.push_back(sem_code);

        // Fast AR: generate remaining num_cb-1 codebooks
        // (seeded frames take all codebooks verbatim from the silence codes)
        auto t_fa0 = clk::now();
        if (seeded) {
            for (int32_t cb_idx = 1; cb_idx < num_cb; ++cb_idx) {
                codebooks_cb.push_back(
                    params.seed_frames[(size_t)step * num_cb + cb_idx]);
            }
        } else
        for (int32_t cb_idx = 1; cb_idx < num_cb; ++cb_idx) {
            // prefix = codebooks_cb[0..cb_idx-1]
            if (!model.fast_decode(state.hidden, codebooks_cb, params.n_threads, fast_logits)) {
                std::cerr << "[Generate] fast_decode failed at cb " << cb_idx << std::endl;
                // fill remaining with zeros
                for (int32_t r = cb_idx; r < num_cb; ++r) {
                    codebooks_cb.push_back(0);
                }
                break;
            }
            int32_t cb_token = sample_token(fast_logits.data(), (int32_t)fast_logits.size(), sparams);
            codebooks_cb.push_back(cb_token);
        }

        ms_fast += ms_since(t_fa0, clk::now());

        // Store frame: codes[cb * n_frames_capacity + step] = codebooks_cb[cb]
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            out.codes[static_cast<size_t>(cb) * params.max_new_tokens + step] = codebooks_cb[cb];
        }
        out.n_frames++;

        // Build step_input: [main_token (vocab space), codebooks_cb[0..num_cb-1] (codebook space)]
        std::vector<int32_t> step_input(num_cb + 1);
        step_input[0] = main_token;
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            step_input[cb + 1] = codebooks_cb[cb];
        }

        auto t_st0 = clk::now();
        if (!model.step(step_input, params.n_threads, state)) {
            std::cerr << "[Generate] step() failed at step " << step << std::endl;
            break;
        }
        ms_step += ms_since(t_st0, clk::now());

        step++;
        if (params.verbose && step % 50 == 0) {
            std::cout << "\r[Generate] " << step << " / " << params.max_new_tokens << " tokens..." << std::flush;
        }

        // Apply semantic mask and sample next main token
        auto t_sm0 = clk::now();
        bool block_next_end = (step < params.min_tokens_before_end);
        main_token = apply_mask_and_sample(state.logits, block_next_end);
        ms_sample += ms_since(t_sm0, clk::now());
    }

    if (params.verbose) {
        const double ms_loop = ms_since(t_loop0, clk::now());
        std::cout << std::endl;
        std::cout << "[Generate] Done: " << out.n_frames << " frames generated." << std::endl;
        if (out.n_frames > 0) {
            const double f = out.n_frames;
            std::cout << "[Timing] prefill " << (int)ms_prefill << "ms; loop "
                      << (int)ms_loop << "ms for " << out.n_frames << " frames ("
                      << (int)(f * 1000.0 / ms_loop) << " fps) — per frame: step "
                      << ms_step / f << "ms, fastAR " << ms_fast / f
                      << "ms, sample " << ms_sample / f << "ms, other "
                      << (ms_loop - ms_step - ms_fast - ms_sample) / f << "ms"
                      << std::endl;
        }
    }

    // Compact codes from (num_cb, max_tokens) stride to (num_cb, n_frames) row-major
    const int32_t n_frames = out.n_frames;
    if (n_frames < params.max_new_tokens) {
        std::vector<int32_t> compacted(static_cast<size_t>(num_cb) * n_frames);
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            for (int32_t t = 0; t < n_frames; ++t) {
                compacted[static_cast<size_t>(cb) * n_frames + t] =
                    out.codes[static_cast<size_t>(cb) * params.max_new_tokens + t];
            }
        }
        out.codes = std::move(compacted);
    } else {
        out.codes.resize(static_cast<size_t>(num_cb) * n_frames);
    }

    return out;
}

} // namespace s2
