#include "../include/s2_pipeline.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <gguf.h>  // ← AJOUTE CETTE LIGNE

namespace s2 {

Pipeline::Pipeline() {}
Pipeline::~Pipeline() {
    std::cout << "[Pipeline] Cleanup complete" << std::endl;
    // model_ se détruit automatiquement → libère ctx_kv_, kv_buf_, etc.
}

bool Pipeline::init(const PipelineParams & params) {
    std::cout << "--- Pipeline Init ---" << std::endl;

    // Determine GPU devices (-1 = CPU, -2 = unset/follow model)
    int model_gpu = params.vulkan_device;
    int codec_gpu = params.codec_vulkan_device;
    // Only inherit model GPU if codec device was truly unset (use -2 for unset)
    if (codec_gpu == -2) codec_gpu = model_gpu;

    std::cout << "GPU assignment: model -> GPU " << model_gpu
              << ", codec -> GPU " << codec_gpu << std::endl;

    // Charger le tokenizer
    if (!tokenizer_.load(params.tokenizer_path)) {
        std::cerr << "Pipeline error: could not load tokenizer from "
                  << params.tokenizer_path << std::endl;
        return false;
    }

    // Charger le modèle principal sur le GPU assigné
    std::cout << "Loading model from: " << params.model_path << std::endl;
    std::cout << "Loading model on GPU " << model_gpu << "..." << std::endl;
    if (!model_.load(params.model_path, model_gpu)) {
        std::cerr << "Pipeline error: could not load model on GPU " << model_gpu << std::endl;
        return false;
    }
    std::cout << "Model loaded on GPU " << model_gpu << "." << std::endl;

    // NOUVEAU: Choisir le fichier pour le codec
    std::string codec_path = params.codec_model_path.empty() 
        ? params.model_path 
        : params.codec_model_path;
    
    std::cout << "Loading codec from: " << codec_path << std::endl;

    // Load codec — use CPU directly when codec_gpu < 0, or try GPU then CPU
    std::cout << "Loading codec (device=" << codec_gpu << ")..." << std::endl;
    bool codec_loaded = false;

    if (codec_gpu >= 0) {
        std::cout << "Trying codec on GPU " << codec_gpu << "..." << std::endl;
        if (codec_.load(codec_path, codec_gpu)) {
            std::cout << "Codec loaded on GPU " << codec_gpu << "." << std::endl;
            codec_loaded = true;
            codec_on_gpu_ = true;  // actual outcome, not requested config
        } else {
            std::cout << "Codec GPU failed, loading on CPU..." << std::endl;
        }
    }

    if (!codec_loaded) {
        if (codec_.load(codec_path, -1)) {
            std::cout << "Codec loaded on CPU." << std::endl;
            codec_loaded = true;
        }
    }

    if (!codec_loaded) {
        std::cerr << "Pipeline error: could not load codec from " << codec_path << std::endl;
        return false;
    }

    // Synchroniser la config tokenizer avec le modèle
    {
        const ModelHParams & hp = model_.hparams();
        TokenizerConfig & tc    = tokenizer_.config();
        if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
        if (hp.semantic_end_id   > 0) tc.semantic_end_id   = hp.semantic_end_id;
        if (hp.num_codebooks     > 0) tc.num_codebooks     = hp.num_codebooks;
        if (hp.codebook_size     > 0) tc.codebook_size     = hp.codebook_size;
        if (hp.vocab_size        > 0) tc.vocab_size        = hp.vocab_size;
    }

    // Init-time encodes (reference + silence) run on a TEMPORARY CPU codec
    // when the serving codec is on GPU: encode's compute buffer is ~590MB
    // for the 7s reference (measured 2026-06-10) and would otherwise stay
    // resident on a card that doesn't have it to spare. The temp instance
    // frees when init() returns; decode stays on the GPU codec.
    AudioCodec init_cpu_codec;
    AudioCodec * init_encoder = &codec_;
    if (codec_on_gpu_ && init_cpu_codec.load(codec_path, -1)) {
        init_encoder = &init_cpu_codec;
        std::cout << "[init] Using temporary CPU codec for init-time encodes." << std::endl;
    }

    // --- Charger la référence audio une seule fois au démarrage ---
    if (std::FILE* file = std::fopen("reference.wav", "rb")) {
        std::fclose(file);
        AudioData ref_audio;
        if (load_audio("reference.wav", ref_audio, codec_.sample_rate())) {
            std::vector<int32_t> ref_codes;
            int32_t T_prompt = 0;
            if (init_encoder->encode(ref_audio.samples.data(), (int32_t)ref_audio.samples.size(),
                              params.gen.n_threads, ref_codes, T_prompt)) {
                reference_embedding_ = std::string((const char*)ref_codes.data(),
                                                    ref_codes.size() * sizeof(int32_t));
                reference_loaded_ = true;
                std::cout << "Reference audio loaded." << std::endl;
            }
        }
    }

    // --- Charger le texte de référence une seule fois au démarrage ---
    if (std::FILE* file = std::fopen("reference.txt", "r")) {
        std::fclose(file);
        std::ifstream txt_file("reference.txt");
        std::getline(txt_file, reference_text_);
        std::cout << "Reference text loaded." << std::endl;
    }

    // --- Canonical silence codes for the generation runway ---
    // Encode 750ms of digital silence and keep middle frames (encoder edge
    // windows excluded). These are ground-truth "silence" in VQ space,
    // independent of how tightly the reference clip was cut.
    {
        std::vector<float> zeros((size_t)codec_.sample_rate() * 3 / 4, 0.0f);
        std::vector<int32_t> codes;
        int32_t n_frames = 0;
        if (init_encoder->encode(zeros.data(), (int32_t)zeros.size(),
                          params.gen.n_threads, codes, n_frames) &&
            n_frames >= 12 && !codes.empty()) {
            const int32_t num_cb = (int32_t)(codes.size() / n_frames);
            const int32_t keep = 8, start = (n_frames - keep) / 2;
            silence_codes_.resize((size_t)num_cb * keep);
            for (int32_t cb = 0; cb < num_cb; ++cb) {
                for (int32_t t = 0; t < keep; ++t) {
                    silence_codes_[(size_t)cb * keep + t] =
                        codes[(size_t)cb * n_frames + start + t];
                }
            }
            silence_frames_ = keep;
            std::cout << "[init] Silence codes ready (" << keep
                      << " frames); semantic row:";
            for (int32_t t = 0; t < keep; ++t) {
                std::cout << " " << silence_codes_[t];
            }
            std::cout << std::endl;
        } else {
            std::cout << "[init] Silence-code encoding unavailable; runway disabled."
                      << std::endl;
        }
    }

    initialized_ = true;
    return true;
}

// Chunked codec decode: bounds the decode compute buffer. A whole-clip GPU
// decode of 512 frames needs a ~2.9GB CUDA buffer (measured 2026-06-10);
// decoding fixed windows caps it at ~64 frames (~360MB) regardless of reply
// length. Each window carries OVL frames of lead context so conv/attention
// edges heal, and the splice is crossfaded over ~1 frame of samples.
static bool decode_chunked(AudioCodec & codec,
                           const int32_t * codes, int32_t n_frames,
                           int32_t num_cb, int32_t n_threads,
                           std::vector<float> & audio_out) {
    // 24+8 frames/window ≈ 135MB decode buffer — sized to fit the ~776MB
    // free-but-fragmented VRAM measured 2026-06-10 (48+8 wanted 271MB and
    // still failed contiguous allocation on this card).
    const int32_t CHUNK = 24;
    const int32_t OVL   = 8;
    if (num_cb <= 0 || n_frames <= 0) return false;
    if (n_frames <= CHUNK + OVL) {
        return codec.decode(codes, n_frames, n_threads, audio_out);
    }
    audio_out.clear();
    std::vector<int32_t> win;
    std::vector<float> win_audio;
    int32_t spf = 0;  // samples per frame, learned from the first window
    int32_t start = 0;
    while (start < n_frames) {
        const int32_t lead = (start == 0) ? 0 : OVL;
        const int32_t core = std::min(CHUNK, n_frames - start);
        const int32_t w0   = start - lead;
        const int32_t wlen = lead + core;
        win.resize((size_t)num_cb * wlen);
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            for (int32_t t = 0; t < wlen; ++t) {
                win[(size_t)cb * wlen + t] = codes[(size_t)cb * n_frames + w0 + t];
            }
        }
        win_audio.clear();
        if (!codec.decode(win.data(), wlen, n_threads, win_audio)) return false;
        if (spf == 0) {
            spf = (int32_t)(win_audio.size() / wlen);
            if (spf <= 0) return false;
        }
        const size_t keep_begin = (size_t)lead * spf;
        // Crossfade the previous window's tail into this window's rendering
        // of the same time span (both have full decode context there).
        const size_t xf = (start == 0)
            ? 0
            : std::min({(size_t)spf, audio_out.size(), keep_begin});
        const size_t out_tail = audio_out.size() - xf;
        for (size_t i = 0; i < xf; ++i) {
            const float a = audio_out[out_tail + i];
            const float b = win_audio[keep_begin - xf + i];
            const float w = (float)(i + 1) / (float)(xf + 1);
            audio_out[out_tail + i] = a * (1.0f - w) + b * w;
        }
        audio_out.insert(audio_out.end(),
                         win_audio.begin() + keep_begin, win_audio.end());
        start += core;
    }
    return true;
}

bool Pipeline::synthesize(const PipelineParams & params) {
    if (!initialized_) {
        std::cerr << "Pipeline not initialized." << std::endl;
        return false;
    }

    std::cout << "--- Pipeline Synthesize ---" << std::endl;
    std::cout << "Text: " << params.text << std::endl;

    const int32_t num_codebooks = model_.hparams().num_codebooks;

    // 1. Audio Prompt Loading
    std::vector<int32_t> ref_codes;
    int32_t T_prompt = 0;
    if (!params.prompt_audio_path.empty()) {
        std::cout << "Loading reference audio: " << params.prompt_audio_path << std::endl;
        AudioData ref_audio;
        if (load_audio(params.prompt_audio_path, ref_audio, codec_.sample_rate())) {
            if (!codec_.encode(ref_audio.samples.data(), (int32_t)ref_audio.samples.size(),
                               params.gen.n_threads, ref_codes, T_prompt)) {
                std::cerr << "Pipeline warning: encode failed." << std::endl;
                ref_codes.clear();
                T_prompt = 0;
            }
        }
    }

    // 2. Build Prompt Tensor
    PromptTensor prompt = build_prompt(
        tokenizer_, params.text, params.prompt_text,
        ref_codes.empty() ? nullptr : ref_codes.data(),
        num_codebooks, T_prompt);

    // Dans synthesize_to_buffer() - APRÈS la construction du prompt (~ligne 150-170)

    // 3. Setup KV Cache - FIX SERVEUR (ne plus appeler init_kv_cache() à chaque fois)
    GenerateParams gen = params.gen;

    // Silence runway: force canonical silence codes (encoded from digital
    // silence at init — see silence_codes_) as the opening of every
    // generation, so the first phoneme never lands on frame zero.
    const int32_t SEED_N = 6;
    if (silence_frames_ >= SEED_N &&
        silence_codes_.size() == (size_t)num_codebooks * silence_frames_) {
        gen.seed_frames.resize((size_t)SEED_N * num_codebooks);
        for (int32_t t = 0; t < SEED_N; ++t) {
            for (int32_t cb = 0; cb < num_codebooks; ++cb) {
                gen.seed_frames[(size_t)t * num_codebooks + cb] =
                    silence_codes_[(size_t)cb * silence_frames_ + t];
            }
        }
    }

    int32_t max_seq_len = prompt.cols + std::max(gen.max_new_tokens, gen.kv_reserve_tokens);

    if (!kv_cache_initialized_ || max_seq_len > kv_cache_max_len_) {
        std::cout << "[INFO] Pipeline: init/reinit KV cache (max_seq_len="
                  << max_seq_len << ", prev=" << kv_cache_max_len_ << ")" << std::endl;

        if (model_.init_kv_cache(max_seq_len)) {
            kv_cache_initialized_ = true;
            kv_cache_max_len_ = max_seq_len;
        } else if (kv_cache_initialized_ && prompt.cols + 32 <= kv_cache_max_len_) {
            // Growing the cache needs a fresh GPU allocation that can fail on
            // a full card (init_kv_cache leaves the old cache intact on
            // failure). Degrade instead of failing: clamp the generation
            // budget to the existing cache — shorter audio beats an HTTP 500
            // and a silent cloud fallback.
            gen.max_new_tokens = kv_cache_max_len_ - prompt.cols;
            std::cerr << "[WARN] Pipeline: KV regrow failed; clamping max_new_tokens to "
                      << gen.max_new_tokens << " (cache max=" << kv_cache_max_len_ << ")" << std::endl;
        } else {
            std::cerr << "Pipeline error: init_kv_cache failed." << std::endl;
            return false;
        }

        // Reset position dans le cache pour nouvelle génération
        model_.reset();

    } else {
        std::cout << "[INFO] Pipeline: reusing KV cache (max=" << kv_cache_max_len_ << ")" << std::endl;
        model_.reset();  // Reset n_past_ = 0 pour nouvelle génération
    }

    // 4. Generate
    GenerateResult res = generate(model_, tokenizer_.config(), prompt, gen);
    if (res.n_frames == 0) {
        std::cerr << "Pipeline error: generation produced no frames." << std::endl;
        return false;
    }

    // 5. Decode
    std::vector<float> audio_out;
    const bool decode_ok = codec_on_gpu_
        ? decode_chunked(codec_, res.codes.data(), res.n_frames,
                         res.num_codebooks, params.gen.n_threads, audio_out)
        : codec_.decode(res.codes.data(), res.n_frames,
                        params.gen.n_threads, audio_out);
    if (!decode_ok) {
        std::cerr << "Pipeline error: decode failed." << std::endl;
        return false;
    }

    // 6. Save
    if (!save_audio(params.output_path, audio_out, codec_.sample_rate())) {
        std::cerr << "Pipeline error: save_audio failed to " << params.output_path << std::endl;
        return false;
    }

    std::cout << "Saved audio to: " << params.output_path << std::endl;
    return true;
}

// --- synthesize_to_buffer avec chronométrage détaillé ---
bool Pipeline::synthesize_to_buffer(const PipelineParams & params, std::vector<char> & output_buffer) {
    if (!initialized_) {
        std::cerr << "Pipeline not initialized." << std::endl;
        return false;
    }

    std::cout << "--- Pipeline Synthesize (to buffer) ---" << std::endl;
    std::cout << "Text: " << params.text << std::endl;

    const int32_t num_codebooks = model_.hparams().num_codebooks;

    // 1. Audio Prompt Loading
    auto t0 = std::chrono::steady_clock::now();

    std::vector<int32_t> ref_codes;
    int32_t T_prompt = 0;
    if (reference_loaded_) {
        ref_codes.resize(reference_embedding_.size() / sizeof(int32_t));
        memcpy(ref_codes.data(), reference_embedding_.data(), reference_embedding_.size());
        T_prompt = ref_codes.size() / num_codebooks;
        std::cout << "Using pre-loaded reference audio." << std::endl;
    } else if (!params.prompt_audio_path.empty()) {
        std::cout << "Loading reference audio: " << params.prompt_audio_path << std::endl;
        AudioData ref_audio;
        if (load_audio(params.prompt_audio_path, ref_audio, codec_.sample_rate())) {
            if (!codec_.encode(ref_audio.samples.data(), (int32_t)ref_audio.samples.size(),
                               params.gen.n_threads, ref_codes, T_prompt)) {
                std::cerr << "Pipeline warning: encode failed." << std::endl;
                ref_codes.clear();
                T_prompt = 0;
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    std::cout << "[TIMING] Reference loading: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms" << std::endl;

    // 2. Build Prompt Tensor
    PromptTensor prompt = build_prompt(
        tokenizer_, params.text, reference_text_,
        ref_codes.empty() ? nullptr : ref_codes.data(),
        num_codebooks, T_prompt);

    // Dans synthesize_to_buffer() - remplacer la section KV cache :
    auto t2 = std::chrono::steady_clock::now();
    std::cout << "[TIMING] Build prompt: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " ms" << std::endl;

    // 3. Setup KV Cache - CORRIGÉ
    GenerateParams gen = params.gen;

    // Silence runway (same as synthesize() above): force canonical silence
    // codes from init as the opening of every generation.
    const int32_t SEED_N = 6;
    if (silence_frames_ >= SEED_N &&
        silence_codes_.size() == (size_t)num_codebooks * silence_frames_) {
        gen.seed_frames.resize((size_t)SEED_N * num_codebooks);
        for (int32_t t = 0; t < SEED_N; ++t) {
            for (int32_t cb = 0; cb < num_codebooks; ++cb) {
                gen.seed_frames[(size_t)t * num_codebooks + cb] =
                    silence_codes_[(size_t)cb * silence_frames_ + t];
            }
        }
    }

    int32_t max_seq_len = prompt.cols + std::max(gen.max_new_tokens, gen.kv_reserve_tokens);
    if (!kv_cache_initialized_ || max_seq_len > kv_cache_max_len_) {
        std::cout << "[INFO] Re-init KV cache for seq_len=" << max_seq_len << std::endl;
        if (model_.init_kv_cache(max_seq_len)) {
            kv_cache_initialized_ = true;
            kv_cache_max_len_ = max_seq_len;
        } else if (kv_cache_initialized_ && prompt.cols + 32 <= kv_cache_max_len_) {
            // Failed grow leaves the old cache intact (transactional
            // init_kv_cache) — clamp the generation budget and continue
            // rather than 500ing into a silent cloud fallback.
            gen.max_new_tokens = kv_cache_max_len_ - prompt.cols;
            std::cerr << "[WARN] Pipeline: KV regrow failed; clamping max_new_tokens to "
                      << gen.max_new_tokens << " (cache max=" << kv_cache_max_len_ << ")" << std::endl;
        } else {
            std::cerr << "Pipeline error: init_kv_cache failed." << std::endl;
            return false;
        }
        model_.reset();
    } else {
        std::cout << "[INFO] Reusing KV cache (max_len=" << kv_cache_max_len_ << ")" << std::endl;
    model_.reset();
    }

    auto t3 = std::chrono::steady_clock::now();
    std::cout << "[TIMING] KV cache init: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms" << std::endl;

    // 4. Generate
    GenerateResult res = generate(model_, tokenizer_.config(), prompt, gen);
    if (res.n_frames == 0) {
        std::cerr << "Pipeline error: generation produced no frames." << std::endl;
        return false;
    }

    auto t4 = std::chrono::steady_clock::now();
    std::cout << "[TIMING] Generate (" << res.n_frames << " frames): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()
              << " ms" << std::endl;

    // 5. Decode
    std::vector<float> audio_out;
    const bool decode_ok = codec_on_gpu_
        ? decode_chunked(codec_, res.codes.data(), res.n_frames,
                         res.num_codebooks, params.gen.n_threads, audio_out)
        : codec_.decode(res.codes.data(), res.n_frames,
                        params.gen.n_threads, audio_out);
    if (!decode_ok) {
        std::cerr << "Pipeline error: decode failed." << std::endl;
        return false;
    }

    auto t5 = std::chrono::steady_clock::now();
    std::cout << "[TIMING] Codec decode: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()
              << " ms" << std::endl;

    // 6. Convertir en WAV en mémoire
    const int32_t sample_rate = codec_.sample_rate();
    const int32_t num_samples = static_cast<int32_t>(audio_out.size());
    const int16_t bits_per_sample = 16;
    const int32_t bytes_per_sample = bits_per_sample / 8;
    const int32_t num_channels = 1;
    const int32_t block_align = num_channels * bytes_per_sample;
    const int32_t byte_rate = sample_rate * block_align;
    const int32_t data_size = num_samples * bytes_per_sample;
    const int32_t wav_header_size = 44;
    const int32_t file_size = wav_header_size + data_size;

    output_buffer.clear();
    output_buffer.resize(wav_header_size + data_size);

    char* ptr = output_buffer.data();

    // Header WAV
    std::memcpy(ptr + 0, "RIFF", 4);
    std::memcpy(ptr + 4, &file_size, 4);
    std::memcpy(ptr + 8, "WAVE", 4);
    std::memcpy(ptr + 12, "fmt ", 4);
    int32_t fmt_chunk_size = 16;
    std::memcpy(ptr + 16, &fmt_chunk_size, 4);
    int16_t audio_format = 1;
    std::memcpy(ptr + 20, &audio_format, 2);
    std::memcpy(ptr + 22, &num_channels, 2);
    std::memcpy(ptr + 24, &sample_rate, 4);
    std::memcpy(ptr + 28, &byte_rate, 4);
    std::memcpy(ptr + 32, &block_align, 2);
    std::memcpy(ptr + 34, &bits_per_sample, 2);
    std::memcpy(ptr + 36, "data", 4);
    std::memcpy(ptr + 40, &data_size, 4);

    // Convertir float -> int16
    std::vector<int16_t> pcm_samples(num_samples);
    for (int32_t i = 0; i < num_samples; ++i) {
        float s = std::max(-1.0f, std::min(1.0f, audio_out[i]));
        pcm_samples[i] = static_cast<int16_t>(s * 32767.0f);
    }
    std::memcpy(ptr + wav_header_size, pcm_samples.data(), data_size);

    auto t6 = std::chrono::steady_clock::now();
    std::cout << "[TIMING] WAV buffer build: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count()
              << " ms" << std::endl;

    std::cout << "[TIMING] TOTAL: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t0).count()
              << " ms" << std::endl;

    std::cout << "Generated " << num_samples << " samples (" << data_size << " bytes) in buffer." << std::endl;
    return true;
}

} // namespace s2
