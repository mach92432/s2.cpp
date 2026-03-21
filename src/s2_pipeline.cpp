#include "../include/s2_pipeline.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <chrono>

namespace s2 {

Pipeline::Pipeline() {}
Pipeline::~Pipeline() {
    std::cout << "[Pipeline] Cleanup complete" << std::endl;
    // model_ se détruit automatiquement → libère ctx_kv_, kv_buf_, etc.
}

bool Pipeline::init(const PipelineParams & params) {
    std::cout << "--- Pipeline Init ---" << std::endl;

    // Déterminer les GPU
    int model_gpu = params.vulkan_device;
    int codec_gpu = params.codec_vulkan_device;
    // Si codec_vulkan_device non spécifié, utiliser le même GPU que le modèle
    if (codec_gpu < 0) codec_gpu = model_gpu;

    std::cout << "GPU assignment: model -> GPU " << model_gpu
              << ", codec -> GPU " << codec_gpu << std::endl;

    // Charger le tokenizer
    if (!tokenizer_.load(params.tokenizer_path)) {
        std::cerr << "Pipeline error: could not load tokenizer from "
                  << params.tokenizer_path << std::endl;
        return false;
    }

    // Charger le modèle principal sur le GPU assigné
    std::cout << "Loading model on GPU " << model_gpu << "..." << std::endl;
    if (!model_.load(params.model_path, model_gpu)) {
        std::cerr << "Pipeline error: could not load model on GPU " << model_gpu << std::endl;
        return false;
    }
    std::cout << "Model loaded on GPU " << model_gpu << "." << std::endl;

    // Charger le codec sur le GPU assigné, avec fallback
    std::cout << "Loading codec on GPU " << codec_gpu << "..." << std::endl;
    if (!codec_.load(params.model_path, codec_gpu)) {
        std::cerr << "Pipeline warning: codec failed on GPU " << codec_gpu
                  << ", trying GPU " << model_gpu << "." << std::endl;
        if (!codec_.load(params.model_path, model_gpu)) {
            std::cerr << "Pipeline warning: codec failed on GPU, falling back to CPU." << std::endl;
            if (!codec_.load(params.model_path, -1)) {
                std::cerr << "Pipeline error: could not load codec." << std::endl;
                return false;
            }
            std::cout << "Codec loaded on CPU (fallback)." << std::endl;
        } else {
            std::cout << "Codec loaded on GPU " << model_gpu << " (shared)." << std::endl;
        }
    } else {
        std::cout << "Codec loaded on GPU " << codec_gpu << " (dedicated)." << std::endl;
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

    // --- Charger la référence audio une seule fois au démarrage ---
    if (std::FILE* file = std::fopen("reference.wav", "rb")) {
        std::fclose(file);
        AudioData ref_audio;
        if (load_audio("reference.wav", ref_audio, codec_.sample_rate())) {
            std::vector<int32_t> ref_codes;
            int32_t T_prompt = 0;
            if (codec_.encode(ref_audio.samples.data(), (int32_t)ref_audio.samples.size(),
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

    initialized_ = true;
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
    int32_t max_seq_len = prompt.cols + params.gen.max_new_tokens;

    if (!kv_cache_initialized_ || max_seq_len > kv_cache_max_len_) {
        std::cout << "[INFO] Pipeline: init/reinit KV cache (max_seq_len=" 
                  << max_seq_len << ", prev=" << kv_cache_max_len_ << ")" << std::endl;
        
        if (!model_.init_kv_cache(max_seq_len)) {
            std::cerr << "Pipeline error: init_kv_cache failed." << std::endl;
            return false;
        }
        
        kv_cache_initialized_ = true;
        kv_cache_max_len_ = max_seq_len;
        
        // Reset position dans le cache pour nouvelle génération
        model_.reset();
        
    } else {
        std::cout << "[INFO] Pipeline: reusing KV cache (max=" << kv_cache_max_len_ << ")" << std::endl;
        model_.reset();  // Reset n_past_ = 0 pour nouvelle génération
    }

    // 4. Generate
    GenerateResult res = generate(model_, tokenizer_.config(), prompt, params.gen);
    if (res.n_frames == 0) {
        std::cerr << "Pipeline error: generation produced no frames." << std::endl;
        return false;
    }

    // 5. Decode
    std::vector<float> audio_out;
    if (!codec_.decode(res.codes.data(), res.n_frames, params.gen.n_threads, audio_out)) {
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
    int32_t max_seq_len = prompt.cols + params.gen.max_new_tokens;
    if (!kv_cache_initialized_ || max_seq_len > kv_cache_max_len_) {
        std::cout << "[INFO] Re-init KV cache for seq_len=" << max_seq_len << std::endl;
        if (!model_.init_kv_cache(max_seq_len)) {
            std::cerr << "Pipeline error: init_kv_cache failed." << std::endl;
            return false;
        }
        kv_cache_initialized_ = true;
        kv_cache_max_len_ = max_seq_len;
    } else {
        std::cout << "[INFO] Reusing KV cache (max_len=" << kv_cache_max_len_ << ")" << std::endl;
    }

    auto t3 = std::chrono::steady_clock::now();
    std::cout << "[TIMING] KV cache init: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms" << std::endl;

    // 4. Generate
    GenerateResult res = generate(model_, tokenizer_.config(), prompt, params.gen);
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
    if (!codec_.decode(res.codes.data(), res.n_frames, params.gen.n_threads, audio_out)) {
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
