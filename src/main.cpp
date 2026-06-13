#include "s2_pipeline.h"
#include <crow.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <mutex>
#include <thread>

int main(int argc, char** argv) {
    // --- Paramètres par défaut ---
    s2::PipelineParams params;
    params.model_path = "model.gguf";
    params.tokenizer_path = "tokenizer.json";
    params.vulkan_device = 0;
    params.codec_vulkan_device = -2;  // -2 = unset, inherit from model; -1 = force CPU
    params.gen.n_threads = 4;
    params.gen.max_new_tokens = 1024;
    params.gen.temperature = 0.7f;
    params.gen.top_p = 0.7f;
    params.gen.top_k = 30;

    int port = 8080;

    // --- Parse des arguments ---
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (arg == "--model-codec" && i + 1 < argc) {
            params.codec_model_path = argv[++i];
        } else if ((arg == "-t" || arg == "--tokenizer") && i + 1 < argc) {
            params.tokenizer_path = argv[++i];
        } else if ((arg == "-v" || arg == "--vulkan") && i + 1 < argc) {
            params.vulkan_device = std::stoi(argv[++i]);
        } else if (arg == "--codec-vulkan" && i + 1 < argc) {
            params.codec_vulkan_device = std::stoi(argv[++i]);
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if ((arg == "-threads" || arg == "--threads") && i + 1 < argc) {
            params.gen.n_threads = std::stoi(argv[++i]);
        } else if ((arg == "--max-tokens") && i + 1 < argc) {
            params.gen.max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: s2 [options]\n"
                      << "Options:\n"
                      << "  -m, --model <path>       Path to unified GGUF model (default: model.gguf)\n"
                      << "  --model-codec <path>     Path to codec-only GGUF (optional)\n"
                      << "  -t, --tokenizer <path>   Path to tokenizer.json (default: tokenizer.json)\n"
                      << "  -v, --vulkan <device>    Vulkan device for model (default: 0)\n"
                      << "  --codec-vulkan <device>  Vulkan device for codec (default: 0)\n"
                      << "  -p, --port <N>           HTTP port (default: 8080)\n"
                      << "  --threads <N>            CPU threads (default: 12)\n"
                      << "  --max-tokens <N>         Max tokens to generate (default: 1024)\n"
                      << std::endl;
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            std::cerr << "Use --help for usage information." << std::endl;
            return 1;
        }
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Model:        " << params.model_path << std::endl;
    std::cout << "  Codec model:  " << (params.codec_model_path.empty() ? params.model_path : params.codec_model_path) << std::endl;
    std::cout << "  Tokenizer:    " << params.tokenizer_path << std::endl;
    std::cout << "  Model GPU:    " << params.vulkan_device << std::endl;
    std::cout << "  Codec GPU:    " << params.codec_vulkan_device << std::endl;
    std::cout << "  Port:         " << port << std::endl;
    std::cout << "  Threads:      " << params.gen.n_threads << std::endl;
    std::cout << "  Max tokens:   " << params.gen.max_new_tokens << std::endl;

    // --- Charger le modèle ---
    s2::Pipeline pipeline;
    if (!pipeline.init(params)) {
        std::cerr << "Pipeline initialization failed." << std::endl;
        return 1;
    }

    // The Fish Audio Peter Parker checkpoint mangles the first 1-2 syllables of
    // every prompt — verified on both the cloud API and this local s2.cpp build.
    // Other interjections like "Okay, " or "Hey, " get substituted into a second
    // "Hey" by the Peter character bias; "Listen, " is phonetically distant
    // enough that the model absorbs it as a slight unintelligible artifact and
    // the caller's actual first word survives intact. Configurable via the
    // S2_LEADING_PREFIX env var; set it to "" to disable entirely.
    const char * env_prefix = std::getenv("S2_LEADING_PREFIX");
    const std::string leading_prefix = env_prefix ? std::string(env_prefix) : std::string("Listen, ");
    if (!leading_prefix.empty()) {
        std::cout << "Leading prefix: \"" << leading_prefix
                  << "\" (set S2_LEADING_PREFIX=\"\" to disable)" << std::endl;
    } else {
        std::cout << "Leading prefix: disabled" << std::endl;
    }

    // --- Warm-up: claim GPU compute buffers at startup, not at request time ---
    // Pass 1 (short text, full max_new_tokens) sizes the KV cache for the
    // longest allowed generation. Pass 2 (long text, 4 tokens) sizes the
    // persistent prefill buffer for a worst-case prompt. Both allocations are
    // retained for the process lifetime, so requests never cudaMalloc on a
    // full card. Retries cover the boot race with TurboQuant's model load
    // (After= orders process start, not VRAM allocation).
    {
        const std::string worst_case_text =
            "Here is the complete list you asked for. First, the living room lights are at "
            "forty percent and the kitchen lights are off. Second, the thermostat is holding "
            "sixty-eight degrees with the fan on auto. Third, the front door is locked, the "
            "garage door is closed, and the back door was opened twice this afternoon. Fourth, "
            "the washer finished its cycle about twenty minutes ago and the dryer has eleven "
            "minutes remaining. Fifth, the cameras saw the usual delivery around three thirty. "
            "Sixth, tomorrow looks cloudy with a high of fifty-nine, so nothing unusual there. "
            "Let me know if you want me to change any of these or hear anything again.";
        bool warmed = false;
        for (int attempt = 1; attempt <= 60 && !warmed; ++attempt) {
            s2::PipelineParams warm = params;
            std::vector<char> discard;
            // Long text sizes the prefill buffer worst-case; kv_reserve_tokens
            // sizes the KV cache to the serving ceiling while only 8 frames
            // are actually generated/decoded (boot ~60s, not ~5min). The
            // codec decode buffer no longer needs worst-case warm-up sizing:
            // decode_chunked() bounds it to one fixed window (and on CPU the
            // buffer is plain RAM anyway — GPU codec is blocked on upstream
            // CUDA bugs: IM2COL invalid-argument + 590MB encode buffer,
            // measured 2026-06-10).
            warm.text = worst_case_text;
            warm.gen.kv_reserve_tokens = params.gen.max_new_tokens;
            // 64 ≥ CHUNK+OVL+CHUNK (24+8+24): the warm-up decode must produce
            // at least one MIDDLE window (lead 8 + core 24 = 32 frames) so the
            // GPU decode buffer reserves at its true worst-case size. 40 was
            // not enough — its largest window was 24 frames, and the first
            // real 64-frame reply then re-reserved ~190MB at request time on
            // a full card (failed, measured 2026-06-11).
            warm.gen.max_new_tokens = 64;
            bool ok = pipeline.synthesize_to_buffer(warm, discard);
            if (ok) {
                warmed = true;
                std::cout << "[warmup] GPU buffers reserved (attempt " << attempt << ")" << std::endl;
            } else {
                std::cerr << "[warmup] attempt " << attempt
                          << " failed (GPU busy/full?); retrying in 5s" << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        }
        if (!warmed) {
            std::cerr << "[warmup] giving up after 60 attempts — serving anyway; "
                         "requests may fail to cloud fallback until VRAM frees" << std::endl;
        }
    }

    // --- Serveur HTTP ---
    crow::SimpleApp app;

    // ================================================================
    // Helper : traitement commun de synthèse
    // ================================================================
    auto do_synthesize = [&](const crow::json::rvalue& json) -> crow::response {
        // Crow dispatches handlers on 8 threads but Pipeline/SlowARModel share
        // one KV cache, n_past_, and the allocr swap in prefill() — concurrent
        // synthesis corrupts state. Serialize: the GPU AR loop is serial
        // anyway, so queueing costs latency only (e.g. probe overlapping a
        // real request).
        static std::mutex synth_mutex;
        std::lock_guard<std::mutex> synth_lock(synth_mutex);

        s2::PipelineParams synth_params = params;

        // Compatibilité Fish Audio : "text" est le champ principal
        if (json.has("text")) {
            synth_params.text = json["text"].s();
        } else if (json.has("input")) {
            // Certaines implémentations utilisent "input"
            synth_params.text = json["input"].s();
        } else {
            return crow::response(400, "Missing 'text' field");
        }

        // Workaround for the leading-word drop quirk of the Peter Parker checkpoint:
        // prepend a throwaway interjection that the model will discard, so the
        // caller's actual first word survives. Skip if the caller already starts
        // with the same prefix to avoid double-prepending.
        // 2026-06-09: live evidence the prefix is audibly spoken (Sean asked
        // Peter why he always says "Listen") — per-request opt-out below lets
        // A/B testing settle whether the mangling claim still holds without
        // env churn + service restarts. {"leading_prefix": false} disables.
        bool want_prefix = true;
        if (json.has("leading_prefix")) {
            want_prefix = json["leading_prefix"].b();
        }
        if (want_prefix && !leading_prefix.empty() && !synth_params.text.empty() &&
            synth_params.text.compare(0, leading_prefix.size(), leading_prefix) != 0) {
            synth_params.text = leading_prefix + synth_params.text;
        }

        // Paramètres de génération (format s2.cpp natif)
        if (json.has("temperature")) synth_params.gen.temperature = json["temperature"].d();
        if (json.has("top_p"))       synth_params.gen.top_p = json["top_p"].d();
        if (json.has("top_k"))       synth_params.gen.top_k = json["top_k"].i();
        if (json.has("threads"))     synth_params.gen.n_threads = json["threads"].i();

        // Paramètres Fish Audio (ignorés gracieusement si non pertinents)
        // reference_id, chunk_length, normalize, format, mp3_bitrate, opus_bitrate, latency
        // On les accepte sans erreur pour la compatibilité

        // Déterminer le format de sortie
        std::string format = "wav";
        if (json.has("format")) {
            format = json["format"].s();
        }

        // Synthèse
        std::vector<char> audio_buffer;
        if (!pipeline.synthesize_to_buffer(synth_params, audio_buffer)) {
            return crow::response(500, "Synthesis failed");
        }

        crow::response res;

        if (format == "wav") {
            res.set_header("Content-Type", "audio/wav");
            res.body.assign(audio_buffer.data(), audio_buffer.size());
        } else if (format == "pcm") {
            // PCM brut sans header WAV (skip les 44 premiers octets)
            res.set_header("Content-Type", "audio/pcm");
            res.set_header("X-Sample-Rate", std::to_string(pipeline.sample_rate()));
            if (audio_buffer.size() > 44) {
                res.body.assign(audio_buffer.data() + 44, audio_buffer.size() - 44);
            }
        } else {
            // Par défaut WAV (mp3/opus non supportés nativement)
            res.set_header("Content-Type", "audio/wav");
            res.body.assign(audio_buffer.data(), audio_buffer.size());
        }

        return res;
    };

    // ================================================================
    // Route Fish Audio compatible : POST /v1/tts
    // ================================================================
    CROW_ROUTE(app, "/v1/tts")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        auto json = crow::json::load(req.body);
        if (!json) {
            return crow::response(400, "Invalid JSON");
        }
        return do_synthesize(json);
    });

    // ================================================================
    // Route legacy : POST /synthesize (compatibilité avec vos tests)
    // ================================================================
    CROW_ROUTE(app, "/synthesize")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        auto json = crow::json::load(req.body);
        if (!json) {
            return crow::response(400, "Invalid JSON");
        }
        return do_synthesize(json);
    });

    // ================================================================
    // Route OpenAI compatible : POST /v1/audio/speech
    // ================================================================
    CROW_ROUTE(app, "/v1/audio/speech")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        auto json = crow::json::load(req.body);
        if (!json) {
            return crow::response(400, "Invalid JSON");
        }
        return do_synthesize(json);
    });

    // ================================================================
    // Health check & info
    // ================================================================
    CROW_ROUTE(app, "/v1/models")
    .methods("GET"_method)
    ([&]() {
        crow::json::wvalue resp;
        resp["object"] = "list";
        crow::json::wvalue model;
        model["id"] = "s2-pro-local";
        model["object"] = "model";
        model["owned_by"] = "local";
        resp["data"][0] = std::move(model);
        return crow::response(200, resp);
    });

    CROW_ROUTE(app, "/health")
    .methods("GET"_method)
    ([]() {
        return crow::response(200, "OK");
    });

    CROW_ROUTE(app, "/")
    ([&port]() {
        crow::json::wvalue info;
        info["status"] = "running";
        info["port"] = port;
        info["endpoints"][0] = "/v1/tts";
        info["endpoints"][1] = "/synthesize";
        info["endpoints"][2] = "/v1/audio/speech";
        info["endpoints"][3] = "/v1/models";
        info["endpoints"][4] = "/health";
        return crow::response(200, info);
    });

    std::cout << "\nEndpoints:" << std::endl;
    std::cout << "  POST /v1/tts           (Fish Audio compatible)" << std::endl;
    std::cout << "  POST /synthesize       (legacy)" << std::endl;
    std::cout << "  POST /v1/audio/speech  (OpenAI compatible)" << std::endl;
    std::cout << "  GET  /v1/models        (model list)" << std::endl;
    std::cout << "  GET  /health           (health check)" << std::endl;
    std::cout << std::endl;

    std::cout << "Starting server on port " << port << "..." << std::endl;
    app.port(port).multithreaded().run();
    return 0;
}
