#include "s2_pipeline.h"
#include <crow.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>

int main(int argc, char** argv) {
    // --- Paramètres par défaut ---
    s2::PipelineParams params;
    params.model_path = "model.gguf";
    params.tokenizer_path = "tokenizer.json";
    params.vulkan_device = 0;
    params.codec_vulkan_device = 0;
    params.gen.n_threads = 4;
    params.gen.max_new_tokens = 2048;
    params.gen.temperature = 0.7f;
    params.gen.top_p = 0.7f;
    params.gen.top_k = 30;

    int port = 8080;

    // --- Parse des arguments ---
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if ((arg == "-t" || arg == "--tokenizer") && i + 1 < argc) {
            params.tokenizer_path = argv[++i];
        } else if ((arg == "-v" || arg == "--vulkan") && i + 1 < argc) {
            params.vulkan_device = std::stoi(argv[++i]);
        } else if (arg == "--codec-vulkan" && i + 1 < argc) {
            params.codec_vulkan_device = std::stoi(argv[++i]);
        } else if ((arg == "-p" || arg == "-port" || arg == "--port") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if ((arg == "-threads" || arg == "--threads") && i + 1 < argc) {
            params.gen.n_threads = std::stoi(argv[++i]);
        } else if ((arg == "-max-tokens" || arg == "--max-tokens") && i + 1 < argc) {
            params.gen.max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: s2 [options]\n"
                      << "Options:\n"
                      << "  -m, --model <path>       Path to unified GGUF model (default: model.gguf)\n"
                      << "  -t, --tokenizer <path>   Path to tokenizer.json (default: tokenizer.json)\n"
                      << "  -v, --vulkan <device>    Vulkan device for model (default: 0)\n"
                      << "  --codec-vulkan <device>  Vulkan device for codec (default: 0)\n"
                      << "  -p, -port, --port <N>    HTTP port (default: 8080)\n"
                      << "  -threads, --threads <N>  CPU threads (default: 4)\n"
                      << "  -max-tokens <N>          Max tokens to generate (default: 512)\n"
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

    // --- Serveur HTTP ---
    crow::SimpleApp app;

    // ================================================================
    // Helper : traitement commun de synthèse
    // ================================================================
    auto do_synthesize = [&](const crow::json::rvalue& json) -> crow::response {
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
