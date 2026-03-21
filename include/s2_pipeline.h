#pragma once
// s2_pipeline.h — End-to-end TTS pipeline

#include "s2_audio.h"
#include "s2_codec.h"
#include "s2_generate.h"
#include "s2_model.h"
#include "s2_tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace s2 {

struct PipelineParams {
    // Paths
    std::string model_path;
    std::string tokenizer_path;

    // Input
    std::string text;
    std::string prompt_text;
    std::string prompt_audio_path;
    std::string output_path;

    // Generation
    GenerateParams gen;

    // Backend
    int32_t vulkan_device = -1;        // GPU for model (-1 = CPU)
    int32_t codec_vulkan_device = -1;  // GPU for codec (-1 = same as model)
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    bool init(const PipelineParams & params);
    bool synthesize(const PipelineParams & params);
    bool synthesize_to_buffer(const PipelineParams & params, std::vector<char> & output_buffer);

    int32_t sample_rate() const { return codec_.sample_rate(); }

private:
    Tokenizer   tokenizer_;
    SlowARModel model_;
    AudioCodec  codec_;
    bool initialized_ = false;
    
    // ÉTAT KV CACHE - FIX FUITES MÉMOIRE
    bool        kv_cache_initialized_ = false;
    int32_t     kv_cache_max_len_     = 0;
    
    // Reference audio and text management
    bool reference_loaded_ = false;
    std::string reference_embedding_;
    std::string reference_text_;
};

} // namespace s2
