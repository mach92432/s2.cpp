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
    std::string codec_model_path;  // NOUVEAU: codec-only GGUF (optionnel)

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
    // True when the codec runs on a GPU backend: decode is then chunked to
    // bound the compute buffer (whole-clip = ~2.9GB for 512 frames). On CPU
    // the buffer is plain RAM and chunking costs ~33% extra decode time
    // (overlap re-decode), so whole-clip decode is used there.
    bool codec_on_gpu_ = false;

    bool reference_loaded_ = false;
    std::string reference_embedding_;
    std::string reference_text_;

    // Canonical silence VQ codes for the generation runway: produced at init
    // by encoding digital silence and keeping middle frames (encoder edge
    // windows excluded). Layout (num_codebooks, T) row-major like encode().
    std::vector<int32_t> silence_codes_;
    int32_t silence_frames_ = 0;
};

} // namespace s2
