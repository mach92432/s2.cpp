# s2.cpp

> **ALPHA — EXPERIMENTAL SOFTWARE**
> This is an early-stage, community-built inference engine. Expect rough edges, missing features, and breaking changes. Not production-ready.

**s2.cpp** — Fish Audio's S2 Pro Dual-AR text-to-speech model running locally via a pure C++/GGML inference engine with CPU, Vulkan backends. No Python runtime required after build.

> **Built on Fish Audio S2 Pro**
> The model weights are licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved.
> See [LICENSE.md](LICENSE.md) for full terms. Commercial use requires a separate license from Fish Audio — contact [business@fish.audio](mailto:business@fish.audio).

---

## What this is

This repository is a fork of https://github.com/rodrigomatta/s2.cpp which is currently a command-line script.

This version of s2.cpp is an API server version, compatible with the Fish Audio API. On good GPUs, it can achieve real-time performance while taking advantage of quantized models. If needed, VRAM savings can be used to run multiple services simultaneously.

Tested only with Nvidia GPU on Debian Linux platform.

This repository contains:

- **`s2.cpp`** — a self-contained C++17 inference engine built on [ggml](https://github.com/ggml-org/ggml), handling tokenization, Dual-AR generation, audio codec encode/decode, and WAV output throw API similar with Fish Audio API
- **`tokenizer.json`** — Qwen3 BPE tokenizer with ByteLevel pre-tokenization
- GGUF model files are **not included** here — see [Model variants](#model-variants) below

The engine runs the full pipeline: text via API → tokens → Slow-AR transformer (with KV cache) → Fast-AR codebook decoder → audio codec → WAV file return via API.

---

## Model variants

GGUF files are available at [rodrigomt/s2-pro-gguf](https://huggingface.co/rodrigomt/s2-pro-gguf) on Hugging Face.

| File | Size | Notes |
|---|---|---|
| `s2-pro-f16.gguf` | 9.3 GB | Full precision — reference quality 19+ GB VRAM |
| `s2-pro-q8_0.gguf` | 5.7 GB | Near-lossless — recommended for 12+ GB VRAM |
| `s2-pro-q6_k.gguf` | 4.8 GB | Good quality/size balance — recommended for 11+ GB VRAM |
| `s2-pro-q3_k.gguf` | 4.0 GB | Good quality/size balance — recommended for 8+ GB VRAM |



All variants include both the transformer weights and the audio codec in a single file.

---

## Requirements

### Build dependencies

- CMake ≥ 3.14
- C++17 compiler (GCC ≥ 10, Clang ≥ 11, MSVC 2019+)
- For Vulkan GPU support: Vulkan SDK and `glslc`

```bash
# Ubuntu / Debian
sudo apt install cmake build-essential
```

# Vulkan (optional, recommended for GPU acceleration on AMD/Intel/Nvidia)

```bash
sudo apt install vulkan-tools libvulkan-dev glslc
```

### Runtime

No Python or PyTorch required. The binary links only against the ggml shared libraries built alongside it.

---

## Building

Clone (ggml is NOT a submodule):

```bash
git clone https://github.com/mach92432/s2.cpp.git
cd s2.cpp
```


### With Vulkan GPU support

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_VULKAN=ON
cmake --build build --parallel $(nproc)
```

The binary is produced at `build/s2`.

---

## Usage

### Basic server launch for GPU Vulkan (ex: Nvidia)

Put model.gguf (or link) in s2.cpp directory
Only for cloning put reference.wav and reference.txt in s2.cpp directory

```bash
build/s2 -v 0 --codec-vulkan 0 -port 8081
```

`--model model.gguf` to specify the path to a GGUF model (default model.gguf)
`--model-codec` to specify the path to a GGUF model for 'codec' processing only. By default, it's the model specified by '--model' or 'model.gguf'. 
`-v 0` selects the first Vulkan device. The transformer runs on GPU.
`--codec-vulkan 0` selects the first Vulkan device for audio codec. It is possible to use the CPU instead whith `--codec-vulkan -1`.
`--port 8081` : port to listen
`--help` for other options

### GPU inference via Vulkan with curl

```bash
curl -X POST http://localhost:8081/v1/tts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d "{\"text\":\"[emphasis] Bonjour Bob ! [pause] Je suis bien là, mais il semble y avoir une petite confusion : je ne suis pas Samantha. Je suis **Anna** (ou **Anaïs**, selon l'humeur du jour !). [laughing] Je suis parfaitement réveillée et prête à t'aider. Que souhaites-tu faire ?\",\"format\":\"wav\"}" \
  -o output.wav
```



### All options

| Flag | Default | Description |
|---|---|---|
| `-v ` | Vulkan device id for Model (e.g. `-v 0`)|
| `--codec-vulkan ` | Vulkan device id for Codec (e.g. `--codec-vulkan 0`) |
| `--port ` | Server port ti listen (e.g. `--port 8081`) |

---

## Choosing a model

| VRAM available | Recommended model |
|---|---|
| 8 GB | `q3_k` — good quality/size balance |
| 11 GB | `q6_k` — good quality/size balance |
| 12 GB | `q8_0` — near-lossless quality |
| 19 GB | `f16` — near-lossless quality |

VRAM usage at runtime is approximately double of the file size (because codec runs on GPU).

## Benchmark

For speed, I don't recommend using the CPU for the codec. Using the CPU for the codec doubles the total processing time.

I suggest choosing a quantized version that can fit twice in the allocated VRAM. It's possible to use two GPUs.

The audio generation speed is approximately 0.8x on an RTX3090. 

The speed is roughly the same regardless of the model.

The sound quality remains acceptable for the smallest model. 

Voice cloning works correctly. 

Tags may be less respected with high levels of quantization.

Generating short texts often results in artifacts at the end. Whenever possible, long texts should be split into segments of at least 100 characters. 

---

## Architecture notes

S2 Pro uses a **Dual-AR** architecture:

- **Slow-AR** — a 36-layer Qwen3-based transformer (4.13B params) that processes the full token sequence with GQA (32 heads, 8 KV heads), RoPE at 1M base, QK norm, and a persistent KV cache
- **Fast-AR** — a 4-layer transformer (0.42B params) that autoregressively generates 10 acoustic codebook tokens from the Slow-AR hidden state for each semantic step
- **Audio codec** — a convolutional encoder/decoder with residual vector quantization (RVQ, 10 codebooks × 4096 entries) that converts between audio waveforms and discrete codes

Total: ~4.56B parameters.

---

## Implementation notes

The C++ engine (`src/`) is built entirely on [ggml](https://github.com/ggml-org/ggml) include in this code. Key design decisions:

- **Reference audio** To avoid processing the reference audio during each request, it is processed at server startup and kept in memory.
- **Separate persistent `gallocr` allocators** for Slow-AR and Fast-AR — each path keeps its own compute buffer, avoiding memory re-planning per token
- **Temporary prefill allocator** — freed immediately after prefill, so the large compute buffer does not persist into the generation loop
- **Codec on GPU or CPU** — the audio codec executes on codec-vulkan id. -1 value for CPU
- **posix_fadvise(DONTNEED)** after mmap — releases the GGUF file from kernel page cache after weights are loaded to VRAM, preventing RAM duplication equal to the model file size
- **Correct ByteLevel tokenization** — the GPT-2 byte-to-unicode table is applied before BPE, producing token IDs identical to the HuggingFace reference tokenizer

---

## Known limitations (alpha)

- No streaming output — WAV is return to the client only after full generation completes
- No batch inference
- Voice cloning quality depends heavily on reference audio length and SNR
- Windows is untested
- macOS is untested

---

## License

The model weights and associated materials are licensed under the **Fish Audio Research License**. Key points:

- **Research and non-commercial use:** free, under the terms of this Agreement
- **Commercial use:** requires a separate written license from Fish Audio
- When distributing, you must include a copy of the license and the attribution notice
- Attribution: *"This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved."*

Full license: [LICENSE.md](LICENSE.md)

Commercial licensing: [https://fish.audio](https://fish.audio) · [business@fish.audio](mailto:business@fish.audio)

The inference engine source code (`src/`) is a Derivative Work of the Fish Audio Materials as defined in the Agreement and is distributed under the same Fish Audio Research License terms.
