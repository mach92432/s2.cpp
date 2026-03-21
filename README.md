# s2.cpp

> **ALPHA — EXPERIMENTAL SOFTWARE**
> This is an early-stage, community-built inference engine. Expect rough edges, missing features, and breaking changes. Not production-ready.

**s2.cpp** — Fish Audio's S2 Pro Dual-AR text-to-speech model running locally via a pure C++/GGML inference engine with CPU, Vulkan backends. No Python runtime required after build.

> **Built on Fish Audio S2 Pro**
> The model weights are licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved.
> See [LICENSE.md](LICENSE.md) for full terms. Commercial use requires a separate license from Fish Audio — contact [business@fish.audio](mailto:business@fish.audio).

---

## What this is

This repository contains:

- **`s2.cpp`** — a self-contained C++17 inference engine built on [ggml](https://github.com/ggml-org/ggml), handling tokenization, Dual-AR generation, audio codec encode/decode, and WAV output throw API similar with Fish Audio API
- **`tokenizer.json`** — Qwen3 BPE tokenizer with ByteLevel pre-tokenization
- GGUF model files are **not included** here — see [Model variants](#model-variants) below

The engine runs the full pipeline: text → tokens → Slow-AR transformer (with KV cache) → Fast-AR codebook decoder → audio codec → WAV file.

---

## Model variants

GGUF files are available at [rodrigomt/s2-pro-gguf](https://huggingface.co/rodrigomt/s2-pro-gguf) on Hugging Face.

| File | Size | Notes |
|---|---|---|
| `s2-pro-f16.gguf` | 9.3 GB | Full precision — reference quality |
| `s2-pro-q8_0.gguf` | 5.7 GB | Near-lossless — recommended for 12+ GB VRAM |
| `s2-pro-q6_k.gguf` | 4.8 GB | Good quality/size balance — recommended for 11+ GB VRAM |
| `s2-pro-Q3k_m.gguf` | 4.0 GB | Good quality/size balance — recommended for 8+ GB VRAM |



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

# Vulkan (optional, recommended for GPU acceleration on AMD/Intel/NVIDIA)
sudo apt install vulkan-tools libvulkan-dev glslc

### Runtime

No Python or PyTorch required. The binary links only against the ggml shared libraries built alongside it.

---

## Building

Clone (ggml is NOT a submodule):

```bash
git clone https://github.com/mach92432/s2.cpp.git
cd s2.cpp
```

### CPU only (never tested)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
```

### With Vulkan GPU support

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_VULKAN=ON
cmake --build build --parallel $(nproc)
```

The binary is produced at `build/s2`.

---

## Usage

### Basic server launch for GPU Vulkan (ex: NVidia)

Put model.gguf (or link) in s2.cpp directory
Only for cloning put reference.wav and reference.txt in s2.cpp directory

```bash
build/s2 -v 0 --codec-vulkan 0 -port 8081
```

`-v 0` selects the first Vulkan device. The transformer runs on GPU.
`--codec-vulkan 0` selects the first Vulkan device for audio codec. It should be possible to use the CPU instead (not tested).
`-port 8081` : port to listen
The other basic options are hard-coded

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
| `-port ` | Server port ti listen (e.g. `-port 8081`) |

---

## Choosing a model

| VRAM available | Recommended model |
|---|---|
| 8 GB | `Q3_k_m` — good quality/size balance |
| 12 GB | `q8_0` — near-lossless quality |
| 11 GB | `q6_k` — good quality/size balance |
| < 6 GB | `f16` on CPU (slow) — no GPU variant at this quality level is currently available |

VRAM usage at runtime is approximately equal to the file size (transformer weights only; codec runs on CPU).

---

## Architecture notes

S2 Pro uses a **Dual-AR** architecture:

- **Slow-AR** — a 36-layer Qwen3-based transformer (4.13B params) that processes the full token sequence with GQA (32 heads, 8 KV heads), RoPE at 1M base, QK norm, and a persistent KV cache
- **Fast-AR** — a 4-layer transformer (0.42B params) that autoregressively generates 10 acoustic codebook tokens from the Slow-AR hidden state for each semantic step
- **Audio codec** — a convolutional encoder/decoder with residual vector quantization (RVQ, 10 codebooks × 4096 entries) that converts between audio waveforms and discrete codes

Total: ~4.56B parameters.

---

## Implementation notes

The C++ engine (`src/`) is built entirely on [ggml](https://github.com/ggml-org/ggml) (unmodified, pinned as a submodule). Key design decisions:

- **Separate persistent `gallocr` allocators** for Slow-AR and Fast-AR — each path keeps its own compute buffer, avoiding memory re-planning per token
- **Temporary prefill allocator** — freed immediately after prefill, so the large compute buffer does not persist into the generation loop
- **Codec on CPU** — the audio codec executes exactly twice per synthesis (encode reference + decode output), so running it on CPU has zero impact on generation throughput
- **posix_fadvise(DONTNEED)** after mmap — releases the GGUF file from kernel page cache after weights are loaded to VRAM, preventing RAM duplication equal to the model file size
- **Correct ByteLevel tokenization** — the GPT-2 byte-to-unicode table is applied before BPE, producing token IDs identical to the HuggingFace reference tokenizer

---

## Known limitations (alpha)

- No streaming output — WAV is written only after full generation completes
- No batch inference
- Voice cloning quality depends heavily on reference audio length and SNR
- Windows: CUDA and Vulkan backends are supported; when using MSVC 2019+, ensure CUDA ≥ 12.4 is installed before building
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
