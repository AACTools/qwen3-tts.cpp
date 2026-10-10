/* audio_tokenizer_codec_encoder.cpp — Mimi encoder (stub, Phase 1 WIP)
 *
 * Architecture: SEANet convs (strides 8,6,5,4 → 25Hz) → causal
 * transformer (8L × 512H, LayerNorm+GELU, sliding window 250) →
 * stride-2 downsample → 12.5Hz → split RVQ (1 semantic + 15 acoustic).
 *
 * Status: architecture documented in the header, load()/encode() are
 * stubs. See docs/ICL_PLAN.md for the full implementation plan and
 * review findings.
 *
 * Next steps (in order):
 * 1. Extend GGUFLoader to enumerate + load tok_enc.* tensors
 * 2. Implement SEANet conv graph (match decoder's im2col patterns)
 * 3. Implement transformer (ggml_rope_ext, sliding-window mask)
 * 4. Implement split RVQ (L2 argmin)
 * 5. Parity test vs Python Mimi encoder
 */

#include "audio_tokenizer_codec_encoder.h"
#include <cstdio>

#define ENC_LOG(...) do { fprintf(stderr, "[codec-encoder] " __VA_ARGS__); } while (0)

bool CodecEncoder::load(const qwen3_tts::GGUFLoader & gguf) {
    (void)gguf;
    ENC_LOG("load: stub — see implementation notes in header\n");
    model_.loaded = false;
    error_msg_ = "CodecEncoder::load not yet implemented";
    return false;
}

bool CodecEncoder::encode(
    const float * audio, int32_t n_samples,
    std::vector<int32_t> & codec_tokens, int32_t & n_frames)
{
    (void)audio; (void)n_samples; (void)codec_tokens; (void)n_frames;
    if (!model_.loaded) {
        error_msg_ = "encoder not loaded";
        return false;
    }
    ENC_LOG("encode: stub\n");
    error_msg_ = "CodecEncoder::encode not yet implemented";
    return false;
}
