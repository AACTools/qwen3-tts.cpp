/* audio_tokenizer_codec_encoder.h — Mimi encoder (audio → codec tokens)
 *
 * Port of the encoder half of Qwen3-TTS-Tokenizer-12Hz. The decoder is
 * already implemented in audio_tokenizer_decoder.cpp; this adds the
 * reverse path needed for in-context voice cloning (ICL mode).
 *
 * ⚠️ This is a MIMI encoder (Kyuzan/Mimi), NOT WavTokenizer:
 * - ELU activation (no Snake, no weight-norm)
 * - SEANet convs with strides [8, 6, 5, 4] → 25 Hz
 * - Causal transformer: LayerNorm(w+b), GELU non-gated MLP, layer scales,
 *   sliding window 250, RoPE θ=10000
 * - Post-transformer stride-2 downsample conv → 12.5 Hz
 * - Split RVQ: parallel semantic (1×2048) + acoustic (15 of 31 layers,
 *   2048 each) with L2 argmin codebook search
 *
 * Architecture verified against the checkpoint tensors and
 * modeling_mimi.py from the Python reference.
 */
#ifndef AUDIO_TOKENIZER_CODEC_ENCODER_H
#define AUDIO_TOKENIZER_CODEC_ENCODER_H

#include "gguf_loader.h"
#include <ggml.h>
#include <vector>
#include <string>

// ─── Configuration ───────────────────────────────────────────────────

struct MimiEncoderConfig {
    // SEANet conv parameters (strides 8,6,5,4 → 25 Hz)
    static constexpr int NUM_CONV_STAGES = 4;
    static constexpr int CONV_STRIDES[NUM_CONV_STAGES] = {8, 6, 5, 4};
    static constexpr int SEANET_KERNEL = 7;      // base kernel for expansion blocks
    static constexpr int SEANET_LAYERS_PER_STAGE = 2; // residual blocks per stage

    // Transformer @ 25 Hz
    int32_t hidden_size = 512;
    int32_t num_layers = 8;
    int32_t num_heads = 8;
    int32_t sliding_window = 250;
    float rope_theta = 10000.0f;
    // LayerNorm (not RMSNorm), GELU (not SiLU), with bias + layer scales

    // Post-transformer downsample (stride 2 → 12.5 Hz)
    static constexpr int DOWNSAMPLE_KERNEL = 4;
    static constexpr int DOWNSAMPLE_STRIDE = 2;

    // Quantizer
    int32_t semantic_codebook_size = 2048;
    int32_t semantic_codebook_dim = 256;
    int32_t acoustic_codebook_size = 2048;
    int32_t acoustic_codebook_dim = 256;
    int32_t num_acoustic_quantizers = 15;  // valid layers 0-14 of 31 stored
    int32_t num_semantic_quantizers = 1;
    int32_t total_codebooks() const { return num_acoustic_quantizers + num_semantic_quantizers; }

    // Audio
    int32_t sample_rate = 24000;
    float frame_rate = 12.5f;  // after the post-transformer stride-2

    int32_t head_dim() const { return hidden_size / num_heads; }
};

// ─── Model tensors ───────────────────────────────────────────────────

struct MimiEncoderModel {
    MimiEncoderConfig config;
    bool loaded = false;

    // SEANet encoder convolutions (4 stages, each with residual blocks)
    // Stage structure: conv (stride) → [res_block → res_block] × stages
    // Each res_block: conv (dilated) → ELU → conv → ELU → skip connection
    struct SeanetStage {
        // Strided entry conv
        struct ggml_tensor* entry_weight = nullptr; // [out_ch, in_ch, k, 1]
        struct ggml_tensor* entry_bias = nullptr;

        // Residual blocks (2 per stage)
        struct ggml_tensor* res_conv1_weight[2] = {};
        struct ggml_tensor* res_conv1_bias[2] = {};
        struct ggml_tensor* res_conv2_weight[2] = {};
        struct ggml_tensor* res_conv2_bias[2] = {};
    };
    SeanetStage seanet[MimiEncoderConfig::NUM_CONV_STAGES];

    // Final conv (before transformer)
    struct ggml_tensor* pre_tfm_conv_weight = nullptr;
    struct ggml_tensor* pre_tfm_conv_bias = nullptr;

    // Transformer layers (8 layers)
    struct ggml_tensor* tfm_attn_q[8] = {};
    struct ggml_tensor* tfm_attn_k[8] = {};
    struct ggml_tensor* tfm_attn_v[8] = {};
    struct ggml_tensor* tfm_attn_o[8] = {};
    struct ggml_tensor* tfm_attn_norm_w[8] = {};
    struct ggml_tensor* tfm_attn_norm_b[8] = {};
    struct ggml_tensor* tfm_attn_scale[8] = {};
    struct ggml_tensor* tfm_ffn_fc1[8] = {};
    struct ggml_tensor* tfm_ffn_fc2[8] = {};
    struct ggml_tensor* tfm_ffn_norm_w[8] = {};
    struct ggml_tensor* tfm_ffn_norm_b[8] = {};
    struct ggml_tensor* tfm_ffn_scale[8] = {};

    // Post-transformer downsample conv
    struct ggml_tensor* downsample_weight = nullptr; // [512, 512, 4, 1]
    struct ggml_tensor* downsample_bias = nullptr;

    // Quantizer projections
    struct ggml_tensor* vq_semantic_in = nullptr;   // input_proj [256, 512]
    struct ggml_tensor* vq_semantic_out = nullptr;  // output_proj [512, 256]
    struct ggml_tensor* vq_acoustic_in = nullptr;   // input_proj [256, 512]
    struct ggml_tensor* vq_acoustic_out = nullptr;  // output_proj [512, 256]

    // RVQ codebooks: [codebook_size, codebook_dim], centroids (not EMA sums)
    struct ggml_tensor* vq_semantic_cb = nullptr;
    struct ggml_tensor* vq_acoustic_cb[15] = {};

    // Companion usage tensors for normalization verification
    struct ggml_tensor* vq_semantic_usage = nullptr;
    struct ggml_tensor* vq_acoustic_usage[15] = {};
};

// ─── Encoder ─────────────────────────────────────────────────────────

class CodecEncoder {
public:
    CodecEncoder() = default;
    ~CodecEncoder() = default;

    bool load(const GGUFLoader& gguf);
    bool is_loaded() const { return model_.loaded; }

    /// Encode 24kHz mono f32 audio → codec tokens.
    /// codec_tokens: [n_frames × 16] (1 semantic + 15 acoustic per frame)
    /// n_frames: number of 12.5 Hz frames
    /// Returns false on failure (check get_error()).
    bool encode(const float* audio, int32_t n_samples,
                std::vector<int32_t>& codec_tokens,
                int32_t& n_frames);

    const char* get_error() const { return error_msg_.c_str(); }
    const MimiEncoderConfig& config() const { return model_.config; }

private:
    MimiEncoderModel model_;
    std::string error_msg_;

    // Graph building helpers
    struct ggml_tensor* build_seanet_stage(
        struct ggml_context* ctx, struct ggml_tensor* input, int stage_idx);

    struct ggml_tensor* build_transformer_layer(
        struct ggml_context* ctx, struct ggml_tensor* input,
        int layer_idx, struct ggml_tensor* positions);

    struct ggml_tensor* build_rvq_quantize(
        struct ggml_context* ctx, struct ggml_tensor* input,
        bool semantic_path);

    // L2 argmin for one codebook entry
    static int32_t rvq_l2_argmin(
        const float* codebook, int32_t cb_size, int32_t cb_dim,
        const float* vector);
};

#endif // AUDIO_TOKENIZER_CODEC_ENCODER_H
