/* audio_tokenizer_codec_encoder.h — WavTokenizer encoder (audio → codec tokens)
 *
 * Port of the encoder half of Qwen3-TTS-Tokenizer-12Hz. The decoder is
 * already implemented in audio_tokenizer_decoder.cpp; this adds the reverse
 * path needed for in-context voice cloning (ICL mode).
 *
 * Architecture: downsample conv → conv blocks (Snake + weight-norm) →
 * transformer (8L × 512H × 8 heads) → dual RVQ (1 semantic + 15 acoustic).
 * Single forward pass — no autoregressive generation.
 */
#ifndef AUDIO_TOKENIZER_CODEC_ENCODER_H
#define AUDIO_TOKENIZER_CODEC_ENCODER_H

#include "gguf_loader.h"
#include <ggml.h>
#include <vector>
#include <memory>

struct CodecEncoderConfig {
    int32_t hidden_size = 512;
    int32_t num_layers = 8;
    int32_t num_heads = 8;
    int32_t codebook_size_acoustic = 2048;
    int32_t codebook_size_semantic = 4096;
    int32_t codebook_dim = 256;
    int32_t num_acoustic_quantizers = 15;
    int32_t num_semantic_quantizers = 1;
    int32_t frame_rate = 12.5;
    int32_t sample_rate = 24000;

    // Derived: total codebooks the decoder expects (16)
    int32_t total_codebooks() const {
        return num_acoustic_quantizers + num_semantic_quantizers;
    }
};

struct CodecEncoderModel {
    CodecEncoderConfig config;

    // Downsample
    struct ggml_tensor* downsample_weight = nullptr;  // [1, 1, stride, in_ch]

    // Conv blocks (pre-transformer feature extraction)
    struct ggml_tensor* conv_weight[8] = {};   // tok_enc.conv.{i}
    struct ggml_tensor* conv_bias[8] = {};
    struct ggml_tensor* res_weight[4][4] = {}; // tok_enc.res.{i}.blk.{j}
    struct ggml_tensor* res_bias[4][4] = {};

    // Transformer layers
    struct ggml_tensor* blk_attn_q[8] = {};
    struct ggml_tensor* blk_attn_k[8] = {};
    struct ggml_tensor* blk_attn_v[8] = {};
    struct ggml_tensor* blk_attn_o[8] = {};
    struct ggml_tensor* blk_attn_norm_w[8] = {};
    struct ggml_tensor* blk_attn_norm_b[8] = {};
    struct ggml_tensor* blk_ffn_up[8] = {};
    struct ggml_tensor* blk_ffn_down[8] = {};
    struct ggml_tensor* blk_ffn_norm_w[8] = {};
    struct ggml_tensor* blk_ffn_norm_b[8] = {};

    // Quantizer projections
    struct ggml_tensor* vq_semantic_in = nullptr;   // input_proj
    struct ggml_tensor* vq_semantic_out = nullptr;  // output_proj
    struct ggml_tensor* vq_acoustic_in = nullptr;
    struct ggml_tensor* vq_acoustic_out = nullptr;

    // RVQ codebooks (embed_sum → nearest-neighbour)
    struct ggml_tensor* vq_semantic_cb[1] = {};     // [codebook_size, dim]
    struct ggml_tensor* vq_acoustic_cb[15] = {};

    bool loaded = false;
};

class CodecEncoder {
public:
    CodecEncoder() = default;
    ~CodecEncoder() = default;

    bool load(const GGUFLoader& gguf);
    bool is_loaded() const { return model_.loaded; }

    // Encode 24kHz mono f32 audio → codec tokens.
    // codec_tokens: [n_frames * 16] (16 codebooks per frame, matching
    // the decoder's format). Frame rate: 12.5 Hz.
    // Returns false on failure (check get_error()).
    bool encode(const float* audio, int32_t n_samples,
                std::vector<int32_t>& codec_tokens,
                int32_t& n_frames);

    const char* get_error() const { return error_msg_.c_str(); }
    const CodecEncoderConfig& config() const { return model_.config; }

private:
    CodecEncoderModel model_;
    std::string error_msg_;

    // Internal: build and run the GGML compute graph
    struct ggml_cgraph* build_encode_graph(
        struct ggml_context* ctx,
        struct ggml_tensor* inp_audio,
        int32_t n_samples,
        struct ggml_tensor** out_tokens);

    // Internal: RVQ nearest-neighbour for one codebook
    static int32_t rvq_quantize(
        const struct ggml_tensor* codebook, // [cb_size, dim]
        const float* vector,                // [dim]
        int32_t dim);
};

#endif // AUDIO_TOKENIZER_CODEC_ENCODER_H
