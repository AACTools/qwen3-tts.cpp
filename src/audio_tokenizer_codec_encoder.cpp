/* audio_tokenizer_codec_encoder.cpp — Mimi encoder implementation
 *
 * Single forward pass: audio → SEANet convs → causal transformer →
 * stride-2 downsample → split RVQ → codec tokens [n_frames × 16].
 *
 * Architecture verified from checkpoint tensors + config — see
 * docs/ENCODER_ARCHITECTURE.md for the complete reference.
 *
 * Conventions follow AGENTS.md: C++17, no exceptions, bool returns,
 * error_msg_ member, snake_case, qwen3_tts namespace, fprintf(stderr).
 */

#include "audio_tokenizer_codec_encoder.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace qwen3_tts {

#define ENC_LOG(...) do { fprintf(stderr, "[codec-encoder] " __VA_ARGS__); } while (0)

// ─── Load from GGUF ────────────────────────────────────────────────────

bool CodecEncoder::load(const std::string & model_path) {
    unload();

    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }

    model_.config.sample_rate = loader.get_u32("qwen3-tts.tokenizer.sample_rate", 24000);

    int64_t n_tensors = loader.get_n_tensors();

    // Count encoder tensors
    int enc_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "tok_enc.", 8) == 0) {
            enc_tensor_count++;
        }
    }

    if (enc_tensor_count == 0) {
        error_msg_ = "No encoder tensors (tok_enc.*) found in model";
        return false;
    }

    ENC_LOG("Found %d encoder tensors\n", enc_tensor_count);

    // Create GGML context for tensor metadata
    size_t ctx_size = ggml_tensor_overhead() * enc_tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to initialize GGML context";
        return false;
    }

    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();

    // Map tensors from metadata to our context
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "tok_enc.", 8) != 0) {
            continue;
        }

        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta_ctx, name);
        if (!meta_tensor) {
            continue;
        }

        struct ggml_tensor * tensor = ggml_dup_tensor(model_.ctx, meta_tensor);
        ggml_set_name(tensor, name);
        model_.tensors[name] = tensor;

        std::string sname(name);

        // ── SEANet entry convs ──
        if (sname == "tok_enc.conv.0.weight")  model_.conv_entry_w[0] = tensor;
        if (sname == "tok_enc.conv.0.bias")    model_.conv_entry_b[0] = tensor;
        if (sname == "tok_enc.conv.3.weight")  model_.conv_entry_w[1] = tensor;
        if (sname == "tok_enc.conv.3.bias")    model_.conv_entry_b[1] = tensor;
        if (sname == "tok_enc.conv.6.weight")  model_.conv_entry_w[2] = tensor;
        if (sname == "tok_enc.conv.6.bias")    model_.conv_entry_b[2] = tensor;
        if (sname == "tok_enc.conv.9.weight")  model_.conv_entry_w[3] = tensor;
        if (sname == "tok_enc.conv.9.bias")    model_.conv_entry_b[3] = tensor;

        // ── SEANet bottleneck convs ──
        if (sname == "tok_enc.conv.12.weight") model_.bottleneck_expand_w = tensor;
        if (sname == "tok_enc.conv.12.bias")   model_.bottleneck_expand_b = tensor;
        if (sname == "tok_enc.conv.14.weight") model_.bottleneck_contract_w = tensor;
        if (sname == "tok_enc.conv.14.bias")   model_.bottleneck_contract_b = tensor;

        // ── Residual blocks (4 stages × 2 convs each) ──
        // res.{stage}.blk.{conv_idx}.{weight,bias}
        for (int stage = 0; stage < 4; stage++) {
            for (int ci = 0; ci < 2; ci++) {
                int layer_idx = stage * 3 + 1; // 1, 4, 7, 10
                int conv_idx = ci == 0 ? 1 : 3; // blk.1, blk.3
                std::string wname = "tok_enc.res." + std::to_string(layer_idx) +
                    ".blk." + std::to_string(conv_idx) + ".weight";
                std::string bname = "tok_enc.res." + std::to_string(layer_idx) +
                    ".blk." + std::to_string(conv_idx) + ".bias";
                if (sname == wname) model_.res_w[stage][ci] = tensor;
                if (sname == bname) model_.res_b[stage][ci] = tensor;
            }
        }

        // ── Transformer layers ──
        for (int layer = 0; layer < 8; layer++) {
            std::string prefix = "tok_enc.blk." + std::to_string(layer) + ".";
            if (sname == prefix + "attn_q.weight")         model_.tfm_q[layer] = tensor;
            if (sname == prefix + "attn_k.weight")         model_.tfm_k[layer] = tensor;
            if (sname == prefix + "attn_v.weight")         model_.tfm_v[layer] = tensor;
            if (sname == prefix + "attn_output.weight")    model_.tfm_o[layer] = tensor;
            if (sname == prefix + "attn_norm.weight")      model_.tfm_attn_ln_w[layer] = tensor;
            if (sname == prefix + "attn_norm.bias")        model_.tfm_attn_ln_b[layer] = tensor;
            if (sname == prefix + "attn_scale")            model_.tfm_attn_scale[layer] = tensor;
            if (sname == prefix + "ffn_up.weight")         model_.tfm_ffn_up[layer] = tensor;
            if (sname == prefix + "ffn_down.weight")       model_.tfm_ffn_down[layer] = tensor;
            if (sname == prefix + "ffn_norm.weight")       model_.tfm_ffn_ln_w[layer] = tensor;
            if (sname == prefix + "ffn_norm.bias")         model_.tfm_ffn_ln_b[layer] = tensor;
            if (sname == prefix + "ffn_scale")             model_.tfm_ffn_scale[layer] = tensor;
        }

        // ── Post-transformer downsample ──
        if (sname == "tok_enc.downsample.weight") model_.downsample_w = tensor;

        // ── Quantizer projections ──
        if (sname == "tok_enc.vq_semantic.input_proj.weight")  model_.vq_sem_in = tensor;
        if (sname == "tok_enc.vq_semantic.output_proj.weight") model_.vq_sem_out = tensor;
        if (sname == "tok_enc.vq_acoustic.input_proj.weight")  model_.vq_aco_in = tensor;
        if (sname == "tok_enc.vq_acoustic.output_proj.weight") model_.vq_aco_out = tensor;

        // ── Semantic codebook ──
        if (sname == "tok_enc.vq_semantic.0.codebook") model_.vq_sem_cb = tensor;

        // ── Acoustic codebooks (layers 0-14 of 31) ──
        for (int q = 0; q < 15; q++) {
            std::string cbname = "tok_enc.vq_acoustic." + std::to_string(q) + ".codebook";
            if (sname == cbname) model_.vq_aco_cb[q] = tensor;
        }
    }

    // Load actual tensor data from file into backend buffer
    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx,
                                     model_.tensors, model_.buffer, error_msg_)) {
        error_msg_ = "Failed to load encoder tensor data: " + error_msg_;
        return false;
    }

    model_.loaded = true;
    ENC_LOG("Encoder loaded: %d tensors\n", enc_tensor_count);
    return true;
}

void CodecEncoder::unload() {
    if (model_.buffer) {
        ggml_backend_buffer_free(model_.buffer);
        model_.buffer = nullptr;
    }
    if (model_.ctx) {
        ggml_free(model_.ctx);
        model_.ctx = nullptr;
    }
    model_.tensors.clear();
    model_.loaded = false;
}

// ─── Encode (stub — forward pass implementation pending) ─────────────

bool CodecEncoder::encode(
    const float * audio,
    int32_t n_samples,
    std::vector<int32_t> & codec_tokens,
    int32_t & n_frames)
{
    if (!model_.loaded) {
        error_msg_ = "encoder not loaded";
        return false;
    }

    // TODO: implement the forward pass per docs/ENCODER_ARCHITECTURE.md:
    // 1. SEANet stages (conv + res blocks, ELU)
    // 2. Bottleneck (conv.12 + conv.14)
    // 3. Transformer (8 layers, LayerNorm+GELU, sliding window 250)
    // 4. Downsample (stride 2)
    // 5. Split RVQ (L2 argmin)

    ENC_LOG("encode: forward pass not yet implemented\n");
    error_msg_ = "CodecEncoder::encode forward pass not yet implemented";
    return false;
}

} // namespace qwen3_tts
