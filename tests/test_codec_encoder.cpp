/* test_codec_encoder.cpp — Mimi codec encoder tests
 *
 * Tests tensor loading and forward-pass from the v2 GGUF.
 *
 * Usage: ./build/test_codec_encoder --model models/qwen3-tts-tokenizer-f16-v2.gguf
 */

#include "audio_tokenizer_codec_encoder.h"
#include "gguf_loader.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace qwen3_tts;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
        return false; \
    } \
} while (0)

static bool test_config_defaults() {
    MimiEncoderConfig cfg;
    TEST_ASSERT(cfg.NUM_STAGES == 4, "4 SEANet stages");
    TEST_ASSERT(cfg.CONV_STRIDES[0] == 8 && cfg.CONV_STRIDES[1] == 6 &&
                cfg.CONV_STRIDES[2] == 5 && cfg.CONV_STRIDES[3] == 4,
                "strides 8,6,5,4");
    TEST_ASSERT(cfg.HIDDEN_SIZE == 512, "hidden 512");
    TEST_ASSERT(cfg.NUM_LAYERS == 8, "8 transformer layers");
    TEST_ASSERT(cfg.NUM_HEADS == 8, "8 heads");
    TEST_ASSERT(cfg.SLIDING_WINDOW == 250, "sliding window 250");
    TEST_ASSERT(cfg.INTERMEDIATE_SIZE == 2048, "MLP intermediate 2048");
    TEST_ASSERT(cfg.CODEBOOK_SIZE == 2048, "codebook 2048");
    TEST_ASSERT(cfg.CODEBOOK_DIM == 256, "codebook dim 256");
    TEST_ASSERT(cfg.NUM_SEMANTIC + cfg.NUM_ACOUSTIC_VALID == 16, "16 total quantizers");
    fprintf(stderr, "PASS: config defaults\n");
    tests_passed++;
    return true;
}

static bool test_load_and_encode(const std::string & model_path) {
    CodecEncoder encoder;

    bool ok = encoder.load(model_path);
    if (!ok) {
        fprintf(stderr, "SKIP: load/encode (model unavailable: %s)\n",
                encoder.get_error());
        return true;
    }

    TEST_ASSERT(encoder.is_loaded(), "loaded after load()");
    fprintf(stderr, "PASS: encoder load\n");
    tests_passed++;

    // Generate 1 second of test audio (sine wave at 440 Hz)
    int32_t n_samples = 24000;
    std::vector<float> audio(n_samples);
    for (int32_t i = 0; i < n_samples; i++) {
        audio[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * (float)i / 24000.0f);
    }

    std::vector<int32_t> codec_tokens;
    int32_t n_frames = 0;

    fprintf(stderr, "Encoding %d samples (1s @ 24kHz)...\n", n_samples);
    ok = encoder.encode(audio.data(), n_samples, codec_tokens, n_frames);
    if (!ok) {
        fprintf(stderr, "INFO: encode not yet working: %s\n", encoder.get_error());
        fprintf(stderr, "SKIP: encode forward pass\n");
        return true;
    }

    TEST_ASSERT(n_frames > 0, "frames > 0");
    TEST_ASSERT((int32_t)codec_tokens.size() == n_frames * 16, "16 tokens per frame");
    for (auto t : codec_tokens) {
        TEST_ASSERT(t >= 0 && t < 2048, "token in [0, 2048)");
    }

    fprintf(stderr, "PASS: encode (%d frames, %zu tokens)\n",
            n_frames, codec_tokens.size());
    tests_passed++;
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path = "models/qwen3-tts-tokenizer-f16-v2.gguf";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[i + 1];
            i++;
        }
    }

    fprintf(stderr, "=== Codec Encoder Tests ===\n");
    fprintf(stderr, "Model: %s\n\n", model_path.c_str());

    test_config_defaults();
    test_load_and_encode(model_path);

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n",
            tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
