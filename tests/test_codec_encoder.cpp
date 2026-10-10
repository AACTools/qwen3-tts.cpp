/* test_codec_encoder.cpp — Mimi codec encoder tests
 *
 * Tests tensor loading from the v2 GGUF (with normalized codebooks).
 * Forward-pass tests will be added as the implementation progresses.
 *
 * Usage: ./build/test_codec_encoder --model models/qwen3-tts-tokenizer-f16-v2.gguf
 */

#include "audio_tokenizer_codec_encoder.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

static bool test_load(const std::string & model_path) {
    CodecEncoder encoder;
    TEST_ASSERT(!encoder.is_loaded(), "not loaded before load()");

    bool ok = encoder.load(model_path);
    if (!ok) {
        fprintf(stderr, "SKIP: load test (model not available: %s)\n",
                encoder.get_error());
        return true; // Don't fail if model isn't present
    }

    TEST_ASSERT(encoder.is_loaded(), "loaded after load()");
    TEST_ASSERT(encoder.config().sample_rate == 24000, "sample rate 24000");

    fprintf(stderr, "PASS: encoder load\n");
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

    fprintf(stderr, "═══ Codec Encoder Tests ═══\n");
    fprintf(stderr, "Model: %s\n\n", model_path.c_str());

    test_config_defaults();
    test_load(model_path);

    fprintf(stderr, "\n═══ Results: %d passed, %d failed ═══\n",
            tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
