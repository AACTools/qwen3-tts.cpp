# In-Context Cloning + Word Timings for qwen3-tts.cpp

## Problem

The current qwen3-tts.cpp implements **x-vector-only cloning** (ECAPA speaker
embedding): the timbre is cloned but accent/prosody come from the model's
language prior. This is the official `x_vector_only_mode=true` — the Python
docs call it "reduced quality". Issue #18 tracks the missing transcript.

The fix: port the **WavTokenizer encoder** (audio → codec tokens) and implement
**in-context learning (ICL) prefill**, where the reference audio's codec tokens
and its transcript serve as context before the target text — carrying accent,
rhythm, and delivery along with the timbre.

## Architecture

```
CURRENT (x-vector only):
  text tokens → [TTS pad + speaker_embd] → [target text] → generate codes

ICL MODE (full context):
  text tokens → [ref_text tokens + ref_codec_tokens] → [target text] → generate codes
                        ↑ accent/prosody travel here
```

## What exists already

- Encoder weights ARE in the GGUF (`tok_enc.*` in `convert_tokenizer_to_gguf.py`)
  — downsample conv, conv blocks, transformer layers, RVQ codebooks (acoustic
  + semantic). The inference path simply isn't implemented.
- The decoder (`audio_tokenizer_decoder.cpp`) is fully ported — same style
  of GGML graph building.
- `verify_encoder.py` has the Python-side reference for comparison.
- Deterministic reference harness (`generate_deterministic_reference.py`)
  for parity testing.

## Encoder architecture (from config + weight names)

1. **Downsample**: strided conv (24kHz → 12.5 frames/s, stride ~1920)
2. **Conv blocks**: `tok_enc.conv.{i}` + residual `tok_enc.res.{i}.blk.{j}`
   (Snake activation, weight-normed convs)
3. **Transformer**: `tok_enc.blk.{i}` — self-attn (Q/K/V/O) + FFN,
   pre-LayerNorm, 8 layers × 512 hidden × 8 heads
4. **Quantizer**: two RVQ paths:
   - Semantic: `tok_enc.vq_semantic` (1 quantizer, dim 256, codebook 4096)
   - Acoustic: `tok_enc.vq_acoustic` (15 quantizers, dim 256, codebook 2048)
   - Combined: 16 codebooks total (matches the 16 the decoder consumes)

## Implementation plan

### Phase 1: WavTokenizer encoder port (~1 week)

`audio_tokenizer_codec_encoder.{h,cpp}` — new files:

```cpp
struct CodecEncoderConfig {
    int32_t hidden_size;      // 512
    int32_t num_layers;       // 8
    int32_t num_heads;        // 8
    int32_t codebook_size;    // 2048 (acoustic) / 4096 (semantic)
    int32_t codebook_dim;     // 256
    int32_t num_quantizers;   // 15 acoustic + 1 semantic = 16 total
    int32_t frame_rate;       // 12.5 Hz
};

class CodecEncoder {
    // Encode audio (24kHz f32 mono) → 16 codebook token sequences
    // (N frames × 16 codebooks), same format the decoder consumes.
    bool encode(const float* audio, int32_t n_samples,
                std::vector<int32_t>& codec_tokens,  // [n_frames * 16]
                int32_t& n_frames);
};
```

Graph building follows `audio_tokenizer_decoder.cpp`'s pattern:
- Build GGML compute graph for: downsample → conv blocks → transformer →
  project to quantizer dims → RVQ (nearest-neighbour lookup per codebook)
- One-shot inference (no autoregressive generation needed — encoder is
  a single forward pass)

### Phase 2: ICL prefill (~3 days)

Modify `build_prefill_graph` in `tts_transformer.cpp`:

```cpp
// New overload accepting reference context:
bool build_prefill_graph_icl(
    const int32_t* text_tokens, int32_t n_tokens,     // target text
    const int32_t* ref_text_tokens, int32_t n_ref_text, // transcript
    const int32_t* ref_codec_tokens,                    // from Phase 1
    int32_t n_ref_frames,                               // frames × 16 codebooks
    const float* speaker_embd,                          // still used
    int32_t language_id,
    std::vector<float>& prefill_embd, ...
);
```

Prefill layout becomes:
```
[role tokens (3)] [think/lang (4)] [speaker_embd (1)] [codec pad+BOS (2)]
[ref_text projected (n)] [ref_codec embedding (n_frames)]
[target_text projected (n)] [codec BOS]
```

### Phase 3: CLI + C API (~2 days)

```bash
# CLI:
./qwen3-tts-cli -m models -t "Hello" -r ref.wav --ref-text "The transcript..."
```

```c
// C API (upward-compatible addition):
Qwen3TtsAudio* qwen3_tts_synthesize_in_context(
    Qwen3Tts* tts, const char* text,
    const char* ref_audio_path, const char* ref_text,
    const Qwen3TtsParams* params);
```

### Phase 4: Word timings via attention taps (~1 week)

Instrument the transformer decode loop to capture cross-attention between
generated frames and the text-token positions in the KV cache. Each frame's
attention peak over text tokens identifies the currently-spoken word.
Convert frame index → audio time (12.5 frames/s at 24kHz = 80ms per frame).

Expose via:
```c
typedef struct Qwen3TtsWordTiming {
    int32_t text_token_start;  // token index in the text sequence
    int32_t text_token_end;
    int32_t frame_start;       // audio frame index
    int32_t frame_end;
} Qwen3TtsWordTiming;

int32_t qwen3_tts_get_word_timings(Qwen3Tts* tts,
    Qwen3TtsWordTiming* out, int32_t max_count);
```

## Testing

Every phase gets deterministic parity tests against the Python pipeline,
following the existing `generate_deterministic_reference.py` pattern:
- Phase 1: codec tokens cosine similarity vs Python encoder
- Phase 2: prefill logits cosine similarity with ICL-mode Python
- Phase 3: E2E audio comparison
- Phase 4: word timing accuracy vs forced alignment

## Success criteria

- Clone quality matches DashScope cloud enrollment (accent + prosody preserved)
- Word timings within 50ms of forced-alignment ground truth
- No regression on the existing x-vector path (backward compatible)
