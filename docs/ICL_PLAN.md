# In-Context Cloning + Word Timings for qwen3-tts.cpp (v2, post-review)

## Problem

The current qwen3-tts.cpp implements **x-vector-only cloning** (ECAPA speaker
embedding): the timbre is cloned but accent/prosody come from the model's
language prior. Issue #18 tracks the missing transcript.

The fix: port the **codec encoder** (audio → discrete tokens) and implement
**in-context learning (ICL) prefill** where reference audio tokens and their
transcript serve as context before the target text.

## ⚠️ ARCHITECTURE CORRECTION (from checkpoint + Python reference)

The tokenizer encoder is **Mimi** (Kyuzan/Mimi), NOT WavTokenizer:
- `modeling_qwen3_tts_tokenizer_v2.py:899`: `Qwen3TTSTokenizerV2Encoder(MimiModel)`
- Activation: **ELU** (no Snake, no weight-norm in the encoder path)
- Topology: SEANet convs (strides 8,6,5,4 → **25 Hz**) → causal transformer
  (8L × 512H, **LayerNorm with bias**, **GELU** non-gated MLP, **layer
  scales**, **sliding window 250**, RoPE θ=10000) → downsample conv
  [512,512,4] stride 2 → **12.5 Hz** → quantizer
- The transformer sits at 25 Hz (2× the positions), the stride-2 conv
  comes **after** the transformer
- Encoder transformer ≠ decoder pre_tfm: LayerNorm (not RMSNorm), GELU
  non-gated MLP (not SiLU gated), layer scales present, causal attention

## Converter bug (must fix FIRST)

`convert_tokenizer_to_gguf.py` normalizes only tensors containing the
substring `"embedding_sum"` — encoder tensors are named `codebook.embed_sum`
which does NOT contain that substring. Result: **codebooks are written as
raw EMA sums, not centroids**. The `cluster_usage` tensors needed for
normalization are also missing from the GGUF.

Python reference (`modeling_mimi.py:1197-1199`):
```python
embed = embed_sum / cluster_usage.clamp(min=eps)
```

**Fix**: update the converter to normalize `codebook.embed_sum` tensors
matching `embed_sum` (not `embedding_sum`), write `cluster_usage` as
companion tensors, and regenerate the GGUF. Version-bump the GGUF.

## Quantizer facts (corrected)

- Semantic codebook: **2048 × 256** (not 4096; the config's 4096 is dead)
- Acoustic RVQ stores **31 layers** (0–30); valid codes = `[:, :16]` =
  semantic + acoustic[0..14]
- Semantic and acoustic chains are **parallel** (both quantize the same
  projected input; residuals chain *within* each path)
- Codebook search: plain **L2 argmin** (`torch.cdist … argmin`), not cosine
- Total 16 codebooks consumed by the decoder ✓ (but ordering matters)

## Implementation plan (revised)

### Phase 0: Converter fix + GGUF regeneration (~2 days)

1. Fix `convert_tokenizer_to_gguf.py`:
   - Change the normalization trigger from `"embedding_sum" in name` to
     `"embed_sum" in name or "embedding_sum" in name`
   - Write `cluster_usage` tensors alongside codebooks
   - Add GGUF version metadata (bump to v2)
2. Regenerate the tokenizer GGUF from the same HF weights
3. Update the decoder's codebook loading to use the companion usage
   tensors when present (backward compatible)
4. Verify: decode a known codec sequence, confirm audio parity

### Phase 1: Mimi encoder port (~2–3 weeks)

`audio_tokenizer_codec_encoder.{h,cpp}` — new files:

```
Architecture:
  input: f32 audio [n_samples] @ 24kHz
  ↓ SEANet encoder: conv chain with strides [8, 6, 5, 4], ELU activation
    → [n_samples/960] positions @ 25 Hz, hidden 512
  ↓ causal transformer: 8 layers × 512 hidden × 8 heads
    LayerNorm(w+b), GELU non-gated MLP (fc1→fc2), layer scales,
    sliding window 250, RoPE θ=10000
  ↓ downsample conv [512, 512, kernel=4] stride 2
    → [n_positions/2] positions @ 12.5 Hz
  ↓ split RVQ:
    semantic path: input_proj → 1× RVQ (2048-entry codebook, 256-dim)
    acoustic path: input_proj → 15× RVQ (2048-entry codebooks, 256-dim,
      residual-chained within acoustic path)
  output: codec_tokens [n_frames × 16] (1 semantic + 15 acoustic)
```

Key implementation notes:
- Causal conv padding: pad left only (Mimi is a causal model)
- Layer scales: `attn_scale` and `ffn_scale` per layer (scalar multipliers)
- Sliding window 250: mask attention beyond 250 positions back
- L2 argmin quantization: for each frame × codebook, find the nearest
  codebook entry by Euclidean distance
- Single forward pass (no autoregressive generation)

### Phase 2: ICL prefill (~4–6 days, after redesign)

**The prefill layout is ADDITIVE INTERLEAVING, not sequential blocks.**

From `modeling_qwen3_tts.py:1968–2019` (streaming mode):

```
Base prefix (identical to x-vector mode, 9 tokens):
  [role(3)] [think/lang(3–4)] [speaker_embd(1)] [codec_pad ⊕ tts_bos(1)]

ICL context (the new part):
  text stream: ref_text_tokens + target_text_tokens + tts_eos
  codec stream: codec_bos_embedding + Σₖ₌₀¹⁵ embedding(ref_code[:,k])
  
  Per-position: text_embed[i] ⊕ codec_embed[i] (element-wise SUM)
  If text is longer: leftover text becomes trailing stream (tts_pad padded)
  If codec is longer: codec frames without text get tts_pad overlay
```

Critical details:
- ref_text must use the same chat template tokenization, sliced `[:, 3:-2]`
- cb0 of ref codes uses `talker.codec_embd`; cb1–15 use
  `code_predictor.codec_embd[i-1]`; summed over all 16 per frame
- The `first_text ⊕ codec_bos` fusion token from x-vector mode does NOT
  exist in ICL mode
- Streaming mode matches the existing C++ (non-streaming differs)

### Phase 3: CLI + C API (~2 days)

```bash
# CLI:
./qwen3-tts-cli -m models -t "Hello" -r ref.wav --ref-text "The transcript..."
```

```c
// C API additions (upward-compatible):
Qwen3TtsAudio* qwen3_tts_synthesize_in_context(
    Qwen3Tts* tts, const char* text,
    const char* ref_audio_path, const char* ref_text,
    const Qwen3TtsParams* params);

Qwen3TtsAudio* qwen3_tts_synthesize_in_context_samples(
    Qwen3Tts* tts, const char* text,
    const float* ref_samples, int32_t n_ref_samples,
    const char* ref_text,
    const Qwen3TtsParams* params);
```

### Phase 4: Word timings (ICL-mode only, rescoped)

**Correction**: In x-vector mode there are NO text-token positions in the
KV cache (trailing text is added to the step embedding). Only ICL mode
creates mixed text⊕codec positions that attention can be tapped from.

Mechanism: `ggml_set_output` on the post-softmax KQ in `build_step_graph`,
read back `[n_kv × n_heads]` per layer per frame. Identify which layers/heads
correlate with word boundaries empirically.

Resolution floor: 80ms per frame (12.5 Hz). Success criterion: **≥1 frame
accuracy**, not 50ms (which is below the frame resolution).

## Testing strategy

Extend `generate_deterministic_reference.py` with an ICL mode run:
- Codec encoder: exact-token match rate (integers, not cosine) with
  tolerance floor for f16 argmin ties
- ICL prefill: prefill-logit cosine similarity vs Python ICL mode
- E2E: audio comparison against Python `generate_voice_clone` output

## Success criteria

- Clone quality: accent/prosody preserved from the reference (the gap
  between LOCAL and CLOUD that motivated this work)
- No regression on x-vector path (backward compatible)
- Word timings: ≥1 frame (80ms) accuracy, ICL mode only

## Effort estimate (revised)

| Phase | Estimate | Notes |
|---|---|---|
| 0: Converter fix | 2 days | Blocking dependency |
| 1: Encoder port | 2–3 weeks | The big rock: Mimi ≠ WavTokenizer |
| 2: ICL prefill | 4–6 days | Additive interleaving, not sequential |
| 3: CLI/C API | 2 days | Straightforward |
| 4: Word timings | 2–3 days infra; accuracy open-ended | ICL only, ≥80ms floor |
| **Total** | **~4–5 weeks** | |
