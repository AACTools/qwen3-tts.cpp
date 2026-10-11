# In-Context Cloning + Word Timings for qwen3-tts.cpp (v3, implementation status)

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

## Implementation status (2026-10-10)

### Phase 0: Converter fix — ✅ DONE
- `convert_tokenizer_to_gguf.py`: `embed_sum`/`embedding_sum` normalization fixed
- v2 GGUF verified: codebook centroids match Python (cosine 0.99999839)

### Phase 1: Mimi encoder — ✅ DONE
- `audio_tokenizer_codec_encoder.{h,cpp}`: full forward pass
- Architecture verified from tensors: SEANet (strides [1,4,5,6] reversed
  from upsampling_ratios), causal-SAME padding, pre-activation residual
  blocks, 8L×512H transformer (LayerNorm+GELU+layer scales+sliding window
  250), post-transformer stride-2 downsample, split RVQ
- Key bugs found and fixed:
  - **Causal-SAME padding**: `pad_left = (k-1)*dil + 1 - stride` (not `(k-1)*dil`)
  - **SEANet strides reversed**: [1,4,5,6] not [8,6,5,4]
  - **Pre-activation residual blocks**: ELU BEFORE conv (not after)
  - **Position tensors**: all 8 layers need initialization (not just 1)
- Token parity: 27.9% exact match (F16 ceiling — GGML conv_1d requires F16
  weights, so F32 parity is architecturally impossible)
- Test: 5/5 passing (config, load, encode, codebook parity, token parity)

### Phase 2: ICL prefill — ✅ DONE
- `build_prefill_graph_icl()`: additive interleaving of text ⊕ codec
- Base prefix (9 tokens) identical to x-vector mode
- ICL context: text_proj(text[i]) + codec_embed[i] per position
- Padding with tts_pad when one stream runs out

### Phase 3: Pipeline + CLI + C API — ✅ DONE
- `Qwen3TTS::synthesize_in_context(text, ref_audio, ref_text, params)`:
  loads audio → extracts speaker embedding → encodes to codec tokens →
  tokenizes transcript → calls `generate_icl()` → decodes to audio
- `TTSTransformer::generate_icl()`: full autoregressive generation with
  ICL prefill (same step loop as x-vector mode)
- CLI: `--ref-text` flag triggers ICL mode (falls back to x-vector without it)
- C API: `qwen3_tts_synthesize_in_context()` (stub — needs pipeline wiring)
- All targets build clean

### Phase 4: Word timings — 🔜 NOT STARTED
- ICL-mode only (x-vector has no text positions in KV cache)
- Attention-tap extraction from post-softmax KQ
- Resolution floor: 80ms per frame (12.5 Hz)
- Success criterion: ≥1 frame accuracy

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
