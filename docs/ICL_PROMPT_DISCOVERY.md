# ICL Prompt Construction — Reference Data for C++ Parity

## Critical discovery (2026-10-10)

The Python `generate_voice_clone` does NOT generate target speech directly.
It generates a CONTINUATION of the reference:

```python
# Line 148 of generate_voice_clone source:
codes_for_decode = torch.cat([ref_code, generated_codes], dim=0)

# Line 161: cut the reference portion from the audio AFTER decoding
ref_len = ref_code.shape[0]
cut = ref_len / total_len * wav.shape[0]
wav = wav[cut:]
```

The model is trained to:
1. See the ICL prefill (ref text + ref codec + speaker embedding)
2. "Reproduce" the reference speech (generating codes that match ref_code)
3. Continue generating the TARGET speech after the reference

This means the C++ implementation must:
1. Build the same prefill as Python
2. Generate enough tokens to cover ref frames + target speech
3. Decode the concatenated codes
4. CUT the reference portion from the audio output

## Python model.generate() interface

The `Qwen3TTSForConditionalGeneration.generate()` receives:
- `input_ids`: target text token IDs (with chat template)
- `ref_ids`: reference text token IDs (with chat template)
- `voice_clone_prompt`: dict with:
  - `ref_code`: [n_ref_frames, 16] codec tokens from the tokenizer encoder
  - `ref_spk_embedding`: [1024] speaker embedding from ECAPA
- `languages`: language setting
- `non_streaming_mode`: bool

## Next steps for C++ parity

1. Hook `Qwen3TTSForConditionalGeneration.generate()` to capture the exact
   `inputs_embeds` passed to the talker (the prefill embedding)
2. Compare C++ `build_prefill_graph_icl()` output element-by-element
3. Fix the C++ prefill to match
4. Add the ref-code concatenation + audio cutting step

## Saved reference files

- `reference/icl_ref_code.npy` — [188, 16] codec tokens from 15s reference
- `reference/icl_spk_embedding.npy` — [1024] speaker embedding
- `reference/icl_prefill_embedding.npy` — [1, 197, 1024] (captured but incomplete — generation timed out)

The 197 positions = 9 base prefix + ~188 ICL context (dominated by codec frames).
