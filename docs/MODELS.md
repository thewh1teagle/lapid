# Adding a model

A model is only its own math. Everything else (loading, pipelines, conversion, verification,
benchmarks, serving) comes from the shared framework in `lapid-hub/src/runtime`,
`lapid-convert` and `lapid-verify`. LFM3 (`crates/lapid-hub/src/lfm3`, 500 lines) is the
worked example used below.

## Files

```
crates/lapid-hub/src/<model>/
  mod.rs       public type alias, descriptor, registry entry
  config.rs    hyper-parameters from the HF config.json keys
  model.rs     weights and the forward graph
  convert.rs   name table, role table, small hooks
tools/reference/<model>_ref.py   reference run of the original model
data/reference/<model>/          result.json + activations.safetensors
```

## 1. The math

Pick the pipeline trait that fits and implement only that:

| Kind | Trait | Module |
|---|---|---|
| Decoder-only LM | `CausalLm` | `runtime::causal_lm` |
| RNN-T / TDT speech recognition | `Transducer`, `StreamingTransducer` | `runtime::transducer` |
| Encoder-decoder (Whisper style) | `Seq2Seq` | `runtime::seq2seq` |
| Text embeddings | `TextEmbedder` | `runtime::embedding` |
| Speech synthesis | `SpeechModel` | `runtime::tts` |

For a causal LM (LFM3, `lfm3/model.rs`):

```rust
impl CausalLm for Lfm3Model {
    const ARCH: &'static str = "lfm3";          // .lapid architecture + metadata key
    type State = LmCache;                        // KV cache + per-layer recurrent slots
    fn load(l: &Loader<'_>, meta: &Value) -> Result<Self>;
    fn dims(&self) -> LmDims;
    fn new_state(&self) -> LmCache;
    fn end_of_turn(&self) -> &'static [&'static str] { &["<|im_end|>"] }
    fn forward(&self, ids: &[u32], st: &mut LmCache, rows: Rows, taps: &mut Taps)
        -> Result<Option<Tensor>>;
}
pub type Lfm3 = Llm<Lfm3Model>;
```

`forward` builds the lazy graph for new tokens at `st.offset()`: append K/V with
`st.kv.append(layer, ..)`, keep conv or SSM state with `st.with_slot(i, |s| ..)`, call
`taps.tap("layers.{i}", &x)` for every intermediate the reference names, and return the logits
of `rows`. For activation steering ([STEERING.md](STEERING.md)) pass the attention output,
the FFN output and the residual after each block through `taps.steer(Hook::AttnOut | FfnOut |
Resid, i, &x)` and list them in `CausalLm::steer_hooks` (identity when no file is loaded). `Llm<M>` then gives `from_hub`, `load`, `load_bytes`, chunked prefill, decode,
sessions with prefix reuse, chat templates (Jinja), sampling, stop tokens and strings,
streaming, cancellation and timing. Templates that read globals beyond messages / tools / thinking
(Llama 3.x `date_string`) get them from `CausalLm::template_globals` (model defaults) or
`ChatOptions::template_vars` (per call); `lapid verify` passes a reference's
`template_globals` / `date_string`.

Building blocks live in `lapid-nn` (`Attention`, `KvCache`, `Rope`, `ZeroTailRope`,
`GatedMlp`, `Conv1d`, norms, `Loader::linear_checked`, `vision` for image towers: images,
patching, 2-D RoPE, pooling; `llm` for decoder LLMs: MoE experts and routing, gated attention,
Gated DeltaNet, hyper-connections, RoPE variants, LM head ...). If a model needs a piece that is
not there and another model could use it, add it there instead of to the model.

## 2. Conversion

Declare a `ConvertSpec` (`lfm3/convert.rs`): accepted HF `model_type`s and GGUF
architectures, name rules (`{i}` captures a segment, `*` matches anything, an empty target
drops the tensor) and role rules for the dtype policy:

```rust
pub static CONVERT: ConvertSpec = ConvertSpec {
    architecture: "lfm3", license: "lfm1.0",
    hf_model_types: &["lfm3"], gguf_architectures: &["lfm3"],
    hf_names: &[Rule::drop("lm_head.weight")],
    gguf_names: &[Rule::new("blk.{i}.attn_q.{w}", "model.layers.{i}.self_attn.q_proj.{w}"), ...],
    roles: &[RoleRule::new(EMBED, Role::LinearInt8)],
    text_assets: true,      // tokenizer.json, chat template, special tokens, generation config
    hooks: &Hooks,
};
```

`ConvertHooks` covers what a table cannot: `expected` (required), `hf_config` /
`gguf_config` (config object, validated), `reshape`, `row_order` (raw row permutation, e.g.
`transforms::llama_qk_unpermute`, keeps quantized blocks), `transform` (fold BatchNorm,
split fused tensors, combine tensors) and `metadata`. The engine handles HF repos and
directories (sharded safetensors included), GGUF with K-quant passthrough, blobs, metadata
and the dtype policy.

## 3. Verification and benchmarks

Write `tools/reference/<model>_ref.py` in the shared format (see `lapid_verify::harness`)
and give a `Descriptor`:

```rust
impl LmDef for Lfm3Model {
    const DESCRIPTOR: Descriptor = Descriptor { name: "lfm3", default_model: "LiquidAI/LFM3-1.7B",
        default_dtype: "int8", reference_dir: "data/reference/lfm3", strict_dtypes: &["fp32", "fp16"], ... };
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;
}
pub static ENTRY: ModelEntry = ModelEntry::causal_lm::<Lfm3Model>();
```

Add `&crate::<model>::ENTRY` to `runtime::registry::MODELS`. That one line gives
`lapid convert`, `lapid verify <model>` (template, tokenizer, layer by layer taps, cached
decode, greedy tokens, fake-quant kernel check for quantized files), `lapid bench <model>`,
calibrated int4, `lapid run` and `lapid serve`.

Other pipelines follow the same pattern with their own definition trait and entry:

| Kind | Definition | Entry | Model-specific verify part |
|---|---|---|---|
| Text embeddings | `embedding::EmbedderDef` (`DESCRIPTOR`, `CONVERT`, `from_file`) | `ModelEntry::embedder::<M>()` | none: taps from `encode` / `project`, cosine of `embed` vs the reference, Matryoshka sizes |
| Speech synthesis | `tts::SpeechDef` (adds `request_options`, `tokens`) | `ModelEntry::speech::<M>()` | `reference_stages` (teacher-forced stages) and `reference_speech` (free run on the reference noise) |
| Speech recognition, other audio | `audio::AudioDef` (adds `stages`, `end_to_end`, `bench_run`); recognizers also `audio::AsrModel` | `audio::asr_entry::<M>()`, `audio::audio_entry::<M>()` | `stages` (rows on the reference inputs: front end, encoder taps, step logits), `end_to_end` (tokens / text checks), `bench_run` (stage times) |
| Image + text LM | `vlm::VisionLm` on top of `CausalLm` (`prepare_image`, `encode_image`, `forward_media`, `image_tokens`) | `ModelEntry::causal_lm::<M>()` with `verify` / `bench` set to `vlm_verify::{verify, bench}` | `reference_image` (the reference's prepared pixels), `media_taps` (encoder outputs the decoder check can take from the reference) |

The generic harnesses are `runtime::embedding_harness`, `runtime::tts_harness` and
`runtime::audio_verify` / `audio_bench` (EmbeddingGemma, Phonon, and Whisper (seq2seq),
Parakeet (offline TDT) and Zipformer (streaming RNN-T) are the worked examples). Checkpoints with more than
`config.json` + weights (sentence-transformers modules, voice files, model cards) use
`ConvertHooks::hf_extras` to add weight files, blobs and metadata.

Vision-language models (Gemma 4 is the worked example) get chat with image parts
(`vision::Message`), placeholder expansion, image encoding and image-aware session reuse
from `Llm::mm_prompt` / `mm_generate` / `mm_chat_stream`. A GGUF model whose vision tower
ships as a separate llama.cpp `mmproj` converts as one file through
`ConvertHooks::gguf_companion` (usually `engine::sibling_mmproj`), `companion_names` and
`companion_config`.

## 4. Check

```
lapid verify <model> --dtype fp16 --device cpu      # greedy must be token-exact
lapid verify <model> --device cuda                  # int8: kernels vs fake-quant
lapid bench <model> --device cuda
```
