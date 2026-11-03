# `.lapid` model file format, version 1 (format ids `lapid/1` and `lapid/1.1`)

Reference implementation: `crates/lapid-format` (`Reader`, `Writer`, `cmd::info`).
All integers are little endian. Hashes are BLAKE3 (32 bytes; hex = 64 lowercase chars).

## Versions

The binary header version stays `1`; minor revisions are named by metadata `"format"`.

| format | adds |
|---|---|
| `lapid/1` | header, section table, METADATA, TENSOR_INDEX, TENSOR_DATA |
| `lapid/1.1` | BLOBS section (named byte blobs), `components`, `kind` / `extends` (extras files) |

Readers accept both ids. The 1.1 metadata keys (`components`, `kind`, `extends`) are only
interpreted in `lapid/1.1` files; in `lapid/1` files they are free-form like any other key.
A 1.0 reader rejects 1.1 files (format id check; BLOBS is a required section). The
reference writer always writes `lapid/1.1` (a given `"lapid/1"` is upgraded).

One file holds a whole model: several components (e.g. a language model, an audio codec
and default voices) namespaced by tensor prefix, plus non-tensor assets (tokenizer files,
chat templates, voice files) as blobs.

## Overall structure

```
[header 64 B] [section table] [sections ...]
```

Sections may appear in any order and at any offset. The reference writer produces:

```
0       header (64 B)
64      section table (3 or 4 entries x 56 B), zero pad to 1024
1024    METADATA (JSON)          (format 1.0 writers put it at 256)
        64 KiB of zero padding (room for in-place metadata edits)
D       TENSOR_DATA, D = align_up(end_of_metadata + 65536, 4096); each tensor 256-aligned
I       TENSOR_INDEX (JSON), I = align_up(end_of_tensor_data, 256)
B       BLOBS (only if the file has blobs), B = align_up(end_of_index, 4096)
```

## Header (64 bytes)

| offset | size | field | value |
|---|---|---|---|
| 0 | 4 | magic | `"LPID"` (4C 50 49 44) |
| 4 | 4 | version | u32 = 1 |
| 8 | 8 | required_features | u64 bit set; v1 defines no bits, must be 0 |
| 16 | 8 | table_offset | u64, absolute offset of the section table |
| 24 | 8 | table_count | u64, number of entries (1..=64) |
| 32 | 32 | reserved | zero when written, ignored when read |

## Section table entry (56 bytes)

| offset | size | field |
|---|---|---|
| 0 | 4 | kind (u32) |
| 4 | 4 | flags (u32): bit 0 = required |
| 8 | 8 | offset (u64, absolute) |
| 16 | 8 | length (u64, bytes) |
| 24 | 32 | blake3 of the `length` bytes at `offset` |

Kinds: `1` METADATA, `2` TENSOR_INDEX, `3` TENSOR_DATA, `4` BLOBS (1.1). The first
three must appear exactly once; BLOBS at most once (absent = no blobs). The writer marks
all sections required. Unknown kinds: ignored when the required bit is clear, rejected
(`unsupported_section`) when set.

TENSOR_DATA's hash covers the whole section including inter-tensor padding. Readers do not
check it on open (that would read the whole file); `verify_tensors` checks the per-tensor
hashes from the index instead.

## METADATA (kind 1)

A UTF-8 JSON object, at most 16 MiB. Required keys:

| key | type | meaning |
|---|---|---|
| `format` | string | `"lapid/1"` or `"lapid/1.1"` |
| `name` | string | model name, e.g. `"whisper-small"` |
| `architecture` | string | e.g. `"whisper"` |
| `dtype` | string | main weight precision: `"fp32"`, `"fp16"`, `"int8"`, `"int4"`, `"q4_k"`, `"q5_k"`, `"q6_k"`, `"q4_k_m"` (llama.cpp's Q4_K_M mix) |
| `license` | string | SPDX id (lowercase ok) or `"unknown"` |
| `source` | object | `{"repo": string, "revision"?: string, "file_hash"?: "blake3:<hex>", ...}` |
| `converter` | string | `"lapid-convert <version>"` |

Missing keys → `missing_metadata` (writer and reader). Other keys are free-form; by
convention a model stores its configuration in an object named after its architecture.

Optional keys (format 1.1):

| key | type | meaning |
|---|---|---|
| `kind` | string | `"model"` (default) or `"extra"`; anything else → `unsupported` |
| `extends` | object | extras only, required there: `{"architecture": string, "component": string}` |
| `components` | object | named sub-models, see below |
| `quantization` | object | how quantized weights were made when not plain round-to-nearest, e.g. calibrated int4 `{"method": "gptq", "calibration": "builtin", ...}` (docs/spec/quant.md) |

### Components

```json
"components": {
  "lm":            {"architecture": "llama", "config": "llama", "license": "apache-2.0", "tensor_prefix": "lm."},
  "codec":         {"architecture": "mimi",  "config": {"...": "..."},                   "tensor_prefix": "codec."},
  "voice.default": {"architecture": "voice", "config": {"dim": 1024},                    "tensor_prefix": "voice.default."}
}
```

- Name (the key): 1..=256 bytes. 1..=64 components.
- `architecture` (string, required).
- `config` (required): an object (inline), or a string naming a top-level metadata key
  whose value is an object (avoids duplicating the conventional `"<architecture>"`
  config object). A dangling reference → `missing_metadata`.
- `license` (optional SPDX string): overrides the file's `license` for this component
  (e.g. an Apache LM bundled with an MIT codec). The file's `license` should then describe
  the combination (e.g. the most restrictive one).
- `tensor_prefix` (string, required): `""` or ending in `.`, ≤ 256 bytes, distinct across
  components. A component's tensors are the tensors whose names start with its prefix
  (longest prefix wins); names below the prefix are the component's own naming.
- Without `components`, a model file has one implicit component named after
  `architecture`, config = metadata[`architecture`] (if an object), prefix `""`.

### Extras files

A `.lapid` file with `"kind": "extra"` is not loadable on its own: it adds tensors and/or
blobs to a component of a model of `extends.architecture` (a user voice, later a LoRA
adapter). Rules:

- All required metadata keys still apply; `architecture` must equal
  `extends.architecture` (`format` otherwise).
- Tensor and blob names are relative to the extended component: a loader resolves an
  extra's tensor `voice.mine.embedding` as the component's own name, exactly as if it were
  stored after the component's `tensor_prefix` in the base file.
- An extra may carry its own `components` (rare); without it, it has none.
- How an architecture consumes extras (e.g. "a voice extra provides `voice.<id>.*`") is
  model-specific and documented with the model. No model uses extras yet.
### `whisper` object (architecture = "whisper", written by `lapid-hub`'s Whisper converter)

```
n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, n_audio_ffn,
n_text_ctx, n_text_state, n_text_head, n_text_layer, n_text_ffn, n_vocab   (integers)
multilingual: bool        -- false for *.en models: the prompt is just <|startoftranscript|>
activation: "gelu"
tokens: [string; n_vocab] -- id-ordered; byte-level BPE form (GPT-2 bytes_to_unicode,
                             e.g. " the" = "Ġthe"); special tokens verbatim ("<|en|>")
special: { sot, eot, transcribe, translate, no_timestamps, timestamp_begin
           (= no_timestamps + 1), no_speech, prev, sot_lm? }
languages: { "en": 50259, ... }       -- code → token id (empty for English-only models)
suppress_tokens: [id], begin_suppress_tokens: [id]
alignment_heads?: [[layer, head], ...]
max_initial_timestamp_index?: 50
```

In format 1.1 files the token list is not in metadata: `tokens_blob` (string, default
`"tokenizer/tokens.json"`) names a blob (`application/json`) holding the same id-ordered
array of `n_vocab` strings. `lapid/1` files keep `tokens` inline. Whisper files declare one
component: `{"whisper": {"architecture": "whisper", "config": "whisper", "license", "tensor_prefix": ""}}`.

For `openai/whisper-small`: sot 50258, eot 50257, en 50259, translate 50358,
transcribe 50359, sot_lm 50360, prev 50361, no_speech 50362, no_timestamps 50363,
timestamp_begin 50364, n_vocab 51865.

## TENSOR_INDEX (kind 2)

A UTF-8 JSON array (≤ 64 MiB, ≤ 1 000 000 entries) of:

```json
{"name": "model.encoder.conv1.weight", "dtype": "f16", "shape": [768, 80, 3],
 "offset": 69632, "nbytes": 368640, "hash": "<64 hex chars, blake3 of the bytes>"}
{"name": "model.encoder.layers.0.fc1.weight", "dtype": "int8", "shape": [3072, 768],
 "offset": 438272, "nbytes": 2506752, "hash": "...", "block_size": 32}
```

- `dtype`: `f32`, `f16`, `bf16`, `i32`, `int8`, `int4`, `q4_k`, `q5_k`, `q6_k`, `mxfp4`,
  `iq2_xxs`, `iq2_s`, `iq3_xxs` (see docs/spec/quant.md; the k-quants and i-quants are ggml
  super-blocks of 256 elements, stored as in GGUF).
- `block_size` (quantized dtypes only, omitted otherwise): elements per scale along the
  last axis. Writers always emit it; v1 supports only `32` — any other value is rejected
  with `unsupported`. A quantized entry without it (files from before the attribute) is
  read as `32`; `block_size` on a non-quantized dtype is `format`.
- `nbytes` must equal `product(shape) * size(dtype)` for plain dtypes; for quantized
  dtypes `shape` must be `[rows, cols]` with `cols % block_elems == 0` (32 for int8/int4,
  256 for the k-quants) and `nbytes` must equal `lapid_quant::layout(..).total`; a 3-D
  `[E, rows, cols]` quantized tensor is an expert stack, `E` such layouts back to back.
- k-quant dtypes have their block structure fixed by the dtype; writers emit `block_size: 256`
  for them (lapid/1.1 writers emitted 32) and readers ignore `block_size` for k-quant dtypes.
- `offset` is absolute, a multiple of 256, and `[offset, offset+nbytes)` lies inside
  TENSOR_DATA. Non-empty tensors do not overlap. Names are unique, 1..=1024 bytes.
- Element data is little endian, row-major (C order).

## TENSOR_DATA (kind 3)

Starts at a 4096-aligned offset. Holds the tensor bytes at the offsets listed in the index;
gaps are zero. Loaders mmap the file (or take the whole image from memory: `Reader::from_bytes` /
`from_static`, e.g. `include_bytes!` or on wasm, which has no mmap) and use tensor slices
directly (zero copy).

## BLOBS (kind 4, format 1.1)

Named byte blobs for non-tensor assets. Offsets are relative to the section start, so the
section can be copied between files verbatim:

```
0       u64 index_length L
8       index: UTF-8 JSON array (L bytes, ≤ 16 MiB, ≤ 65 536 entries)
        zero pad
...     blob bytes, each blob at a 256-aligned offset ≥ 8 + L; gaps zero
```

```json
{"name": "tokenizer/tokens.json", "media_type": "application/json",
 "offset": 4352, "nbytes": 812345, "hash": "<64 hex chars, blake3 of the bytes>"}
```

- `name`: 1..=1024 bytes, unique. Convention: `/`-separated paths, component assets under
  the component name (`lm/tokenizer.json`, `lm/chat_template.jinja`, `voice/default.bin`).
- `media_type`: `type/subtype[;params]`, 1..=255 printable ASCII bytes, no spaces.
- `[offset, offset + nbytes)` inside the section, after the index; non-empty blobs do
  not overlap.
- The whole section is ≤ 4 TiB. Its section hash is checked on open when the section is
  ≤ 256 MiB (tokenizers and other small assets); larger sections (raw tables streamed
  from a source file, e.g. embedding hash tables) are only index-validated on open.
  Per-blob hashes are checked by `verify_tensors` / `lapid info --verify`.

## Reader validation (in order)

1. File ≥ 64 B, magic, `version == 1` (`unsupported_version`), no unknown
   `required_features` bits (`unsupported_feature`).
2. `1 <= table_count <= 64`, table inside the file, after the header.
3. Every section range inside the file (no u64 overflow); header, table and all non-empty
   sections pairwise disjoint; no duplicate known kinds; unknown required kinds rejected.
4. Exactly one METADATA / TENSOR_INDEX / TENSOR_DATA, at most one BLOBS; size limits
   (`limit_exceeded`).
5. METADATA and TENSOR_INDEX blake3 match (`hash_mismatch`); both parse as JSON;
   required metadata keys present; `format` is `lapid/1` or `lapid/1.1`.
6. Every index entry validated as above.
7. (1.1) `kind` / `extends`, then `components`, validated as above.
8. BLOBS (if present): blake3 of the section (sections ≤ 256 MiB), then the blob index as
   above.

Error codes: `format` (malformed), `unsupported` (quantization `block_size` ≠ 32),
`unsupported_version`, `unsupported_feature`, `unsupported_section`, `limit_exceeded`, `hash_mismatch`, `missing_metadata`, `io`.

The mapping assumes the file is not modified while open.

## Writer

`Writer::create(path, metadata)` validates metadata (setting `"format": "lapid/1.1"`;
components / kind / extends checked up front), writes to `<path>.partial`; `add` streams
each tensor immediately (hashing as it goes, on all cores for large tensors);
`add_blob(name, media_type, bytes)` copies a small blob into memory;
`add_blob_shared(name, media_type, image, range)` records a slice of a shared image (e.g.
a source mmap), hashes it, and streams it in chunks at `finish` (memory stays bounded for
blobs of any size); `finish` writes the index, the BLOBS section, the table and the header,
fsyncs and renames to `path`. A writer dropped without `finish` deletes its partial file.

Reader API: `metadata`, `tensors`, `tensor(name)`, `tensor_bytes`, `blobs()`,
`blob(name) -> Option<&[u8]>`, `components()`, `kind()`, `extends()`, `verify_tensors()`.
`lapid::LapidFile` exposes `blob`, `blobs`, `components`, `extends`.

## In-place metadata edits

A tool may rewrite METADATA in place if the new JSON fits in `length + 65536` bytes before
the next section: write the bytes, then update the table entry's length and hash.

## `lapid info`

`lapid_format::cmd::info(path, verify)` returns:

```json
{"path": "...", "file_size": 0, "format": "lapid/1.1", "version": 1, "kind": "model",
 "metadata": {"...": "...", "whisper": {"tokens_blob": "tokenizer/tokens.json", "...": "..."}},
 "components": [{"name": "whisper", "architecture": "whisper", "license": "mit",
                 "tensor_prefix": "", "tensor_count": 479, "tensor_bytes": 0}],
 "blobs": [{"name": "tokenizer/tokens.json", "media_type": "application/json", "nbytes": 0}],
 "tensor_count": 479, "dtypes": {"f16": {"count": 0, "bytes": 0, "params": 0}},
 "total_params": 0, "tensor_bytes": 0, "verified": true}
```
Extras files add `"extends": {"architecture", "component"}`. Component `license` falls
back to the file's.
Every metadata array longer than 32 items (at any depth) is shown as `"<array of N>"`. With `verify`, a hash mismatch is returned as
an error (`hash_mismatch`, naming the tensor).
