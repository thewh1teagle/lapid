# Activation steering file format, version 1 (`lapid-steer`)

Reference implementation: `crates/lapid-hub/src/runtime/steer.rs` (`SteerFile::open`,
`SteerFile::save`, `Steering::apply`). Writer without lapid: `tools/steer/heretic_to_lapid.py`.
Usage: [`docs/STEERING.md`](../STEERING.md).

A steering file is a plain [safetensors](https://github.com/huggingface/safetensors) file:
`u64` little-endian header length, a JSON header, then the tensor bytes. It is small (one
`hidden_size` vector per hooked layer, under 1 MB for a 5120-wide, 40-layer model) and any
numpy / torch tooling can write it, which is why it is not a `.lapid` file.

## Metadata (`__metadata__`, all values strings)

| key | required | meaning |
|---|---|---|
| `format` | yes | `lapid-steer` |
| `version` | yes | `1` (readers reject newer versions) |
| `architecture` | yes | the `.lapid` `architecture` it was made for (`deepseek41`, `llama`, ...) |
| `hidden_size` | yes | width of every vector; must equal the model's |
| `name` | no | id used by `--steer` listings and the API (default: file stem) |
| `description`, `source`, `license` | no | free text |
| `scale` | no | default scale of every entry (default `1`) |
| `scale.<tensor>` | no | scale of one entry (overrides `scale`) |

## Tensors

One F32 (F16 / BF16 accepted) tensor of shape `[hidden_size]` per entry, named
`<hook>.<layer>.<op>`; nothing else may be in the file.

| hook | where (per layer `layer`, 0-based) |
|---|---|
| `attn_out` | the attention / token-mixer output added to the residual stream (after any post-norm); with hyper-connections (DeepSeek V4 / V4.1 mHC, Qwen4-exp) the block output before it is expanded to the streams. DeepSeek V4.1: right after `attn.wo_b`. |
| `ffn_out` | the feed-forward / MoE output, likewise |
| `resid` | the residual stream after the layer (each stream with hyper-connections) |

| op | effect on the activation `y` (last axis), with scale `s` |
|---|---|
| `ablate` | `y <- y - s * d * (d . y)`, `d` = the vector normalized at load (`s = 1` removes the direction, `s = 2.5` flips and amplifies it like heretic's weight edit) |
| `add` | `y <- y + s * v` |

The effective scale is the entry's file scale times the multiplier chosen at run time
(`--steer FILE:MULT`, the API's `"steer": {"name": MULT}`). Entries at the same hook and
layer apply in order: ablations before additions, files in the order given. Activations are
steered in f32 and cast back to their dtype.

## Validation on load

The file must parse as above (finite values, non-zero `ablate` vectors, at least one
entry) and match the model: same `architecture`, same `hidden_size`, every layer below the
model's layer count, every hook supported by the model (`CausalLm::steer_hooks`; error
code `unsupported` otherwise, `steering` for the other mismatches).

Equivalence: `ablate` at `attn_out` of a layer whose attention ends in a linear `W`
(`y = W x`) gives exactly the model with `W' = W - s d dᵀ W` (abliteration's weight edit).
