# Architecture

lapid turns model code into fused kernels for the device it runs on.

```
model code (hub)          Whisper, … written with tensor ops
      │
tensor graph              lazy DAG of ~25 primitives + 3 composites
      │  compile
schedule                  fused kernels + buffer plan (kernel IR)
      │  execute
backend                   CPU · Vulkan · Metal · CUDA
```

## Layers

| Layer | Crates | Responsibility |
|---|---|---|
| Hub | `lapid-hub` | Ready models: weights → graphs, tokenizers, decoding, model-specific conversion |
| Model library | `lapid-nn` | Layers, attention + KV cache, RoPE, MLPs, audio front-end, tokenizers, sampling |
| Core API | `lapid` | `Tensor`, `Device`, `.lapid` loading |
| Graph | `lapid-tensor` | Dtypes, ops, lazy DAG, shape checks, reference decompositions |
| Compiler | `lapid-compiler` | Views, fusion, memory planning → kernel IR; the `Backend` trait |
| Backends | `lapid-cpu`, `lapid-vulkan` | Implement every kernel kind of the IR |
| Data | `lapid-format`, `lapid-quant`, `lapid-convert` | `.lapid` files, quantization, generic importers |
| Tooling | `lapid-verify`, `lapid-cli` | Reference comparison, benchmarks, the `lapid` binary |

Dependencies point downward only. Core crates know nothing about specific models.

## The graph

Tensors are immutable handles to nodes of a DAG. Building ops only records them; nothing runs
until `Device::eval`.

- **Primitives:** elementwise (unary, binary, where, cast), reduce, movement (reshape, permute,
  expand, slice, pad, cat), matmul, gather.
- **Composites:** softmax, layer norm, attention. Each has a reference decomposition into
  primitives; backends may run them natively.

## The compiler

1. **Views.** Movement ops become strided views: no copies for reshape, permute, broadcast or slice.
2. **Fusion.** Elementwise chains are inlined into their consumer: matmul epilogues
   (`linear + bias + gelu` is one kernel), reduce prologues, attention inputs.
3. **Lowering.** The result is a list of launches over a small kernel IR: elementwise, reduce,
   matmul, gather, softmax, layer norm, attention.
4. **Memory.** Temporaries share memory slots by liveness.

Each kernel has a *spec* (structure, cached) and *dims* (sizes, passed at launch), so one compiled
kernel serves every shape: e.g. a growing KV cache never recompiles.

## Backends

- **CPU:** hand-written SIMD kernels (NEON/dotprod, AVX2, scalar) with runtime feature detection;
  fused expressions run through a compiled bytecode over vectorized chunks; quantized matmuls use
  int8 dot products; flash-style attention.
- **Vulkan:** kernels are WGSL templates with small substitution slots, compiled in-process to
  SPIR-V with `naga` (cooperative matrices where available); one command buffer per evaluation;
  Vulkan is loaded at runtime.
- **Metal:** kernels are `.metal` templates with slots (`simdgroup_matrix` matmul and attention),
  compiled by the system Metal compiler at runtime; unified-memory buffers, zero-copy uploads.
- **CUDA:** kernels are PTX templates with slots, JIT-compiled by the driver; the driver is loaded
  at runtime: no CUDA toolkit needed to build or run.

## Data

- **`.lapid` files:** header, section table, JSON metadata (identity, license, source, model config),
  tensor index and 256-byte-aligned tensor data with per-tensor hashes. Loaded by mmap, zero copy.
- **Weight residency:** weights are copied to the device once (default when the model fits), or
  read in place from the mmap (`--mmap`, `LAPID_MMAP=1`; automatic when the model does not fit):
  the page cache streams them from disk. GPUs that read host memory through the host page tables
  get the host pointer; the pages a GPU eval reads are re-mapped and locked around it.
- **Quantization:** `int8` / `int4` with a per-block scale; block size is an attribute of the tensor.
- **Import:** generic readers (safetensors, …); each model's conversion lives with the model.

## Adding a model

A model is only its own math: about 300 to 500 lines. Everything else is shared and reused:

- **Pipelines:** causal LM (prefill, KV cache, sampling, chat), transducer, seq2seq, streaming, TTS, embedding.
- **Building blocks:** attention variants, convolutions, norms, MLPs, RoPE, audio features, tokenizers.
- **Conversion:** a generic converter plus a declarative weight-name map per model.
- **Verification and benchmarks:** a generic harness driven by a short per-model description.

If a model needs more than that, the missing piece belongs in the shared layer, not in the model.
The step-by-step guide, with LFM3 as the worked example, is [`MODELS.md`](MODELS.md).

## Verification

A reference run of the original PyTorch model saves named activations. lapid runs the same inputs,
compares layer by layer within dtype-specific tolerances, and checks the final output exactly.
Quantized models are checked twice: kernels against a fake-quant reference (the same
weights dequantized to f32, tight tolerance), and quality loss against the original model
(reported separately). The comparison engine is generic; each model's wiring lives in the hub.
Benchmarks replay real kernel launches and compare against established engines on the same shapes.
