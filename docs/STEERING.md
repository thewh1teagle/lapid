# Activation steering

A steering file is a small file that changes a model's behavior at run time: it projects
directions out of (or adds vectors to) the activations at chosen layers. One file works on
every quantization of the model, so an "abliterated" (refusal-removed) variant is a 1 MB file
next to the base model instead of a second copy of the weights. Format: [spec/steering.md](spec/steering.md).

Supported: DeepSeek V4.1 and V4, the Llama family (Llama 2 / 3, Mistral, Qwen2 / 2.5, Qwen3,
Qwen3-MoE, Qwen3-VL text), Qwen3.5, Qwen4-exp, LFM3 and Gemma 4, at hooks `attn_out`, `ffn_out`
and `resid`. Without `--steer` the model graph is unchanged; with it, each hooked layer costs
one dot product and one axpy.

## Command line

```sh
lapid steer deepseek41-heretic.safetensors          # print what a file does (JSON)

lapid run ~/models/deepseek41-flash-int8.lapid \
    --steer deepseek41-heretic.safetensors -p "..."  # steered chat

lapid serve model.lapid --steer a.safetensors --steer b.safetensors:0.5
```

`--steer FILE[:SCALE]` is repeatable. `SCALE` multiplies the file's own scales: `1` (default)
applies the file as made, `0.5` half as strong, `0` loads it but leaves it off until a request
turns it on. Files are checked against the model at load (architecture, hidden size, layers,
supported hooks) and fail with a clear error otherwise.

## Server API

Steering files are loaded at startup and listed per model in `GET /api/info`
(`models[].steering`: name, description, `default_scale`, entries). Chat and text completion
requests take an extra field:

```json
{"messages": [...], "steer": {"deepseek41-heretic": 1.0}}
```

- `"steer"` omitted or `null`: the startup selection (every `--steer` file at its scale).
- `"steer": {"name": scale, ...}`: exactly these files with these multipliers.
- `"steer": {}` or `false`: no steering.
- An unknown name is a 400 error.

Prompt-prefix caching stays correct: a cached session is only reused by a request with the
same steering (files and scales); otherwise its prompt is computed again.

The web UI shows the loaded files under **Chat settings → Steering**, each with an on/off switch
and a multiplier; "Server default" goes back to the startup selection.

## Making files

- From heretic's abliteration directions (DeepSeek V4.1 Flash, `pqhaz/DeepSeek-V4.1-Flash-1017-heretic`):

  ```sh
  uv run tools/steer/heretic_to_lapid.py directions.npz deepseek41-heretic.safetensors
  ```

  `directions.npz` holds the float32 `[layers, hidden]` arrays of heretic's
  `ablation_directions.pt` (read the tensors without unpickling; never load untrusted `.pt`
  files). The file ablates the `projected` direction of layers 2 to 39 at `attn_out` with scale
  2.5, which equals heretic's edit of `attn.wo_b` (`W - 2.5 d dᵀ W`).
- Any tool that writes safetensors can make one: f32 `[hidden_size]` tensors named
  `<hook>.<layer>.<ablate|add>` plus the metadata in the spec. `SteerFile::save` writes them
  from Rust.
