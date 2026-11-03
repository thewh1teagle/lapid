# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Turn heretic's DeepSeek-V4.1-Flash abliteration directions into a lapid steering file.

heretic (huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic, MIT; ablation_spec
`m:2.5:0:0:2:1`) edits only `layers.L.attn.wo_b.weight` for L = 2..39:
`W' = W - s d dᵀ W` with s = 2.5 and d the unit `projected` direction of layer L. At runtime
that is exactly `y <- y - s d (d . y)` on the attention output after `wo_b`: lapid's
`attn_out` hook with op `ablate`, so one small file steers any quantization of the model.

Input: an `.npz` with float32 `[layers, hidden]` arrays (`projected`, `directions`, ...),
extracted from heretic's `ablation_directions.pt` without unpickling. Never load the `.pt`.

    uv run tools/steer/heretic_to_lapid.py directions.npz deepseek41-heretic.safetensors

Format: docs/STEERING.md (safetensors, `__metadata__` strings, one f32 `[hidden]` tensor
`<hook>.<layer>.<op>` per vector).
"""

import argparse
import json
import struct

import numpy as np


def write_steering(path, vectors, meta):
    """vectors: {name: float32 1-D array}; meta: {str: str}."""
    header, blobs, off = {}, [], 0
    for name, v in vectors.items():
        b = np.ascontiguousarray(v, dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": [int(v.shape[0])], "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = meta
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="directions .npz (float32 [layers, hidden] arrays)")
    ap.add_argument("out", help="output steering file (.safetensors)")
    ap.add_argument("--key", default="projected", help="array to use (default: projected)")
    ap.add_argument("--scale", type=float, default=2.5, help="ablation strength s (default 2.5)")
    ap.add_argument("--first-layer", type=int, default=2, help="first ablated layer (default 2)")
    ap.add_argument("--architecture", default="deepseek41")
    ap.add_argument("--name", default="deepseek41-heretic")
    args = ap.parse_args()

    dirs = np.load(args.npz, allow_pickle=False)[args.key].astype(np.float64)
    if dirs.ndim != 2:
        raise SystemExit(f"{args.key}: want [layers, hidden], got {dirs.shape}")
    layers, hidden = dirs.shape
    vectors = {}
    for layer in range(args.first_layer, layers):
        d = dirs[layer]
        n = np.linalg.norm(d)
        if not np.isfinite(n) or n < 1e-12:
            raise SystemExit(f"layer {layer}: zero or non-finite direction")
        vectors[f"attn_out.{layer}.ablate"] = (d / n).astype(np.float32)
    meta = {
        "format": "lapid-steer",
        "version": "1",
        "architecture": args.architecture,
        "hidden_size": str(hidden),
        "name": args.name,
        "description": f"Refusal direction ablation (heretic): attention output after wo_b, layers "
        f"{args.first_layer}-{layers - 1}, y <- y - {args.scale:g} d (d . y), {args.key} directions",
        "source": "huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic (ablation_spec m:2.5:0:0:2:1)",
        "license": "MIT",
        "scale": f"{args.scale:g}",
    }
    write_steering(args.out, vectors, meta)
    print(f"{args.out}: {len(vectors)} vectors, hidden {hidden}, layers {args.first_layer}-{layers - 1}, scale {args.scale:g}")


if __name__ == "__main__":
    main()
