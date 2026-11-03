"""Engram hash constants and test vectors for `deepseek41_ref.py` (helper module), computed with the
official `inference/engram.py` (`EngramLayout`, `NgramHashState`, `build_compressed_token_map`,
`compute_hash_multipliers`) from a config + the real tokenizer. No weights are involved, so the
REAL config's constants (needed by the Rust converter / model) come out of the same code path.

Row semantics: `rows[t][e][c]` is the row of layer `layer_ids[e]`'s own table
(`layers.{id}.engram.embed`) for token t and column c = (n - 2) * n_heads + head (n-gram size n =
2..max_ngram_size), i.e. `hash % primes[e][n-2][head] + offsets[e][c]`. llama.cpp concatenates
the tables of all Engram layers: `concat_offsets[e][c] = offsets[e][c] + sum(num_embeddings[:e])`.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch


def cached_token_map(engram_mod):
    """Memoize build_compressed_token_map (one pass over the 129k-token vocab) per tokenizer content."""
    orig = engram_mod.build_compressed_token_map
    memo: dict[str, tuple[list[int], int]] = {}

    def build(tokenizer):
        key = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        if key not in memo:
            memo[key] = orig(tokenizer)
        return memo[key]

    engram_mod.build_compressed_token_map = build


def hash_state(E, args, tokenizer):
    layout = E.EngramLayout.from_args(args)
    sums = [sum(p for per in layer for p in per) for layer in layout.primes]
    assert tuple(sums) == tuple(args.engram_num_embeddings), (sums, args.engram_num_embeddings)
    return E.NgramHashState(args, layout, tokenizer), layout


def rows_for(state, ids: list[int]) -> torch.Tensor:
    """[L, n_layers, n_cols] hashed rows; also checks prefill-then-decode gives the same rows."""
    t = torch.tensor([ids])
    full = state(t, 0)[0].clone()
    k = max(1, len(ids) // 2)
    parts = [state(t[:, :k], 0)[0].clone()] + [state(t[:, j : j + 1], j)[0].clone() for j in range(k, len(ids))]
    assert torch.equal(torch.cat(parts), full), "incremental engram hashing differs from prefill"
    return full


def constants(state, layout, args, cases: dict[str, list[int]]) -> tuple[dict, np.ndarray]:
    token_map = state.token_map.numpy().astype(np.int32)
    n_cols = (layout.max_ngram_size - 1) * layout.n_heads
    offsets = state.offsets.tolist()
    bases = np.cumsum([0, *layout.num_embeddings[:-1]]).tolist()
    out = {
        "layer_ids": list(layout.layer_ids),
        "max_ngram_size": layout.max_ngram_size,
        "n_heads": layout.n_heads,
        "head_dim": layout.head_dim,
        "n_cols": n_cols,
        "engram_vocab_size": args.engram_vocab_size,
        "compressed_vocab_size": int(token_map.max()) + 1,
        "pad_token_id": args.engram_pad_id,
        "pad_compressed_id": int(state.pad_id),
        "num_embeddings": list(layout.num_embeddings),
        "multipliers": state.multipliers.tolist(),
        "primes": [[list(per) for per in layer] for layer in layout.primes],
        "primes_flat": [[p for per in layer for p in per] for layer in layout.primes],
        "offsets": offsets,
        "concat_offsets": [[o + b for o in row] for row, b in zip(offsets, bases)],
        "token_map": {
            "file": "engram_token_map.npy",
            "dtype": "int32",
            "len": int(token_map.size),
            "sha256_le_int32": hashlib.sha256(token_map.astype("<i4").tobytes()).hexdigest(),
            "head": token_map[:64].tolist(),
        },
        "hash_cases": [{"name": n, "tokens": ids, "rows": rows_for(state, ids).tolist()} for n, ids in cases.items()],
        "row_semantics": "rows[t][e][c]: row in layers.{layer_ids[e]}.engram.embed; c = (n-2)*n_heads + head",
    }
    return out, token_map


def write(out_dir, tiny: tuple[dict, np.ndarray], real: tuple[dict, np.ndarray]) -> None:
    assert np.array_equal(tiny[1], real[1]), "tiny and real token maps differ"
    np.save(out_dir / "engram_token_map.npy", tiny[1])
    (out_dir / "engram.json").write_text(json.dumps({"tiny": tiny[0], "real": real[0]}, ensure_ascii=False) + "\n")
