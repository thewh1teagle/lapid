"""The official DeepSeek-V4.1 inference code (`inference/model.py`, `engram.py`,
`encoding/encoding.py` of the pinned HF revision) with `kernel.py` (tilelang, GPU-only) replaced by
pure-torch equivalents, for `deepseek41_ref.py` (helper module, not run on its own).

* `sparse_attn`: gather the kv rows named by topk_idxs (-1 = none), softmax over them with the
  attention sink as one extra logit in the denominator, K = V = the gathered rows (the kernel's
  online softmax computes the same; an all-masked row gives 0 in both).
* `hc_split_sinkhorn`: same formula and op order as the kernel.
* `act_quant` / `fp4_act_quant` (in-place activation fake-quant: fp8 e4m3 per 32 of the window KV
  and of the attention output before wo_a, e2m1 per 32 / e8m0 of the indexer q and k, e2m1 per 16
  with e4m3 scales of the compressed KV): identity unless `EMULATE["act_quant"]`, then an exact
  emulation of the kernels (scale amax/qmax, pow2-rounded for e8m0, clamp, round to nearest even).
* `fp8_gemm` / `fp4_gemm`: unused (weights are dequantized to fp32 so `linear` takes F.linear).
* `ParallelEngramEmbedding.forward` (world size 1): e4m3 rows times their e8m0 scales, returned in
  fp32 instead of the official (exact, lossless) bf16 cast.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

import deepseek41_ckpt as ckpt_fmt

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
OFFICIAL_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "inference/*.py", "inference/config.json", "encoding/encoding.py"]


def official_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, revision=REVISION, allow_patterns=OFFICIAL_FILES))


EMULATE = {"act_quant": False}


def _act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """kernel.act_quant: per block_size group fp8 e4m3, scale amax/448 (pow2-rounded when scale_fmt)."""
    if not inplace:
        raise NotImplementedError("non-inplace act_quant only feeds the fp8 GEMM, unused at fp32")
    if not EMULATE["act_quant"]:
        return x
    g = x.float().unflatten(-1, (-1, block_size))
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = ckpt_fmt.pow2_ceil(amax / 448.0) if scale_fmt is not None else amax / 448.0
    y = (g / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
    x.copy_(y.flatten(-2))
    return x


def _fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """kernel.fp4_act_quant: per group e2m1, scale pow2(amax/6) (e8m0) or e4m3(amax/6)."""
    if not inplace:
        raise NotImplementedError
    if not EMULATE["act_quant"]:
        return x
    g = x.float().unflatten(-1, (-1, block_size))
    amax = g.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        s = ckpt_fmt.pow2_ceil(amax.clamp_min(6 * 2.0**-126) / 6.0)
    y = ckpt_fmt.fp4_round((g / s).clamp(-6, 6)) * s
    x.copy_(y.flatten(-2))
    return x


def _sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.sparse_attn: per query, softmax over the gathered kv rows (-1 = none) with the sink as
    an extra logit in the denominator; K = V = the gathered rows."""
    b = q.size(0)
    valid = topk_idxs >= 0
    g = kv[torch.arange(b)[:, None, None], topk_idxs.clamp_min(0).long()].float()  # [b,s,k,d]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), g) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], -torch.inf)
    sink = attn_sink.float().view(1, 1, -1, 1).expand(*scores.shape[:3], 1)
    p = torch.cat([scores, sink], dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", p, g).to(q.dtype)


def _hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _unused(*_a, **_k):
    raise NotImplementedError("fp8/fp4 GEMMs are not used: weights are dequantized to fp32")


def load_official(src: Path):
    """Import the official inference/model.py with kernel.py replaced by the pure-torch versions."""
    sys.dont_write_bytecode = True  # never write into the official checkout / HF cache
    kernel = types.ModuleType("kernel")
    kernel.act_quant, kernel.fp4_act_quant = _act_quant, _fp4_act_quant
    kernel.sparse_attn, kernel.hc_split_sinkhorn = _sparse_attn, _hc_split_sinkhorn
    kernel.fp8_gemm = kernel.fp4_gemm = _unused
    sys.modules["kernel"] = kernel
    sys.path.insert(0, str(src / "inference"))
    model = importlib.import_module("model")
    engram = importlib.import_module("engram")

    def engram_lookup(self, indices):  # ParallelEngramEmbedding.forward, world_size 1, fp32 out
        values = F.embedding(indices, self.weight).float().unflatten(-1, (-1, self.block_size))
        return (values * F.embedding(indices, self.scale).float().unsqueeze(-1)).flatten(-2)

    model.ParallelEngramEmbedding.forward = engram_lookup
    spec = importlib.util.spec_from_file_location("encoding_dsv41", src / "encoding" / "encoding.py")
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    return model, engram, enc
