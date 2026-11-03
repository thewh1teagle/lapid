"""The official DeepSeek-V4.1 on-disk checkpoint format, for `deepseek41_ref.py` (helper module,
not run on its own).

Format (read off the headers of the real `deepseek-ai/DeepSeek-V4.1-Flash-1017` shards):

* FP8 linears (`attn.wq_a/wq_b/wkv/wo_a/wo_b`, `attn.indexer.wq_b`, `ffn.shared_experts.w1/w2/w3`,
  `engram.wkv`, `mtp.*.main_proj`): `<name>.weight` F8_E4M3 `[N, K]` + `<name>.scale` F8_E8M0
  `[N/32, K/32]` (one power-of-two scale per 32x32 block, `w = q * 2^(e-127)`).
* routed experts `ffn.experts.{e}.w1/w2/w3`: `.weight` I8 `[N, K/2]`, two e2m1 codes per byte
  (low nibble = element 2j, high nibble = element 2j+1, `{0,.5,1,1.5,2,3,4,6}` + sign bit 3) +
  `.scale` F8_E8M0 `[N, K/32]` (one scale per 32 consecutive elements of a row).
* Engram tables `layers.{i}.engram.embed.weight` F8_E4M3 `[rows, D]` + `.scale` F8_E8M0
  `[rows, D/32]` (per row, per 32 values).
* F32: `attn.attn_sink`, `ffn.gate.bias`, `ffn.gate.bias_vl`, `hc_{attn,ffn}_{fn,base,scale}`.
* BF16: everything else (embed, head, norms, compressor, indexer wk / k_norm / weights_proj,
  router `ffn.gate.weight`, `engram.q_weight` / `k_weight`, vision, aligner, image_*, mtp heads).

Shards `model-0000k-of-0000N.safetensors` + `model.safetensors.index.json`, laid out like the
real repo: vision/aligner, embed/image_*, one shard per layer, head/norm, mtp, one per Engram table.
Quantization of the random fp32 weights uses power-of-two scales `2^ceil(log2(amax / qmax))`
(e4m3 qmax 448, e2m1 qmax 6), round to nearest even.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

E4M3 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_BOUNDS = (FP4_GRID[1:] + FP4_GRID[:-1]) / 2
FP4_TABLE = torch.cat([FP4_GRID, -FP4_GRID])  # code -> value (code 8 = -0)

FP8_LINEARS = re.compile(
    r"(attn\.(wq_a|wq_b|wkv|wo_a|wo_b)|attn\.indexer\.wq_b|ffn\.shared_experts\.w[123]|engram\.wkv|main_proj)\.weight$"
)
FP4_EXPERTS = re.compile(r"ffn\.experts\.\d+\.w[123]\.weight$")
ENGRAM_TABLE = re.compile(r"engram\.embed\.weight$")
F32_NAMES = re.compile(r"(attn_sink|ffn\.gate\.bias(_vl)?|hc_(attn|ffn)_(fn|base|scale))$")


def disk_kind(name: str) -> str:
    if FP8_LINEARS.search(name):
        return "fp8"
    if FP4_EXPERTS.search(name):
        return "fp4"
    if ENGRAM_TABLE.search(name):
        return "engram"
    if F32_NAMES.search(name):
        return "f32"
    return "bf16"


# ---------------------------------------------------------------- quantization (pure torch)


def pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(x)) for x > 0, exact (kernel.py `fast_round_scale`: exponent + (mantissa != 0))."""
    m, e = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), torch.where(m == 0.5, e - 1, e))


def fp4_codes(x: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 code (ties to even) of x, |x| <= 6 expected (saturates)."""
    mag = x.abs().clamp(max=6.0)
    idx = torch.searchsorted(FP4_BOUNDS, mag.contiguous())
    tie = (idx < 7) & (FP4_BOUNDS[idx.clamp(max=6)] == mag)
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    return torch.where(x < 0, idx + 8, idx).to(torch.uint8)


def fp4_round(x: torch.Tensor) -> torch.Tensor:
    return FP4_TABLE[fp4_codes(x).long()]


def quant_fp8_blocks(w: torch.Tensor, block: int = 32):
    n, k = w.shape
    assert n % block == 0 and k % block == 0, (w.shape, block)
    b = w.float().reshape(n // block, block, k // block, block)
    scale = pow2_ceil(b.abs().amax(dim=(1, 3)).clamp(min=2.0**-100) / 448.0)
    q = (b / scale[:, None, :, None]).clamp(-448, 448).to(E4M3).reshape(n, k)
    return q, scale.to(E8M0)


def dequant_fp8_blocks(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    bn, bk = n // scale.shape[0], k // scale.shape[1]
    s = scale.float().repeat_interleave(bn, 0).repeat_interleave(bk, 1)
    return q.float() * s


def quant_fp8_rows(w: torch.Tensor, group: int = 32):
    r, d = w.shape
    g = w.float().reshape(r, d // group, group)
    scale = pow2_ceil(g.abs().amax(-1).clamp(min=2.0**-100) / 448.0)
    q = (g / scale[..., None]).clamp(-448, 448).to(E4M3).reshape(r, d)
    return q, scale.to(E8M0)


def quant_fp4(w: torch.Tensor, group: int = 32):
    n, k = w.shape
    g = w.float().reshape(n, k // group, group)
    scale = pow2_ceil(g.abs().amax(-1).clamp(min=2.0**-100) / 6.0)
    codes = fp4_codes(g / scale[..., None]).reshape(n, k // 2, 2)
    packed = (codes[..., 0] | (codes[..., 1] << 4)).view(torch.int8)
    return packed, scale.to(E8M0)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor, group: int = 32) -> torch.Tensor:
    b = packed.view(torch.uint8)
    codes = torch.stack([b & 0x0F, (b >> 4) & 0x0F], dim=-1).flatten(1)
    return FP4_TABLE[codes.long()] * scale.float().repeat_interleave(group, 1)


# ---------------------------------------------------------------- random weights


def init_value(name: str, shape, g: torch.Generator) -> torch.Tensor:
    def normal(mean, std):
        return torch.randn(shape, generator=g) * std + mean

    if name.endswith("engram.embed.weight"):
        return normal(0.0, 1.0)
    if re.search(r"(norm\.weight|_norm\.weight)$", name) or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
        return normal(1.0, 0.1)
    if name.endswith("attn_sink"):
        return normal(0.0, 1.0)
    if re.search(r"hc_(attn|ffn)_base$", name):
        return normal(0.0, 0.5)
    if re.search(r"hc_(attn|ffn)_scale$", name):
        return normal(1.0, 0.2)
    if re.search(r"ffn\.gate\.bias(_vl)?$", name):
        return normal(0.0, 0.1)
    if name.endswith(("engram.q_weight", "engram.k_weight")):
        return normal(1.0, 0.3)
    if name in ("embed.weight",) or name.startswith("image_") or name.endswith("markov_head.embed.weight"):
        return normal(0.0, 1.0)
    if name.endswith(".bias"):
        return normal(0.0, 0.02)
    if name == "head.weight":
        return normal(0.0, 0.5)
    if len(shape) == 2:  # every other projection: unit-variance outputs
        return normal(0.0, shape[1] ** -0.5)
    return normal(0.0, 0.1)


# ---------------------------------------------------------------- shards


def shard_of(name: str, n_layers: int) -> str:
    """Shard key in the real repo's order (sorted below)."""
    if name.startswith(("vision.", "aligner.")):
        return "a"
    if name.startswith(("embed.", "image_")):
        return "b"
    m = re.match(r"layers\.(\d+)\.engram\.", name)
    if m:
        return f"z{int(m.group(1)):04d}"
    m = re.match(r"layers\.(\d+)\.", name)
    if m:
        return f"c{int(m.group(1)):04d}"
    if name in ("head.weight", "norm.weight"):
        return "d"
    m = re.match(r"mtp\.(\d+)\.", name)
    if m:
        return f"e{int(m.group(1)):04d}"
    raise ValueError(name)


def write_checkpoint(fp32: dict[str, torch.Tensor], ckpt: Path, n_layers: int) -> dict[str, str]:
    """fp32 weights (official module names) -> official-format shards + index. Returns name -> kind."""
    from safetensors.torch import save_file

    disk: dict[str, torch.Tensor] = {}
    kinds = {}
    for name, w in fp32.items():
        kind = kinds[name] = disk_kind(name)
        if kind == "fp8":
            disk[name], disk[name[: -len("weight")] + "scale"] = quant_fp8_blocks(w)
        elif kind == "fp4":
            disk[name], disk[name[: -len("weight")] + "scale"] = quant_fp4(w)
        elif kind == "engram":
            disk[name], disk[name[: -len("weight")] + "scale"] = quant_fp8_rows(w)
        elif kind == "f32":
            disk[name] = w.float().contiguous()
        else:
            disk[name] = w.to(torch.bfloat16).contiguous()
    groups: dict[str, list[str]] = {}
    for n in disk:
        groups.setdefault(shard_of(n, n_layers), []).append(n)
    keys = sorted(groups)
    total = len(keys)
    weight_map, size = {}, 0
    ckpt.mkdir(parents=True, exist_ok=True)
    for old in ckpt.glob("model-*.safetensors"):
        old.unlink()
    for i, k in enumerate(keys):
        fname = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
        part = {n: disk[n].contiguous() for n in sorted(groups[k])}
        save_file(part, str(ckpt / fname))
        for n, t in part.items():
            weight_map[n] = fname
            size += t.numel() * t.element_size()
    index = {"metadata": {"total_size": size}, "weight_map": dict(sorted(weight_map.items()))}
    (ckpt / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    return kinds


def read_checkpoint(ckpt: Path) -> dict[str, torch.Tensor]:
    """Official-format shards -> fp32 tensors under the module names, exactly dequantized.
    Engram tables stay as (e4m3 values, e8m0 scales) in fp32: the official module multiplies
    them on lookup. FP8 linears / FP4 experts come back as w = q * scale."""
    from safetensors import safe_open

    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    raw: dict[str, torch.Tensor] = {}
    for fname in sorted(set(index["weight_map"].values())):
        with safe_open(str(ckpt / fname), framework="pt") as f:
            for n in f.keys():
                raw[n] = f.get_tensor(n)
    out = {}
    for name, t in raw.items():
        if name.endswith(".scale") and name[: -len("scale")] + "weight" in raw:
            continue
        kind = disk_kind(name)
        if kind == "fp8":
            assert t.dtype == E4M3
            out[name] = dequant_fp8_blocks(t, raw[name[: -len("weight")] + "scale"])
        elif kind == "fp4":
            assert t.dtype == torch.int8
            out[name] = dequant_fp4(t, raw[name[: -len("weight")] + "scale"])
        elif kind == "engram":
            assert t.dtype == E4M3
            out[name] = t.float()
            out[name[: -len("weight")] + "scale"] = raw[name[: -len("weight")] + "scale"].float()
        else:
            assert t.dtype == (torch.float32 if kind == "f32" else torch.bfloat16), (name, t.dtype)
            out[name] = t.float()
    return out
