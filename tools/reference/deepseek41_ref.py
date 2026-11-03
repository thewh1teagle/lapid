# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.8",
#   "transformers>=5.6",
#   "tokenizers",
#   "numpy",
#   "safetensors>=0.6",
#   "sympy",
#   "pillow",
#   "huggingface_hub",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Reference data for lapid's DeepSeek V4.1 verification: the OFFICIAL PyTorch inference code
(`deepseek-ai/DeepSeek-V4.1-Flash-1017` `inference/model.py` + `engram.py`, pinned revision),
fp32, CPU, greedy. HF transformers has no `deepseek_v41` model (5.22 checked).

    uv run tools/reference/deepseek41_ref.py --tiny --out data/reference/deepseek41/tiny

The real checkpoint (~475 GB) does not fit this machine, so the proof is `--tiny`: random weights
with the real V4.1 structure, every feature exercised, narrow dims (see `tiny_config`):
11 layers, ratios [0,0,2,2,2,2,1,1,1,1,0] (+1 MTP stage, ratio 0), kv sources [2,4,6] (ratio 2 /
ratio 2 / ratio 1), index sources [2,4,5,6,8] (5 = index source reading layer 4's KV), candidate
source 6 (block 4, 3 blocks kept) feeding index source 8, index_topk 6 (32 heads x 128), window
16, dim 64, 8 heads, 8 routed experts (3 active) + 1 shared, expert dim 64, Engram on
layers 1 and 4 (real hashing: 4-grams, 2 heads, real tokenizer token map / multipliers / primes at
engram_vocab_size 1000), head dim 512 + rope 64, q/o low rank, grouped wo_a, sinks, YaRN on
compressed layers, hc 4 + Sinkhorn 20 with the lagged pre-mix, sqrtsoftplus MoE + bias, shared
expert, SwiGLU clamp, no hash routing, real tokenizer + official `encoding/encoding.py`.

The tiny checkpoint is saved in the OFFICIAL on-disk format (`deepseek41_ckpt.py`: FP8 e4m3 +
e8m0 32x32 linears, I8-packed e2m1 + e8m0/32 experts, FP8 Engram tables + e8m0 [rows, D/32],
BF16/F32 rest, real tensor names incl. bias_vl / vision / aligner / image_* / mtp.0, real
config.json layout with text_config / vision_config / quantization_config, sharded with
model.safetensors.index.json, tokenizer files) in `<out>/model/`; the reference reads it back and
runs on the exactly dequantized weights.

Writes:
  <out>/activations.safetensors (f32; S = prompt rows, D = hidden 64, V = 129280)
      short.embed [S,D]; short.layers.{i} [S,4D] (hc streams after layer i, flattened hc-major);
      short.layers.{i}.op [S,D] (attention output, after wo_b); short.norm [S,D] (final norm,
      after the lagged hc pre-mix); short.logits [S,V]; short.gen.logits [K,V] (prefill + cached
      greedy decode steps); long.layers.{i}.tail [T,4D]; long.logits.last [1,V]; long.gen.logits [K,V]
  <out>/debug.safetensors (not read by `lapid verify`; for debugging the port), per case c:
      c.layers.{i}.engram [S,4D] (streams after the Engram add, Engram layers only);
      c.layers.{i}.engram_rows int64 [S,(N-1)*H] (hashed rows into that layer's table);
      c.layers.{i}.compressed [S/ratio,512] (kv-source compressor latent, before RoPE);
      c.layers.{i}.index_k [S/ratio,128] (kv-source index keys, after RoPE);
      c.layers.{i}.topk int32 [S,k] (index-source selection, compressed row ids, -1 = none);
      c.layers.{i}.candidates uint8 [S,S/ratio] (candidate-source block mask)
  <out>/result.json: cases.{short,long} (messages_json, thinking / enable_thinking, prompt_text,
      prompt_tokens, generated_tokens, generated_text), template_cases (official encoding),
      tokenizer_cases, checks (self-consistency + feature coverage), engram summary
  <out>/engram.json: Engram constants for the tiny AND the real config (multipliers, primes,
      per-layer offsets, llama.cpp-style concatenated offsets, num_embeddings, pad ids, token map
      summary) + hash test vectors (token ids -> rows [L][n_engram_layers][(N-1)*H]); the full
      token map is <out>/engram_token_map.npy (int32 [129280], identical for tiny and real).

Deviations from the official code (details in deepseek41_official.py):
  * kernel.py (tilelang, GPU-only) is replaced by pure torch: sparse_attn and hc_split_sinkhorn
    with the same math; the kernels' on-the-fly activation fake-quant (fp8 window KV and wo_a
    input, fp4 indexer q/k and compressed KV) is the IDENTITY: lapid, like llama.cpp and HF, keeps
    activations in f32. The short prompt is also run with an exact emulation of those kernels and
    the logits difference is reported (checks.act_quant_emulation), for information only.
  * Weights are dequantized exactly to fp32 (fp32 F.linear instead of the fp8/fp4 GEMMs; wo_a,
    which convert.py dequantizes to bf16, in fp32 too); the Engram lookup multiplies the e4m3 rows
    by their e8m0 scales as upstream but returns fp32 (upstream casts to bf16, which is exact).
    Torch default dtype fp32 throughout.
  * The forward loop is the official Transformer.forward body (text only) with taps, and the head
    computes all rows (upstream returns the last row only).
  * Greedy decoding = generate.py at temperature 0 (argmax), batch 1, eos id 1.
  * The tiny model has index_n_heads 32 (as the real one): with few heads, relu(q.k) is often 0
    for every head, the index scores tie exactly at 0 and top-k picks are implementation-defined.
    checks.topk_min_relative_margin reports how close the indexer / router top-k decisions are.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deepseek41_ckpt as ckpt_fmt  # noqa: E402
from deepseek41_official import EMULATE, REPO, REVISION, load_official, official_dir  # noqa: E402


LONG_TEXT = """The history of the printing press is a story of many small improvements rather than a single \
invention. Long before movable type reached Europe, craftsmen in East Asia carved whole pages into wooden \
blocks, inked them, and pressed paper against the surface to make copies of religious texts, calendars and \
official notices. Later, artisans experimented with individual characters made of baked clay, wood and \
eventually metal, which could be arranged, printed and then reused for another page. In the fifteenth \
century a goldsmith in Mainz combined several existing ideas: a screw press adapted from wine making, an \
oil-based ink that clung to metal, and a hand mould that could cast large numbers of identical letters \
quickly. The result was not just faster copying but a new economy of knowledge. Printers needed paper \
mills, type founders, editors, binders and merchants who could carry finished books along trade routes. \
Within fifty years presses operated in hundreds of towns, and millions of volumes circulated across the \
continent. Scholars could compare identical copies of a text, cite page numbers, and build on each other's \
work with a precision that had been impossible when every manuscript differed."""

SHORT_MESSAGES = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
LONG_MESSAGES = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": LONG_TEXT + "\n\nSummarize the passage above in three bullet points."},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
        },
    }
]

TEMPLATE_CASES = [
    {"messages": [{"role": "user", "content": "Hello!"}], "thinking": False},
    {"messages": [{"role": "user", "content": "Hello!"}], "thinking": True},
    {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], "thinking": False},
    {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], "thinking": True},
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 = 4.", "reasoning_content": "Simple sum."},
            {"role": "user", "content": "And 3+3?"},
        ],
        "thinking": True,
    },
    {
        "messages": [
            {"role": "user", "content": "Name a color."},
            {"role": "assistant", "content": "Blue.", "reasoning_content": "Pick any color."},
        ],
        "thinking": True,
    },
    {
        "messages": [
            {"role": "user", "content": "Name a color."},
            {"role": "assistant", "content": "Blue."},
            {"role": "user", "content": "Another?"},
        ],
        "thinking": False,
    },
    {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "system", "content": "From now on answer in French."},
        ],
        "thinking": False,
    },
    {
        "messages": [
            {"role": "system", "content": "Use tools when helpful.", "tools": TOOLS},
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Need the tool.",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 2}"}}
                ],
            },
            {"role": "tool", "content": "{\"temp\": 18, \"sky\": \"clear\"}", "tool_call_id": "c1"},
            {"role": "tool", "content": "second", "tool_call_id": "c1"},
        ],
        "thinking": True,
    },
    {
        "messages": [
            {"role": "system", "content": "Use tools.", "tools": TOOLS},
            {"role": "user", "content": "Weather in Rome?"},
        ],
        "thinking": False,
    },
    {"messages": [{"role": "user", "content": "Ünïcödé — 日本語 🙂 tabs\tand\nnewlines"}], "thinking": False},
]

TOKENIZER_CASES = [
    "Hello, world!",
    " leading space and  double  spaces",
    "Numbers: 1234567 and 3.14159, year 2026.",
    "def f(x):\n    return x * 2  # comment\n\n\n",
    "Ünïcödé — 日本語 🙂 emoji and accents: café naïve",
    "It's we're they'll I'd I'M",
    "<｜User｜>special tokens inline<｜Assistant｜></think>",
    "中文分词测试，日本語のテキスト。",
]

# Engram hash vectors: texts whose n-grams collide after normalization ("The" / " the" / "THE"),
# plus raw ids: bos/eos/pad, special tokens, the last id, partial-UTF-8 byte tokens.
ENGRAM_TEXTS = ["The cat sat on the mat. THE CAT SAT ON THE MAT. the  Cat\tsat", "Café cafe CAFÉ naïve naive 日本語"]
ENGRAM_RAW_IDS = [0, 2, 2, 5, 128803, 1, 129279, 300, 300, 300, 2, 7, 129263, 45, 45]


def f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())


# ------------------------------------------------------------------ configs


def tiny_config(real: dict) -> dict:
    """config.json of the tiny model: the real file's layout and keys, narrow values."""
    cfg = copy.deepcopy(real)
    t = cfg["text_config"]
    t.update(
        hidden_size=64,
        moe_intermediate_size=64,
        num_hidden_layers=11,
        num_attention_heads=8,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        swiglu_limit=1.5,
        n_routed_experts=8,
        num_experts_per_tok=3,
        sliding_window=16,
        compress_ratios=[0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0],
        kv_source_layer_ids=[2, 4, 6],
        index_source_layer_ids=[2, 4, 5, 6, 8],
        index_topk=6,
        candidate_source_layer_id=6,
        candidate_topk_blocks=3,
        candidate_block_size=4,
        engram_layer_ids=[1, 4],
        engram_vocab_size=1000,
        engram_n_heads=2,
        engram_head_dim=64,
        num_nextn_predict_layers=1,
        dspark_target_layer_ids=[8, 9],
        dspark_markov_rank=32,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
    )
    cfg["vision_config"].update(num_hidden_layers=1, hidden_size=32, num_attention_heads=2, intermediate_size=64)
    return cfg


def model_args(M, E, cfg: dict, max_seq_len: int = 1024, with_extras: bool = True):
    """HF config.json -> official ModelArgs (the inference/config.json key names)."""
    t, v = cfg["text_config"], cfg.get("vision_config") or {}
    rs = t["rope_scaling"]
    a = dict(
        max_batch_size=1, max_seq_len=max_seq_len, temperature=0.0, dtype="bf16", expert_dtype=None,
        vocab_size=t["vocab_size"], dim=t["hidden_size"], moe_inter_dim=t["moe_intermediate_size"],
        n_layers=t["num_hidden_layers"], n_mtp_layers=t["num_nextn_predict_layers"], n_heads=t["num_attention_heads"],
        n_routed_experts=t["n_routed_experts"], n_shared_experts=t["n_shared_experts"],
        n_activated_experts=t["num_experts_per_tok"], score_func=t["scoring_func"], norm_topk_prob=t["norm_topk_prob"],
        route_scale=t["routed_scaling_factor"], swiglu_limit=t["swiglu_limit"], q_lora_rank=t["q_lora_rank"],
        head_dim=t["head_dim"], rope_head_dim=t["qk_rope_head_dim"], norm_eps=t["rms_norm_eps"], o_groups=t["o_groups"],
        o_lora_rank=t["o_lora_rank"], window_size=t["sliding_window"], compress_ratios=tuple(t["compress_ratios"]),
        kv_source_layers=tuple(t["kv_source_layer_ids"]), index_source_layers=tuple(t["index_source_layer_ids"]),
        compress_rope_theta=t["compress_rope_theta"], original_seq_len=rs["original_max_position_embeddings"],
        rope_theta=t["rope_theta"], rope_factor=rs["factor"], beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"],
        index_n_heads=t["index_n_heads"], index_head_dim=t["index_head_dim"], index_topk=t["index_topk"],
        candidate_source_layer=t["candidate_source_layer_id"], candidate_topk_blocks=t["candidate_topk_blocks"],
        candidate_block_size=t["candidate_block_size"], hc_mult=t["hc_mult"], hc_sinkhorn_iters=t["hc_sinkhorn_iters"],
        hc_eps=t["hc_eps"], engram_layer_ids=tuple(t["engram_layer_ids"]),
        engram_num_embeddings=tuple(t["engram_num_embeddings"]), engram_max_ngram_size=t["engram_max_ngram_size"],
        engram_vocab_size=t["engram_vocab_size"], engram_n_heads=t["engram_n_heads"],
        engram_head_dim=t["engram_head_dim"], engram_pad_id=t["engram_pad_token_id"],
        engram_compressed_vocab_size=t["engram_compressed_vocab_size"], image_token_id=cfg["image_token_id"],
    )
    if with_extras:  # vision tower + DSpark stage, only so their tensors exist in the checkpoint
        a.update(
            vision_n_layers=v["num_hidden_layers"], vision_dim=v["hidden_size"], vision_n_heads=v["num_attention_heads"],
            vision_inter_dim=v["intermediate_size"], vision_patch_size=v["patch_size"], vision_rope_theta=v["rope_theta"],
            vision_downsample_ratio=v["downsample_ratio"], vision_max_n_token=v["max_image_tokens"],
            vision_min_pixels=v["min_pixels"], vision_max_wh_ratio=v["max_wh_ratio"],
            dspark_block_size=t["dspark_block_size"], dspark_noise_token_id=t["dspark_noise_token_id"],
            dspark_target_layer_ids=tuple(t["dspark_target_layer_ids"]), dspark_markov_rank=t["dspark_markov_rank"],
            dspark_n_routed_experts=t["dspark_n_routed_experts"],
            dspark_n_activated_experts=t["dspark_num_experts_per_tok"],
        )
    return M.ModelArgs(**a)


def check_mapping(M, E, real: dict, official_args: dict):
    """model_args(real HF config) must agree with the official inference/config.json."""
    a = model_args(M, E, real).__dict__
    skip = {"max_batch_size", "max_seq_len", "temperature", "dtype", "expert_dtype", "vision_max_wh_ratio"}
    bad = {k: (a[k], v) for k, v in official_args.items() if k in a and k not in skip and (tuple(v) if isinstance(v, list) else v) != a[k]}
    assert not bad, f"HF config -> ModelArgs mapping disagrees with inference/config.json: {bad}"


# ------------------------------------------------------------------ the tiny model


def build_tiny(M, E, src: Path, ckpt: Path, tokenizer_src, seed: int):
    """Random weights -> official-format checkpoint -> read back -> official Transformer (fp32)."""
    real = json.loads((src / "config.json").read_text())
    cfg = tiny_config(real)
    t = cfg["text_config"]
    layout = E.EngramLayout.from_args(types.SimpleNamespace(
        engram_layer_ids=t["engram_layer_ids"], engram_max_ngram_size=t["engram_max_ngram_size"],
        engram_n_heads=t["engram_n_heads"], engram_vocab_size=t["engram_vocab_size"], engram_head_dim=t["engram_head_dim"],
        engram_num_embeddings=()))
    t["engram_num_embeddings"] = [sum(p for per in layer for p in per) for layer in layout.primes]
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(tokenizer_src / f, ckpt / f)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    args = model_args(M, E, cfg)
    torch.manual_seed(0)
    model = M.Transformer(args, tok).float().eval()
    tied = {n for n in model.state_dict() if n.startswith("mtp.") and n.split(".", 2)[-1] in ("embed.weight", "head.weight")}
    g = torch.Generator().manual_seed(seed)
    fp32 = {n: ckpt_fmt.init_value(n, tuple(p.shape), g) for n, p in model.state_dict().items()
            if n not in tied and not n.endswith("engram.embed.scale")}
    kinds = ckpt_fmt.write_checkpoint(fp32, ckpt, args.n_layers)
    weights = ckpt_fmt.read_checkpoint(ckpt)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    assert set(missing) == tied and not unexpected, (missing, unexpected)
    for n in tied:  # DSpark stages alias the backbone embed / head
        assert model.state_dict()[n].data_ptr() == model.state_dict()[n.split(".", 2)[-1]].data_ptr()
    return model, tok, cfg, args, kinds


# ------------------------------------------------------------------ forward with taps


class Taps:
    """Forward hooks: attention outputs, compressor latents, index keys, top-k, candidate masks."""

    def __init__(self, M, model):
        self.M, self.model, self.out, self.on = M, model, {}, False
        self.clamp = [0, 0]
        for i, layer in enumerate(model.layers):
            layer.attn.register_forward_hook(self._save(f"layers.{i}.op"))
            if layer.attn.compressor is not None:
                layer.attn.compressor.register_forward_hook(self._save(f"layers.{i}.compressed", clone=True))
            if layer.attn.indexer is not None:
                layer.attn.indexer.register_forward_hook(self._topk(i))
            for ex in [*layer.ffn.experts, layer.ffn.shared_experts]:
                ex.w1.register_forward_hook(self._clamp(gate=True))
                ex.w3.register_forward_hook(self._clamp(gate=False))
        self.limit = model.layers[0].ffn.shared_experts.swiglu_limit
        self.margin: dict[str, float] = {}
        orig = torch.Tensor.topk

        def topk(t, k, *a, **kw):  # relative gap between the k-th and (k+1)-th score
            caller = type(sys._getframe(1).f_locals.get("self")).__name__
            if self.on and caller in ("Indexer", "Gate") and t.size(-1) > k:
                v = orig(t.float(), k + 1, dim=-1).values
                hi, lo = v[..., k - 1], v[..., k]
                ok = torch.isfinite(hi) & torch.isfinite(lo)
                if ok.any():
                    m = float(((hi - lo) / (hi.abs() + lo.abs() + 1e-30))[ok].min())
                    self.margin[caller] = min(self.margin.get(caller, 1.0), m)
            return orig(t, k, *a, **kw)

        torch.Tensor.topk = topk

    def _save(self, key, clone=False):
        def f(_m, _i, out):
            if self.on and out is not None:
                self.out[key] = out[0].clone() if clone else out[0]
        return f

    def _topk(self, i):
        def f(mod, inp, out):
            if not self.on:
                return
            offset = inp[4]
            self.out[f"layers.{i}.topk"] = torch.where(out[0] >= 0, out[0] - offset, -1).int()
            sa = self.M.shared_attn
            if mod.owns_k:
                n = inp[0].size(1) // mod.compress_ratio
                self.out[f"layers.{i}.index_k"] = sa.index_k[0, :n].clone()
            if mod.is_candidate_source:
                self.out[f"layers.{i}.candidates"] = sa.candidates[0].to(torch.uint8)
        return f

    def _clamp(self, gate):
        def f(_m, _i, out):
            if self.on:
                v = out.float()
                self.clamp[0] += int((v > self.limit).sum() if gate else (v.abs() > self.limit).sum())
                self.clamp[1] += v.numel()
        return f


def forward(M, model, ids: torch.Tensor, start_pos: int, taps: Taps | None = None):
    """Official Transformer.forward (text only) with taps and all-row logits. ids [1, S]."""
    if taps is not None:
        taps.out, taps.on = {}, True
    hashes = model.engram_hash(ids, start_pos, None)
    h = model.embed(ids)
    rec = taps.out if taps is not None else {}
    rec["embed"] = h[0]
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, model.hc_mult)
    for i, layer in enumerate(model.layers):
        if layer.engram is not None:
            rows = hashes[:, :, layer.engram.layer_hash_index, :]
            h = layer.engram(h, rows, None)
            rec[f"layers.{i}.engram"] = h[0].flatten(1)
            rec[f"layers.{i}.engram_rows"] = rows[0]
        h, pre_mix = layer(h, start_pos, pre_mix, None)
        rec[f"layers.{i}"] = h[0].flatten(1)
    h = layer.hc_pre(h, pre_mix)
    n = model.norm(h)
    rec["norm"] = n[0]
    logits = F.linear(n.float(), model.head.weight)[0]
    if taps is not None:
        taps.on = False
    return logits, rec


def greedy(M, model, prompt: list[int], max_new: int, eos: int):
    """generate.py at temperature 0: prefill, then one token per step. Returns tokens, step logits."""
    logits, _ = forward(M, model, torch.tensor([prompt]), 0)
    steps, gen = [logits[-1]], []
    for _ in range(max_new):
        nxt = int(steps[-1].argmax())
        gen.append(nxt)
        if nxt == eos or len(gen) == max_new:
            break
        logits, _ = forward(M, model, torch.tensor([[nxt]]), len(prompt) + len(gen) - 1)
        steps.append(logits[-1])
    return gen, torch.stack(steps)


# ------------------------------------------------------------------ checks


def coverage(rec: dict, args, prompt_len: int) -> dict:
    """Do the sparse paths actually bite on this prompt?"""
    out = {}
    for i in args.index_source_layers:
        tk = rec.get(f"layers.{i}.topk")
        if tk is None:
            continue
        ratio = args.compress_ratios[i]
        visible = torch.arange(1, prompt_len + 1) // ratio
        chosen = (tk >= 0).sum(-1)
        out[f"layers.{i}.topk"] = {"rows_with_more_visible_than_topk": int((visible > args.index_topk).sum()),
                                   "max_selected": int(chosen.max()), "rows": prompt_len}
    c = args.candidate_source_layer
    if f"layers.{c}.candidates" in rec:
        cand = rec[f"layers.{c}.candidates"].bool()
        ratio = args.compress_ratios[c]
        vis = torch.arange(cand.size(1))[None, :] < (torch.arange(1, prompt_len + 1) // ratio)[:, None]
        out["candidates"] = {"visible": int(vis.sum()), "visible_kept": int((cand & vis).sum())}
        for i in args.index_source_layers:
            if i > c and f"layers.{i}.topk" in rec:
                tk = rec[f"layers.{i}.topk"].long()
                inside = torch.gather(cand, 1, tk.clamp_min(0))
                assert bool(inside[tk >= 0].all()), f"layer {i} picked rows outside the candidate blocks"
                out["candidates"][f"layers.{i}.topk_inside_candidates"] = True
    for i in args.engram_layer_ids:
        before = rec[f"layers.{i - 1}"] if i > 0 else rec["embed"].repeat(1, args.hc_mult)
        out[f"layers.{i}.engram_mean_abs_delta"] = float((rec[f"layers.{i}.engram"] - before).abs().mean())
    return out


def decode_vs_full(M, model, prompt: list[int], gen: list[int], gen_logits: torch.Tensor) -> dict:
    """Cached decode logits vs one uncached forward over prompt + generated tokens."""
    k = gen_logits.shape[0]
    full, _ = forward(M, model, torch.tensor([prompt + gen[: k - 1]]), 0)
    rows = full[len(prompt) - 1 : len(prompt) - 1 + k]
    d = (rows - gen_logits).abs()
    return {"steps": k, "max_abs_diff": float(d.max()), "logit_scale": float(gen_logits.abs().max()),
            "argmax_equal": bool((rows.argmax(-1) == gen_logits.argmax(-1)).all())}


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/reference/deepseek41/tiny", type=Path)
    ap.add_argument("--tiny", action="store_true", help="random tiny model (the only mode: the real checkpoint is 475 GB)")
    ap.add_argument("--official", default=None, help="local checkout of the HF repo's code/config/tokenizer (default: download the pinned revision)")
    ap.add_argument("--max-new-tokens", default=40, type=int)
    ap.add_argument("--gen-logit-steps", default=12, type=int)
    ap.add_argument("--long-tail", default=8, type=int)
    ap.add_argument("--seed", default=3, type=int, help="weight seed (the default keeps every top-k decision clear of ties)")
    args_cli = ap.parse_args()
    if not args_cli.tiny:
        print("only --tiny is supported (the DeepSeek V4.1 checkpoint does not fit this machine)", file=sys.stderr)
        return 2
    import deepseek41_engram as eng

    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float32)
    out = args_cli.out
    out.mkdir(parents=True, exist_ok=True)
    src = official_dir(args_cli.official)
    M, E, enc = load_official(src)
    eng.cached_token_map(E)
    real_cfg = json.loads((src / "config.json").read_text())
    check_mapping(M, E, real_cfg, json.loads((src / "inference" / "config.json").read_text()))
    model, tok, cfg, args, kinds = build_tiny(M, E, src, out / "model", src, args_cli.seed)
    taps = Taps(M, model)
    eos = cfg["eos_token_id"]

    tensors: dict[str, np.ndarray] = {}
    debug: dict[str, np.ndarray] = {}
    cases, checks = {}, {"coverage": {}, "decode_vs_full": {}}
    for cname, messages, thinking in (("short", SHORT_MESSAGES, False), ("long", LONG_MESSAGES, False)):
        prompt = enc.encode_messages(messages, thinking_mode="thinking" if thinking else "chat")
        ids = tok(prompt, add_special_tokens=False).input_ids
        logits, rec = forward(M, model, torch.tensor([ids]), 0, taps)
        if cname == "short":
            for k in ["embed", *[f"layers.{i}{s}" for i in range(args.n_layers) for s in ("", ".op")], "norm"]:
                tensors[f"short.{k}"] = f32(rec[k])
            tensors["short.logits"] = f32(logits)
        else:
            for i in range(args.n_layers):
                tensors[f"long.layers.{i}.tail"] = f32(rec[f"layers.{i}"][-args_cli.long_tail :])
            tensors["long.logits.last"] = f32(logits[-1:])
        for k, v in rec.items():
            if any(k.endswith(s) for s in (".engram", ".engram_rows", ".compressed", ".index_k", ".topk", ".candidates")):
                debug[f"{cname}.{k}"] = np.ascontiguousarray(v.cpu().numpy() if v.dtype != torch.float32 else f32(v))
        checks["coverage"][cname] = coverage(rec, args, len(ids))
        checks.setdefault("topk_min_relative_margin", {})[cname] = dict(taps.margin)
        taps.margin = {}
        gen, gen_logits = greedy(M, model, ids, args_cli.max_new_tokens, eos)
        tensors[f"{cname}.gen.logits"] = f32(gen_logits[: args_cli.gen_logit_steps])
        checks["decode_vs_full"][cname] = decode_vs_full(M, model, ids, gen, gen_logits[: args_cli.gen_logit_steps])
        cases[cname] = {
            "messages_json": json.dumps(messages, ensure_ascii=False),
            "thinking": thinking,
            "enable_thinking": thinking,
            "prompt_text": prompt,
            "prompt_tokens": ids,
            "generated_tokens": gen,
            "generated_text": tok.decode(gen, skip_special_tokens=False),
        }
        print(f"{cname}: {len(ids)} prompt tokens -> {tok.decode(gen)!r}", file=sys.stderr)
    checks["swiglu_clamped_fraction"] = taps.clamp[0] / max(taps.clamp[1], 1)

    # the official kernels' activation fake-quant, emulated (informational)
    EMULATE["act_quant"] = True
    q_logits, _ = forward(M, model, torch.tensor([cases["short"]["prompt_tokens"]]), 0)
    EMULATE["act_quant"] = False
    ref = torch.from_numpy(tensors["short.logits"])
    checks["act_quant_emulation"] = {
        "short_logits_max_abs_diff": float((q_logits - ref).abs().max()),
        "short_logits_rel_l2": float((q_logits - ref).norm() / ref.norm()),
        "argmax_agree_fraction": float((q_logits.argmax(-1) == ref.argmax(-1)).float().mean()),
    }

    template_cases = [
        {
            "messages_json": json.dumps(c["messages"], ensure_ascii=False),
            "thinking": c["thinking"],
            "enable_thinking": c["thinking"],
            "text": enc.encode_messages(c["messages"], thinking_mode="thinking" if c["thinking"] else "chat"),
        }
        for c in TEMPLATE_CASES
    ]
    tokenizer_cases = [{"text": t, "ids": tok(t, add_special_tokens=False).input_ids} for t in TOKENIZER_CASES]

    # Engram constants + hash vectors, tiny and real
    hash_cases = {"short": cases["short"]["prompt_tokens"], "long": cases["long"]["prompt_tokens"], "raw_ids": ENGRAM_RAW_IDS}
    hash_cases.update({f"text{j}": tok(t, add_special_tokens=False).input_ids for j, t in enumerate(ENGRAM_TEXTS)})
    tiny_state, tiny_layout = eng.hash_state(E, args, tok)
    tiny = eng.constants(tiny_state, tiny_layout, args, hash_cases)
    for i, e in zip(args.engram_layer_ids, range(len(args.engram_layer_ids))):
        assert np.array_equal(np.array(tiny[0]["hash_cases"][0]["rows"])[:, e], debug[f"short.layers.{i}.engram_rows"])
    real_args = model_args(M, E, real_cfg, max_seq_len=4096, with_extras=False)
    real_state, real_layout = eng.hash_state(E, real_args, tok)
    real = eng.constants(real_state, real_layout, real_args, hash_cases)
    eng.write(out, tiny, real)

    save_file(tensors, str(out / "activations.safetensors"))
    save_file(debug, str(out / "debug.safetensors"))
    import transformers

    result = {
        "model": "deepseek41-tiny (random)",
        "seed": args_cli.seed,
        "dtype": "fp32",
        "device": "cpu",
        "reference": f"official {REPO}@{REVISION} inference/model.py (pure-torch kernels, fp32)",
        "cases": cases,
        "template_cases": template_cases,
        "tokenizer_cases": tokenizer_cases,
        "special": {"eos": [eos]},
        "n_layers": args.n_layers,
        "compress_ratios": list(args.compress_ratios[: args.n_layers]),
        "kv_source_layers": list(args.kv_source_layers),
        "index_source_layers": list(args.index_source_layers),
        "candidate_source_layer": args.candidate_source_layer,
        "engram_layers": list(args.engram_layer_ids),
        "checkpoint_kinds": {k: sum(1 for v in kinds.values() if v == k) for k in sorted(set(kinds.values()))},
        "checks": checks,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0]},
    }
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(checks, indent=1), file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
