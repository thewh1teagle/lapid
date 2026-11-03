//! Qwen3-VL: the generic vision encoder (`vision::VisionEncoder`, merger + DeepStack) and the
//! Qwen3 text decoder of `crate::llama` (HF `Qwen3VLTextModel`).
//!
//! The first DeepStack layers add the encoder's DeepStack features to the image rows after
//! the layer. RoPE is M-RoPE (`lapid_nn::llm::rope::MRope`): each token has (t, h, w)
//! positions, each frequency is driven by one of them; text tokens have t = h = w, an image of
//! `H x W` merged tokens at cursor `c` gets (c, c + row, c + col) and moves the cursor to
//! `c + max(H, W)` (HF `get_rope_index`). State: KV cache plus that cursor.

use lapid::{DType, Error, Result, Tensor};
use lapid_nn::llm::rope::mrope_positions;
use lapid_nn::{KvCache, Loader};
use lapid_verify::harness::Reference;
use serde_json::Value;

use super::config::Qwen3VlConfig;
use crate::llama::model::LlamaModel;
use crate::runtime::{CausalLm, Hook, LmDims, LmState, Media, MediaPrompt, Rows, Taps, VisionLm};
use crate::vision::{Image, ImageProcessor, Patches, VisionEncoder};

pub use crate::llama::model::{EMBED, LM_HEAD};

/// KV cache + M-RoPE cursor (next text position; ahead of the token count after an image).
pub struct MropeCache {
    pub kv: KvCache,
    pos: usize,
    next: usize,
    staged: Option<usize>,
}

impl LmState for MropeCache {
    fn offset(&self) -> usize {
        self.pos
    }

    fn staged(&self) -> Vec<&Tensor> {
        self.kv.staged()
    }

    fn commit(&mut self, realized: Vec<Tensor>, n: usize) -> Result<()> {
        self.kv.commit(realized)?;
        self.pos += n;
        self.next = self.staged.take().unwrap_or(self.next + n);
        Ok(())
    }

    fn discard(&mut self) {
        self.kv.discard();
        self.staged = None;
    }

    fn clear(&mut self) {
        self.kv.clear();
        (self.pos, self.next, self.staged) = (0, 0, None);
    }

    /// Text-only sequences (positions = token indices) can rewind.
    fn truncate(&mut self, len: usize) -> bool {
        if len > self.pos || self.next != self.pos || self.kv.truncate(len).is_err() {
            return false;
        }
        (self.pos, self.next) = (len, len);
        true
    }
}

pub struct Qwen3VlModel {
    pub cfg: Qwen3VlConfig,
    vision: VisionEncoder,
    processor: ImageProcessor,
    text: LlamaModel,
}

impl Qwen3VlModel {
    /// Pixel budget per image (`max_pixels`), e.g. to bound memory.
    pub fn set_max_pixels(&mut self, n: usize) {
        self.processor.max_pixels = n.max(self.processor.min_pixels);
    }
}

impl CausalLm for Qwen3VlModel {
    const ARCH: &'static str = "qwen3vl";
    type State = MropeCache;

    fn load(l: &Loader<'_>, meta: &Value) -> Result<Qwen3VlModel> {
        let cfg = Qwen3VlConfig::from_json(meta, None)?;
        Ok(Qwen3VlModel {
            vision: VisionEncoder::load(l, &cfg.vision)?,
            processor: cfg.processor.clone(),
            text: LlamaModel::from_config(l, cfg.text.base.clone())?,
            cfg,
        })
    }

    fn dims(&self) -> LmDims {
        self.text.dims()
    }

    fn new_state(&self) -> MropeCache {
        MropeCache { kv: KvCache::new(self.cfg.text.base.n_layers), pos: 0, next: 0, staged: None }
    }

    fn steer_hooks(&self) -> &'static [Hook] {
        &Hook::ALL
    }

    fn end_of_turn(&self) -> &'static [&'static str] {
        &["<|im_end|>", "<|endoftext|>"]
    }

    fn forward(&self, ids: &[u32], st: &mut MropeCache, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        self.forward_media(ids, &[], st, rows, taps)
    }
}

impl VisionLm for Qwen3VlModel {
    type Image = Patches;

    fn has_vision(&self) -> bool {
        true
    }

    fn image_tokens(&self) -> (u32, Option<u32>, Option<u32>) {
        (self.cfg.image_token_id, None, None)
    }

    fn prepare_image(&self, img: &Image) -> Result<Patches> {
        self.processor.process(img)
    }

    fn image_size(&self, p: &Patches) -> (usize, (usize, usize)) {
        let m = self.cfg.vision.spatial_merge_size;
        (p.merged_tokens(m), (p.grid_h / m, p.grid_w / m))
    }

    /// Merged tokens, then the DeepStack features (taps `vision.*`).
    fn encode_image(&self, p: &Patches, taps: &mut Taps) -> Result<Vec<Tensor>> {
        self.vision.build(p, taps)
    }

    /// Taps: `embeds`, `layers.{i}.op`, `layers.{i}` (before DeepStack), `norm`.
    fn forward_media(&self, ids: &[u32], media: &[Media], st: &mut MropeCache, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        let (n, d) = (ids.len(), self.cfg.text.base.hidden_size);
        let runs: Vec<_> = media.iter().map(|m| (m.start, m.len, m.grid)).collect();
        let (pos, next) = mrope_positions(n, &runs, st.next)?;
        st.staged = Some(next);
        let rows_of = |t: &Tensor, m: &Media| t.cast(DType::F32)?.reshape(&[1, m.len, d]);
        let mut x = self.text.head.embed(ids)?;
        if !media.is_empty() {
            let (mut parts, mut at) = (Vec::new(), 0);
            for m in media {
                if m.start > at {
                    parts.push(x.slice(1, at, m.start)?);
                }
                parts.push(rows_of(&m.embeds[0], m)?);
                at = m.start + m.len;
            }
            if at < n {
                parts.push(x.slice(1, at, n)?);
            }
            x = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?;
        }
        taps.tap("embeds", &x);
        let cs = self.cfg.text.mrope.tables(&self.text.inv_freq, &pos)?;
        self.text.decode(x, &mut st.kv, &cs, rows, taps, |i, mut x| {
            for m in media {
                if let Some(ds) = m.embeds.get(1 + i) {
                    x = x.add(&rows_of(ds, m)?.pad(1, m.start, n - m.start - m.len, 0.0)?)?;
                }
            }
            Ok(x)
        })
    }

    fn image_pixels<'a>(&self, p: &'a Patches) -> &'a [f32] {
        &p.data
    }

    /// `<case>.pixel_values` [N, 3 P P] (first temporal frame) and `grid_thw`.
    fn reference_image(&self, r: &Reference, case: &str) -> Result<Patches> {
        let c = r.case(case);
        let g = |i: usize| c.pointer(&format!("/grid_thw/{i}")).and_then(Value::as_u64).map(|x| x as usize);
        let (Some(gh), Some(gw)) = (g(1), g(2)) else { return Err(Error::new("reference", format!("qwen3vl: case {case} has no grid_thw"))) };
        let data = r.acts.get(&format!("{case}.pixel_values"))?;
        Ok(Patches { dim: data.len() / (gh * gw).max(1), data, grid_h: gh, grid_w: gw })
    }

    fn media_taps(&self) -> Vec<String> {
        let mut v = vec!["vision.merged".to_string()];
        v.extend((0..self.cfg.vision.deepstack_layers.len()).map(|j| format!("vision.deepstack.{j}")));
        v
    }

    /// The prompt's M-RoPE positions vs the reference `positions` ([3][n]).
    fn check_prompt(&self, p: &MediaPrompt<Patches>, case: &Value) -> Option<bool> {
        let want = case.get("positions")?.as_array()?;
        let row = |k: usize| want.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect::<Vec<_>>());
        let runs: Vec<_> = p.images.iter().map(|i| (i.start, i.len, i.grid)).collect();
        let Ok((got, _)) = mrope_positions(p.ids.len(), &runs, 0) else { return Some(false) };
        let (t, h, w) = (row(0)?, row(1)?, row(2)?);
        Some(t.len() == got.len() && got.iter().enumerate().all(|(i, x)| *x == [t[i], h[i], w[i]]))
    }
}
