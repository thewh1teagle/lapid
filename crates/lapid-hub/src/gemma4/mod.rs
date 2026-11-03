//! Google Gemma 4: one model for the dense (E2B, E4B, 31B) and MoE (26B-A4B) text decoders
//! and the vision tower (`config`: `moe`, `vision`). Only the model's math, config,
//! conversion tables and verification descriptors; the rest is `crate::runtime`
//! (`Llm` for text chat, `runtime::vlm` for image + text chat).
//!
//! ```no_run
//! use lapid_hub::gemma4::{ChatOptions, Gemma4};
//! use lapid_hub::vision::{Image, Message};
//! let device = lapid::Device::best();
//! let llm = Gemma4::from_hub("google/gemma-4-E2B-it", "int8", &device)?;
//! let msg = Message::user_with_images(vec![Image::open("photo.jpg".as_ref())?], "Describe this image.");
//! let mut session = llm.new_session();
//! for token in llm.mm_chat_stream(&mut session, &[msg], &ChatOptions::default())? {
//!     print!("{}", token?.text);
//! }
//! # Ok::<(), lapid::Error>(())
//! ```
//!
//! Verification names: `gemma4` (text reference, plus its `inject` case), `gemma4moe` (the
//! tiny MoE text reference by default), `gemma4vl` (image references).

pub mod config;
pub mod convert;
pub mod model;
pub mod vision;

use std::path::Path;

use lapid::{Error, Result, Tensor, bail};
use lapid_convert::engine::ConvertSpec;
use lapid_nn::Loader;
use lapid_verify::Descriptor;
use lapid_verify::harness::Reference;
use serde_json::Value;

pub use crate::runtime::{CancelToken, ChatOptions, FinishReason, GenerateOptions, GenerationStats, Message, Token, TokenStream};
pub use config::{Gemma4Config, LayerType};
use crate::cmd::ModelArgs;
use crate::runtime::{CausalLm, Hook, LmCache, LmDef, LmDims, Llm, Media, ModelEntry, Rows, Taps, VisionLm, lm_bench, vlm_verify};
use config::ImageProcessorConfig;
use model::Decoder;
use vision::{PreparedImage, VisionTower};

/// A loaded Gemma 4 model (text, and images when the file has the vision tower).
pub type Gemma4 = Llm<Gemma4Model>;
/// The MoE variants are the same model.
pub type Gemma4Moe = Gemma4;
/// Image + text chat: `Gemma4::mm_chat_stream` (`runtime::vlm`).
pub type Gemma4Vl = Gemma4;
pub type Session = crate::runtime::Session<LmCache>;

pub struct Gemma4Model {
    pub decoder: Decoder,
    tower: Option<VisionTower>,
    processor: ImageProcessorConfig,
    image_tokens: (u32, u32, u32),
}

impl Gemma4Model {
    pub fn config(&self) -> &Gemma4Config {
        &self.decoder.cfg
    }

    /// Soft-token budget per image (HF `max_soft_tokens`: 70, 140, 280, 560 or 1120; more
    /// tokens = higher resolution). GGUF projectors do not record it (default 280).
    pub fn set_max_soft_tokens(&mut self, n: usize) -> Result<()> {
        if ![70, 140, 280, 560, 1120].contains(&n) {
            bail!("invalid_argument", "gemma4: max_soft_tokens {n} (expected 70, 140, 280, 560 or 1120)");
        }
        self.processor.max_soft_tokens = n;
        Ok(())
    }
}

impl CausalLm for Gemma4Model {
    const ARCH: &'static str = "gemma4";
    type State = LmCache;

    fn load(l: &Loader<'_>, meta: &Value) -> Result<Gemma4Model> {
        let cfg = Gemma4Config::from_json(meta)?;
        let decoder = Decoder::load(l, &cfg)?;
        // The tower is optional: text-only files (and GGUFs without a projector) have none.
        let tower = match &cfg.vision {
            Some(vc) if l.contains(vision::EMBED_PROJ) => Some(VisionTower::load(l, vc, cfg.hidden_size)?),
            _ => None,
        };
        let [image, boi, eoi] = cfg.image_tokens;
        let image_tokens = match (image, boi, eoi) {
            (Some(i), Some(b), Some(e)) => (i, b, e),
            _ if tower.is_none() => (u32::MAX, u32::MAX, u32::MAX),
            _ => bail!("metadata", "gemma4: the vision tower needs image_token_id, boi_token_id and eoi_token_id"),
        };
        if let Some(t) = &tower
            && (cfg.processor.patch_size != t.cfg.patch_size || cfg.processor.pooling_kernel_size != t.cfg.pooling_kernel_size)
        {
            bail!("metadata", "gemma4: image_processor patch / pooling sizes differ from vision_config");
        }
        Ok(Gemma4Model { processor: cfg.processor.clone(), decoder, tower, image_tokens })
    }

    fn dims(&self) -> LmDims {
        let c = self.config();
        LmDims { vocab: c.vocab_size, hidden: c.hidden_size, layers: c.n_layers }
    }

    fn new_state(&self) -> LmCache {
        LmCache::new(self.decoder.new_cache(), 0)
    }

    fn steer_hooks(&self) -> &'static [Hook] {
        &Hook::ALL
    }

    fn end_of_turn(&self) -> &'static [&'static str] {
        &["<turn|>"]
    }

    fn think_markers(&self) -> (&'static str, &'static str) {
        ("<|channel>thought\n", "<channel|>")
    }

    /// Taps: `embed`, `layers.{i}.attn`, `layers.{i}`, `norm`.
    fn forward(&self, ids: &[u32], st: &mut LmCache, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        self.decoder.forward(ids, &[], &mut st.kv, rows, taps)
    }
}

impl VisionLm for Gemma4Model {
    type Image = PreparedImage;

    fn has_vision(&self) -> bool {
        self.tower.is_some()
    }

    fn image_tokens(&self) -> (u32, Option<u32>, Option<u32>) {
        let (i, b, e) = self.image_tokens;
        (i, Some(b), Some(e))
    }

    fn prepare_image(&self, img: &crate::vision::Image) -> Result<PreparedImage> {
        vision::prepare(img, &self.processor)
    }

    fn image_size(&self, img: &PreparedImage) -> (usize, (usize, usize)) {
        let k = self.processor.pooling_kernel_size;
        (img.n_soft, (img.grid.0 / k, img.grid.1 / k))
    }

    fn encode_image(&self, img: &PreparedImage, taps: &mut Taps) -> Result<Vec<Tensor>> {
        let t = self.tower.as_ref().ok_or_else(|| Error::new("unsupported", "gemma4: no vision tower"))?;
        Ok(vec![t.build(img, taps)?])
    }

    fn forward_media(&self, ids: &[u32], media: &[Media], st: &mut LmCache, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>> {
        self.decoder.forward(ids, media, &mut st.kv, rows, taps)
    }

    fn image_pixels<'a>(&self, img: &'a PreparedImage) -> &'a [f32] {
        &img.patches
    }

    /// `<case>.pixel_values` [N, 3 p p] and `<case>.positions` [N, 2] (x, y).
    fn reference_image(&self, r: &Reference, case: &str) -> Result<PreparedImage> {
        let positions: Vec<[usize; 2]> = r.acts.get(&format!("{case}.positions"))?.chunks(2).map(|c| [c[0] as usize, c[1] as usize]).collect();
        let k = self.processor.pooling_kernel_size;
        let (cols, rows) = (positions.iter().map(|p| p[0] + 1).max().unwrap_or(0), positions.iter().map(|p| p[1] + 1).max().unwrap_or(0));
        if rows * cols != positions.len() || rows % k != 0 || cols % k != 0 {
            bail!("reference", "gemma4: patch grid {rows}x{cols} for {} patches", positions.len());
        }
        Ok(PreparedImage { patches: r.acts.get(&format!("{case}.pixel_values"))?, n_soft: positions.len() / (k * k), positions, grid: (rows, cols) })
    }

    fn media_taps(&self) -> Vec<String> {
        vec!["vision.projected".into()]
    }

    /// The resize is torchvision-exact.
    fn preprocess_tolerance(&self) -> (f32, f32) {
        (1e-6, 1e-6)
    }

    fn configure(&mut self, reference: &Value) -> Result<()> {
        match reference.get("max_soft_tokens").and_then(Value::as_u64) {
            Some(n) => self.set_max_soft_tokens(n as usize),
            None => Ok(()),
        }
    }
}

const fn descriptor(name: &'static str, model: &'static str, dtype: &'static str, dir: &'static str, script: &'static str) -> Descriptor {
    Descriptor {
        name,
        default_model: model,
        default_dtype: dtype,
        reference_dir: dir,
        reference_script: script,
        reference_label: "PyTorch fp32 (HF transformers Gemma 4)",
        strict_dtypes: &["fp32", "fp16"],
        bench_prompt_tokens: 512,
        bench_gen_tokens: 128,
    }
}

impl LmDef for Gemma4Model {
    const DESCRIPTOR: Descriptor = descriptor("gemma4", "google/gemma-4-E2B-it", "int8", "data/reference/gemma4", "tools/reference/gemma4_ref.py");
    const CONVERT: &'static ConvertSpec = &convert::CONVERT;
}

const MOE: Descriptor = descriptor("gemma4moe", "google/gemma-4-26B-A4B-it", "int4", "data/reference/gemma4moe/tiny", "tools/reference/gemma4moe_ref.py");
const VL: Descriptor = descriptor("gemma4vl", "google/gemma-4-E2B-it", "fp16", "data/reference/gemma4vl", "tools/reference/gemma4vl_ref.py");

fn verify(r: Option<&Path>, a: &ModelArgs) -> Result<Value> {
    vlm_verify::verify_text::<Gemma4Model>(&Gemma4Model::DESCRIPTOR, r, a)
}

fn verify_moe(r: Option<&Path>, a: &ModelArgs) -> Result<Value> {
    vlm_verify::verify_text::<Gemma4Model>(&MOE, r, a)
}

fn bench_moe(i: Option<&Path>, a: &ModelArgs, runs: usize) -> Result<Value> {
    lm_bench::bench::<Gemma4Model>(&MOE, i, a, runs)
}

fn verify_vl(r: Option<&Path>, a: &ModelArgs) -> Result<Value> {
    vlm_verify::verify::<Gemma4Model>(&VL, r, a)
}

fn bench_vl(i: Option<&Path>, a: &ModelArgs, runs: usize) -> Result<Value> {
    vlm_verify::bench::<Gemma4Model>(&VL, i, a, runs)
}

pub static ENTRY: ModelEntry = ModelEntry { verify, ..ModelEntry::causal_lm::<Gemma4Model>() };
pub static MOE_ENTRY: ModelEntry = ModelEntry { name: "gemma4moe", verify: verify_moe, bench: bench_moe, ..ModelEntry::causal_lm::<Gemma4Model>() };
pub static VL_ENTRY: ModelEntry = ModelEntry { name: "gemma4vl", verify: verify_vl, bench: bench_vl, ..ModelEntry::causal_lm::<Gemma4Model>() };

impl Gemma4 {
    pub fn config(&self) -> &Gemma4Config {
        self.model().config()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::cmd::ModelArgs;

    /// The tiny references end to end on the CPU (skipped when `data/reference` lacks them).
    #[test]
    fn tiny_references_verify() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reference");
        for (entry, dir) in [(&super::ENTRY, "gemma4/tiny"), (&super::MOE_ENTRY, "gemma4moe/tiny"), (&super::VL_ENTRY, "gemma4vl/tiny"), (&super::VL_ENTRY, "gemma4vl/tiny_causal")] {
            let dir = root.join(dir);
            if !dir.join("model").is_dir() {
                eprintln!("skipped: {} not found", dir.display());
                continue;
            }
            let args = ModelArgs { model: Some(dir.join("model").display().to_string()), dtype: Some("fp32".into()), device: Some("cpu".into()) };
            let r = (entry.verify)(Some(&dir), &args).expect("verify");
            assert_eq!(r["ok"], true, "{}: {r:#}", dir.display());
        }
    }
}
