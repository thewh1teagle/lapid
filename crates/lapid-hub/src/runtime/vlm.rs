//! Image + text causal LMs. A [`VisionLm`] is a [`CausalLm`] with a vision tower: it
//! prepares an image (resize, patchify), encodes it into embedding rows, and its decoder
//! takes those rows at the image's placeholder tokens ([`Media`]). The pipeline here does
//! the rest for every such model: chat messages with image parts (`vision::Message`)
//! rendered by the model's template, placeholder expansion, image encoding, prefill chunks
//! that never split an image, and sessions whose prefix reuse stops before an image that
//! is not the one in the cache.
//!
//! ```ignore
//! let vlm = Gemma4::from_hub("google/gemma-4-E2B-it", "int8", &device)?;
//! let msg = Message::user_with_images(vec![Image::open("photo.jpg".as_ref())?], "Describe this image.");
//! let mut session = vlm.new_session();
//! for t in vlm.mm_chat_stream(&mut session, &[msg], &ChatOptions::default())? { print!("{}", t?.text); }
//! ```

use std::sync::Arc;

use lapid::{Device, Error, Result, Tensor};
use lapid_verify::harness::Reference;
use serde_json::Value;

use super::causal_lm::{CausalLm, Llm, Rows, Taps, eval_commit, step};
use super::chat::ChatOptions;
use super::generate::{GenerateOptions, TokenStream};
use super::jinja;
use super::session::{LmState, Session};
use super::steer::Steering;
use crate::vision::{Image, Message, collect_images};

/// Embedding rows that replace a run of image placeholder tokens.
#[derive(Clone, Debug)]
pub struct Media {
    /// First placeholder token: an absolute prompt index, relative to the forward's ids
    /// inside [`VisionLm::forward_media`].
    pub start: usize,
    /// Placeholder tokens (= embedding rows).
    pub len: usize,
    /// Token grid (rows, columns) of the image (M-RoPE positions).
    pub grid: (usize, usize),
    /// `[0]`: embeddings `[len, D]`; further tensors are model extras aligned with the rows
    /// (e.g. DeepStack features).
    pub embeds: Vec<Tensor>,
}

/// A causal LM with a vision tower: only its own math and the verification glue.
pub trait VisionLm: CausalLm {
    /// One image ready for the vision tower.
    type Image: Clone + Send + Sync;

    /// Whether the loaded file has a vision tower.
    fn has_vision(&self) -> bool;
    /// The placeholder the template renders per image (repeated once per embedding row
    /// after expansion) and the tokens wrapped around an expanded run.
    fn image_tokens(&self) -> (u32, Option<u32>, Option<u32>);
    /// Resize / rescale / patchify.
    fn prepare_image(&self, img: &Image) -> Result<Self::Image>;
    /// Placeholder tokens and token grid (rows, columns) of a prepared image.
    fn image_size(&self, img: &Self::Image) -> (usize, (usize, usize));
    /// Lazy embeddings of one image in [`Media::embeds`] order (taps: `vision.*`).
    fn encode_image(&self, img: &Self::Image, taps: &mut Taps) -> Result<Vec<Tensor>>;
    /// [`CausalLm::forward`] with image rows (`start` relative to `ids`, never split).
    fn forward_media(&self, ids: &[u32], media: &[Media], state: &mut Self::State, rows: Rows, taps: &mut Taps) -> Result<Option<Tensor>>;

    /// Pixel values of a prepared image as the reference stores them (`<case>.pixel_values`).
    fn image_pixels<'a>(&self, img: &'a Self::Image) -> &'a [f32];
    /// The prepared image the reference computed (`<case>.pixel_values` and friends).
    fn reference_image(&self, r: &Reference, case: &str) -> Result<Self::Image>;
    /// Tap names of the [`encode_image`](Self::encode_image) outputs; when the reference has
    /// them all, the decoder is checked on the reference values.
    fn media_taps(&self) -> Vec<String>;
    /// Preprocessing tolerance vs the reference pixels: (max abs, mean abs).
    fn preprocess_tolerance(&self) -> (f32, f32) {
        (0.04, 2e-3)
    }
    /// Apply reference settings (e.g. an image token budget from `result.json`).
    fn configure(&mut self, _reference: &Value) -> Result<()> {
        Ok(())
    }
    /// A model-specific check of a prompt against its reference case (e.g. M-RoPE
    /// positions); `None` = nothing to check.
    fn check_prompt(&self, _prompt: &MediaPrompt<Self::Image>, _case: &Value) -> Option<bool> {
        None
    }
}

/// One image of a [`MediaPrompt`].
#[derive(Clone)]
pub struct PromptImage<I> {
    pub start: usize,
    pub len: usize,
    pub grid: (usize, usize),
    pub image: I,
    /// [`Image::fingerprint`] of the source picture.
    pub hash: u64,
}

/// Token ids with the image placeholders expanded, and the images in order.
#[derive(Clone)]
pub struct MediaPrompt<I> {
    pub ids: Vec<u32>,
    pub images: Vec<PromptImage<I>>,
}

/// Expand each rendered placeholder (one token, or a run already as long as the image) to
/// `begin + placeholder * n + end`, taking the images in order.
pub fn expand<M: VisionLm>(model: &M, ids: &[u32], images: Vec<(M::Image, u64)>) -> Result<MediaPrompt<M::Image>> {
    let (pad, begin, end) = model.image_tokens();
    let (mut out, mut placed) = (Vec::with_capacity(ids.len()), Vec::new());
    let mut it = images.into_iter();
    let mut i = 0;
    while i < ids.len() {
        if ids[i] != pad {
            out.push(ids[i]);
            i += 1;
            continue;
        }
        let run = ids[i..].iter().take_while(|&&t| t == pad).count();
        let (image, hash) = it.next().ok_or_else(|| Error::new("invalid_argument", "more image placeholders than images"))?;
        let (n, grid) = model.image_size(&image);
        if run != 1 && run != n {
            return Err(Error::new("invalid_argument", format!("image placeholder run of {run} tokens for an image of {n}")));
        }
        out.extend(begin);
        placed.push(PromptImage { start: out.len(), len: n, grid, image, hash });
        out.extend(std::iter::repeat_n(pad, n));
        out.extend(end);
        i += run;
    }
    if it.next().is_some() {
        return Err(Error::new("invalid_argument", "more images than image placeholders in the prompt"));
    }
    Ok(MediaPrompt { ids: out, images: placed })
}

/// Prefill chunks `[a, b)` of at most `chunk` tokens, moved so that no image run is split.
pub fn chunk_bounds(n: usize, media: &[Media], chunk: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut a = 0;
    while a < n {
        let mut b = (a + chunk.max(1)).min(n);
        for m in media {
            let (s, e) = (m.start, m.start + m.len);
            if s < b && e > b {
                b = if s > a { s } else { e };
            }
        }
        out.push((a, b));
        a = b;
    }
    out
}

fn check_media(n: usize, media: &[Media]) -> Result<()> {
    let mut end = 0;
    for m in media {
        if m.start < end || m.start + m.len > n || m.embeds.is_empty() {
            return Err(Error::new("invalid_argument", format!("image rows {}..{} overlap or exceed {n} tokens", m.start, m.start + m.len)));
        }
        end = m.start + m.len;
    }
    Ok(())
}

/// [`step`] with image rows (absolute `start`s within `ids`).
pub fn step_media<M: VisionLm>(model: &M, device: &Device, ids: &[u32], media: &[Media], state: &mut M::State, steering: Option<&Arc<Steering>>) -> Result<Vec<f32>> {
    if media.is_empty() {
        return step(model, device, ids, state, steering);
    }
    check_media(ids.len(), media)?;
    let chunks = chunk_bounds(ids.len(), media, model.prefill_chunk());
    let last = chunks.len() - 1;
    for (ci, &(a, b)) in chunks.iter().enumerate() {
        let rows = if ci == last { Rows::Last } else { Rows::None };
        let local: Vec<Media> = media.iter().filter(|m| m.start >= a && m.start < b).map(|m| Media { start: m.start - a, ..m.clone() }).collect();
        let out = match model.forward_media(&ids[a..b], &local, state, rows, &mut Taps::off().with_steering(steering.cloned())) {
            Ok(o) => o,
            Err(e) => {
                state.discard();
                return Err(e);
            }
        };
        let outputs: Vec<&Tensor> = out.iter().collect();
        let realized = eval_commit(device, &outputs, state, b - a)?;
        if ci == last {
            let logits = realized.first().ok_or_else(|| Error::new("backend", "no logits realized"))?;
            return device.to_vec_f32(logits);
        }
    }
    Err(Error::new("internal", "prefill produced no logits"))
}

/// Realize the named taps (plus `logits`) of a forward.
fn realize(device: &Device, mut list: Vec<(String, Tensor)>, logits: Option<Tensor>) -> Result<Vec<(String, Vec<f32>)>> {
    if let Some(l) = logits {
        list.push(("logits".to_string(), l));
    }
    let refs: Vec<&Tensor> = list.iter().map(|(_, t)| t).collect();
    let realized = device.eval(&refs)?;
    list.iter().zip(&realized).map(|((n, _), t)| Ok((n.clone(), device.to_vec_f32(t)?))).collect()
}

/// One uncached forward with image rows, every tap realized (plus `logits`).
pub fn forward_media_traced<M: VisionLm>(model: &M, device: &Device, ids: &[u32], media: &[Media], rows: Rows) -> Result<Vec<(String, Vec<f32>)>> {
    check_media(ids.len(), media)?;
    let mut state = model.new_state();
    let mut taps = Taps::on();
    let logits = model.forward_media(ids, media, &mut state, rows, &mut taps)?;
    realize(device, taps.into_vec(), logits)
}

/// Realized embeddings of one image, and its realized taps when `trace`.
pub fn encode_image<M: VisionLm>(model: &M, device: &Device, img: &M::Image, trace: bool) -> Result<(Vec<Tensor>, Vec<(String, Vec<f32>)>)> {
    let mut taps = if trace { Taps::on() } else { Taps::off() };
    let outs = model.encode_image(img, &mut taps)?;
    let taps = taps.into_vec();
    let mut refs: Vec<&Tensor> = outs.iter().collect();
    refs.extend(taps.iter().map(|(_, t)| t));
    let mut realized = device.eval(&refs)?;
    let tap_vals = realized.split_off(outs.len());
    let traced = taps.iter().zip(&tap_vals).map(|((n, _), t)| Ok((n.clone(), device.to_vec_f32(t)?))).collect::<Result<_>>()?;
    Ok((realized, traced))
}

impl<M: VisionLm> Llm<M> {
    /// Render `messages` (text and image parts) with the chat template and the generation
    /// prompt, tokenize, preprocess the images and expand their placeholders.
    pub fn mm_prompt(&self, messages: &[Message], opts: &ChatOptions) -> Result<MediaPrompt<M::Image>> {
        let msgs = jinja::Value::List(messages.iter().map(Message::to_value).collect());
        let text = self.template().render_values(msgs, jinja::Value::None, true, opts.enable_thinking, &opts.extra_globals())?;
        self.mm_prompt_ids(&self.encode(&text)?, &collect_images(messages))
    }

    /// [`mm_prompt`](Self::mm_prompt) from rendered token ids and the images in order.
    pub fn mm_prompt_ids(&self, ids: &[u32], images: &[&Image]) -> Result<MediaPrompt<M::Image>> {
        if !images.is_empty() && !self.model().has_vision() {
            return Err(Error::new("unsupported", "this model file has no vision tower"));
        }
        let prepared = images.iter().map(|i| Ok((self.model().prepare_image(i)?, i.fingerprint()))).collect::<Result<Vec<_>>>()?;
        expand(self.model(), ids, prepared)
    }

    /// Realized embeddings of one prepared image.
    pub fn encode_image(&self, img: &M::Image) -> Result<Vec<Tensor>> {
        Ok(encode_image(self.model(), self.device(), img, false)?.0)
    }

    /// Stream tokens after `prompt`. The session's cached prefix is reused up to the first
    /// image that is not the same picture at the same place; the other images are encoded.
    pub fn mm_generate<'a>(&'a self, session: &'a mut Session<M::State>, prompt: &MediaPrompt<M::Image>, opts: GenerateOptions) -> Result<TokenStream<'a>> {
        let n = prompt.ids.len();
        let limit = prompt.images.iter().filter(|im| !session.media.contains(&(im.start, im.hash))).map(|im| im.start).min().unwrap_or(n);
        let mut cached = session.reuse_prefix(&prompt.ids[..(limit + 1).min(n)]);
        if prompt.images.iter().any(|im| im.start < cached && cached < im.start + im.len) {
            session.reset();
            cached = 0;
        }
        session.media.retain(|&(s, _)| s < cached);
        let mut media = Vec::new();
        for im in prompt.images.iter().filter(|im| im.start >= cached) {
            media.push(Media { start: im.start, len: im.len, grid: im.grid, embeds: self.encode_image(&im.image)? });
            session.media.push((im.start, im.hash));
        }
        let (model, device) = (self.model(), self.device());
        let steering = session.steering.clone();
        let step_fn = move |ids: &[u32]| -> Result<Vec<f32>> {
            let at = session.tokens.len();
            let local: Vec<Media> = media.iter().filter(|m| m.start >= at && m.start < at + ids.len()).map(|m| Media { start: m.start - at, ..m.clone() }).collect();
            let logits = step_media(model, device, ids, &local, &mut session.state, steering.as_ref())?;
            session.tokens.extend_from_slice(ids);
            Ok(logits)
        };
        TokenStream::new(step_fn, self.tokenizer(), &prompt.ids, cached, opts, self.stop_tokens())
    }

    /// Stream the assistant reply to `messages` (text and image parts).
    pub fn mm_chat_stream<'a>(&'a self, session: &'a mut Session<M::State>, messages: &[Message], opts: &ChatOptions) -> Result<TokenStream<'a>> {
        let prompt = self.mm_prompt(messages, opts)?;
        self.mm_generate(session, &prompt, opts.generate.clone())
    }

    /// The full assistant reply to `messages` (non-streaming convenience).
    pub fn mm_chat(&self, messages: &[Message], opts: &ChatOptions) -> Result<String> {
        let mut session = self.new_session();
        let (text, _, _) = self.mm_chat_stream(&mut session, messages, opts)?.collect_text()?;
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(start: usize, len: usize) -> Media {
        Media { start, len, grid: (1, len), embeds: vec![Tensor::zeros(&[len, 4], lapid::DType::F32)] }
    }

    #[test]
    fn chunks_keep_images_whole() {
        assert_eq!(chunk_bounds(1000, &[], 512), vec![(0, 512), (512, 1000)]);
        assert_eq!(chunk_bounds(1000, &[m(500, 20)], 512), vec![(0, 500), (500, 1000)]);
        assert_eq!(chunk_bounds(1300, &[m(0, 600)], 512), vec![(0, 600), (600, 1112), (1112, 1300)]);
        assert!(check_media(5, &[m(4, 2)]).is_err());
        assert!(check_media(5, &[m(1, 2), m(2, 1)]).is_err());
    }
}
