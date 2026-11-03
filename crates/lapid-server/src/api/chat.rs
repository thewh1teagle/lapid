//! `POST /v1/chat/completions` and `POST /v1/completions` (streaming SSE and plain JSON).

use std::convert::Infallible;

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::stream::{self, Stream, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{ApiError, ApiResult, Shared, new_id, now_secs, parse_json};
use lapid_hub::runtime::serve::{Capability, ChatParams, Delta, GenParams, GenSummary};
use crate::worker::Worker;

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

impl OneOrMany {
    fn into_vec(self) -> Vec<String> {
        match self {
            OneOrMany::One(s) => vec![s],
            OneOrMany::Many(v) => v,
        }
    }
}

#[derive(Deserialize, Default)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

/// Sampling fields shared by both endpoints.
#[derive(Deserialize, Default)]
struct Sampling {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i64>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    stop: Option<OneOrMany>,
    seed: Option<i64>,
    repetition_penalty: Option<f32>,
    n: Option<usize>,
    /// lapid extension: steering files by name with scales (`{}` or `false`: none).
    steer: Option<Value>,
}

/// `"steer": {"name": scale, ...}` (or `false` / `{}` for none; `null` = server default).
fn parse_steer(v: Option<Value>) -> ApiResult<Option<Vec<(String, f32)>>> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(false)) => Ok(Some(Vec::new())),
        Some(Value::Object(m)) => m
            .into_iter()
            .map(|(k, v)| match v.as_f64() {
                Some(x) if x.is_finite() => Ok((k, x as f32)),
                _ => Err(ApiError::bad_request(format!("\"steer\": the scale of {k:?} must be a number"))),
            })
            .collect::<ApiResult<Vec<_>>>()
            .map(Some),
        Some(_) => Err(ApiError::bad_request("\"steer\" must be an object {\"name\": scale} or false")),
    }
}

impl Sampling {
    fn params(self) -> ApiResult<GenParams> {
        if self.n.is_some_and(|n| n != 1) {
            return Err(ApiError::bad_request("only n = 1 is supported"));
        }
        Ok(GenParams {
            max_tokens: self.max_completion_tokens.or(self.max_tokens),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k.map(|k| k.max(0) as usize),
            repetition_penalty: self.repetition_penalty,
            seed: self.seed.map(|s| s as u64),
            stop: self.stop.map(OneOrMany::into_vec).unwrap_or_default(),
            steer: parse_steer(self.steer)?,
        })
    }
}

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<Value>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    tools: Option<Value>,
    chat_template_kwargs: Option<Value>,
    #[serde(flatten)]
    sampling: Sampling,
}

#[derive(Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: OneOrMany,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    #[serde(flatten)]
    sampling: Sampling,
}

/// OpenAI message content (string or text parts) as a plain string; other fields kept.
fn normalize_messages(messages: Vec<Value>) -> ApiResult<Value> {
    if messages.is_empty() {
        return Err(ApiError::bad_request("\"messages\" must not be empty"));
    }
    let mut out = Vec::with_capacity(messages.len());
    for mut m in messages {
        let Some(obj) = m.as_object_mut() else {
            return Err(ApiError::bad_request("each message must be an object"));
        };
        if !obj.get("role").is_some_and(Value::is_string) {
            return Err(ApiError::bad_request("each message needs a string \"role\""));
        }
        let content = match obj.remove("content") {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(s)) => s,
            Some(Value::Array(parts)) => {
                let mut text = Vec::new();
                for p in parts {
                    match p.get("type").and_then(Value::as_str) {
                        Some("text") => text.push(p.get("text").and_then(Value::as_str).unwrap_or("").to_string()),
                        Some(other) => return Err(ApiError::bad_request(format!("content part type {other:?} is not supported"))),
                        None => return Err(ApiError::bad_request("content parts need a \"type\"")),
                    }
                }
                text.join("\n")
            }
            Some(_) => return Err(ApiError::bad_request("message content must be a string or an array of parts")),
        };
        obj.insert("content".into(), Value::String(content));
        out.push(m);
    }
    Ok(Value::Array(out))
}

/// What the worker sends back while generating.
enum GenEvent {
    Delta(Delta),
    Done(GenSummary),
    Error(lapid::Error),
}

fn spawn_gen(
    worker: &Worker,
    run: impl FnOnce(&mut dyn lapid_hub::AnyModel, &mut dyn FnMut(Delta) -> bool) -> lapid::Result<GenSummary> + Send + 'static,
) -> ApiResult<mpsc::Receiver<GenEvent>> {
    let (tx, rx) = mpsc::channel::<GenEvent>(256);
    worker.submit(move |s| {
        if tx.is_closed() {
            return; // the client left while queued
        }
        let r = run(s, &mut |d| tx.blocking_send(GenEvent::Delta(d)).is_ok());
        let _ = tx.blocking_send(match r {
            Ok(sum) => GenEvent::Done(sum),
            Err(e) => GenEvent::Error(e),
        });
    })?;
    Ok(rx)
}

fn usage(s: &GenSummary) -> Value {
    json!({
        "prompt_tokens": s.prompt_tokens,
        "completion_tokens": s.completion_tokens,
        "total_tokens": s.prompt_tokens + s.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": s.cached_tokens},
    })
}

fn finish(s: &GenSummary) -> &str {
    if s.finish_reason == "length" { "length" } else { "stop" }
}

/// Drain a generation: (content, reasoning, summary).
async fn collect(mut rx: mpsc::Receiver<GenEvent>) -> ApiResult<(String, String, GenSummary)> {
    let (mut content, mut reasoning) = (String::new(), String::new());
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Delta(Delta::Content(t)) => content.push_str(&t),
            GenEvent::Delta(Delta::Reasoning(t)) => reasoning.push_str(&t),
            GenEvent::Done(s) => return Ok((content, reasoning, s)),
            GenEvent::Error(e) => return Err(e.into()),
        }
    }
    Err(ApiError::from(lapid::Error::new("internal", "generation ended without a result")))
}

fn data(v: &Value) -> Result<Event, Infallible> {
    Ok(Event::default().data(v.to_string()))
}

/// SSE response: `chunk(delta_or_none, finish)` builds each chunk; usage chunk on request.
fn sse(
    rx: mpsc::Receiver<GenEvent>,
    first: Option<Value>,
    include_usage: bool,
    chunk: impl Fn(Option<&Delta>, Option<&GenSummary>) -> Value + Send + 'static,
    usage_chunk: impl Fn(&GenSummary) -> Value + Send + 'static,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let head = stream::iter(first.map(|v| data(&v)));
    let body = ReceiverStream::new(rx).flat_map(move |ev| {
        let mut out = Vec::new();
        match ev {
            GenEvent::Delta(d) => out.push(data(&chunk(Some(&d), None))),
            GenEvent::Done(s) => {
                out.push(data(&chunk(None, Some(&s))));
                if include_usage {
                    out.push(data(&usage_chunk(&s)));
                }
                out.push(Ok(Event::default().data("[DONE]")));
            }
            GenEvent::Error(e) => {
                out.push(data(&ApiError::from(e).body()));
                out.push(Ok(Event::default().data("[DONE]")));
            }
        }
        stream::iter(out)
    });
    Sse::new(head.chain(body)).keep_alive(KeepAlive::default())
}

pub async fn chat_completions(State(st): State<Shared>, body: Bytes) -> ApiResult<Response> {
    let req: ChatRequest = parse_json(&body)?;
    let worker = st.find(req.model.as_deref(), Capability::Chat)?;
    let model = worker.meta().id.clone();
    let enable_thinking = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|k| k.get("enable_thinking"))
        .and_then(Value::as_bool);
    let params = ChatParams {
        messages: normalize_messages(req.messages)?,
        tools: req.tools.filter(|t| !t.is_null()),
        enable_thinking,
        sampling: req.sampling.params()?,
    };
    let rx = spawn_gen(worker, move |s, out| s.chat(&params, out))?;
    let (id, created) = (new_id("chatcmpl"), now_secs());
    if !req.stream {
        let (content, reasoning, s) = collect(rx).await?;
        let mut message = json!({"role": "assistant", "content": content});
        if !reasoning.is_empty() {
            message["reasoning_content"] = json!(reasoning);
        }
        return Ok(axum::Json(json!({
            "id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish(&s), "logprobs": null}],
            "usage": usage(&s),
            "timings": s.timings(),
        }))
        .into_response());
    }
    let include_usage = req.stream_options.unwrap_or_default().include_usage;
    let base = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model});
    let first = {
        let mut v = base.clone();
        v["choices"] = json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]);
        v
    };
    let b1 = base.clone();
    let chunk = move |d: Option<&Delta>, s: Option<&GenSummary>| {
        let mut v = b1.clone();
        match (d, s) {
            (Some(Delta::Content(t)), _) => v["choices"] = json!([{"index": 0, "delta": {"content": t}, "finish_reason": null}]),
            (Some(Delta::Reasoning(t)), _) => {
                v["choices"] = json!([{"index": 0, "delta": {"reasoning_content": t}, "finish_reason": null}])
            }
            (None, Some(s)) => {
                v["choices"] = json!([{"index": 0, "delta": {}, "finish_reason": finish(s)}]);
                v["timings"] = s.timings();
            }
            (None, None) => {}
        }
        v
    };
    let usage_chunk = move |s: &GenSummary| {
        let mut v = base.clone();
        v["choices"] = json!([]);
        v["usage"] = usage(s);
        v
    };
    Ok(sse(rx, Some(first), include_usage, chunk, usage_chunk).into_response())
}

pub async fn completions(State(st): State<Shared>, body: Bytes) -> ApiResult<Response> {
    let req: CompletionRequest = parse_json(&body)?;
    let worker = st.find(req.model.as_deref(), Capability::Completion)?;
    let model = worker.meta().id.clone();
    let prompts = req.prompt.into_vec();
    let [prompt] = <[String; 1]>::try_from(prompts).map_err(|_| ApiError::bad_request("exactly one prompt is supported"))?;
    let sampling = req.sampling.params()?;
    let rx = spawn_gen(worker, move |s, out| s.complete(&prompt, &sampling, out))?;
    let (id, created) = (new_id("cmpl"), now_secs());
    if !req.stream {
        let (content, _, s) = collect(rx).await?;
        return Ok(axum::Json(json!({
            "id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": content, "finish_reason": finish(&s), "logprobs": null}],
            "usage": usage(&s),
            "timings": s.timings(),
        }))
        .into_response());
    }
    let include_usage = req.stream_options.unwrap_or_default().include_usage;
    let base = json!({"id": id, "object": "text_completion", "created": created, "model": model});
    let b1 = base.clone();
    let chunk = move |d: Option<&Delta>, s: Option<&GenSummary>| {
        let mut v = b1.clone();
        match (d, s) {
            (Some(Delta::Content(t) | Delta::Reasoning(t)), _) => {
                v["choices"] = json!([{"index": 0, "text": t, "finish_reason": null, "logprobs": null}])
            }
            (None, Some(s)) => {
                v["choices"] = json!([{"index": 0, "text": "", "finish_reason": finish(s), "logprobs": null}]);
                v["timings"] = s.timings();
            }
            (None, None) => {}
        }
        v
    };
    let usage_chunk = move |s: &GenSummary| {
        let mut v = base.clone();
        v["choices"] = json!([]);
        v["usage"] = usage(s);
        v
    };
    Ok(sse(rx, None, include_usage, chunk, usage_chunk).into_response())
}
