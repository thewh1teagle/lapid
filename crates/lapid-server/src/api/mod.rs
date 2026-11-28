//! HTTP layer: the axum router, shared state, OpenAI-style errors and model lookup.

mod audio;
mod chat;
mod misc;
mod openapi;
mod realtime;
mod dsml;

use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};
use tower_http::compression::CompressionLayer;
use tower_http::compression::predicate::{DefaultPredicate, NotForContentType, Predicate};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use lapid_hub::runtime::serve::Capability;
use crate::worker::Worker;

/// Shared server state.
pub struct AppState {
    pub workers: Vec<Worker>,
    pub started: Instant,
    /// The device models run on.
    pub device: String,
    pub devices: Vec<String>,
}

pub type Shared = Arc<AppState>;

/// An OpenAI-style error response.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub code: String,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> ApiError {
        ApiError { status: StatusCode::BAD_REQUEST, kind: "invalid_request_error", code: "invalid_argument".into(), message: message.into() }
    }

    pub fn not_found(code: &str, message: impl Into<String>) -> ApiError {
        ApiError { status: StatusCode::NOT_FOUND, kind: "invalid_request_error", code: code.into(), message: message.into() }
    }

    pub fn internal(message: impl Into<String>) -> ApiError {
        ApiError {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            kind: "internal_error",
            code: "internal_error".into(),
            message: message.into(),
        }
    }

    pub fn body(&self) -> Value {
        json!({"error": {"message": self.message, "type": self.kind, "code": self.code, "param": null}})
    }
}

impl From<lapid::Error> for ApiError {
    fn from(e: lapid::Error) -> ApiError {
        let (status, kind) = match e.code {
            "invalid_argument" | "unsupported" | "unsupported_dtype" | "wav" | "audio" | "tokenizer" => {
                (StatusCode::BAD_REQUEST, "invalid_request_error")
            }
            "not_found" | "internal_error" => (StatusCode::NOT_FOUND, "invalid_request_error"),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        };
        ApiError { status, kind, code: e.code.to_string(), message: e.message }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, axum::Json(self.body())).into_response()
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

impl AppState {
    /// The worker for `model` that has `cap`. A missing or unknown model name falls back to
    /// the only loaded model with `cap`, so single-model servers accept any name.
    pub fn find(&self, model: Option<&str>, cap: Capability) -> ApiResult<&Worker> {
        let able: Vec<&Worker> = self.workers.iter().filter(|w| w.meta().has(cap)).collect();
        if let Some(name) = model.filter(|m| !m.is_empty()) {
            if let Some(w) = self.workers.iter().find(|w| w.meta().id == name || w.meta().name == name) {
                if !w.meta().has(cap) {
                    return Err(ApiError::bad_request(format!("model {name:?} does not support {}", cap_name(cap))));
                }
                return Ok(w);
            }
            if able.len() == 1 {
                return Ok(able[0]);
            }
            let ids: Vec<&str> = able.iter().map(|w| w.meta().id.as_str()).collect();
            return Err(ApiError::not_found(
                "model_not_found",
                format!("model {name:?} is not loaded (models with {}: {})", cap_name(cap), ids.join(", ")),
            ));
        }
        match able.as_slice() {
            [w] => Ok(w),
            [] => Err(ApiError::not_found("model_not_found", format!("no loaded model supports {}", cap_name(cap)))),
            _ => Err(ApiError::bad_request("\"model\" is required when several models are loaded")),
        }
    }
}

fn cap_name(cap: Capability) -> &'static str {
    match cap {
        Capability::Chat => "chat completions",
        Capability::Completion => "text completions",
        Capability::Embedding => "embeddings",
        Capability::Transcription => "transcription",
        Capability::Speech => "speech",
    }
}

/// Parse a JSON request body with OpenAI-style errors.
pub fn parse_json<T: serde::de::DeserializeOwned>(body: &[u8]) -> ApiResult<T> {
    serde_json::from_slice(body).map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// A unique-enough response id ("chatcmpl-…").
pub fn new_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    format!("{prefix}-{:x}{:04x}", t, n & 0xffff)
}

pub fn router(state: Shared) -> Router {
    let compress = CompressionLayer::new().compress_when(DefaultPredicate::new().and(NotForContentType::const_new("audio/")));
    Router::new()
        .route("/health", get(misc::health))
        .route("/openapi.json", get(openapi::spec))
        .route("/docs", get(openapi::docs))
        .route("/api/info", get(misc::info))
        .route("/v1/models", get(misc::models))
        .route("/v1/models/{id}", get(misc::model))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(chat::completions))
        .route("/v1/embeddings", post(misc::embeddings))
        .route("/v1/audio/transcriptions", post(audio::transcriptions))
        .route("/v1/audio/translations", post(audio::translations))
        .route("/v1/audio/speech", post(audio::speech))
        .route("/v1/realtime", get(realtime::realtime))
        .fallback(crate::assets::serve_asset)
        .layer(DefaultBodyLimit::max(512 << 20))
        .layer(compress)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
