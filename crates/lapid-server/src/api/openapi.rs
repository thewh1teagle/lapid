//! `/openapi.json` (OpenAPI 3.1 description of the routes in [`super::router`]) and `/docs`
//! (an API reference page rendering it). Keep the spec in step with the request structs.

use axum::Json;
use axum::response::Html;
use serde_json::{Value, json};

pub async fn spec() -> Json<Value> {
    Json(build())
}

pub async fn docs() -> Html<&'static str> {
    Html(DOCS)
}

const DOCS: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>lapid API</title>
</head>
<body>
<noscript>The API description is at <a href="/openapi.json">/openapi.json</a>.</noscript>
<script id="api-reference" data-url="/openapi.json"></script>
<script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference@1.25.0"></script>
</body>
</html>
"#;

fn ok(description: &str, content_type: &str, schema: Value) -> Value {
    json!({"200": {"description": description, "content": {content_type: {"schema": schema}}}})
}

fn json_body(schema: Value) -> Value {
    json!({"required": true, "content": {"application/json": {"schema": schema}}})
}

fn build() -> Value {
    let model = json!({"type": "string", "description": "Model id from /v1/models (default: the first model that can do the task)"});
    let sampling = json!({
        "temperature": {"type": "number"},
        "top_p": {"type": "number"},
        "top_k": {"type": "integer"},
        "max_tokens": {"type": "integer"},
        "max_completion_tokens": {"type": "integer"},
        "stop": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
        "seed": {"type": "integer"},
        "repetition_penalty": {"type": "number"},
        "n": {"type": "integer", "enum": [1]},
        "steer": {"oneOf": [{"type": "object", "additionalProperties": {"type": "number"}}, {"type": "boolean", "enum": [false]}], "description": "lapid: steering files (loaded with --steer) by name with scale multipliers, e.g. {\"deepseek41-heretic\": 1}; {} or false = none; omitted = the server's default (docs/STEERING.md)"},
        "stream": {"type": "boolean", "description": "Server-sent events"},
        "stream_options": {"type": "object", "properties": {"include_usage": {"type": "boolean"}}}
    });
    let with = |extra: Value, required: &[&str]| {
        let mut props = sampling.as_object().unwrap().clone();
        props.insert("model".into(), model.clone());
        props.extend(extra.as_object().unwrap().clone());
        json!({"type": "object", "required": required, "properties": props})
    };
    let chat = with(
        json!({
            "messages": {"type": "array", "items": {"type": "object", "required": ["role"], "properties": {
                "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
                "content": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]}
            }}},
            "tools": {"type": "array", "items": {"type": "object"}},
            "chat_template_kwargs": {"type": "object", "description": "Passed to the chat template, e.g. {\"enable_thinking\": false}"}
        }),
        &["messages"],
    );
    let completion = with(json!({"prompt": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}}), &["prompt"]);
    let object = json!({"type": "object"});
    let transcription_form = json!({"type": "object", "required": ["file"], "properties": {
        "file": {"type": "string", "format": "binary", "description": "Audio file (WAV)"},
        "model": model,
        "language": {"type": "string"},
        "response_format": {"type": "string", "enum": ["json", "text", "srt", "vtt", "verbose_json"]},
        "timestamp_granularities[]": {"type": "array", "items": {"type": "string", "enum": ["segment", "word"]}}
    }});
    let audio_op = |summary: &str| json!({"post": {
        "summary": summary, "tags": ["Audio"],
        "requestBody": {"required": true, "content": {"multipart/form-data": {"schema": transcription_form}}},
        "responses": ok("Transcript", "application/json", json!({"type": "object"}))
    }});
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "lapid",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "OpenAI-compatible API served by `lapid serve`."
        },
        "paths": {
            "/health": {"get": {"summary": "Liveness check", "tags": ["Server"], "responses": ok("ok", "application/json", object.clone())}},
            "/api/info": {"get": {"summary": "Server, devices and loaded models", "tags": ["Server"], "responses": ok("Info", "application/json", object.clone())}},
            "/v1/models": {"get": {"summary": "List models", "tags": ["Models"], "responses": ok("Models", "application/json", object.clone())}},
            "/v1/models/{id}": {"get": {
                "summary": "Get a model", "tags": ["Models"],
                "parameters": [{"name": "id", "in": "path", "required": true, "schema": {"type": "string"}}],
                "responses": ok("Model", "application/json", object.clone())
            }},
            "/v1/chat/completions": {"post": {
                "summary": "Chat completion", "tags": ["Text"],
                "requestBody": json_body(chat), "responses": ok("Completion, or an SSE stream when stream is true", "application/json", object.clone())
            }},
            "/v1/completions": {"post": {
                "summary": "Text completion", "tags": ["Text"],
                "requestBody": json_body(completion), "responses": ok("Completion", "application/json", object.clone())
            }},
            "/v1/embeddings": {"post": {
                "summary": "Embeddings", "tags": ["Text"],
                "requestBody": json_body(json!({"type": "object", "required": ["input"], "properties": {
                    "model": model,
                    "input": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                    "dimensions": {"type": "integer", "description": "Truncate (Matryoshka models)"},
                    "encoding_format": {"type": "string", "enum": ["float", "base64"]},
                    "prompt_name": {"type": "string", "description": "Prompt from the model's table, e.g. query or document"}
                }})),
                "responses": ok("Embeddings", "application/json", object.clone())
            }},
            "/v1/audio/transcriptions": audio_op("Speech to text"),
            "/v1/audio/translations": audio_op("Speech to English text"),
            "/v1/audio/speech": {"post": {
                "summary": "Text to speech", "tags": ["Audio"],
                "requestBody": json_body(json!({"type": "object", "required": ["input"], "properties": {
                    "model": model,
                    "input": {"type": "string"},
                    "voice": {"type": "string"},
                    "response_format": {"type": "string", "enum": ["wav", "pcm"]},
                    "stream_format": {"type": "string", "enum": ["audio", "sse"]},
                    "seed": {"type": "integer"},
                    "temperature": {"type": "number"}
                }})),
                "responses": ok("Audio", "audio/wav", json!({"type": "string", "format": "binary"}))
            }}
        }
    })
}
