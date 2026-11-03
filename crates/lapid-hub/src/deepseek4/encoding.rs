//! DeepSeek V4 prompt encoding: a port of the upstream `encoding/encoding_dsv4.py`
//! (`encode_messages`). V4 ships no Jinja chat template; this is the reference format.
//!
//! Supported: roles system / user / assistant / tool / developer / latest_reminder, tools on
//! the system (or developer) message (OpenAI format, DSML tool-call rendering), tool results
//! merged into user turns (`merge_tool_messages`, sorted by call order), `reasoning_content`,
//! chat vs thinking mode with `drop_thinking`, `response_format`, `wo_eos`. Not supported:
//! the internal classification `task` tokens, reasoning-effort prefixes and `context`.

use lapid::{Error, Result};

use crate::runtime::jinja::{Template, Value};

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const LATEST_REMINDER: &str = "<｜latest_reminder｜>";
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const DSML: &str = "｜DSML｜";

const TOOLS_TEMPLATE: &str = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{dsml}tool_calls>\" block like the following:\n\n<{dsml}tool_calls>\n<{dsml}invoke name=\"$TOOL_NAME\">\n<{dsml}parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{dsml}parameter>\n...\n</{dsml}invoke>\n<{dsml}invoke name=\"$TOOL_NAME2\">\n...\n</{dsml}invoke>\n</{dsml}tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{schemas}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

fn err(m: impl Into<String>) -> Error {
    Error::new("template", format!("deepseek4 encoding: {}", m.into()))
}

/// `json.dumps(v, ensure_ascii=False)` through the Jinja `tojson` filter.
pub(crate) fn to_json(v: &Value) -> Result<String> {
    Template::parse("{{ v | tojson }}")?.render(&[("v", v.clone())])
}

pub(crate) fn get<'a>(m: &'a Value, key: &str) -> Option<&'a Value> {
    match m {
        Value::Map(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

pub(crate) fn set(m: &mut Value, key: &str, v: Value) {
    if let Value::Map(pairs) = m {
        match pairs.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = v,
            None => pairs.push((key.to_string(), v)),
        }
    }
}

pub(crate) fn remove(m: &mut Value, key: &str) {
    if let Value::Map(pairs) = m {
        pairs.retain(|(k, _)| k != key);
    }
}

pub(crate) fn s(v: Option<&Value>) -> String {
    match v {
        Some(Value::Str(x)) => x.clone(),
        Some(Value::None) | Some(Value::Undefined) | None => String::new(),
        Some(other) => to_json(other).unwrap_or_default(),
    }
}

pub(crate) fn role(m: &Value) -> String {
    s(get(m, "role"))
}

pub(crate) fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::None) | Some(Value::Undefined) | Some(Value::Bool(false)) => false,
        Some(Value::Str(x)) => !x.is_empty(),
        Some(Value::List(x)) => !x.is_empty(),
        Some(Value::Map(x)) => !x.is_empty(),
        _ => true,
    }
}

pub(crate) fn map(pairs: Vec<(&str, Value)>) -> Value {
    Value::Map(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

pub(crate) fn list(v: Option<&Value>) -> &[Value] {
    match v {
        Some(Value::List(x)) => x,
        _ => &[],
    }
}

fn render_tools(tools: &[Value]) -> Result<String> {
    let schemas: Vec<String> = tools.iter().map(|t| to_json(get(t, "function").unwrap_or(t))).collect::<Result<_>>()?;
    Ok(TOOLS_TEMPLATE.replace("{dsml}", DSML).replace("{schemas}", &schemas.join("\n")))
}

fn encode_arguments(args: Option<&Value>) -> Result<String> {
    let parsed = match args {
        Some(Value::Str(raw)) => match Value::from_json_str(raw) {
            Ok(m @ Value::Map(_)) => m,
            _ => map(vec![("arguments", Value::Str(raw.clone()))]),
        },
        Some(other) => map(vec![("arguments", other.clone())]),
        None => map(vec![("arguments", Value::None)]),
    };
    let Value::Map(pairs) = parsed else { return Err(err("tool arguments")) };
    let mut out = Vec::new();
    for (k, v) in &pairs {
        let (is_str, value) = match v {
            Value::Str(x) => ("true", x.clone()),
            other => ("false", to_json(other)?),
        };
        out.push(format!("<{DSML}parameter name=\"{k}\" string=\"{is_str}\">{value}</{DSML}parameter>"));
    }
    Ok(out.join("\n"))
}

fn merge_tool_messages(messages: &[Value]) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::new();
    for msg in messages {
        let r = role(msg);
        let last_user_blocks = |m: &Vec<Value>| m.last().is_some_and(|l| role(l) == "user" && get(l, "content_blocks").is_some());
        if r == "tool" {
            let block = map(vec![
                ("type", Value::str("tool_result")),
                ("tool_use_id", Value::str(s(get(msg, "tool_call_id")))),
                ("content", get(msg, "content").cloned().unwrap_or(Value::str(""))),
            ]);
            if last_user_blocks(&merged) {
                if let Some(Value::Map(pairs)) = merged.last_mut()
                    && let Some((_, Value::List(blocks))) = pairs.iter_mut().find(|(k, _)| k == "content_blocks")
                {
                    blocks.push(block);
                }
            } else {
                merged.push(map(vec![("role", Value::str("user")), ("content_blocks", Value::List(vec![block]))]));
            }
        } else if r == "user" {
            let content = get(msg, "content").cloned().unwrap_or(Value::str(""));
            let block = map(vec![("type", Value::str("text")), ("text", content.clone())]);
            if last_user_blocks(&merged) && merged.last().is_some_and(|l| get(l, "task").is_none()) {
                if let Some(Value::Map(pairs)) = merged.last_mut()
                    && let Some((_, Value::List(blocks))) = pairs.iter_mut().find(|(k, _)| k == "content_blocks")
                {
                    blocks.push(block);
                }
            } else {
                let mut m = map(vec![("role", Value::str("user")), ("content", content), ("content_blocks", Value::List(vec![block]))]);
                for key in ["task", "wo_eos", "mask"] {
                    if let Some(v) = get(msg, key) {
                        set(&mut m, key, v.clone());
                    }
                }
                merged.push(m);
            }
        } else {
            merged.push(msg.clone());
        }
    }
    merged
}

pub(crate) fn sort_tool_results(messages: &mut [Value]) {
    let mut order: Vec<(String, usize)> = Vec::new();
    for msg in messages.iter_mut() {
        let r = role(msg);
        if r == "assistant" && truthy(get(msg, "tool_calls")) {
            order.clear();
            for (i, tc) in list(get(msg, "tool_calls")).iter().enumerate() {
                let mut id = s(get(tc, "id"));
                if id.is_empty() {
                    id = s(get(tc, "function").and_then(|f| get(f, "id")));
                }
                if !id.is_empty() {
                    order.push((id, i));
                }
            }
        } else if r == "user" && !order.is_empty() {
            let blocks = list(get(msg, "content_blocks")).to_vec();
            let mut tools: Vec<Value> = blocks.iter().filter(|b| s(get(b, "type")) == "tool_result").cloned().collect();
            if tools.len() < 2 {
                continue;
            }
            let rank = |b: &Value| order.iter().find(|(id, _)| *id == s(get(b, "tool_use_id"))).map_or(0, |(_, i)| *i);
            tools.sort_by_key(rank);
            let mut it = tools.into_iter();
            let sorted: Vec<Value> = blocks
                .into_iter()
                .map(|b| if s(get(&b, "type")) == "tool_result" { it.next().unwrap_or(b) } else { b })
                .collect();
            set(msg, "content_blocks", Value::List(sorted));
        }
    }
}

fn last_user_index(messages: &[Value]) -> Option<usize> {
    messages.iter().rposition(|m| matches!(role(m).as_str(), "user" | "developer"))
}

fn drop_thinking_messages(messages: Vec<Value>) -> Vec<Value> {
    let last = last_user_index(&messages);
    drop_thinking_before(messages, last)
}

/// Drop `reasoning_content` from assistant turns before `last` (the last user turn) and any
/// message outside the kept roles there; `None` keeps everything.
pub(crate) fn drop_thinking_before(messages: Vec<Value>, last: Option<usize>) -> Vec<Value> {
    let mut out = Vec::new();
    for (i, mut m) in messages.into_iter().enumerate() {
        let r = role(&m);
        let after = last.is_none_or(|l| i >= l);
        if matches!(r.as_str(), "user" | "system" | "tool" | "latest_reminder" | "direct_search_results") || after {
            out.push(m);
        } else if r == "assistant" {
            remove(&mut m, "reasoning_content");
            out.push(m);
        }
    }
    out
}

pub(crate) fn render_user(msg: &Value) -> String {
    let blocks = list(get(msg, "content_blocks"));
    if blocks.is_empty() {
        return s(get(msg, "content"));
    }
    let parts: Vec<String> = blocks
        .iter()
        .map(|b| match s(get(b, "type")).as_str() {
            "text" => s(get(b, "text")),
            "tool_result" => {
                let content = match get(b, "content") {
                    Some(Value::List(items)) => items
                        .iter()
                        .map(|x| if s(get(x, "type")) == "text" { s(get(x, "text")) } else { format!("[Unsupported {}]", s(get(x, "type"))) })
                        .collect::<Vec<_>>()
                        .join("\n\n"),
                    other => s(other),
                };
                format!("<tool_result>{content}</tool_result>")
            }
            other => format!("[Unsupported {other}]"),
        })
        .collect();
    parts.join("\n\n")
}

fn render_message(index: usize, messages: &[Value], thinking: bool, drop: bool) -> Result<String> {
    let msg = &messages[index];
    let last_user = last_user_index(messages);
    let r = role(msg);
    let mut prompt = String::new();
    let extras = |p: &mut String| -> Result<()> {
        let tools = list(get(msg, "tools"));
        if !tools.is_empty() {
            p.push_str("\n\n");
            p.push_str(&render_tools(tools)?);
        }
        if let Some(rf) = get(msg, "response_format").filter(|v| truthy(Some(v))) {
            p.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
            p.push_str(&to_json(rf)?);
        }
        Ok(())
    };
    match r.as_str() {
        "system" => {
            prompt.push_str(&s(get(msg, "content")));
            extras(&mut prompt)?;
        }
        "developer" => {
            prompt.push_str(USER);
            prompt.push_str(&s(get(msg, "content")));
            extras(&mut prompt)?;
        }
        "user" => {
            prompt.push_str(USER);
            prompt.push_str(&render_user(msg));
        }
        "latest_reminder" => {
            prompt.push_str(LATEST_REMINDER);
            prompt.push_str(&s(get(msg, "content")));
        }
        "assistant" => {
            let mut tc = String::new();
            let calls = list(get(msg, "tool_calls"));
            if !calls.is_empty() {
                let mut items = Vec::new();
                for c in calls {
                    let f = get(c, "function").unwrap_or(c);
                    let args = encode_arguments(get(f, "arguments"))?;
                    items.push(format!("<{DSML}invoke name=\"{}\">\n{args}\n</{DSML}invoke>", s(get(f, "name"))));
                }
                tc = format!("\n\n<{DSML}tool_calls>\n{}\n</{DSML}tool_calls>", items.join("\n"));
            }
            if thinking && (!drop || last_user.is_none_or(|l| index > l)) {
                prompt.push_str(&s(get(msg, "reasoning_content")));
                prompt.push_str(THINK_END);
            }
            prompt.push_str(&s(get(msg, "content")));
            prompt.push_str(&tc);
            if !truthy(get(msg, "wo_eos")) {
                prompt.push_str(EOS);
            }
        }
        "tool" => return Err(err("tool messages must be merged into user turns")),
        other => return Err(err(format!("unknown role {other:?}"))),
    }
    if get(msg, "task").is_some() {
        return Err(err("classification tasks are not supported"));
    }
    if index + 1 < messages.len() && !matches!(role(&messages[index + 1]).as_str(), "assistant" | "latest_reminder") {
        return Ok(prompt);
    }
    if matches!(r.as_str(), "user" | "developer") {
        prompt.push_str(ASSISTANT);
        let open = thinking && (!drop || last_user.is_none_or(|l| index >= l));
        prompt.push_str(if open { THINK_START } else { THINK_END });
    }
    Ok(prompt)
}

/// `encode_messages(messages, thinking_mode, drop_thinking=True)` for a JSON list of messages.
pub fn encode(messages: &Value, thinking: bool) -> Result<String> {
    let Value::List(items) = messages else { return Err(err("messages must be a list")) };
    let mut full = merge_tool_messages(items);
    sort_tool_results(&mut full);
    let drop = !full.iter().any(|m| truthy(get(m, "tools")));
    if thinking && drop {
        full = drop_thinking_messages(full);
    }
    let mut prompt = BOS.to_string();
    for i in 0..full.len() {
        prompt.push_str(&render_message(i, &full, thinking, drop)?);
    }
    Ok(prompt)
}

/// Render messages given as JSON (OpenAI style; key order kept); `tools_json` (a list of
/// OpenAI tools) goes on the system message, where upstream V4 carries tools.
pub fn render_json(messages_json: &str, tools_json: Option<&str>, thinking: bool) -> Result<String> {
    let mut messages = Value::from_json_str(messages_json)?;
    if let Some(t) = tools_json {
        messages = with_tools(messages, Value::from_json_str(t)?);
    }
    encode(&messages, thinking)
}

/// Attach OpenAI-format `tools` (JSON list) to the first system message (inserting an empty
/// one at the front when there is none): upstream V4 carries tools on the system message.
pub fn with_tools(messages: Value, tools: Value) -> Value {
    let Value::List(mut items) = messages else { return messages };
    if !truthy(Some(&tools)) {
        return Value::List(items);
    }
    match items.iter_mut().find(|m| role(m) == "system") {
        Some(sys) => set(sys, "tools", tools),
        None => items.insert(0, map(vec![("role", Value::str("system")), ("content", Value::str("")), ("tools", tools)])),
    }
    Value::List(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(json: &str, thinking: bool) -> String {
        encode(&Value::from_json_str(json).expect("json"), thinking).expect("encode")
    }

    #[test]
    fn basic_chat() {
        let j = r#"[{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"}]"#;
        assert_eq!(enc(j, false), format!("{BOS}S{USER}Hi{ASSISTANT}{THINK_END}"));
        assert_eq!(enc(j, true), format!("{BOS}S{USER}Hi{ASSISTANT}{THINK_START}"));
    }

    #[test]
    fn drops_old_reasoning() {
        let j = r#"[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "r"}, {"role": "user", "content": "c"}]"#;
        assert_eq!(enc(j, true), format!("{BOS}{USER}a{ASSISTANT}{THINK_END}b{EOS}{USER}c{ASSISTANT}{THINK_START}"));
    }
}
