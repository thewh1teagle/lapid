//! DeepSeek V4.1 prompt encoding: a port of the upstream `encoding/encoding.py`
//! (`encode_messages`, text path). Like V4 there is no Jinja chat template.
//!
//! V4.1 vs V4: DSML tags with a leading space (`<｜DSML｜ calls>`, ` invoke`, ` parameter`),
//! a numeric reasoning-effort prefix behind `<｜System｜>` (thinking mode, index 0 only),
//! mid-conversation system messages (`<｜System｜>`, they open an assistant turn like a user
//! message), tool namespaces (`ns::name`), tool-result-only user turns not counting as a new
//! turn for `drop_thinking`, structured (list) content, and no `developer` role.
//!
//! Supported: roles system / user / assistant / tool / latest_reminder, tools on system
//! messages, tool results merged into user turns and sorted by call order, quick-instruction
//! `task` tokens, `drop_thinking`, `response_format`, `wo_eos`. Not supported: image content
//! blocks (rejected with code `unsupported`) and `context` / `add_default_bos_token`.

use lapid::{Error, Result};

use crate::deepseek4::encoding::{
    drop_thinking_before, get, list, map, remove, render_user, role, s, set, sort_tool_results, to_json, truthy,
};
use crate::runtime::jinja::Value;

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
pub const USER: &str = "<｜User｜>";
pub const ASSISTANT: &str = "<｜Assistant｜>";
pub const SYSTEM: &str = "<｜System｜>";
pub const LATEST_REMINDER: &str = "<｜latest_reminder｜>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const DSML: &str = "｜DSML｜";
/// Image placeholder of the vision path; rejected in text content here.
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";
/// Budget used when no reasoning effort is given (`"high"`).
pub const DEFAULT_REASONING_EFFORT: i64 = 75;

const CALLS: &str = "<｜DSML｜ calls>";
const CALLS_END: &str = "</｜DSML｜ calls>";
const INVOKE: &str = "｜DSML｜ invoke";
const PARAMETER: &str = "｜DSML｜ parameter";

const TASKS: [(&str, &str); 6] = [
    ("action", "<｜action｜>"),
    ("query", "<｜query｜>"),
    ("authority", "<｜authority｜>"),
    ("domain", "<｜domain｜>"),
    ("title", "<｜title｜>"),
    ("read_url", "<｜read_url｜>"),
];

const TOOLS_HEAD: &str = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n";
const TOOLS_TAIL: &str = "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

fn err(m: impl Into<String>) -> Error {
    Error::new("template", format!("deepseek41 encoding: {}", m.into()))
}

/// Python's `x is not None` for a dict lookup.
fn present(v: Option<&Value>) -> Option<&Value> {
    v.filter(|x| !matches!(x, Value::None | Value::Undefined))
}

/// Resolve a reasoning effort (`None`, an int in `[1, 100]`, or `"low"`/`"high"`/`"max"`) to
/// its numeric budget.
pub fn reasoning_budget(effort: Option<&Value>) -> Result<i64> {
    match present(effort) {
        None => Ok(DEFAULT_REASONING_EFFORT),
        Some(Value::Int(n)) if (1..=100).contains(n) => Ok(*n),
        Some(Value::Str(x)) if x == "low" => Ok(50),
        Some(Value::Str(x)) if x == "high" => Ok(75),
        Some(Value::Str(x)) if x == "max" => Ok(100),
        Some(other) => Err(err(format!(
            "invalid reasoning effort {}, should be int within [1,100] or [\"low\", \"high\", \"max\"]",
            to_json(other).unwrap_or_default()
        ))),
    }
}

// ---- namespaces and tools ----

/// A namespace value as a name: a dict contributes its `name`.
fn namespace_name(v: Option<&Value>) -> Option<String> {
    match present(v)? {
        ns @ Value::Map(_) => Some(s(get(ns, "name"))),
        other => Some(s(Some(other))),
    }
}

/// `_split_tool_name`: split `ns::name` and check it against an explicit namespace.
fn split_tool_name(name: &str, namespace: Option<String>) -> Result<(Option<String>, String)> {
    let (mut ns, mut bare) = (namespace, name.to_string());
    if let Some((prefix, rest)) = name.split_once("::") {
        if ns.as_deref().is_some_and(|n| n != prefix) {
            return Err(err(format!("Conflicting tool namespaces: {} != {prefix}", ns.unwrap_or_default())));
        }
        (ns, bare) = (Some(prefix.to_string()), rest.to_string());
    }
    if bare.contains("::") {
        return Err(err(format!("Tool name must not contain '::': {bare}")));
    }
    if let Some(n) = ns.as_deref().filter(|n| n.contains("::")) {
        return Err(err(format!("Tool namespace must not contain '::': {n}")));
    }
    Ok((ns, bare))
}

fn qualified_name(name: &str, namespace: Option<String>) -> Result<String> {
    Ok(match split_tool_name(name, namespace)? {
        (Some(ns), bare) => format!("{ns}::{bare}"),
        (None, bare) => bare,
    })
}

/// `tools_from_openai_format` for one tool: the function schema with a qualified name and the
/// namespace description prepended.
fn tool_schema(tool: &Value) -> Result<Value> {
    let mut f = get(tool, "function").filter(|f| matches!(f, Value::Map(_))).cloned().ok_or_else(|| err("tool without a function"))?;
    if let Some(ns) = present(get(tool, "namespace")) {
        set(&mut f, "namespace", ns.clone());
    }
    let ns = present(get(&f, "namespace")).cloned();
    let name = qualified_name(&s(get(&f, "name")), namespace_name(ns.as_ref()))?;
    set(&mut f, "name", Value::Str(name));
    remove(&mut f, "namespace");
    if let Some(ns @ Value::Map(_)) = &ns
        && truthy(get(ns, "description"))
    {
        let description = format!("{}\n{}", s(get(ns, "description")), s(get(&f, "description")));
        set(&mut f, "description", Value::Str(description));
    }
    Ok(f)
}

fn render_tools(schemas: &[Value]) -> Result<String> {
    let json: Vec<String> = schemas.iter().map(to_json).collect::<Result<_>>()?;
    Ok(format!("{TOOLS_HEAD}{}{TOOLS_TAIL}", json.join("\n")))
}

/// `encode_arguments_to_dsml`: arguments as a dict, a JSON string (possibly double-encoded),
/// or anything else wrapped as `{"arguments": ...}`.
fn encode_arguments(raw: &Value) -> Result<String> {
    let mut args = raw.clone();
    for _ in 0..2 {
        let Value::Str(text) = &args else { break };
        match Value::from_json_str(text) {
            Ok(v) => args = v,
            Err(_) => break,
        }
    }
    let pairs = match args {
        Value::Map(pairs) => pairs,
        _ => vec![("arguments".to_string(), raw.clone())],
    };
    let mut out = Vec::new();
    for (k, v) in &pairs {
        let (is_str, value) = match v {
            Value::Str(x) => ("true", x.clone()),
            other => ("false", to_json(other)?),
        };
        out.push(format!("<{PARAMETER} name=\"{k}\" string=\"{is_str}\">{value}</{PARAMETER}>"));
    }
    Ok(out.join("\n"))
}

/// One OpenAI tool call rendered as a DSML `invoke` block.
fn render_tool_call(call: &Value) -> Result<String> {
    let f = get(call, "function").ok_or_else(|| err("tool call without a function"))?;
    let ns = if truthy(get(call, "namespace")) { get(call, "namespace") } else { get(f, "namespace") };
    let (ns, name) = split_tool_name(&s(get(f, "name")), namespace_name(ns))?;
    let args = get(f, "arguments").ok_or_else(|| err("tool call without arguments"))?;
    let name = qualified_name(&name, ns)?;
    Ok(format!("<{INVOKE} name=\"{name}\">\n{}\n</{INVOKE}>", encode_arguments(args)?))
}

// ---- preprocessing ----

/// `_validate_no_image_sp_tokens`.
fn check_no_placeholder(msg: &Value) -> Result<()> {
    for key in ["content", "reasoning_content"] {
        if let Some(Value::Str(x)) = get(msg, key)
            && x.contains(IMAGE_PLACEHOLDER)
        {
            return Err(err(format!("{key} contains the image special token {IMAGE_PLACEHOLDER}")));
        }
    }
    Ok(())
}

/// The text side of `_process_image_blocks`: image blocks are rejected (text-only port).
fn check_blocks(blocks: &[Value]) -> Result<()> {
    for b in blocks {
        match s(get(b, "type")).as_str() {
            "image" | "image_url" => {
                return Err(Error::new("unsupported", "deepseek41 encoding: image content blocks are not supported"));
            }
            "tool_result" => check_blocks(list(get(b, "content")))?,
            "text" if s(get(b, "text")).contains(IMAGE_PLACEHOLDER) => {
                return Err(err(format!("text block contains the image placeholder {IMAGE_PLACEHOLDER}")));
            }
            _ => {}
        }
    }
    Ok(())
}

/// `process_image_messages` for one message: list content becomes `content_blocks`, and a
/// non-string `content` is replaced by the joined text blocks.
fn normalize_content(msg: &mut Value) -> Result<()> {
    check_no_placeholder(msg)?;
    if let Some(content @ Value::List(_)) = get(msg, "content").cloned()
        && get(msg, "content_blocks").is_none()
    {
        remove(msg, "content");
        set(msg, "content_blocks", content);
    }
    let blocks = list(get(msg, "content_blocks")).to_vec();
    if !blocks.is_empty() {
        check_blocks(&blocks)?;
        if !matches!(get(msg, "content"), Some(Value::Str(_))) {
            let texts: Vec<String> = blocks.iter().filter(|b| s(get(b, "type")) == "text").map(|b| s(get(b, "text"))).collect();
            set(msg, "content", Value::Str(texts.join("\n\n")));
        }
    } else if matches!(get(msg, "content"), Some(Value::List(_))) {
        // An empty list renders as `content or ""`.
        set(msg, "content", Value::str(""));
    }
    Ok(())
}

fn append_blocks(msg: &mut Value, extra: Vec<Value>) -> Result<()> {
    if let Value::Map(pairs) = msg
        && let Some((_, Value::List(blocks))) = pairs.iter_mut().find(|(k, _)| k == "content_blocks")
    {
        blocks.extend(extra);
        return Ok(());
    }
    Err(err("content_blocks must be a list"))
}

/// `merge_tool_messages`: tool messages become `tool_result` blocks of a user turn; user
/// messages get `content_blocks` (and keep all their metadata).
fn merge_tool_messages(messages: &[Value]) -> Result<Vec<Value>> {
    let mut merged: Vec<Value> = Vec::new();
    for msg in messages {
        let r = role(msg);
        let last_user = merged.last().filter(|l| role(l) == "user" && get(l, "content_blocks").is_some());
        let merge_user = last_user.is_some_and(|l| present(get(l, "task")).is_none());
        let merge_tool = last_user.is_some();
        if r == "tool" {
            let mut content = get(msg, "content").cloned().unwrap_or(Value::str(""));
            if truthy(get(msg, "content_blocks")) && !matches!(get(msg, "content"), Some(Value::Str(_))) {
                content = get(msg, "content_blocks").cloned().unwrap_or(Value::None);
            }
            let block = map(vec![
                ("type", Value::str("tool_result")),
                ("tool_use_id", get(msg, "tool_call_id").cloned().unwrap_or(Value::str(""))),
                ("content", content),
            ]);
            match merged.last_mut() {
                Some(last) if merge_tool => append_blocks(last, vec![block])?,
                _ => merged.push(map(vec![("role", Value::str("user")), ("content_blocks", Value::List(vec![block]))])),
            }
        } else if r == "user" {
            let blocks = match present(get(msg, "content_blocks")) {
                Some(Value::List(b)) => b.clone(),
                Some(_) => return Err(err("content_blocks must be a list")),
                None => match get(msg, "content") {
                    Some(Value::List(b)) => b.clone(),
                    c => vec![map(vec![("type", Value::str("text")), ("text", c.cloned().unwrap_or(Value::str("")))])],
                },
            };
            match merged.last_mut() {
                Some(last) if merge_user => append_blocks(last, blocks)?,
                _ => {
                    let mut m = msg.clone();
                    set(&mut m, "content_blocks", Value::List(blocks));
                    merged.push(m);
                }
            }
        } else {
            merged.push(msg.clone());
        }
    }
    Ok(merged)
}

/// `find_last_user_index`: the last user (or mid-conversation system) message that is not
/// only tool results.
fn last_user_index(messages: &[Value]) -> Option<usize> {
    (0..messages.len()).rev().find(|&i| {
        let (m, r) = (&messages[i], role(&messages[i]));
        let blocks = list(get(m, "content_blocks"));
        (r == "user" || (r == "system" && i > 0))
            && (blocks.is_empty() || blocks.iter().any(|b| matches!(s(get(b, "type")).as_str(), "text" | "image" | "image_url")))
    })
}

// ---- rendering ----

fn render_message(index: usize, messages: &[Value], thinking: bool, drop: bool, budget: i64) -> Result<String> {
    let msg = &messages[index];
    let last_user = last_user_index(messages);
    let r = role(msg);
    let schemas: Vec<Value> = list(get(msg, "tools")).iter().map(tool_schema).collect::<Result<_>>()?;
    let calls: Vec<String> = list(get(msg, "tool_calls")).iter().map(render_tool_call).collect::<Result<_>>()?;

    let effort = index == 0 && thinking;
    let mut prompt = String::new();
    if index == 0 && (effort || r == "system") {
        prompt.push_str(SYSTEM);
    }
    if effort {
        prompt.push_str(&format!(
            "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        ));
    }
    match r.as_str() {
        "system" => {
            if index > 0 {
                prompt.push_str(SYSTEM);
            }
            prompt.push_str(&s(get(msg, "content")));
            if !schemas.is_empty() {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(&schemas)?);
            }
            if let Some(rf) = get(msg, "response_format").filter(|v| truthy(Some(v))) {
                prompt.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                prompt.push_str(&to_json(rf)?);
            }
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
            let prev_has_task = index > 0 && present(get(&messages[index - 1], "task")).is_some();
            if thinking && !prev_has_task && (!drop || last_user.is_none_or(|l| index > l)) {
                prompt.push_str(&s(get(msg, "reasoning_content")));
                prompt.push_str(THINK_END);
            }
            prompt.push_str(&s(get(msg, "content")));
            if !calls.is_empty() {
                prompt.push_str(&format!("\n\n{CALLS}\n{}\n{CALLS_END}", calls.join("\n")));
            }
            if !truthy(get(msg, "wo_eos")) {
                prompt.push_str(EOS);
            }
        }
        "tool" => return Err(err("tool messages must be merged into user turns")),
        other => return Err(err(format!("unknown role {other:?}"))),
    }

    if index + 1 < messages.len() && !matches!(role(&messages[index + 1]).as_str(), "assistant" | "latest_reminder") {
        return Ok(prompt);
    }
    if let Some(task) = present(get(msg, "task")) {
        let name = match task {
            Value::Str(x) => x.as_str(),
            _ => "",
        };
        let Some((_, token)) = TASKS.iter().find(|(t, _)| *t == name) else {
            return Err(err(format!("invalid task {}", to_json(task).unwrap_or_default())));
        };
        if name == "action" {
            prompt.push_str(ASSISTANT);
            prompt.push_str(if thinking { THINK_START } else { THINK_END });
        }
        prompt.push_str(token);
    } else if r == "user" || (r == "system" && index > 0) {
        prompt.push_str(ASSISTANT);
        let open = thinking && (!drop || last_user.is_none_or(|l| index >= l));
        prompt.push_str(if open { THINK_START } else { THINK_END });
    }
    Ok(prompt)
}

/// `encode_messages(messages, thinking_mode, drop_thinking=True, reasoning_effort=...)` for a
/// list of OpenAI-style messages. `reasoning_effort` is an int in `[1, 100]` or
/// `"low"`/`"high"`/`"max"`; `None` means `"high"`. It is validated even in chat mode.
pub fn encode(messages: &Value, thinking: bool, reasoning_effort: Option<&Value>) -> Result<String> {
    let Value::List(items) = messages else { return Err(err("messages must be a list")) };
    let budget = reasoning_budget(reasoning_effort)?;
    for m in items {
        check_no_placeholder(m)?;
    }
    let mut full = merge_tool_messages(items)?;
    sort_tool_results(&mut full);
    for m in full.iter_mut() {
        normalize_content(m)?;
    }
    let drop = !full.iter().any(|m| truthy(get(m, "tools")));
    if thinking && drop {
        let last = last_user_index(&full);
        full = drop_thinking_before(full, last);
    }
    let mut prompt = BOS.to_string();
    for i in 0..full.len() {
        prompt.push_str(&render_message(i, &full, thinking, drop, budget)?);
    }
    Ok(prompt)
}

/// Render messages given as JSON (OpenAI style; key order kept). `tools_json` (a list of
/// OpenAI tools) goes on the leading system message ([`with_tools`]). `reasoning_effort` is
/// `"low"`/`"high"`/`"max"` or a decimal integer in `[1, 100]`.
pub fn render_json(messages_json: &str, tools_json: Option<&str>, thinking: bool, reasoning_effort: Option<&str>) -> Result<String> {
    let mut messages = Value::from_json_str(messages_json)?;
    if let Some(t) = tools_json {
        messages = with_tools(messages, Value::from_json_str(t)?);
    }
    let effort = reasoning_effort.map(|e| e.trim().parse::<i64>().map_or_else(|_| Value::str(e), Value::Int));
    encode(&messages, thinking, effort.as_ref())
}

/// Attach OpenAI-format `tools` to the first message when it is a system message (as the
/// upstream `load_cases` does), else insert an empty system message carrying them: V4.1
/// renders tools only on system messages.
pub fn with_tools(messages: Value, tools: Value) -> Value {
    let Value::List(mut items) = messages else { return messages };
    if !truthy(Some(&tools)) {
        return Value::List(items);
    }
    match items.first_mut().filter(|m| role(m) == "system") {
        Some(sys) => set(sys, "tools", tools),
        None => items.insert(0, map(vec![("role", Value::str("system")), ("content", Value::str("")), ("tools", tools)])),
    }
    Value::List(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(json: &str, thinking: bool, effort: Option<Value>) -> Result<String> {
        encode(&Value::from_json_str(json).expect("json"), thinking, effort.as_ref())
    }

    const EFFORT_75: &str = "<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n";

    #[test]
    fn basic_chat_and_thinking() {
        let j = r#"[{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"}]"#;
        assert_eq!(enc(j, false, None).unwrap(), format!("{BOS}{SYSTEM}S{USER}Hi{ASSISTANT}{THINK_END}"));
        assert_eq!(enc(j, true, None).unwrap(), format!("{BOS}{EFFORT_75}S{USER}Hi{ASSISTANT}{THINK_START}"));
    }

    #[test]
    fn effort_prefix() {
        let j = r#"[{"role": "user", "content": "q"}]"#;
        let budget = |e: Value| enc(j, true, Some(e)).unwrap();
        assert!(budget(Value::str("low")).contains("Reasoning Effort: 50 "));
        assert!(budget(Value::str("max")).contains("Reasoning Effort: 100 "));
        assert!(budget(Value::Int(42)).starts_with(&format!("{BOS}{SYSTEM}Reasoning Effort: 42 ")));
        assert_eq!(enc(j, false, Some(Value::str("max"))).unwrap(), format!("{BOS}{USER}q{ASSISTANT}{THINK_END}"));
        for bad in [Value::Int(0), Value::Int(101), Value::str("medium"), Value::Bool(true), Value::Float(1.5)] {
            assert_eq!(enc(j, true, Some(bad)).unwrap_err().code, "template");
        }
        assert_eq!(render_json(j, None, true, Some("42")).unwrap(), budget(Value::Int(42)));
    }

    #[test]
    fn dsml_tool_call() {
        let j = r#"[{"role": "user", "content": "question"}, {"role": "assistant", "reasoning_content": "r", "content": "summary",
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": "{\"query\":\"value\",\"limit\":2}"}}]}]"#;
        let want = format!(
            "{BOS}{EFFORT_75}{USER}question{ASSISTANT}{THINK_START}r{THINK_END}summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">value</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>{EOS}"
        );
        assert_eq!(enc(j, true, None).unwrap(), want);
    }

    #[test]
    fn mid_conversation_system() {
        let j = r#"[{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "system", "content": "mid"}]"#;
        assert_eq!(enc(j, false, None).unwrap(), format!("{BOS}{USER}q1{ASSISTANT}{THINK_END}a1{EOS}{SYSTEM}mid{ASSISTANT}{THINK_END}"));
    }

    #[test]
    fn images_rejected() {
        let j = r#"[{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "x.png"}}]}]"#;
        assert_eq!(enc(j, false, None).unwrap_err().code, "unsupported");
    }

    #[test]
    fn tools_attach_to_system() {
        let tools = r#"[{"type": "function", "function": {"name": "f", "parameters": {}}}]"#;
        let with_sys = render_json(r#"[{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]"#, Some(tools), false, None).unwrap();
        assert!(with_sys.starts_with(&format!("{BOS}{SYSTEM}S\n\n## Tools\n")));
        let without = render_json(r#"[{"role": "user", "content": "q"}]"#, Some(tools), false, None).unwrap();
        assert!(without.starts_with(&format!("{BOS}{SYSTEM}\n\n## Tools\n")) && without.contains("{\"name\": \"f\", \"parameters\": {}}"));
    }

    /// `load_cases` + `encode_case(case, thinking_mode="chat")` from upstream `test_encoding.py`.
    fn fixture(input: &str) -> String {
        let data = Value::from_json_str(input).expect("fixture json");
        let (mut messages, mut thinking, mut effort) = (data.clone(), false, None);
        if matches!(data, Value::Map(_)) {
            messages = get(&data, "messages").cloned().expect("messages");
            if let (Some(tools), Value::List(items)) = (get(&data, "tools"), &mut messages) {
                set(&mut items[0], "tools", tools.clone());
            }
            thinking = s(get(&data, "thinking_mode")) == "thinking";
            effort = present(get(&data, "reasoning_effort")).cloned();
        }
        encode(&messages, thinking, effort.as_ref()).expect("encode fixture")
    }

    macro_rules! fixtures {
        ($($name:ident = $n:literal),*) => {$(
            #[test]
            fn $name() {
                let input = include_str!(concat!("testdata/encoding/test_input_", $n, ".json"));
                let output = include_str!(concat!("testdata/encoding/test_output_", $n, ".txt"));
                assert_eq!(fixture(input), output);
            }
        )*};
    }
    // Fixture 5 needs image blocks (vision path) and is not ported.
    fixtures!(fixture_1 = "1", fixture_2 = "2", fixture_3 = "3", fixture_4 = "4");

    /// Edge cases with outputs generated by the upstream `encoding.py`.
    #[test]
    fn reference_cases() {
        let cases = Value::from_json_str(include_str!("testdata/encoding/extra_cases.json")).expect("json");
        assert!(list(Some(&cases)).len() > 60);
        for case in list(Some(&cases)) {
            let name = s(get(case, "name"));
            let got = encode(get(case, "messages").expect("messages"), truthy(get(case, "thinking")), get(case, "effort"));
            match get(case, "expected") {
                Some(want) => assert_eq!(got.unwrap_or_else(|e| panic!("{name}: {e}")), s(Some(want)), "{name}"),
                None => assert!(got.is_err(), "{name}: expected an error"),
            }
        }
    }
}
