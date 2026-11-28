//! DeepSeek V4.x tool calls: a DSML block in the assistant text becomes OpenAI `tool_calls`.
//!
//! The model writes (V4.1 puts a space after `｜DSML｜`, V4 does not; both are accepted):
//!
//! ```text
//! <｜DSML｜ calls>
//! <｜DSML｜ invoke name="NAME">
//! <｜DSML｜ parameter name="KEY" string="true|false">VALUE</｜DSML｜ parameter>
//! </｜DSML｜ invoke>
//! </｜DSML｜ calls>
//! ```
//!
//! `string="true"`: VALUE is the string as is. `string="false"`: VALUE is JSON.

use serde_json::{Map, Value};

/// Start of any DSML tag; text before it is ordinary content.
const OPEN: &str = "<｜DSML｜";
const MARK: &str = "｜DSML｜";

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// JSON object text, as OpenAI `function.arguments`.
    pub arguments: String,
}

/// Position of the next `<｜DSML｜ kw` (or `</｜DSML｜ kw` when `close`) at or after `from`:
/// (index of the `<`, index just after the keyword).
fn find_tag(s: &str, from: usize, close: bool, kws: &[&str]) -> Option<(usize, usize)> {
    let lead = if close { "</" } else { "<" };
    let mut at = from;
    while let Some(i) = s.get(at..)?.find(MARK) {
        let m = at + i;
        at = m + MARK.len();
        let Some(start) = m.checked_sub(lead.len()) else { continue };
        if s.get(start..m) != Some(lead) {
            continue;
        }
        let rest = s[at..].trim_start_matches(' ');
        let kw_at = s.len() - rest.len();
        for kw in kws {
            if let Some(after) = rest.strip_prefix(kw)
                && !after.starts_with(|c: char| c.is_alphanumeric() || c == '_')
            {
                return Some((start, kw_at + kw.len()));
            }
        }
    }
    None
}

/// `attr="value"` inside an opening tag's text.
fn attr<'a>(tag: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("{key}=\"");
    let mut from = 0;
    while let Some(i) = tag[from..].find(&pat) {
        let at = from + i;
        // whole-word match: `name=` must not be the tail of another attribute.
        if at == 0 || tag.as_bytes()[at - 1].is_ascii_whitespace() {
            let v = &tag[at + pat.len()..];
            return v.find('"').map(|e| &v[..e]);
        }
        from = at + pat.len();
    }
    None
}

/// End of an opening tag: (text between keyword and `>`, index after `>`).
fn open_tag_end(s: &str, after_kw: usize) -> Option<(&str, usize)> {
    let gt = after_kw + s[after_kw..].find('>')?;
    Some((&s[after_kw..gt], gt + 1))
}

fn parse_params(body: &str) -> String {
    let mut args = Map::new();
    let mut at = 0;
    while let Some((_, kw_end)) = find_tag(body, at, false, &["parameter"]) {
        let Some((tag, val_start)) = open_tag_end(body, kw_end) else { break };
        let (val_end, next) = match find_tag(body, val_start, true, &["parameter"]) {
            Some((c, ce)) => (c, body[ce..].find('>').map_or(body.len(), |g| ce + g + 1)),
            // Truncated generation: the value runs to the end of the invoke.
            None => (body.len(), body.len()),
        };
        let raw = &body[val_start..val_end];
        let value = match attr(tag, "string") {
            Some("false") => serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string())),
            _ => Value::String(raw.to_string()),
        };
        if let Some(name) = attr(tag, "name") {
            args.insert(name.to_string(), value);
        }
        at = next;
    }
    Value::Object(args).to_string()
}

/// Calls in a text that starts at (or contains) a `calls` block.
fn parse_calls(block: &str) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some((_, kw_end)) = find_tag(block, at, false, &["invoke"]) {
        let Some((tag, body_start)) = open_tag_end(block, kw_end) else { break };
        let (body_end, next) = match find_tag(block, body_start, true, &["invoke"]) {
            Some((c, ce)) => (c, block[ce..].find('>').map_or(block.len(), |g| ce + g + 1)),
            None => (block.len(), block.len()),
        };
        if let Some(name) = attr(tag, "name") {
            out.push(ToolCall { name: name.to_string(), arguments: parse_params(&block[body_start..body_end]) });
        }
        at = next;
    }
    out
}

/// Whole answer: (content before the block, calls). No block or no parsable call: the text
/// unchanged and no calls.
pub fn split(text: &str) -> (String, Vec<ToolCall>) {
    let Some((start, kw_end)) = find_tag(text, 0, false, &["calls", "tool_calls"]) else {
        return (text.to_string(), Vec::new());
    };
    let end = find_tag(text, kw_end, true, &["calls", "tool_calls"]).map_or(text.len(), |(c, _)| c);
    let calls = parse_calls(&text[kw_end..end]);
    if calls.is_empty() {
        return (text.to_string(), Vec::new());
    }
    (text[..start].trim_end().to_string(), calls)
}

/// Streaming side: passes ordinary text through, holds back anything that may be the start
/// of a DSML block (and the block itself) until the generation ends.
#[derive(Default)]
pub struct Filter {
    held: String,
    block: String,
    in_block: bool,
}

impl Filter {
    /// Text safe to send now for one more delta.
    pub fn push(&mut self, delta: &str) -> String {
        if self.in_block {
            self.block.push_str(delta);
            return String::new();
        }
        self.held.push_str(delta);
        if let Some(i) = self.held.find(OPEN) {
            self.block = self.held.split_off(i);
            self.in_block = true;
            return std::mem::take(&mut self.held);
        }
        // Keep the longest tail that could still grow into `<｜DSML｜`.
        let n = self.held.len();
        let keep = (1..=n.min(OPEN.len() - 1))
            .rev()
            .find(|&k| self.held.is_char_boundary(n - k) && OPEN.starts_with(&self.held[n - k..]))
            .unwrap_or(0);
        let tail = self.held.split_off(n - keep);
        std::mem::replace(&mut self.held, tail)
    }

    /// End of generation: (text still to send, calls). A block that does not parse as a call
    /// is returned as text so nothing is lost.
    pub fn finish(&mut self) -> (String, Vec<ToolCall>) {
        if self.in_block {
            let block = std::mem::take(&mut self.block);
            self.in_block = false;
            let (rest, calls) = split(&block);
            return if calls.is_empty() { (block, calls) } else { (rest, calls) };
        }
        (std::mem::take(&mut self.held), Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V41: &str = "I'll look it up.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">hello world</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";

    fn args(c: &ToolCall) -> Value {
        serde_json::from_str(&c.arguments).unwrap()
    }

    #[test]
    fn v41_block() {
        let (text, calls) = split(V41);
        assert_eq!(text, "I'll look it up.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
        assert_eq!(args(&calls[0]), serde_json::json!({"query": "hello world", "limit": 2}));
    }

    #[test]
    fn v4_without_space_and_tool_calls_name() {
        let t = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"a\" string=\"true\">x</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let (text, calls) = split(t);
        assert_eq!(text, "");
        assert_eq!(args(&calls[0]), serde_json::json!({"a": "x"}));
    }

    #[test]
    fn two_calls_multiline_string_and_json_values() {
        let t = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a.md</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"body\" string=\"true\"># T\n\nuse <b> & \"q\"\n</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"opts\" string=\"false\">{\"x\":[1,2],\"y\":true}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"ls\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let (_, calls) = split(t);
        assert_eq!(calls.len(), 2);
        assert_eq!(args(&calls[0])["body"], "# T\n\nuse <b> & \"q\"\n");
        assert_eq!(args(&calls[0])["opts"], serde_json::json!({"x": [1, 2], "y": true}));
        assert_eq!(calls[1].name, "ls");
        assert_eq!(calls[1].arguments, "{}");
    }

    #[test]
    fn truncated_generation_still_yields_the_call() {
        let t = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"a\" string=\"true\">partial";
        let (_, calls) = split(t);
        assert_eq!(calls[0].name, "f");
        assert_eq!(args(&calls[0]), serde_json::json!({"a": "partial"}));
    }

    #[test]
    fn plain_text_and_prose_about_dsml_are_untouched() {
        for t in ["hello", "the tag <｜DSML｜ is used for tools", ""] {
            let (text, calls) = split(t);
            assert_eq!((text.as_str(), calls.len()), (t, 0));
        }
    }

    #[test]
    fn filter_streams_text_and_holds_the_block_across_any_split() {
        let chars: Vec<char> = V41.chars().collect();
        for step in 1..=7 {
            let mut f = Filter::default();
            let mut sent = String::new();
            for c in chars.chunks(step) {
                sent.push_str(&f.push(&c.iter().collect::<String>()));
            }
            let (tail, calls) = f.finish();
            sent.push_str(&tail);
            assert!(!sent.contains("DSML"), "step {step}: leaked {sent:?}");
            assert_eq!(sent.trim_end(), "I'll look it up.", "step {step}");
            assert_eq!(calls.len(), 1, "step {step}");
        }
    }

    #[test]
    fn filter_releases_a_false_alarm_prefix() {
        let mut f = Filter::default();
        let mut sent = f.push("a <");
        sent.push_str(&f.push("｜D"));
        sent.push_str(&f.push("one"));
        let (tail, calls) = f.finish();
        sent.push_str(&tail);
        assert_eq!(sent, "a <｜Done");
        assert!(calls.is_empty());
    }

    #[test]
    fn unparsable_block_is_returned_as_text() {
        let mut f = Filter::default();
        let mut sent = f.push("x <｜DSML｜ calls> garbage");
        let (tail, calls) = f.finish();
        sent.push_str(&tail);
        assert_eq!(sent, "x <｜DSML｜ calls> garbage");
        assert!(calls.is_empty());
    }
}
