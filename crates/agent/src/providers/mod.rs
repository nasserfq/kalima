//! The AI services: Claude (Anthropic), Gemini (Google), and every OpenAI-compatible service
//! (OpenRouter, Groq, Ollama on this computer, or any other address).

pub mod anthropic;
pub mod gemini;
pub mod openai;

use std::io::BufReader;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::clip;
use crate::types::{Message, Reply, StreamEvent, ToolSpec};
use crate::{Error, Result};

/// The wire protocol a service speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Anthropic,
    Gemini,
    OpenAi,
}

/// A service the app can talk to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provider {
    /// Stable id, also the key's name in the keychain.
    pub id: String,
    pub name: String,
    pub kind: Kind,
    pub base_url: String,
    /// Whether requests need an API key (Ollama on this computer doesn't).
    pub needs_key: bool,
    /// Where to get a key, shown in Settings.
    #[serde(default)]
    pub key_url: String,
    /// Built in (can't be removed).
    #[serde(default)]
    pub builtin: bool,
}

impl Provider {
    fn new(id: &str, name: &str, kind: Kind, base_url: &str, needs_key: bool, key_url: &str) -> Self {
        Provider { id: id.into(), name: name.into(), kind, base_url: base_url.into(), needs_key, key_url: key_url.into(), builtin: true }
    }
}

/// The services offered out of the box.
pub fn builtin_providers() -> Vec<Provider> {
    vec![
        Provider::new("anthropic", "Claude", Kind::Anthropic, "https://api.anthropic.com/v1", true, "https://console.anthropic.com/settings/keys"),
        Provider::new(
            "gemini",
            "Gemini",
            Kind::Gemini,
            "https://generativelanguage.googleapis.com/v1beta",
            true,
            "https://aistudio.google.com/apikey",
        ),
        Provider::new("openrouter", "OpenRouter", Kind::OpenAi, "https://openrouter.ai/api/v1", true, "https://openrouter.ai/keys"),
        Provider::new("groq", "Groq", Kind::OpenAi, "https://api.groq.com/openai/v1", true, "https://console.groq.com/keys"),
        Provider::new("ollama", "Ollama", Kind::OpenAi, "http://localhost:11434/v1", false, "https://ollama.com/download"),
    ]
}

/// A model one service offers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    /// Free to use (a free model, a free tier, or a model running on this computer).
    pub free: bool,
    /// US dollars per million input and output tokens, when known.
    pub price: Option<(f64, f64)>,
}

/// One request to a model.
#[derive(Debug, Clone)]
pub struct Request<'a> {
    pub model: &'a str,
    pub system: &'a str,
    pub messages: &'a [Message],
    pub tools: &'a [ToolSpec],
    /// "low", "medium" or "high" (Claude's effort; ignored where not supported).
    pub effort: &'a str,
}

/// Waits before sending again to a busy service, and how many times it is tried.
const RETRY_WAITS: [Duration; 2] = [Duration::from_secs(2), Duration::from_secs(5)];

/// Whether `e` means the service is busy or briefly down (worth sending the same request again).
/// Nothing has streamed yet when these come back, so sending again is safe.
pub fn is_busy(e: &Error) -> bool {
    matches!(e, Error::Api { status: 500 | 502 | 503 | 504 | 529, .. }) || matches!(e, Error::Api { status: 0, body, .. } if is_overload_text(body))
}

fn is_overload_text(body: &str) -> bool {
    let b = body.to_lowercase();
    b.contains("overloaded") || b.contains("high demand") || b.contains("unavailable")
}

/// Google's free models, best first: when one is busy, out of quota or gone, the next is tried.
const GEMINI_FREE: [&str; 3] = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-flash-lite-latest"];

/// Other models to try, in order, when `model` stays busy or is out of quota: the other free
/// models of the same service. A paid or unknown model has none (the person chose it on purpose).
pub fn fallback_models(p: &Provider, model: &str) -> Vec<&'static str> {
    match p.kind {
        Kind::Gemini if model.contains("flash") || model.contains("gemma") => GEMINI_FREE.iter().copied().filter(|m| *m != model).collect(),
        _ => Vec::new(),
    }
}

/// Send `req` and stream the reply to `on_event`. A busy service is tried again after a short
/// wait, then the next free model where there is one (each change is reported with
/// [`StreamEvent::Retrying`]).
pub fn send(p: &Provider, key: &str, req: &Request, cancel: &Arc<AtomicBool>, on_event: &mut dyn FnMut(StreamEvent)) -> Result<Reply> {
    if p.needs_key && key.trim().is_empty() {
        return Err(Error::NoKey(p.name.clone()));
    }
    let models: Vec<&str> = std::iter::once(req.model).chain(fallback_models(p, req.model).iter().copied().filter(|m| *m != req.model)).collect();
    let mut last = None;
    for (i, model) in models.iter().enumerate() {
        let r = Request { model, ..req.clone() };
        // The chosen model gets its retries; a fallback gets one try.
        let waits: &[Duration] = if i == 0 { &RETRY_WAITS } else { &[] };
        let mut attempt = 0usize;
        loop {
            match send_once(p, key, &r, cancel, on_event) {
                Err(e) if is_busy(&e) => {
                    if i == 0 {
                        last = Some(e);
                    }
                    let Some(wait) = waits.get(attempt) else { break };
                    attempt = attempt.saturating_add(1);
                    on_event(StreamEvent::Retrying { model: (*model).to_string() });
                    sleep_unless_cancelled(*wait, cancel)?;
                }
                // A fallback this key or service doesn't offer (any more): try the next one, and
                // report the chosen model's busy error if none answers.
                Err(Error::Api { status: 400 | 403 | 404, .. }) if i > 0 => break,
                // Out of quota for now: waiting seconds won't help, another free model may.
                Err(e @ Error::Api { status: 429, .. }) if models.len() > 1 => {
                    last.get_or_insert(e);
                    break;
                }
                other => return other,
            }
        }
        if let Some(next) = models.get(i.saturating_add(1)) {
            on_event(StreamEvent::Retrying { model: (*next).to_string() });
        }
    }
    Err(last.unwrap_or_else(|| Error::Invalid("no model".into())))
}

fn sleep_unless_cancelled(d: Duration, cancel: &Arc<AtomicBool>) -> Result<()> {
    let step = Duration::from_millis(100);
    let mut left = d;
    while !left.is_zero() {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let s = left.min(step);
        std::thread::sleep(s);
        left = left.saturating_sub(s);
    }
    Ok(())
}

fn send_once(p: &Provider, key: &str, req: &Request, cancel: &Arc<AtomicBool>, on_event: &mut dyn FnMut(StreamEvent)) -> Result<Reply> {
    match p.kind {
        Kind::Anthropic => anthropic::send(p, key, req, cancel, on_event),
        Kind::Gemini => gemini::send(p, key, req, cancel, on_event),
        Kind::OpenAi => openai::send(p, key, req, cancel, on_event),
    }
}

/// The models `p` offers.
pub fn list_models(p: &Provider, key: &str) -> Result<Vec<ModelInfo>> {
    if p.needs_key && key.trim().is_empty() {
        return Err(Error::NoKey(p.name.clone()));
    }
    let mut v = match p.kind {
        Kind::Anthropic => anthropic::list_models(p, key)?,
        Kind::Gemini => gemini::list_models(p, key)?,
        Kind::OpenAi => openai::list_models(p, key)?,
    };
    v.sort_by(|a, b| b.free.cmp(&a.free).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())));
    Ok(v)
}

/// The cost in US dollars of `input` and `output` tokens at `price`.
/// The model to use on `p` when the person hasn't picked one; empty when the service has no
/// sensible default (pick from its list instead).
pub fn default_model(p: &Provider) -> &'static str {
    match p.kind {
        Kind::Anthropic => anthropic::DEFAULT_MODEL,
        Kind::Gemini => gemini::DEFAULT_MODEL,
        Kind::OpenAi => "",
    }
}

pub fn cost(price: (f64, f64), input: u64, output: u64) -> f64 {
    (input as f64 * price.0 + output as f64 * price.1) / 1_000_000.0
}

// ---- HTTP ----

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(120)))
        .http_status_as_error(false)
        .user_agent(concat!("Kalima-Agent/", env!("CARGO_PKG_VERSION")))
        // The system's own certificate store, as the app's updater uses.
        .tls_config(ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::PlatformVerifier).build())
        .build()
        .new_agent()
}

/// POST `body` as JSON and return the open streaming response, or the service's error.
pub(crate) fn post_stream(service: &str, url: &str, headers: &[(&str, String)], body: &Value) -> Result<BufReader<ureq::BodyReader<'static>>> {
    let mut req = agent().post(url);
    for (k, v) in headers {
        req = req.header(*k, v.as_str());
    }
    let resp = req.header("content-type", "application/json").send(serde_json::to_vec(body)?)?;
    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        let text = resp.into_body().read_to_string().unwrap_or_default();
        return Err(Error::Api { service: service.into(), status, body: error_message(&text) });
    }
    Ok(BufReader::new(resp.into_body().into_reader()))
}

/// GET `url` and parse the JSON answer.
pub(crate) fn get_json(service: &str, url: &str, headers: &[(&str, String)]) -> Result<Value> {
    let mut req = agent().get(url);
    for (k, v) in headers {
        req = req.header(*k, v.as_str());
    }
    let resp = req.call()?;
    let status = resp.status().as_u16();
    let text = resp.into_body().with_config().limit(64 << 20).read_to_string()?;
    if !(200..300).contains(&status) {
        return Err(Error::Api { service: service.into(), status, body: error_message(&text) });
    }
    Ok(serde_json::from_str(&text)?)
}

/// A readable message from an error body: the `error.message` the services send, or the start of it.
pub(crate) fn error_message(body: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        let e = v.get("error").unwrap_or(&v);
        let e = e.as_array().and_then(|a| a.first()).and_then(|x| x.get("error")).unwrap_or(e);
        if let Some(m) = e.get("message").and_then(Value::as_str) {
            return clip(m, 600);
        }
        if let Some(m) = e.as_str() {
            return clip(m, 600);
        }
    }
    clip(body.trim(), 600)
}

/// The JSON object in `s`, or an error naming what was wrong (a tool input that didn't parse).
pub(crate) fn parse_tool_input(s: &str) -> Value {
    if s.trim().is_empty() {
        return Value::Object(Default::default());
    }
    match serde_json::from_str::<Value>(s) {
        Ok(v @ Value::Object(_)) => v,
        _ => serde_json::json!({ crate::INVALID_INPUT: clip(s, 2000) }),
    }
}

/// The bytes of a base64 attachment as text (for text files sent inline).
pub(crate) fn attachment_text(name: &str, data: &str) -> String {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD.decode(data).unwrap_or_default();
    format!("<file name=\"{name}\">\n{}\n</file>", String::from_utf8_lossy(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_readable() {
        assert_eq!(error_message(r#"{"type":"error","error":{"type":"x","message":"bad key"}}"#), "bad key");
        assert_eq!(error_message(r#"[{"error":{"code":400,"message":"API key not valid"}}]"#), "API key not valid");
        assert_eq!(error_message(r#"{"error":"nope"}"#), "nope");
        assert_eq!(error_message("plain"), "plain");
    }

    #[test]
    fn tool_input_parsing_never_fails() {
        assert_eq!(parse_tool_input(""), serde_json::json!({}));
        assert_eq!(parse_tool_input("{\"a\":1}"), serde_json::json!({"a":1}));
        assert!(parse_tool_input("{\"a\":").get(crate::INVALID_INPUT).is_some());
        assert!(parse_tool_input("[1]").get(crate::INVALID_INPUT).is_some());
    }

    #[test]
    fn busy_services_are_retried_and_others_are_not() {
        let api = |status| Error::Api { service: "Gemini".into(), status, body: "x".into() };
        assert!(is_busy(&api(503)));
        assert!(is_busy(&api(500)));
        assert!(!is_busy(&api(400)));
        assert!(!is_busy(&api(429)), "a quota limit doesn't clear in seconds");
        assert!(is_busy(&Error::Api { service: "Gemini".into(), status: 0, body: "The model is overloaded".into() }));
        assert!(!is_busy(&Error::Network("no route".into())));
    }

    #[test]
    fn a_busy_gemini_flash_falls_back_to_other_free_models() {
        let g = builtin_providers().into_iter().find(|p| p.id == "gemini").unwrap();
        assert_eq!(fallback_models(&g, "gemini-flash-latest"), ["gemini-2.5-flash", "gemini-flash-lite-latest"]);
        assert_eq!(fallback_models(&g, "gemini-3.1-flash-lite"), GEMINI_FREE);
        let c = builtin_providers().into_iter().find(|p| p.id == "anthropic").unwrap();
        assert!(fallback_models(&c, "claude-x").is_empty());
    }

    /// A local server that answers 503 for `gemini-flash-latest` and a short reply for any other model.
    #[test]
    fn send_moves_to_the_fallback_model_when_the_chosen_one_stays_busy() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming().take(4) {
                let mut s = stream.unwrap();
                let mut buf = vec![0u8; 1 << 16];
                let n = s.read(&mut buf).unwrap_or(0);
                let head = String::from_utf8_lossy(&buf[..n]).to_string();
                let resp = if head.contains("gemini-flash-latest") {
                    let b = r#"{"error":{"code":503,"message":"This model is currently experiencing high demand."}}"#;
                    format!(
                        "HTTP/1.1 503 Service Unavailable\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{b}",
                        b.len()
                    )
                } else {
                    let b = "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
                    format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{b}", b.len())
                };
                let _ = s.write_all(resp.as_bytes());
            }
        });
        let mut p = builtin_providers().into_iter().find(|p| p.id == "gemini").unwrap();
        p.base_url = format!("http://127.0.0.1:{port}/v1beta");
        let msgs = vec![crate::types::Message::user_text("hi")];
        let req = Request { model: "gemini-flash-latest", system: "", messages: &msgs, tools: &[], effort: "" };
        let mut events = Vec::new();
        let r = send(&p, "k", &req, &Arc::new(AtomicBool::new(false)), &mut |e| events.push(e)).unwrap();
        assert_eq!(r.message.text(), "ok");
        assert_eq!(r.message.model.as_deref(), Some("gemini-2.5-flash"));
        let retries: Vec<_> =
            events.iter().filter_map(|e| if let StreamEvent::Retrying { model } = e { Some(model.as_str()) } else { None }).collect();
        assert_eq!(retries, ["gemini-flash-latest", "gemini-flash-latest", "gemini-2.5-flash"]);
    }

    #[test]
    fn cost_is_per_million() {
        assert!((cost((4.0, 20.0), 1_000_000, 500_000) - 14.0).abs() < 1e-9);
    }
}
