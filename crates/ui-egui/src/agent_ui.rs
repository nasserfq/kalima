//! وكيل (AI Agent): a side pane where the person chats with an AI model that edits the open
//! document live, like Copilot in Word. Home ▸ AI Agent (`ui.agentPane`) opens it.
//!
//! The model runs on a worker thread (`wordcraft_agent::run`). Its tools are the MCP server's own
//! (`wordcraft_mcp::tools`), but they reach this window instead of a TCP port: each call becomes a
//! [`ControlRequest`] on a channel that [`drain`] answers every frame, so every edit happens in the
//! window as the person watches. A whole turn is one Undo step ("AI Agent").
//!
//! Keys are kept in the macOS Keychain (elsewhere in an owner-only file in the settings folder)
//! and sent only to their own service. The web build has no agent: the pane says so.

use crate::WordApp;

/// Strings: Arabic in the Arabic interface, English otherwise.
fn l(ar: &'static str, en: &'static str) -> &'static str {
    if crate::i18n::current().code() == "ar" { ar } else { crate::i18n::t(en) }
}

/// The pane's title and the ribbon button's label.
pub(crate) fn title() -> &'static str {
    l("وكيل", "AI Agent")
}

#[derive(Default)]
pub(crate) struct Runtime {
    pub(crate) open: bool,
    #[cfg(not(target_arch = "wasm32"))]
    state: Option<native::State>,
}

/// Show or hide the pane (`ui.agentPane`, `{"value": bool}` or a toggle).
pub(crate) fn toggle(app: &mut WordApp, value: Option<bool>) {
    app.agent.open = value.unwrap_or(!app.agent.open);
}

/// The agent's own requests and its worker's news, every frame.
pub(crate) fn drain(app: &mut WordApp, ctx: &egui::Context) {
    #[cfg(not(target_arch = "wasm32"))]
    native::drain(app, ctx);
    #[cfg(target_arch = "wasm32")]
    let _ = (app, ctx);
}

/// The pane's body.
pub(crate) fn pane(app: &mut WordApp, ui: &mut egui::Ui) {
    #[cfg(not(target_arch = "wasm32"))]
    native::pane(app, ui);
    #[cfg(target_arch = "wasm32")]
    {
        let _ = app;
        ui.label(l("وكيل متاح في تطبيق كلمة على الجهاز، لا في المتصفح.", "The AI Agent is available in the desktop app, not in the web browser."));
    }
}

/// Send `text` as the person's message, as if typed in the pane (`ui.agentPane` with `ask`):
/// for tests and the control channel.
pub(crate) fn ask(app: &mut WordApp, ctx: &egui::Context, text: &str) -> Result<(), String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        app.agent.open = true;
        native::send(app, ctx, text.to_string())
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (app, ctx, text);
        Err("the AI Agent needs the desktop app".into())
    }
}

impl WordApp {
    /// Whether the AI Agent is working on a turn now (for scripted tests).
    pub fn agent_running(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        return self.agent.state.as_ref().is_some_and(native::State::running);
        #[cfg(target_arch = "wasm32")]
        return false;
    }

    /// The AI Agent's chat so far and its last error, as JSON (for scripted tests).
    pub fn agent_transcript(&self) -> serde_json::Value {
        #[cfg(not(target_arch = "wasm32"))]
        return self.agent.state.as_ref().map(native::State::transcript).unwrap_or_default();
        #[cfg(target_arch = "wasm32")]
        return serde_json::Value::Null;
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{Receiver, Sender, channel};
    use std::time::Duration;

    use egui::{Align, Color32, Key, KeyboardShortcut, Layout, Modifiers, RichText, Ui};
    use serde_json::{Value, json};
    use wordcraft_agent::providers::{ModelInfo, Provider, default_model, list_models};
    use wordcraft_agent::run::{RunConfig, RunEvent, Tools};
    use wordcraft_agent::settings::Settings;
    use wordcraft_agent::types::{Message, Part, Role, ToolSpec};
    use wordcraft_agent::{Error, INVALID_INPUT, MAX_RESULT, secrets};

    use super::l;
    use crate::WordApp;
    use crate::control::{ControlRequest, Outcome};
    use crate::theme::Tokens;

    /// The tools the agent gets: reading the document and running its commands. Opening, saving
    /// and the window itself stay with the person.
    const TOOLS: &[&str] = &["get_text", "inspect_document", "list_commands", "select_text", "type_text", "execute", "batch"];

    /// How long one tool call may wait for the window (a big document's first layout is slow).
    const TOOL_TIMEOUT: Duration = Duration::from_secs(120);

    /// The services offered on the first-run card, the free one first.
    const SETUP: &[&str] = &["gemini", "openrouter", "anthropic", "groq", "ollama"];

    pub(super) struct State {
        dir: Option<PathBuf>,
        settings: Settings,
        /// Keys read so far, by provider id ("" = none saved).
        keys: HashMap<String, String>,
        chat: Vec<Message>,
        input: String,
        key_input: String,
        /// The first-run card is showing, for this provider.
        setup: Option<String>,
        run: Option<Running>,
        /// Models by provider id, once listed.
        models: HashMap<String, Vec<ModelInfo>>,
        loading: Option<(String, Receiver<Result<Vec<ModelInfo>, String>>)>,
        /// The worker's tool calls, answered in [`drain`].
        req_tx: Sender<ControlRequest>,
        req_rx: Receiver<ControlRequest>,
        /// The last error, shown under the chat.
        notice: Option<String>,
        /// Dollars spent in this chat, where the price is known.
        cost: f64,
    }

    struct Running {
        rx: Receiver<Msg>,
        cancel: Arc<AtomicBool>,
        /// The reply as it streams in.
        text: String,
        /// The tool running now: its call id.
        tool: Option<String>,
        /// The model is thinking (its reasoning streams but isn't shown).
        thinking: bool,
        /// The service was busy and the request is being sent again, to this model.
        retrying: Option<String>,
        /// When the turn began, for the seconds shown while waiting.
        started: std::time::Instant,
        /// Undo history position and document when the turn began.
        mark: u64,
        doc: u64,
    }

    enum Msg {
        Event(RunEvent),
        /// The turn ended: why, if it went wrong.
        Done(Option<String>),
    }

    impl State {
        fn new() -> Self {
            let dir = wordcraft_control_key::settings_dir();
            let settings = dir.as_deref().map(Settings::load).unwrap_or_default();
            let (req_tx, req_rx) = channel();
            State {
                dir,
                settings,
                keys: HashMap::new(),
                chat: Vec::new(),
                input: String::new(),
                key_input: String::new(),
                setup: None,
                run: None,
                models: HashMap::new(),
                loading: None,
                req_tx,
                req_rx,
                notice: None,
                cost: 0.0,
            }
        }

        pub(super) fn running(&self) -> bool {
            self.run.is_some()
        }

        pub(super) fn transcript(&self) -> Value {
            json!({"chat": self.chat, "notice": self.notice})
        }

        fn provider(&self) -> Option<Provider> {
            self.settings.provider(&self.settings.provider)
        }

        fn key(&mut self, id: &str) -> String {
            if let Some(k) = self.keys.get(id) {
                return k.clone();
            }
            let k = self.dir.as_deref().map(|d| secrets::get(d, id)).unwrap_or_default();
            self.keys.insert(id.to_string(), k.clone());
            k
        }

        /// Ready to chat: a service is chosen and has the key it needs.
        fn ready(&mut self) -> bool {
            match self.provider() {
                Some(p) => !p.needs_key || !self.key(&p.id).is_empty(),
                None => false,
            }
        }

        /// The model to use: the chosen one, else the service's default, else its first free model.
        fn model(&self) -> String {
            if !self.settings.model.is_empty() {
                return self.settings.model.clone();
            }
            let Some(p) = self.provider() else { return String::new() };
            let d = default_model(&p);
            if !d.is_empty() {
                return d.to_string();
            }
            let list = self.models.get(&p.id).map(Vec::as_slice).unwrap_or_default();
            list.iter().find(|m| m.free).or(list.first()).map(|m| m.id.clone()).unwrap_or_default()
        }

        fn save(&mut self) {
            if let Some(d) = &self.dir
                && let Err(e) = self.settings.save(d)
            {
                self.notice = Some(format!("{} {e}", l("تعذّر حفظ الإعدادات:", "Couldn't save the settings:")));
            }
        }

        /// List the service's models in the background, once.
        fn load_models(&mut self, ctx: &egui::Context) {
            let Some(p) = self.provider() else { return };
            if self.models.contains_key(&p.id) || self.loading.is_some() {
                return;
            }
            let key = self.key(&p.id);
            if p.needs_key && key.is_empty() {
                return;
            }
            let (tx, rx) = channel();
            self.loading = Some((p.id.clone(), rx));
            let ctx = ctx.clone();
            let spawned = std::thread::Builder::new().name("agent-models".into()).spawn(move || {
                let _ = tx.send(list_models(&p, &key).map_err(|e| e.to_string()));
                ctx.request_repaint();
            });
            if spawned.is_err() {
                self.loading = None;
            }
        }
    }

    fn state(app: &mut WordApp) -> &mut State {
        app.agent.state.get_or_insert_with(State::new)
    }

    /// Answer the worker's tool calls in this window, and take in its news.
    pub(super) fn drain(app: &mut WordApp, ctx: &egui::Context) {
        let Some(st) = app.agent.state.as_mut() else { return };
        if let Some((id, rx)) = &st.loading
            && let Ok(r) = rx.try_recv()
        {
            let id = id.clone();
            st.loading = None;
            match r {
                Ok(mut list) => {
                    list.sort_by(|a, b| b.free.cmp(&a.free).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())));
                    st.models.insert(id, list);
                }
                Err(e) => {
                    st.models.insert(id, Vec::new());
                    st.notice = Some(friendly(&e));
                }
            }
        }
        let requests: Vec<ControlRequest> = st.req_rx.try_iter().collect();
        for req in requests {
            let reply = req.reply.clone();
            let v = match crate::control::handle(app, ctx, &req) {
                Outcome::Done(v) => v,
                Outcome::Screenshot { .. } => json!({"ok": false, "error": "screenshots are not available to the agent"}),
            };
            let _ = reply.send(v);
        }
        let Some(st) = app.agent.state.as_mut() else { return };
        let mut done: Option<Option<String>> = None;
        if let Some(run) = st.run.as_mut() {
            while let Ok(msg) = run.rx.try_recv() {
                match msg {
                    Msg::Event(RunEvent::Text(t)) => {
                        run.thinking = false;
                        run.retrying = None;
                        run.text.push_str(&t);
                    }
                    Msg::Event(RunEvent::Thinking(_)) => {
                        run.thinking = true;
                        run.retrying = None;
                    }
                    Msg::Event(RunEvent::Retrying { model }) => {
                        run.thinking = false;
                        run.retrying = Some(model);
                    }
                    Msg::Event(RunEvent::Assistant(m)) => {
                        run.text.clear();
                        run.thinking = false;
                        st.chat.push(m);
                    }
                    Msg::Event(RunEvent::ToolResults(m)) => st.chat.push(m),
                    Msg::Event(RunEvent::ToolStarted { id, .. }) => run.tool = Some(id),
                    Msg::Event(RunEvent::ToolFinished { .. }) => run.tool = None,
                    Msg::Event(RunEvent::Usage { cost, .. }) => st.cost += cost.unwrap_or(0.0),
                    // The agent's tools never ask; refuse anything that does.
                    Msg::Event(RunEvent::Approve { reply, .. }) => {
                        let _ = reply.send(wordcraft_agent::run::Approval::No);
                    }
                    Msg::Event(RunEvent::Finished(_)) => {}
                    Msg::Done(e) => done = Some(e),
                }
            }
        }
        if let Some(error) = done
            && let Some(run) = st.run.take()
        {
            st.notice = error.as_deref().map(friendly);
            // The whole turn is one Undo step, while its document is still the active one.
            if app.session.document_id() == run.doc {
                app.session.squash_undo_since(run.mark, super::title());
            }
        }
    }

    /// An error as the person should read it.
    fn friendly(e: &str) -> String {
        let has = |s: &str| e.contains(s);
        if has(" 401") || has(" 403") || has("API key not valid") || has("invalid x-api-key") {
            l("المفتاح غير صحيح. افتح «الخدمة» وأدخل مفتاحًا جديدًا.", "The key isn't valid. Open Service and enter a new key.").to_string()
        } else if has("returned 5") || has("high demand") || has("overloaded") || has("Internal error") {
            l(
                "الخدمة مزدحمة الآن ولم تستجب بعد عدة محاولات. حاول بعد دقيقة، أو اختر نموذجًا آخر من القائمة.",
                "The service is busy right now and didn't answer after several tries. Try again in a minute, or choose another model.",
            )
            .to_string()
        } else if has("timed out") || has("timeout") {
            l(
                "تأخرت الخدمة في الرد. حاول مرة أخرى، أو اختر نموذجًا أسرع.",
                "The service took too long to answer. Try again, or choose a faster model.",
            )
            .to_string()
        } else if has(" 429") {
            l(
                "بلغتَ الحد المجاني لنماذج هذه الخدمة الآن (في Gemini نحو ٢٠ طلبًا في اليوم لكل نموذج، وكل مهمة تستهلك طلبين أو ثلاثة). يتجدد الحد غدًا، أو اختر خدمة أخرى من «الخدمة».",
                "You've reached this service's free limit for now (Gemini gives about 20 requests a day per model, and each task takes two or three). It renews tomorrow, or choose another service under Service.",
            )
            .to_string()
        } else if has("network") {
            format!("{} {e}", l("تعذّر الاتصال بالخدمة:", "Couldn't reach the service:"))
        } else {
            e.to_string()
        }
    }

    /// The model's instructions: who it is, what it may do, and what the person is looking at.
    fn system_prompt(app: &WordApp) -> String {
        let title = app.session.documents().into_iter().find(|d| d.active).map(|d| d.title).unwrap_or_default();
        let selection = wordcraft_agent::error::clip(&app.session.selected_text(), 4000);
        let mut s = format!(
            "You are وكيل (Wakeel), the AI writing agent inside كلمة (Kalima), an Arabic-first word processor. \
             The person is looking at their open document \"{title}\", and you change it live with your tools: \
             every change happens in their window while they watch.\n\
             \n\
             Quality comes first. Write like a gifted professional author and editor: rich, vivid, well-organised \
             prose with a clear beginning, middle and end, varied sentences and precise vocabulary. In Arabic, write \
             eloquent Modern Standard Arabic with correct grammar, hamzas and punctuation (، ؛ ؟ « »), without full \
             diacritics (tashkeel) unless asked. Never pad and never repeat yourself. A story has a title, \
             characters, a setting and an arc over several developed paragraphs; an article or report has a title, \
             an introduction, headed sections and a conclusion; a letter has a date, the addressee, a greeting, a \
             clear request, a courteous closing and a signature block. Use [square brackets] only for personal \
             details the person must fill in (their name, an ID, a date), never for content you can write. Don't \
             sign as وكيل or invent an author.\n\
             \n\
             Writing new content (a story, article, report, letter, CV, plan, poem…): call write_document ONCE \
             with the complete content as HTML. <h1> is the title, <h2>/<h3> are section headings, one <p> per \
             paragraph, <ul>/<ol> for lists, <strong>/<em> for emphasis, <blockquote> for a quotation, <table> \
             only for real tabular data (never for layout). Use mode \"replace\" only when the document is empty \
             or you rewrite the whole of it (correcting, shortening or translating it, or the person asked to \
             replace it); otherwise \"append\", so the person's text is never lost. Never type long text piece \
             by piece, never write the same content twice, and never write Markdown (** or #): use HTML tags.\n\
             \n\
             Design tastefully with inline CSS in that HTML: centre the title (text-align:center) and give it and \
             the section headings one calm accent colour (such as #1F3A5F, #0F5132 or #7A3E00); a short subtitle \
             under the title may be centred and grey. Justify body paragraphs (text-align:justify) and keep their \
             text black and their size normal. In a table, make the header row <th> cells bold with a light \
             background (such as #E8EEF6). Don't set font-family (the document's Arabic font is used), and don't use \
             <div> wrappers, margins, widths or borders. Restraint looks professional: no rainbow colours, no huge \
             fonts.\n\
             \n\
             Changing an existing document: read it first (get_text; inspect_document shows styles and formatting).\n\
             - Correcting, improving, shortening or translating it: keep its structure. Rewrite the paragraphs as \
             they are; don't add a title, headings or sections it didn't have. If the whole document changes, call \
             write_document with mode \"replace\" and the same structure (its title as <h1> only if it already had \
             one). If only the selection or one passage changes, select_text it and type_text the new wording.\n\
             - Adding to it (a table, a summary, a conclusion, a section): write_document with mode \"append\" adds \
             after the text; never rewrite what is already there to add something.\n\
             - Formatting part of it (bold a heading, colour, size, alignment, style): select_text the exact words, \
             then execute commands on the selection, for example format.bold {{\"value\": true}}, format.growFont, \
             format.size {{\"size\": 20}}, format.color {{\"color\": \"1F3A5F\"}}, para.alignCenter, styles.apply \
             {{\"style\": \"Heading 1\"}}. In batch, every entry is an engine command id (select.text \
             {{\"text\": \"…\"}}, format.bold…), not a tool name. list_commands finds other commands: don't guess ids.\n\
             \n\
             Don't open, save, close or print files: the person does that. Everything you do in one turn is a single \
             Undo step for the person. When you're done, reply in the person's language in one or two short \
             sentences: what you made or changed. Don't paste the content into the chat.\n"
        );
        if selection.trim().is_empty() {
            s.push_str("Nothing is selected now.");
        } else {
            s.push_str(&format!("The person has selected this text:\n<selection>\n{selection}\n</selection>"));
        }
        s
    }

    /// Start a turn with the person's `text`.
    pub(super) fn send(app: &mut WordApp, ctx: &egui::Context, text: String) -> Result<(), String> {
        let text = text.trim().to_string();
        if text.is_empty() {
            return Err("nothing to send".into());
        }
        if !app.has_document() {
            return Err(l("افتح مستندًا أولًا.", "Open a document first.").into());
        }
        let system = system_prompt(app);
        let mark = app.session.undo_mark();
        let doc = app.session.document_id();
        let st = state(app);
        if st.run.is_some() {
            return Err(l("وكيل يعمل الآن.", "The agent is working.").into());
        }
        if !st.ready() {
            return Err(l("اختر خدمة وأدخل مفتاحها أولًا.", "Choose a service and enter its key first.").into());
        }
        let Some(provider) = st.provider() else { return Err("no service".into()) };
        let model = st.model();
        if model.is_empty() {
            return Err(l("اختر نموذجًا أولًا.", "Choose a model first.").into());
        }
        let price = st.models.get(&provider.id).and_then(|v| v.iter().find(|m| m.id == model)).and_then(|m| m.price);
        let cfg = RunConfig { key: st.key(&provider.id), provider, model, system, effort: st.settings.effort.clone(), price };
        st.chat.push(Message::user_text(text));
        st.notice = None;
        let history = st.chat.clone();
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut tools = AppTools { tx: st.req_tx.clone(), ctx: ctx.clone(), read_doc: false };
        let (worker_cancel, worker_ctx) = (cancel.clone(), ctx.clone());
        let spawned = std::thread::Builder::new().name("agent".into()).spawn(move || {
            let mut allowed = std::collections::HashSet::new();
            let r = wordcraft_agent::run::run(&cfg, &mut tools, history, &mut allowed, &worker_cancel, &mut |e| {
                let _ = tx.send(Msg::Event(e));
                worker_ctx.request_repaint();
            });
            let error = match r {
                Ok(()) | Err(Error::Cancelled) => None,
                // Worded in `drain`, on the window's thread, where the interface language is set.
                Err(e) => Some(e.to_string()),
            };
            let _ = tx.send(Msg::Done(error));
            worker_ctx.request_repaint();
        });
        if let Err(e) = spawned {
            st.chat.pop();
            return Err(e.to_string());
        }
        st.run = Some(Running {
            rx,
            cancel,
            text: String::new(),
            tool: None,
            thinking: false,
            retrying: None,
            started: std::time::Instant::now(),
            mark,
            doc,
        });
        Ok(())
    }

    /// The tools, run in the window through the control channel.
    struct AppTools {
        tx: Sender<ControlRequest>,
        ctx: egui::Context,
        /// The model read the document this turn: only then may it replace text that is there.
        read_doc: bool,
    }

    impl wordcraft_mcp::Backend for AppTools {
        fn call(&mut self, method: &str, params: Value) -> Result<Value, String> {
            let (req, rx) = ControlRequest::new(method, params);
            self.tx.send(req).map_err(|_| "the window closed".to_string())?;
            self.ctx.request_repaint();
            let v = rx.recv_timeout(TOOL_TIMEOUT).map_err(|_| "the window did not answer".to_string())?;
            if v.get("ok").and_then(Value::as_bool) == Some(true) {
                Ok(v.get("result").cloned().unwrap_or(Value::Null))
            } else {
                Err(v.get("error").and_then(Value::as_str).unwrap_or("failed").to_string())
            }
        }
        fn has_ui(&self) -> bool {
            true
        }
        fn describe(&self) -> String {
            "the open window".into()
        }
    }

    /// The agent's own tool for writing whole passages, formatted, in one step.
    const WRITE_DOCUMENT: &str = "write_document";

    /// The most HTML one write_document call takes.
    const MAX_WRITE_HTML: usize = 2 << 20;

    fn write_document_spec() -> ToolSpec {
        ToolSpec {
            name: WRITE_DOCUMENT.into(),
            description: "Write formatted content into the document in one step: a story, article, letter, report, \
                          plan or any passage longer than a sentence. Give complete, well-structured HTML: <h1> for the \
                          title, <h2>/<h3> for section headings, <p> for each paragraph, <ul>/<ol> lists, <strong>, <em>, \
                          <blockquote>, <table>. Inline CSS works for color, font-size, font-weight and text-align. \
                          Arabic paragraphs are made right to left automatically. mode \"replace\" replaces the whole \
                          document (use it for new content in an empty or draft document); \"append\" adds after the \
                          existing text."
                .into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "html": {"type": "string", "description": "The content as HTML (body content only)"},
                    "mode": {"type": "string", "enum": ["replace", "append"]}
                },
                "required": ["html"]
            }),
        }
    }

    /// `\n` and `\r\n` written out as characters (a model escaping twice) as real line breaks.
    fn unescape_breaks(t: &str) -> String {
        if t.contains("\\n") { t.replace("\\r\\n", "\n").replace("\\n", "\n") } else { t.to_string() }
    }

    /// Whether `t` uses Markdown marks a model may write instead of formatting: `**bold**`, or a
    /// line starting with `#`.
    fn looks_like_markdown(t: &str) -> bool {
        t.contains("**") || t.lines().any(|l| l.trim_start().starts_with("# ") || l.trim_start().starts_with("## "))
    }

    /// `**bold**` as `<strong>`, and text with no HTML tags at all read as Markdown: `#` headings,
    /// `-`/`*` list items and one paragraph per line, so no mark ends up in the document as text.
    fn markdown_to_html(t: &str) -> String {
        let has_tags = t.contains("</") || t.contains("<p") || t.contains("<h") || t.contains("<br");
        let body = if has_tags {
            t.to_string()
        } else {
            let esc = |s: &str| s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
            let mut out = String::new();
            let mut in_list = false;
            for line in t.lines().map(str::trim) {
                let item = line.strip_prefix("- ").or_else(|| line.strip_prefix("* "));
                if in_list && item.is_none() {
                    out.push_str("</ul>");
                    in_list = false;
                }
                if line.is_empty() {
                    continue;
                }
                if let Some(i) = item {
                    if !in_list {
                        out.push_str("<ul>");
                        in_list = true;
                    }
                    out.push_str(&format!("<li>{}</li>", esc(i)));
                } else if let Some(h) = line.strip_prefix("### ") {
                    out.push_str(&format!("<h3>{}</h3>", esc(h)));
                } else if let Some(h) = line.strip_prefix("## ") {
                    out.push_str(&format!("<h2>{}</h2>", esc(h)));
                } else if let Some(h) = line.strip_prefix("# ") {
                    out.push_str(&format!("<h1>{}</h1>", esc(h)));
                } else {
                    out.push_str(&format!("<p>{}</p>", esc(line)));
                }
            }
            if in_list {
                out.push_str("</ul>");
            }
            out
        };
        // `**x**` pairs as bold; an odd one left over is dropped.
        let mut body = body;
        if body.matches("**").count() % 2 == 1
            && let Some(i) = body.rfind("**")
        {
            body.replace_range(i..i.saturating_add(2), "");
        }
        let mut out = String::with_capacity(body.len());
        let mut open = false;
        let mut rest = body.as_str();
        while let Some(i) = rest.find("**") {
            out.push_str(rest.get(..i).unwrap_or(""));
            out.push_str(if open { "</strong>" } else { "<strong>" });
            open = !open;
            rest = rest.get(i.saturating_add(2)..).unwrap_or("");
        }
        out.push_str(rest);
        out
    }

    /// Whether `t` has Arabic-script letters.
    fn has_arabic(t: &str) -> bool {
        t.chars().any(|c| matches!(c as u32, 0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF))
    }

    impl AppTools {
        fn exec(&mut self, command: &str, params: Value) -> Result<Value, String> {
            wordcraft_mcp::Backend::call(self, "engine.execute", json!({"command": command, "params": params}))
        }

        /// Paste `html` as formatted content (replacing the document, or after its text), then make
        /// its paragraphs right to left when they are Arabic.
        fn write_document(&mut self, input: &Value) -> Result<String, String> {
            let html = input.get("html").and_then(Value::as_str).map(str::trim).unwrap_or("");
            if html.is_empty() {
                return Err("`html` is empty: send the content to write".into());
            }
            if html.len() > MAX_WRITE_HTML {
                return Err(format!("`html` is too long (at most {MAX_WRITE_HTML} bytes); write it in parts with mode \"append\""));
            }
            let html = markdown_to_html(&unescape_breaks(html));
            let existing = self.exec("document.text", json!({}))?;
            let has_text = existing.get("text").and_then(Value::as_str).or(existing.as_str()).is_some_and(|t| !t.trim().is_empty());
            // Replacing text the model never read would throw away the person's work (a light
            // model asked for "a story" in a document with a list in it did): add after it instead.
            let replace = match input.get("mode").and_then(Value::as_str) {
                Some("replace") => !has_text || self.read_doc,
                Some("append") => false,
                _ => !has_text,
            };
            let start = if replace {
                // Empty the body first: Select All with the caret in a table selects only that
                // cell, and the new document would be written into it.
                self.exec("document.setText", json!({"text": ""}))?;
                self.exec("select.all", json!({}))?
            } else {
                self.exec("caret.docEnd", json!({}))?;
                if has_text {
                    self.exec("text.newParagraph", json!({}))?;
                }
                self.exec("caret.docEnd", json!({}))?
            };
            let start = start.get("anchor").cloned().unwrap_or(Value::Null);
            let end = self.exec("edit.pasteSpecial", json!({"as": "html", "html": html}))?;
            let end = end.get("focus").cloned().unwrap_or(Value::Null);
            if has_arabic(&html) && !start.is_null() && !end.is_null() {
                self.exec("select.range", json!({"anchor": start, "focus": end}))?;
                self.exec("para.rtl", json!({"value": true}))?;
                self.rtl_tables(&start, &end)?;
                self.exec("select.range", json!({"anchor": end, "focus": end}))?;
            }
            Ok(if replace { "Wrote the document." } else { "Added the content after the existing text." }.into())
        }
    }

    impl AppTools {
        /// Make the tables written between `start` and `end` right to left, so their first column
        /// stands at the right as in any Arabic table.
        fn rtl_tables(&mut self, start: &Value, end: &Value) -> Result<(), String> {
            let top = |p: &Value| p.get("path").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_u64);
            let (Some(from), Some(to)) = (top(start), top(end)) else { return Ok(()) };
            let doc = self.exec("document.inspect", json!({}))?;
            let tables: Vec<u64> = doc
                .get("blocks")
                .and_then(Value::as_array)
                .map(|b| {
                    b.iter()
                        .filter(|b| b.get("type").and_then(Value::as_str) == Some("table"))
                        .filter_map(|b| b.get("index").and_then(Value::as_u64))
                        .filter(|i| (from..=to).contains(i))
                        .collect()
                })
                .unwrap_or_default();
            for i in tables {
                let mut at = start.clone();
                if let Some(o) = at.as_object_mut() {
                    o.insert("path".into(), json!([i, 0, 0, 0]));
                    o.insert("off".into(), json!(0));
                }
                self.exec("select.range", json!({"anchor": at, "focus": at}))?;
                self.exec("table.rtl", json!({"value": true}))?;
            }
            Ok(())
        }
    }

    /// The command a tool name stands for, when a model sends a tool name as a command id.
    fn command_for_tool(name: &str) -> Option<&'static str> {
        match name {
            "select_text" => Some("select.text"),
            "type_text" => Some("text.insert"),
            "get_text" => Some("document.text"),
            "inspect_document" => Some("document.inspect"),
            _ => None,
        }
    }

    /// Commands the agent may not run: files, the window and Help belong to the person.
    fn blocked(command: &str) -> bool {
        command.starts_with("file.") || command.starts_with("ui.") || command.starts_with("help.") || command == "view.switchWindows"
    }

    impl Tools for AppTools {
        fn specs(&self) -> Vec<ToolSpec> {
            let defs = wordcraft_mcp::tools::tool_definitions();
            defs.as_array()
                .map(Vec::as_slice)
                .unwrap_or_default()
                .iter()
                .filter_map(|d| {
                    let name = d.get("name")?.as_str()?;
                    TOOLS.contains(&name).then_some(())?;
                    let mut schema = d.get("inputSchema").cloned().unwrap_or(json!({"type": "object"}));
                    // One document: the one in this window.
                    if let Some(props) = schema.get_mut("properties").and_then(Value::as_object_mut) {
                        props.remove("doc");
                    }
                    let description = match name {
                        "type_text" => "Type a few words at the caret (replacing the selection), for small edits. Each line of `text` becomes its own paragraph. For whole passages use write_document.".to_string(),
                        _ => d.get("description")?.as_str()?.to_string(),
                    };
                    Some(ToolSpec { name: name.into(), description, schema })
                })
                .chain(std::iter::once(write_document_spec()))
                .collect()
        }

        fn needs_approval(&self, _name: &str) -> bool {
            false
        }

        fn owner(&self, _name: &str) -> String {
            String::new()
        }

        fn call(&mut self, name: &str, input: &Value) -> wordcraft_agent::Result<String> {
            if input.get(INVALID_INPUT).is_some() {
                return Err(Error::Tool("the arguments were not valid JSON; send an object".into()));
            }
            if matches!(name, "get_text" | "inspect_document") {
                self.read_doc = true;
            }
            if name == WRITE_DOCUMENT {
                return self.write_document(input).map_err(Error::Tool);
            }
            if !TOOLS.contains(&name) {
                return Err(Error::Tool(format!("unknown tool `{name}`")));
            }
            // Markdown typed as text would show its ** and # marks: paste it formatted instead.
            if name == "type_text"
                && let Some(t) = input.get("text").and_then(Value::as_str)
                && looks_like_markdown(&unescape_breaks(t))
            {
                let html = markdown_to_html(&unescape_breaks(t));
                self.exec("edit.pasteSpecial", json!({"as": "html", "html": html})).map_err(Error::Tool)?;
                return Ok("Typed the text, formatted.".into());
            }
            let mut input = input.clone();
            if let Some(o) = input.as_object_mut() {
                o.remove("doc");
                if name == "type_text"
                    && let Some(t) = o.get("text").and_then(Value::as_str)
                {
                    let t = unescape_breaks(t);
                    if t.contains('\n') {
                        o.insert("paragraphs".into(), Value::Bool(true));
                    }
                    o.insert("text".into(), Value::String(t));
                }
            }
            // Models often put the tool names in a batch: run the commands they stand for.
            if name == "batch"
                && let Some(list) = input.get_mut("commands").and_then(Value::as_array_mut)
            {
                for c in list.iter_mut() {
                    if let Some(id) = c.get("command").and_then(Value::as_str).and_then(command_for_tool)
                        && let Some(o) = c.as_object_mut()
                    {
                        o.insert("command".into(), Value::String(id.into()));
                    }
                }
            }
            let commands: Vec<&str> = match name {
                "execute" => input.get("command").and_then(Value::as_str).into_iter().collect(),
                "batch" => input
                    .get("commands")
                    .and_then(Value::as_array)
                    .map(|v| v.iter().filter_map(|c| c.get("command").and_then(Value::as_str)).collect())
                    .unwrap_or_default(),
                _ => Vec::new(),
            };
            if let Some(c) = commands.into_iter().find(|c| blocked(c)) {
                return Err(Error::Tool(format!("`{c}` is for the person to run, not the agent")));
            }
            let r = wordcraft_mcp::tools::call_tool(self, name, &input);
            let text: Vec<&str> = r.content.iter().filter_map(|c| c.get("text").and_then(Value::as_str)).collect();
            let text = wordcraft_agent::error::clip(&text.join("\n"), MAX_RESULT);
            if r.is_error { Err(Error::Tool(text)) } else { Ok(text) }
        }
    }

    /// A tool call as one short line: the command's label where there is one.
    fn step_label(app: &WordApp, name: &str, input: &Value) -> String {
        let command = |id: &str| app.session.registry.get(id).map(|c| crate::i18n::t(c.label).to_string()).unwrap_or_else(|| id.to_string());
        match name {
            "execute" => command(input.get("command").and_then(Value::as_str).unwrap_or("")),
            "batch" => {
                let ids: Vec<String> = input
                    .get("commands")
                    .and_then(Value::as_array)
                    .map(|v| v.iter().filter_map(|c| c.get("command").and_then(Value::as_str)).take(4).map(command).collect())
                    .unwrap_or_default();
                ids.join(l("، ", ", "))
            }
            "type_text" => l("كتابة نص", "Typing text").into(),
            WRITE_DOCUMENT => l("كتابة المستند وتنسيقه", "Writing and formatting the document").into(),
            "select_text" => l("تحديد نص", "Selecting text").into(),
            "get_text" => l("قراءة المستند", "Reading the document").into(),
            "inspect_document" => l("فحص المستند", "Inspecting the document").into(),
            "list_commands" => l("البحث عن أمر", "Looking up commands").into(),
            other => other.into(),
        }
    }

    /// The suggestions shown on an empty chat: label and request.
    fn suggestions() -> [(&'static str, &'static str); 5] {
        [
            (
                l("لخّص المستند", "Summarize this document"),
                l("لخّص المستند في فقرة قصيرة في آخره.", "Summarize the document in a short paragraph at its end."),
            ),
            (
                l("حسّن النص المحدد", "Improve the selected text"),
                l("أعد صياغة النص المحدد ليكون أوضح وأسلس.", "Rewrite the selected text so it's clearer and smoother."),
            ),
            (
                l("صحّح الأخطاء", "Fix spelling and grammar"),
                l("صحّح الأخطاء الإملائية والنحوية في المستند.", "Fix the spelling and grammar mistakes in the document."),
            ),
            (
                l("نسّق كتقرير", "Format as a report"),
                l("نسّق المستند كتقرير: عنوان رئيسي وعناوين للأقسام.", "Format the document as a report: a title and section headings."),
            ),
            (l("أضف جدولًا", "Add a table"), l("أضف جدولًا مناسبًا للمحتوى في مكان المؤشر.", "Add a table that suits the content at the caret.")),
        ]
    }

    pub(super) fn pane(app: &mut WordApp, ui: &mut Ui) {
        let t = Tokens::get(ui.ctx());
        let ctx = ui.ctx().clone();
        let ready = state(app).ready();
        if !ready || state(app).setup.is_some() {
            setup(app, ui, &t);
            return;
        }
        state(app).load_models(&ctx);
        header(app, ui);
        ui.separator();
        // The chat fills the pane above the input box.
        let input_height = 110.0;
        let chat_height = (ui.available_height() - input_height).max(80.0);
        let mut ask: Option<String> = None;
        egui::ScrollArea::vertical().id_salt("agent_chat").max_height(chat_height).auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
            ui.set_width(ui.available_width());
            if state(app).chat.is_empty() && state(app).run.is_none() {
                ask = welcome(ui, &t);
            } else {
                messages(app, ui, &t);
            }
        });
        if let Some(n) = state(app).notice.clone() {
            ui.colored_label(ui.visuals().error_fg_color, n);
        }
        if let Some(text) = ask.or_else(|| input(app, ui)) {
            if let Err(e) = send(app, &ctx, text) {
                state(app).notice = Some(e);
            } else {
                state(app).input.clear();
            }
        }
    }

    fn header(app: &mut WordApp, ui: &mut Ui) {
        let st = state(app);
        let providers = st.settings.providers();
        let current = st.provider().map(|p| p.name).unwrap_or_default();
        let model = st.model();
        let models = st.provider().and_then(|p| st.models.get(&p.id).cloned());
        let mut pick_provider: Option<String> = None;
        let mut pick_model: Option<String> = None;
        let mut new_chat = false;
        let w = ui.available_width();
        ui.horizontal(|ui| {
            craft_ui::ComboBox::from_id_salt("agent_provider").width((w * 0.35).max(60.0)).selected_text(current).show_ui(ui, |ui| {
                for p in &providers {
                    if ui.selectable_label(false, &p.name).clicked() {
                        pick_provider = Some(p.id.clone());
                    }
                }
            });
            let label = models.as_ref().and_then(|v| v.iter().find(|m| m.id == model)).map(|m| m.name.clone()).unwrap_or_else(|| model.clone());
            craft_ui::ComboBox::from_id_salt("agent_model").width((w * 0.45).max(60.0)).selected_text(label).show_ui(ui, |ui| match &models {
                None => {
                    ui.label(l("جارٍ تحميل النماذج…", "Loading models…"));
                }
                Some(list) => {
                    egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                        for m in list {
                            let name = if m.free { format!("{} · {}", m.name, l("مجاني", "free")) } else { m.name.clone() };
                            if ui.selectable_label(m.id == model, name).clicked() {
                                pick_model = Some(m.id.clone());
                            }
                        }
                    });
                }
            });
            if ui.button("+").on_hover_text(l("محادثة جديدة", "New chat")).clicked() {
                new_chat = true;
            }
        });
        if let Some(id) = pick_provider {
            st.settings.provider = id;
            st.settings.model.clear();
            st.notice = None;
            if !st.ready() {
                st.setup = Some(st.settings.provider.clone());
            }
            st.save();
        }
        if let Some(m) = pick_model {
            st.settings.model = m;
            st.save();
        }
        if new_chat && st.run.is_none() {
            st.chat.clear();
            st.notice = None;
            st.cost = 0.0;
        }
    }

    /// The empty chat: what the agent does, and requests to start from.
    fn welcome(ui: &mut Ui, t: &Tokens) -> Option<String> {
        ui.add_space(8.0);
        ui.label(RichText::new(l("مرحبًا، أنا وكيل.", "Hi, I'm your AI Agent.")).font(crate::theme::semibold(16.0)).color(t.text));
        ui.label(
            RichText::new(l(
                "اطلب مني ما تريد في هذا المستند، فأكتب وأنسّق وأعدّل أمامك. يمكنك التراجع عن كل ما أفعله بضغطة واحدة.",
                "Ask me for anything in this document and I'll write, format and edit it as you watch. One Undo takes back everything I did.",
            ))
            .color(t.text_dim),
        );
        ui.add_space(10.0);
        let mut ask = None;
        for (label, request) in suggestions() {
            let b = egui::Button::new(RichText::new(label).color(t.text)).min_size(egui::vec2(ui.available_width(), 28.0));
            if ui.add(b).clicked() {
                ask = Some(request.to_string());
            }
        }
        ask
    }

    fn messages(app: &mut WordApp, ui: &mut Ui, t: &Tokens) {
        let Some(st) = app.agent.state.as_ref() else { return };
        // Which tool calls failed, from their results.
        let mut failed: HashMap<&str, bool> = HashMap::new();
        for m in &st.chat {
            for p in &m.parts {
                if let Part::ToolResult { id, is_error, .. } = p {
                    failed.insert(id.as_str(), *is_error);
                }
            }
        }
        let running_tool = st.run.as_ref().and_then(|r| r.tool.clone());
        let copy = l("نسخ", "Copy");
        for m in &st.chat {
            match m.role {
                Role::User => {
                    let text = m.text();
                    if text.trim().is_empty() {
                        continue;
                    }
                    ui.add_space(6.0);
                    ui.with_layout(Layout::top_down(Align::Max), |ui| {
                        egui::Frame::new().fill(t.checked).corner_radius(10).inner_margin(egui::Margin::symmetric(10, 6)).show(ui, |ui| {
                            ui.set_max_width(ui.available_width() * 0.85);
                            ui.label(RichText::new(text).color(t.text));
                        });
                    });
                }
                Role::Assistant => {
                    for p in &m.parts {
                        match p {
                            Part::Text { text } if !text.trim().is_empty() => {
                                ui.add_space(4.0);
                                crate::agent_markdown::show(ui, text, t, copy);
                            }
                            Part::ToolCall { id, name, input } => {
                                let (mark, color) = match failed.get(id.as_str()) {
                                    Some(false) => ("✓", t.accent_text),
                                    Some(true) => ("✗", ui.visuals().error_fg_color),
                                    None if running_tool.as_deref() == Some(id) => ("…", t.text_dim),
                                    None => ("·", t.text_dim),
                                };
                                let label = step_label(app, name, input);
                                ui.label(RichText::new(format!("{mark} {label}")).small().color(color));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        if let Some(run) = &st.run {
            ui.add_space(4.0);
            if run.text.is_empty() {
                let secs = run.started.elapsed().as_secs();
                let status = match &run.retrying {
                    Some(m) => format!("{} ({m})", l("الخدمة مشغولة، أعيد المحاولة…", "The service is busy, trying again…")),
                    None if run.thinking => l("يفكّر…", "Thinking…").to_string(),
                    None => l("يعمل…", "Working…").to_string(),
                };
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new(status).color(t.text_dim));
                    if secs >= 3 {
                        ui.label(RichText::new(format!("{secs} {}", l("ث", "s"))).small().color(t.text_dim));
                    }
                });
                // Keep the seconds counting while nothing else changes.
                ui.ctx().request_repaint_after(std::time::Duration::from_secs(1));
            } else {
                crate::agent_markdown::show(ui, &run.text, t, copy);
            }
        }
        if st.cost > 0.0 {
            ui.add_space(6.0);
            ui.label(RichText::new(format!("${:.4}", st.cost)).small().color(t.text_dim));
        }
    }

    /// The message box and Send/Stop. Returns the text to send.
    fn input(app: &mut WordApp, ui: &mut Ui) -> Option<String> {
        let st = state(app);
        let running = st.run.is_some();
        let w = ui.available_width();
        let edit = egui::TextEdit::multiline(&mut st.input)
            .id_salt("agent_input")
            .hint_text(l("اطلب من وكيل…", "Ask the AI Agent…"))
            .desired_rows(3)
            .desired_width(w)
            // Enter sends; Shift+Enter starts a new line.
            .return_key(Some(KeyboardShortcut::new(Modifiers::SHIFT, Key::Enter)));
        let r = ui.add(edit);
        let enter = r.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) && !i.modifiers.shift);
        let mut send = None;
        ui.horizontal(|ui| {
            if running {
                if ui.button(l("إيقاف", "Stop")).clicked()
                    && let Some(run) = &st.run
                {
                    run.cancel.store(true, Ordering::Relaxed);
                }
            } else if (ui.add_enabled(!st.input.trim().is_empty(), egui::Button::new(l("إرسال", "Send"))).clicked() || enter)
                && !st.input.trim().is_empty()
            {
                send = Some(st.input.clone());
            }
            if ui.small_button(l("الخدمة", "Service")).clicked() {
                st.setup = Some(st.settings.provider.clone());
                st.key_input.clear();
            }
        });
        send
    }

    /// Choosing a service and entering its key.
    fn setup(app: &mut WordApp, ui: &mut Ui, t: &Tokens) {
        let st = state(app);
        let providers = st.settings.providers();
        let mut chosen = st
            .setup
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| if st.settings.provider.is_empty() { "gemini".into() } else { st.settings.provider.clone() });
        ui.add_space(6.0);
        ui.label(RichText::new(l("اختر خدمة الذكاء الاصطناعي", "Choose an AI service")).font(crate::theme::semibold(15.0)).color(t.text));
        ui.label(
            RichText::new(l(
                "‏Gemini مجاني: أنشئ مفتاحًا من Google AI Studio والصقه هنا. ويقدّم OpenRouter نماذج مجانية أيضًا.",
                "Gemini is free: create a key in Google AI Studio and paste it here. OpenRouter has free models too.",
            ))
            .color(t.text_dim),
        );
        ui.add_space(6.0);
        for id in SETUP {
            let Some(p) = providers.iter().find(|p| p.id == *id) else { continue };
            let name = match *id {
                "gemini" => format!("{} ({})", p.name, l("مجاني", "free")),
                "openrouter" => format!("{} ({})", p.name, l("فيه نماذج مجانية", "has free models")),
                "ollama" => format!("{} ({})", p.name, l("على جهازك", "on this computer")),
                _ => p.name.clone(),
            };
            if ui.radio(chosen == p.id, name).clicked() {
                chosen = p.id.clone();
                st.key_input.clear();
            }
        }
        st.setup = Some(chosen.clone());
        let Some(p) = providers.iter().find(|p| p.id == chosen).cloned() else { return };
        ui.add_space(6.0);
        let has_key = !st.key(&p.id).is_empty();
        if p.needs_key {
            ui.horizontal(|ui| {
                ui.label(l("المفتاح:", "Key:"));
                ui.hyperlink_to(l("احصل على مفتاح", "Get a key"), &p.key_url);
            });
            let hint = if has_key {
                l("محفوظ. الصق مفتاحًا جديدًا لتغييره.", "Saved. Paste a new key to change it.")
            } else {
                l("الصق المفتاح هنا", "Paste the key here")
            };
            ui.add(egui::TextEdit::singleline(&mut st.key_input).password(true).hint_text(hint).desired_width(ui.available_width()));
        } else {
            ui.label(
                RichText::new(l("شغّل Ollama على جهازك أولًا، ثم اضغط «ابدأ».", "Start Ollama on this computer first, then press Start."))
                    .color(t.text_dim),
            );
        }
        ui.add_space(6.0);
        let can = !p.needs_key || has_key || !st.key_input.trim().is_empty();
        let mut start = false;
        let mut cancel = false;
        ui.horizontal(|ui| {
            start = ui
                .add_enabled(
                    can,
                    egui::Button::new(RichText::new(l("ابدأ", "Start")).color(if can { t.on_accent } else { t.text_disabled })).fill(if can {
                        t.accent
                    } else {
                        Color32::TRANSPARENT
                    }),
                )
                .clicked();
            if st.ready() && ui.button(l("إلغاء", "Cancel")).clicked() {
                cancel = true;
            }
        });
        if cancel {
            st.setup = None;
            st.key_input.clear();
        }
        if start {
            let key = st.key_input.trim().to_string();
            if !key.is_empty() {
                let saved = st.dir.as_deref().map(|d| secrets::set(d, &p.id, &key));
                match saved {
                    Some(Ok(())) => {
                        st.keys.insert(p.id.clone(), key);
                    }
                    Some(Err(e)) => {
                        st.notice = Some(format!("{} {e}", l("تعذّر حفظ المفتاح:", "Couldn't save the key:")));
                        return;
                    }
                    None => {
                        // No settings folder: keep the key for this run only.
                        st.keys.insert(p.id.clone(), key);
                    }
                }
            }
            if st.settings.provider != p.id {
                st.settings.model.clear();
            }
            st.settings.provider = p.id.clone();
            st.models.remove(&p.id);
            st.setup = None;
            st.key_input.clear();
            st.notice = None;
            st.save();
        }
        if let Some(n) = &st.notice {
            ui.colored_label(ui.visuals().error_fg_color, n);
        }
        ui.add_space(8.0);
        ui.label(
            RichText::new(l(
                "يُحفظ مفتاحك في سلسلة مفاتيح جهازك ولا يُرسل إلا إلى الخدمة التي اخترتها.",
                "Your key is kept in this computer's keychain and only ever sent to the service you chose.",
            ))
            .small()
            .color(t.text_dim),
        );
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_agent_cannot_touch_files_or_the_window() {
            for c in ["file.save", "file.open", "file.close", "ui.dialog", "help.about", "view.switchWindows"] {
                assert!(blocked(c), "{c}");
            }
            for c in ["format.bold", "insert.table", "text.insert", "select.all"] {
                assert!(!blocked(c), "{c}");
            }
        }

        #[test]
        fn the_agent_sees_only_its_tools_and_one_document() {
            let (tx, _rx) = channel();
            let tools = AppTools { tx, ctx: egui::Context::default(), read_doc: false };
            let specs = tools.specs();
            let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
            assert_eq!(names.len(), TOOLS.len() + 1, "{names:?}");
            assert!(names.contains(&WRITE_DOCUMENT));
            for s in &specs {
                assert!(s.schema.get("properties").and_then(|p| p.get("doc")).is_none(), "{} still offers `doc`", s.name);
            }
        }

        #[test]
        fn escaped_line_breaks_become_real_ones() {
            assert_eq!(unescape_breaks("a\\n\\nb"), "a\n\nb");
            assert_eq!(unescape_breaks("a\nb"), "a\nb");
            assert!(has_arabic("قصة") && !has_arabic("story"));
        }

        #[test]
        fn blocked_commands_never_reach_the_window() {
            let (tx, rx) = channel();
            let mut tools = AppTools { tx, ctx: egui::Context::default(), read_doc: false };
            let e = tools.call("execute", &json!({"command": "file.save"}));
            assert!(e.is_err());
            let e = tools.call("batch", &json!({"commands": [{"command": "format.bold"}, {"command": "file.close"}]}));
            assert!(e.is_err());
            let e = tools.call("screenshot", &json!({}));
            assert!(e.is_err());
            assert!(rx.try_recv().is_err(), "nothing was sent to the window");
        }

        #[test]
        fn markdown_marks_become_formatting() {
            assert_eq!(markdown_to_html("<p>هذا **مهم** جدًا</p>"), "<p>هذا <strong>مهم</strong> جدًا</p>");
            assert_eq!(
                markdown_to_html("# العنوان\nفقرة **أولى**\n- بند\n- آخر\nنهاية"),
                "<h1>العنوان</h1><p>فقرة <strong>أولى</strong></p><ul><li>بند</li><li>آخر</li></ul><p>نهاية</p>"
            );
            assert_eq!(markdown_to_html("a ** b"), "<p>a  b</p>");
            assert!(looks_like_markdown("## قسم\nنص"));
            assert!(!looks_like_markdown("نص عادي"));
        }

        #[test]
        fn errors_read_plainly() {
            assert!(friendly("Gemini returned 401: bad").contains(l("المفتاح", "key")));
            assert!(friendly("OpenRouter returned 429: slow down").contains(l("حد", "limit")));
            let busy = friendly("Gemini returned 503: This model is currently experiencing high demand");
            assert!(busy.contains(l("مزدحمة", "busy")), "{busy}");
        }
    }
}
