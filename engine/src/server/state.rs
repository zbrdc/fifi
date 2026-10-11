//! The engines that the server routes to, and the shared state of the routes.
//!
//! Routing rules: `docs/design/scheduler.md` ("Request flow") and `docs/design/server.md`
//! ("Session handling"). Phase 1 has anonymous sessions only.

use super::error::ServerError;
use super::reqlog::{Sink, default_sink};
use super::wire::{EngineInfo, Model, ModelList, state_name};
use crate::EngError;
use crate::family::sampler::SamplingDefaults;
use crate::sched::worker::{EngineHandle, EngineState};
use crate::tokenizer::TokenizerBundle;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The quiet time after which a stream sends a keep-alive comment (`docs/design/server.md`).
const KEEP_ALIVE: Duration = Duration::from_secs(15);

/// One engine as the routes see it.
pub(crate) struct EngineEntry {
    /// The engine id of `x-engine`.
    pub(crate) id: String,
    /// The model name of `model`.
    pub(crate) model: String,
    pub(crate) handle: EngineHandle,
    pub(crate) bundle: Arc<TokenizerBundle>,
    /// The most tokens of prompt plus completion that the engine accepts. The family of the engine
    /// reports it when it starts. The chat handler reads no other source for it.
    pub(crate) context_limit: u32,
    /// True when the chat handler sends the prompt boundary of each job to the engine
    /// (`prefix_reuse` in the configuration file).
    pub(crate) prefix_reuse: bool,
    /// The sampling defaults of this engine: the `temperature`, `top_p`, and `top_k` that its
    /// `model_dir/generation_config.json` sets, with the contract default for an absent file or
    /// key. The chat handler applies them to every field that a request omits.
    pub(crate) defaults: SamplingDefaults,
}

/// The state that every route shares.
pub(crate) struct AppState {
    api_key: Vec<u8>,
    engines: Vec<EngineEntry>,
    started: u64,
    requests: AtomicU64,
    keep_alive: Duration,
    /// The ids of the engines whose failure text the log already holds.
    reported: Mutex<HashSet<String>>,
    /// Receives each log line. The default writes it to standard error.
    log: Box<dyn Fn(&str) + Send + Sync>,
    /// Receives the request log lines. The default is standard error.
    request_log: Sink,
}

/// The current Unix time in seconds.
pub(crate) fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

impl AppState {
    /// Creates the state. The caller has checked that the key is not empty.
    pub(crate) fn new(api_key: &str, engines: Vec<EngineEntry>) -> Self {
        Self {
            api_key: api_key.as_bytes().to_vec(),
            engines,
            started: unix_seconds(),
            requests: AtomicU64::new(0),
            keep_alive: KEEP_ALIVE,
            reported: Mutex::new(HashSet::new()),
            log: Box::new(|line| eprintln!("error: {line}")),
            request_log: default_sink(),
        }
    }

    /// The sink of the request log lines.
    pub(crate) fn request_log(&self) -> Sink {
        Arc::clone(&self.request_log)
    }

    /// Replaces the request log sink. Tests use this to read the request log lines.
    #[cfg(test)]
    pub(crate) fn with_request_log(mut self, sink: impl std::io::Write + Send + 'static) -> Self {
        self.request_log = Arc::new(Mutex::new(sink));
        self
    }

    /// Replaces the log sink. Tests use this to read the log lines.
    #[cfg(test)]
    pub(crate) fn with_log(mut self, sink: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.log = Box::new(sink);
        self
    }

    /// Changes the keep-alive interval. Tests use this to avoid a wait of 15 seconds.
    #[cfg(test)]
    pub(crate) fn with_keep_alive(mut self, interval: Duration) -> Self {
        self.keep_alive = interval;
        self
    }

    /// The quiet time after which a stream sends a keep-alive comment.
    pub(crate) fn keep_alive(&self) -> Duration {
        self.keep_alive
    }

    /// The configured bearer key.
    pub(crate) fn api_key(&self) -> &[u8] {
        &self.api_key
    }

    /// A new completion id that is unique in this process.
    pub(crate) fn next_completion_id(&self) -> String {
        let number = self.requests.fetch_add(1, Ordering::Relaxed) + 1;
        format!("chatcmpl-{number:016x}")
    }

    /// Picks the engine of a request.
    /// Phase 1 has no sessions, so every `x-session` value is `NotFound`.
    /// Without a session, `x-engine` wins over `model`.
    pub(crate) fn resolve(
        &self,
        session: Option<&str>,
        engine: Option<&str>,
        model: Option<&str>,
    ) -> Result<&EngineEntry, ServerError> {
        if let Some(id) = session {
            return Err(ServerError::NotFound(format!(
                "session {id} does not exist"
            )));
        }
        if let Some(id) = engine {
            return self
                .engines
                .iter()
                .find(|entry| entry.id == id)
                .ok_or_else(|| ServerError::NotFound(format!("engine {id} does not exist")));
        }
        if let Some(name) = model {
            return self
                .engines
                .iter()
                .find(|entry| entry.model == name)
                .ok_or_else(|| ServerError::NotFound(format!("model {name} does not exist")));
        }
        Err(ServerError::Eng(EngError::BadRequest(
            "one of model, x-engine, or x-session is required".to_string(),
        )))
    }

    /// True when at least one engine can take a job now.
    pub(crate) fn is_ready(&self) -> bool {
        self.engines
            .iter()
            .any(|entry| matches!(entry.handle.state(), EngineState::Ready | EngineState::Busy))
    }

    /// The message of the `not_ready` answer: the state of every engine. It never shows the text
    /// of a failure, because the route is public. `new_failures` supplies that text for the log.
    pub(crate) fn not_ready_message(&self) -> String {
        if self.engines.is_empty() {
            return "no engine is configured".to_string();
        }
        let parts: Vec<String> = self
            .engines
            .iter()
            .map(|entry| format!("{} is {}", entry.id, state_name(entry.handle.state())))
            .collect();
        format!("no engine is ready: {}", parts.join("; "))
    }

    /// The failure texts that the log does not hold yet, one for each failed engine.
    /// A second call returns nothing for an engine that the first call reported.
    fn new_failures(&self) -> Vec<String> {
        let mut reported = self.reported.lock().unwrap_or_else(PoisonError::into_inner);
        self.engines
            .iter()
            .filter_map(|entry| {
                let cause = entry.handle.failure()?;
                reported
                    .insert(entry.id.clone())
                    .then(|| format!("engine {} failed: {cause}", entry.id))
            })
            .collect()
    }

    /// Writes each new engine failure to the log, once for each failed engine.
    pub(crate) fn log_new_failures(&self) {
        for line in self.new_failures() {
            (self.log)(&line);
        }
    }

    /// The `GET /v1/models` answer. Engines that hold one model share one entry.
    pub(crate) fn models(&self) -> ModelList {
        let mut models: Vec<Model> = Vec::new();
        for entry in &self.engines {
            let info = EngineInfo {
                id: entry.id.clone(),
                model: entry.model.clone(),
                state: state_name(entry.handle.state()),
            };
            match models.iter_mut().find(|model| model.id == entry.model) {
                Some(model) => model.engines.push(info),
                None => models.push(Model::new(entry.model.clone(), self.started, vec![info])),
            }
        }
        ModelList::new(models)
    }
}

#[cfg(test)]
mod tests {
    use super::{AppState, unix_seconds};
    use crate::EngError;
    use crate::sched::worker::EngineState;
    use crate::server::error::ServerError;
    use crate::server::testkit;

    fn two_engines() -> (AppState, Vec<crate::sched::worker::Engine>) {
        let first = testkit::engine(Vec::new());
        let second = testkit::engine(Vec::new());
        let entries = vec![
            testkit::entry("e1", "model-a", &first, 4_096),
            testkit::entry("e2", "model-b", &second, 4_096),
        ];
        (
            AppState::new(testkit::API_KEY, entries),
            vec![first, second],
        )
    }

    #[test]
    fn x_engine_takes_precedence_over_model() {
        let (state, _engines) = two_engines();
        let picked = state
            .resolve(None, Some("e2"), Some("model-a"))
            .expect("resolve");
        assert_eq!(picked.id, "e2");
    }

    #[test]
    fn a_model_name_picks_the_first_engine_that_holds_it() {
        let engines = [
            testkit::engine(Vec::new()),
            testkit::engine(Vec::new()),
            testkit::engine(Vec::new()),
        ];
        let entries = vec![
            testkit::entry("e1", "model-a", &engines[0], 4_096),
            testkit::entry("e2", "model-b", &engines[1], 4_096),
            testkit::entry("e3", "model-b", &engines[2], 4_096),
        ];
        let state = AppState::new(testkit::API_KEY, entries);
        let picked = state.resolve(None, None, Some("model-b")).expect("resolve");
        assert_eq!(picked.id, "e2");
    }

    #[test]
    fn unknown_engine_or_model_returns_404() {
        let (state, _engines) = two_engines();
        let engine = state.resolve(None, Some("e9"), None).err().expect("engine");
        assert_eq!(engine.status().as_u16(), 404);
        let model = state
            .resolve(None, None, Some("nope"))
            .err()
            .expect("model");
        assert_eq!(model.status().as_u16(), 404);
        let both = state
            .resolve(None, Some("e9"), Some("model-a"))
            .err()
            .expect("both");
        assert_eq!(
            both.status().as_u16(),
            404,
            "a bad x-engine does not fall back to model"
        );
    }

    #[test]
    fn x_session_missing_returns_404() {
        let (state, _engines) = two_engines();
        let error = state
            .resolve(Some("s1"), Some("e1"), Some("model-a"))
            .err()
            .expect("session");
        assert_eq!(error.status().as_u16(), 404);
        assert_eq!(error.code(), "not_found");
    }

    #[test]
    fn a_request_without_any_routing_field_is_a_bad_request() {
        let (state, _engines) = two_engines();
        let error = state.resolve(None, None, None).err().expect("no route");
        assert!(matches!(error, ServerError::Eng(EngError::BadRequest(_))));
    }

    #[test]
    fn completion_ids_are_unique_and_fixed_width() {
        let (state, _engines) = two_engines();
        let first = state.next_completion_id();
        let second = state.next_completion_id();
        assert_eq!(first, "chatcmpl-0000000000000001");
        assert_eq!(second, "chatcmpl-0000000000000002");
    }

    #[test]
    fn the_state_is_ready_only_while_an_engine_can_take_a_job() {
        let (state, _engines) = two_engines();
        for entry in &state.engines {
            testkit::wait_for_state(&entry.handle, EngineState::Ready);
        }
        assert!(state.is_ready());
        let broken = testkit::failing_engine();
        let entries = vec![testkit::entry("e1", "model-a", &broken, 4_096)];
        let down = AppState::new(testkit::API_KEY, entries);
        testkit::wait_for_state(&down.engines[0].handle, EngineState::Error);
        assert!(!down.is_ready());
        let empty = AppState::new(testkit::API_KEY, Vec::new());
        assert!(!empty.is_ready());
    }

    #[test]
    fn the_not_ready_message_names_each_engine_and_its_state_and_hides_the_failure() {
        let broken = testkit::failing_engine();
        let (waiting, gate) = testkit::gated_engine(Vec::new());
        let entries = vec![
            testkit::entry("e1", "model-a", &broken, 4_096),
            testkit::entry("e2", "model-b", &waiting, 4_096),
        ];
        let state = AppState::new(testkit::API_KEY, entries);
        testkit::wait_for_state(&state.engines[0].handle, EngineState::Error);
        let message = state.not_ready_message();
        assert!(message.starts_with("no engine is ready: "), "{message}");
        assert!(message.contains("e1 is error"), "{message}");
        assert!(!message.contains("fixture load failure"), "{message}");
        assert!(message.contains("e2 is loading"), "{message}");
        drop(gate);
    }

    #[test]
    fn a_failure_text_goes_to_the_log_once_for_each_failed_engine() {
        let broken = testkit::failing_engine();
        let (waiting, gate) = testkit::gated_engine(Vec::new());
        let entries = vec![
            testkit::entry("e1", "model-a", &broken, 4_096),
            testkit::entry("e2", "model-b", &waiting, 4_096),
        ];
        let state = AppState::new(testkit::API_KEY, entries);
        testkit::wait_for_state(&state.engines[0].handle, EngineState::Error);
        let first = state.new_failures();
        assert_eq!(first.len(), 1, "{first:?}");
        assert!(first[0].starts_with("engine e1 failed: "), "{first:?}");
        assert!(first[0].contains("fixture load failure"), "{first:?}");
        assert!(
            state.new_failures().is_empty(),
            "the text was reported twice"
        );
        drop(gate);
    }

    #[test]
    fn the_not_ready_message_of_a_server_without_engines_says_so() {
        let state = AppState::new(testkit::API_KEY, Vec::new());
        assert_eq!(state.not_ready_message(), "no engine is configured");
    }

    #[test]
    fn the_keep_alive_interval_is_fifteen_seconds_unless_a_test_changes_it() {
        let state = AppState::new(testkit::API_KEY, Vec::new());
        assert_eq!(state.keep_alive(), std::time::Duration::from_secs(15));
        let quick = state.with_keep_alive(std::time::Duration::from_millis(5));
        assert_eq!(quick.keep_alive(), std::time::Duration::from_millis(5));
    }

    #[test]
    fn models_group_engines_by_model_name() {
        let first = testkit::engine(Vec::new());
        let second = testkit::engine(Vec::new());
        let third = testkit::engine(Vec::new());
        let entries = vec![
            testkit::entry("e1", "model-a", &first, 4_096),
            testkit::entry("e2", "model-b", &second, 4_096),
            testkit::entry("e3", "model-a", &third, 4_096),
        ];
        let state = AppState::new(testkit::API_KEY, entries);
        let list = serde_json::to_value(state.models()).expect("serialize");
        let ids: Vec<&str> = list["data"]
            .as_array()
            .expect("data")
            .iter()
            .map(|model| model["id"].as_str().expect("id"))
            .collect();
        assert_eq!(ids, ["model-a", "model-b"]);
        let engines_of_a: Vec<&str> = list["data"][0]["engines"]
            .as_array()
            .expect("engines")
            .iter()
            .map(|engine| engine["id"].as_str().expect("engine id"))
            .collect();
        assert_eq!(engines_of_a, ["e1", "e3"]);
    }

    #[test]
    fn the_model_creation_time_is_the_start_time() {
        let before = unix_seconds();
        let (state, _engines) = two_engines();
        let created =
            serde_json::to_value(state.models()).expect("serialize")["data"][0]["created"]
                .as_u64()
                .expect("created");
        assert!(created >= before && created <= unix_seconds(), "{created}");
    }
}
