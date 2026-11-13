//! The router: the routes of the first server and the authentication layer.
//!
//! Routes: `GET /health`, `GET /ready`, `GET /v1/models`, `POST /v1/chat/completions`,
//! `POST /v1/completions`, `POST /v1/messages`, `POST /v1/messages/count_tokens`, and the server
//! extensions `POST /v1/token/encode`, `POST /v1/token/decode`, `POST /v1/apply-template`, and
//! `GET /v1/recent-requests` (the GUI's live speed panel) and `GET /v1/sampler-trace` (one
//! reply's armed sampler-trace packet; see `sampler_trace.rs`) and `GET /v1/route-trace` (the
//! engine-wide routing stage ring; see `route_trace.rs`). The extensions are not in the
//! client contract.
//! A route that the contract does not define, or that this server does not ship, gives a JSON 404.
//! A method that a served path does not accept counts as an undefined route, because the contract
//! defines no 405. `HEAD` on a `GET` route stays as axum serves it.

use super::anthropic::{count_tokens, messages};
use super::auth::{check_bearer, check_messages};
use super::chat::chat;
use super::completions::completions;
use super::error::ServerError;
use super::route_trace::get_route_trace;
use super::sampler_trace::get_sampler_trace;
use super::state::AppState;
use super::tokenize::{apply_template, decode, encode};
use super::wire::{ModelList, state_name};
use axum::Json;
use axum::Router;
use axum::extract::{Path, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::json;
use std::sync::Arc;

/// `/health` and `/ready` are public: the contract marks them with an empty `security` array.
pub(crate) fn router(state: Arc<AppState>) -> Router {
    let bearer_routes = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/system", get(system))
        .route("/v1/telemetry", get(telemetry))
        .route("/v1/recent-requests", get(recent_requests))
        .route("/v1/sampler-trace", get(get_sampler_trace))
        .route("/v1/route-trace", get(get_route_trace))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/v1/token/encode", post(encode))
        .route("/v1/token/decode", post(decode))
        .route("/v1/apply-template", post(apply_template))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_auth));
    let messages_routes = Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_messages_auth,
        ));
    let mut app = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .merge(bearer_routes)
        .merge(messages_routes);
    if state.web_dir().is_some() {
        // The frontend shell, like `/health`, is public: it is static
        // files with no data and no key. Only `/` and `/assets/` are
        // ever served from disk, and the API routes above keep their
        // authentication and precedence.
        app = app
            .route("/", get(web_index))
            .route("/assets/{*path}", get(web_asset));
    }
    app.fallback(unknown_route)
        .method_not_allowed_fallback(unknown_route)
        .with_state(state)
}

/// One file of the built frontend, read from `web_dir`. The extension
/// whitelist is what a Vite build emits; anything else is the JSON 404,
/// so a stray file in the directory is never served by accident. The
/// canonicalized path must stay inside the canonicalized directory:
/// `..`, absolute paths, and symlink escapes all end at the 404.
fn web_file(state: &AppState, relative: &std::path::Path) -> Option<(Vec<u8>, &'static str)> {
    let content_type = match relative.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("ico") => "image/x-icon",
        Some("png") => "image/png",
        Some("woff2") => "font/woff2",
        _ => return None,
    };
    let root = state.web_dir()?.canonicalize().ok()?;
    let path = root.join(relative).canonicalize().ok()?;
    if !path.starts_with(&root) || !path.is_file() {
        return None;
    }
    std::fs::read(path).ok().map(|body| (body, content_type))
}

fn web_response(body: Vec<u8>, content_type: &'static str, cache: &'static str) -> Response {
    (
        [
            (axum::http::header::CONTENT_TYPE, content_type),
            (axum::http::header::CACHE_CONTROL, cache),
            (
                axum::http::header::HeaderName::from_static("x-content-type-options"),
                "nosniff",
            ),
        ],
        body,
    )
        .into_response()
}

async fn web_index(State(state): State<Arc<AppState>>) -> Response {
    match web_file(&state, std::path::Path::new("index.html")) {
        // The shell names the hashed assets, so it is revalidated every
        // load; a rebuilt frontend is never masked by a cached shell.
        Some((body, content_type)) => web_response(body, content_type, "no-cache"),
        None => unknown_route().await.into_response(),
    }
}

async fn web_asset(State(state): State<Arc<AppState>>, Path(path): Path<String>) -> Response {
    let relative = std::path::Path::new("assets").join(&path);
    match web_file(&state, &relative) {
        // Vite asset names carry a content hash: cache them as such.
        Some((body, content_type)) => {
            web_response(body, content_type, "public, max-age=31536000, immutable")
        }
        None => unknown_route().await.into_response(),
    }
}

async fn require_auth(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    if request
        .headers()
        .get_all(axum::http::header::AUTHORIZATION)
        .iter()
        .count()
        > 1
    {
        return ServerError::Unauthorized.into_response();
    }
    match check_bearer(request.headers(), state.api_key()) {
        Ok(()) => next.run(request).await,
        Err(error) => error.into_response(),
    }
}

/// The auth of the `/v1/messages` routes: the Anthropic `x-api-key` header or the bearer form.
/// Its failures carry the Anthropic error shape, not the chat one.
async fn require_messages_auth(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let failure = if request
        .headers()
        .get_all(axum::http::header::AUTHORIZATION)
        .iter()
        .count()
        > 1
    {
        Some(ServerError::Unauthorized)
    } else {
        check_messages(request.headers(), state.api_key()).err()
    };
    match failure {
        None => next.run(request).await,
        Some(error) => super::anthropic::error_response(&error),
    }
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

/// Only this handler builds the `not_ready` body (the server design, "Error codes").
/// It logs each new engine failure before the readiness check, so a failed engine is logged once
/// even when another engine makes the answer 200.
async fn ready(State(state): State<Arc<AppState>>) -> Response {
    state.log_new_failures();
    if state.is_ready() {
        return Json(json!({"status": "ready"})).into_response();
    }
    let message = state.not_ready_message();
    let body = json!({"error": {"message": message, "code": "not_ready"}});
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
}

/// `GET /v1/system` (authenticated): what this build and process are,
/// without probing devices. Device facts deliberately stay in the
/// short-lived `enginesrv doctor` process: a CUDA query here could
/// latch the serving backend terminal on failure, and a detected
/// device says nothing about what the loaded executors use. So this
/// reports compiled-in backends, host memory and threads, and the
/// engines with their resolved context limits — never a device list.
async fn system(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    use crate::capability::Probe;
    let host = crate::capability::probe::probe_host();
    let backends = crate::capability::probe::CompiledBackends::of_this_build();
    let mut compiled = Vec::new();
    if backends.cuda {
        compiled.push("cuda");
    }
    if backends.mlx {
        compiled.push("mlx");
    }
    let probe_json = |probe: &Probe<u64>| match probe {
        Probe::Measured(bytes) => serde_json::json!(bytes),
        Probe::Unavailable(reason) => serde_json::json!({"unavailable": reason}),
    };
    let engines: Vec<serde_json::Value> = state
        .entries()
        .iter()
        .map(|entry| {
            serde_json::json!({
                "id": entry.id,
                "model": entry.model,
                "state": state_name(entry.handle.state()),
                "queued": entry.handle.queued(),
                "context_limit": entry.context_limit,
                // The retained rows the engine started with (empty where a
                // family composes no per-key origins) — not a re-resolution.
                "settings": entry.effective,
            })
        })
        .collect();
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "backends_compiled": compiled,
        "host": {
            "cpu_threads": host.cpu_threads,
            "mem_total_bytes": probe_json(&host.mem_total_bytes),
            "mem_available_bytes": probe_json(&host.mem_available_bytes),
        },
        "engines": engines,
    }))
}

async fn models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    Json(state.models())
}

/// The live speed panel's data: this server's recent request rows, newest
/// first (see `reqlog` for the allowlist and the endpoint filter). Dumb by
/// agreement: no filtering, no aggregation — the panel computes its own
/// medians.
/// GPU telemetry for the monitor. The probe runs `nvidia-smi` as a
/// subprocess: a failed driver tool cannot latch the serving backend
/// the way an in-process CUDA query could (the rule /v1/system
/// states). No GPU tool, no rows — `error` says why, with status 200,
/// because the absence is data for the panel, not a request failure.
async fn telemetry(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    if let Some(cached) = state.telemetry_cached() {
        return Json(cached);
    }
    let probed = tokio::task::spawn_blocking(probe_gpus).await;
    let value = match probed {
        Ok(value) => value,
        Err(_) => json!({"gpus": [], "source": "nvidia-smi", "error": "probe task failed"}),
    };
    state.telemetry_store(value.clone());
    Json(value)
}

fn probe_gpus() -> serde_json::Value {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            json!({"gpus": parse_gpu_csv(&String::from_utf8_lossy(&out.stdout)), "source": "nvidia-smi", "error": null})
        }
        Ok(out) => json!({
            "gpus": [],
            "source": "nvidia-smi",
            "error": format!("nvidia-smi exited {}", out.status),
        }),
        Err(error) => json!({
            "gpus": [],
            "source": "nvidia-smi",
            "error": format!("nvidia-smi unavailable: {error}"),
        }),
    }
}

/// One JSON object per GPU line of `nvidia-smi` CSV output. Memory
/// converts MiB to bytes; unparseable fields stay null, never zero.
fn parse_gpu_csv(text: &str) -> Vec<serde_json::Value> {
    let number = |field: Option<&str>| -> Option<f64> { field?.trim().parse::<f64>().ok() };
    let mib_bytes = |field: Option<&str>| -> Option<u64> {
        number(field).map(|mib| (mib * 1024.0 * 1024.0) as u64)
    };
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            json!({
                "index": number(fields.first().copied()).map(|v| v as u64),
                "name": fields.get(1).map(|name| name.trim().to_string()),
                "util_percent": number(fields.get(2).copied()),
                "mem_used_bytes": mib_bytes(fields.get(3).copied()),
                "mem_total_bytes": mib_bytes(fields.get(4).copied()),
                "temp_c": number(fields.get(5).copied()),
                "power_w": number(fields.get(6).copied()),
            })
        })
        .collect()
}

async fn recent_requests(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({ "requests": state.recent_ring().rows_newest_first() }))
}

async fn unknown_route() -> ServerError {
    ServerError::NotFound("route not found".to_string())
}

#[cfg(test)]
mod tests {
    use super::router;
    use crate::sched::worker::EngineState;
    use crate::server::contract::{assert_pointer, assert_schema};
    use crate::server::state::AppState;
    use crate::server::testkit;
    use axum::http::StatusCode;
    use std::sync::{Arc, Mutex};

    fn one_engine_state() -> (Arc<AppState>, crate::sched::worker::Engine) {
        let engine = testkit::engine(Vec::new());
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let entry = testkit::entry("e1", "model-a", &engine, 4_096);
        (
            Arc::new(AppState::new(testkit::API_KEY, vec![entry])),
            engine,
        )
    }

    #[tokio::test]
    async fn missing_bearer_returns_401_error_json() {
        let (state, _engine) = one_engine_state();
        let (status, _headers, body) =
            testkit::send(&router(state), testkit::get("/v1/models", None)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let body = testkit::json_of(&body);
        assert_schema("Error", &body);
        assert_eq!(body["error"]["code"], "unauthorized");
    }

    #[tokio::test]
    async fn a_wrong_bearer_returns_401() {
        let (state, _engine) = one_engine_state();
        let request = testkit::get("/v1/models", Some("sk-wrong"));
        let (status, _headers, _body) = testkit::send(&router(state), request).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn two_authorization_headers_are_unauthorized() {
        let (state, _engine) = one_engine_state();
        let mut request = testkit::get("/v1/models", Some(testkit::API_KEY));
        request.headers_mut().append(
            axum::http::header::AUTHORIZATION,
            axum::http::HeaderValue::from_str("Bearer sk-other").expect("header value"),
        );
        let (status, _headers, body) = testkit::send(&router(state), request).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let body = testkit::json_of(&body);
        assert_schema("Error", &body);
        assert_eq!(body["error"]["code"], "unauthorized");
    }

    #[tokio::test]
    async fn public_routes_skip_bearer_check() {
        let (state, _engine) = one_engine_state();
        let app = router(state);
        let (health, _, health_body) = testkit::send(&app, testkit::get("/health", None)).await;
        assert_eq!(health, StatusCode::OK);
        assert_pointer(
            "/paths/~1health/get/responses/200/content/application~1json/schema",
            &testkit::json_of(&health_body),
        );
        let (ready, _, ready_body) = testkit::send(&app, testkit::get("/ready", None)).await;
        assert_eq!(ready, StatusCode::OK);
        assert_pointer(
            "/paths/~1ready/get/responses/200/content/application~1json/schema",
            &testkit::json_of(&ready_body),
        );
    }

    #[tokio::test]
    async fn only_health_ready_and_metrics_answer_without_the_key() {
        let (state, _engine) = one_engine_state();
        let app = router(state);
        for uri in ["/health", "/ready", "/metrics"] {
            let (status, _headers, _body) = testkit::send(&app, testkit::get(uri, None)).await;
            assert_ne!(status, StatusCode::UNAUTHORIZED, "{uri}");
        }
        let (status, _headers, _body) = testkit::send(&app, testkit::get("/v1/models", None)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let chat = testkit::post_bytes("/v1/chat/completions", None, b"{}".to_vec());
        let (status, _headers, _body) = testkit::send(&app, chat).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn ready_returns_503_without_ready_engine() {
        let broken = testkit::failing_engine();
        let entry = testkit::entry("e1", "model-a", &broken, 4_096);
        testkit::wait_for_state(&entry.handle, EngineState::Error);
        let state = Arc::new(AppState::new(testkit::API_KEY, vec![entry]));
        let (status, _headers, body) =
            testkit::send(&router(state), testkit::get("/ready", None)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let body = testkit::json_of(&body);
        assert_pointer(
            "/paths/~1ready/get/responses/503/content/application~1json/schema",
            &body,
        );
        assert_eq!(body["error"]["code"], "not_ready");
        let message = body["error"]["message"].as_str().expect("message");
        assert!(message.contains("e1 is error"), "{message}");
        assert!(
            !message.contains("fixture load failure"),
            "the public body shows the failure text: {message}"
        );
    }

    #[tokio::test]
    async fn models_route_lists_models_with_engines() {
        let (state, _engine) = one_engine_state();
        let request = testkit::get("/v1/models", Some(testkit::API_KEY));
        let (status, _headers, body) = testkit::send(&router(state), request).await;
        assert_eq!(status, StatusCode::OK);
        let body = testkit::json_of(&body);
        assert_pointer(
            "/paths/~1v1~1models/get/responses/200/content/application~1json/schema",
            &body,
        );
        assert_eq!(body["data"][0]["id"], "model-a");
        assert_eq!(body["data"][0]["engines"][0]["id"], "e1");
        assert_eq!(body["data"][0]["engines"][0]["state"], "ready");
    }

    #[tokio::test]
    async fn system_route_reports_process_facts_without_probing_devices() {
        let (state, engine) = one_engine_state();
        let app = router(state);
        let _keep = &engine;
        let (status, _h, _b) = testkit::send(&app, testkit::get("/v1/system", None)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (status, _h, body) =
            testkit::send(&app, testkit::get("/v1/system", Some(testkit::API_KEY))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            !String::from_utf8_lossy(&body).contains(testkit::API_KEY),
            "the api key must never appear"
        );
        let body = testkit::json_of(&body);
        assert_pointer(
            "/paths/~1v1~1system/get/responses/200/content/application~1json/schema",
            &body,
        );
        assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
        // The compiled-in backends this build carries: none on the plain CPU test build,
        // "mlx" on a native-mlx build, "cuda" on a native-cuda build.
        if cfg!(feature = "native-mlx") {
            assert_eq!(body["backends_compiled"], serde_json::json!(["mlx"]));
        } else if cfg!(all(feature = "native-cuda", target_os = "linux")) {
            assert_eq!(body["backends_compiled"], serde_json::json!(["cuda"]));
        } else {
            assert_eq!(body["backends_compiled"], serde_json::json!([]));
        }
        assert!(body["host"]["cpu_threads"].as_u64().is_some_and(|n| n >= 1));
        assert_eq!(body["engines"][0]["id"], "e1");
        assert_eq!(body["engines"][0]["context_limit"], 4_096);
        // No device claims of any kind from a serving process.
        let text = body.to_string();
        assert!(!text.contains("vram") && !text.contains("device"), "{text}");
    }

    #[tokio::test]
    async fn system_route_reports_the_retained_effective_settings() {
        use crate::capability::effective::{EffectiveSetting, Origin};
        let engine = testkit::engine(Vec::new());
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let rows = vec![EffectiveSetting {
            key: "context_limit",
            value: "32768".to_string(),
            origin: Origin::ConfigFile,
        }];
        let entry = testkit::entry_with_effective("e1", "model-a", &engine, 4_096, rows);
        let state = Arc::new(AppState::new(testkit::API_KEY, vec![entry]));
        let app = router(state);
        let _keep = &engine;
        let request = testkit::get("/v1/system", Some(testkit::API_KEY));
        let (status, _headers, body) = testkit::send(&app, request).await;
        assert_eq!(status, StatusCode::OK);
        // The rows carry configuration, never credentials.
        assert!(!String::from_utf8_lossy(&body).contains(testkit::API_KEY));
        let body = testkit::json_of(&body);
        assert_pointer(
            "/paths/~1v1~1system/get/responses/200/content/application~1json/schema",
            &body,
        );
        let rows = &body["engines"][0]["settings"];
        assert_eq!(rows.as_array().map(Vec::len), Some(1));
        assert_eq!(rows[0]["key"], "context_limit");
        assert_eq!(rows[0]["value"], "32768");
        assert_eq!(rows[0]["origin"], "config_file");
    }

    #[tokio::test]
    async fn models_route_reports_the_load_failure_to_an_authenticated_client() {
        let broken = testkit::failing_engine();
        let healthy = testkit::engine(Vec::new());
        let entries = vec![
            testkit::entry("e1", "model-a", &broken, 4_096),
            testkit::entry("e2", "model-b", &healthy, 4_096),
        ];
        testkit::wait_for_state(&entries[0].handle, EngineState::Error);
        testkit::wait_for_state(&entries[1].handle, EngineState::Ready);
        let state = Arc::new(AppState::new(testkit::API_KEY, entries));
        let app = router(state);

        // Unauthenticated: the failure cause is not a public surface.
        let (status, _h, body) = testkit::send(&app, testkit::get("/v1/models", None)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert!(!String::from_utf8_lossy(&body).contains("fixture load failure"));

        let request = testkit::get("/v1/models", Some(testkit::API_KEY));
        let (status, _headers, body) = testkit::send(&app, request).await;
        assert_eq!(status, StatusCode::OK);
        let text = String::from_utf8_lossy(&body).into_owned();
        assert!(
            !text.contains(testkit::API_KEY),
            "the api key must never appear: {text}"
        );
        let body = testkit::json_of(&body);
        assert_pointer(
            "/paths/~1v1~1models/get/responses/200/content/application~1json/schema",
            &body,
        );
        let failed = &body["data"][0]["engines"][0];
        assert_eq!(failed["state"], "error");
        assert_eq!(failed["error"]["code"], "checkpoint_failure");
        assert_eq!(failed["error"]["stage"], "load");
        assert_eq!(failed["error"]["retry"], "after_correction");
        assert!(
            failed["error"]["message"]
                .as_str()
                .expect("message")
                .contains("fixture load failure")
        );
        assert!(failed["error"]["suggestion"].as_str().is_some());
        // The healthy engine's status is neither hidden nor decorated.
        let ready = &body["data"][1]["engines"][0];
        assert_eq!(ready["state"], "ready");
        assert!(ready.get("error").is_none(), "{ready}");
    }

    fn web_state() -> (
        Arc<AppState>,
        crate::sched::worker::Engine,
        std::path::PathBuf,
    ) {
        // One directory per call: the web tests run in parallel in one
        // process, and a shared fixture raced (an index read during
        // another thread's rewrite came back empty).
        let dir = tempfile::tempdir().expect("web dir").keep();
        let assets = dir.join("assets");
        std::fs::create_dir_all(&assets).expect("web dir");
        std::fs::write(dir.join("index.html"), "<html>fifi shell</html>").expect("index");
        std::fs::write(assets.join("app-abc123.js"), "console.log(1)").expect("asset");
        std::fs::write(dir.join("secret.txt"), "not for serving").expect("decoy");
        let engine = testkit::engine(Vec::new());
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let state = Arc::new(
            AppState::new(
                testkit::API_KEY,
                vec![testkit::entry("e1", "model-a", &engine, 4_096)],
            )
            .with_web_dir(dir.clone()),
        );
        (state, engine, dir)
    }

    #[tokio::test]
    async fn the_web_shell_and_hashed_assets_are_served_without_a_key() {
        let (state, _engine, _dir) = web_state();
        let app = router(state);
        let (status, headers, body) = testkit::send(&app, testkit::get("/", None)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(String::from_utf8_lossy(&body), "<html>fifi shell</html>");
        assert_eq!(headers["cache-control"], "no-cache");
        let (status, headers, body) =
            testkit::send(&app, testkit::get("/assets/app-abc123.js", None)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(String::from_utf8_lossy(&body), "console.log(1)");
        assert!(
            headers["cache-control"]
                .to_str()
                .expect("header")
                .contains("immutable"),
            "{headers:?}"
        );
        // The API behind the shell still demands the key.
        let (status, _h, _b) = testkit::send(&app, testkit::get("/v1/models", None)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn web_serving_refuses_traversal_and_unlisted_extensions() {
        let (state, _engine, _dir) = web_state();
        let app = router(state);
        for uri in [
            "/assets/../secret.txt",
            "/assets/%2e%2e/secret.txt",
            "/assets/missing.js",
            "/secret.txt",
        ] {
            let (status, _h, body) = testkit::send(&app, testkit::get(uri, None)).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{uri}");
            assert_eq!(
                testkit::json_of(&body)["error"]["code"],
                "not_found",
                "{uri}"
            );
            assert!(
                !String::from_utf8_lossy(&body).contains("not for serving"),
                "{uri}"
            );
        }
    }

    #[tokio::test]
    async fn web_serving_refuses_a_symlink_out_of_the_web_dir() {
        let (state, _engine, dir) = web_state();
        let outside =
            std::env::temp_dir().join(format!("fifi-web-outside-{}.js", std::process::id()));
        std::fs::write(&outside, "secret").expect("outside file");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, dir.join("assets/linked.js")).expect("symlink");
        let app = router(state);
        let (status, _h, body) = testkit::send(&app, testkit::get("/assets/linked.js", None)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(!String::from_utf8_lossy(&body).contains("secret"));
    }

    #[tokio::test]
    async fn without_a_web_dir_the_root_stays_a_json_404() {
        let (state, _engine) = one_engine_state();
        let (status, _h, body) = testkit::send(&router(state), testkit::get("/", None)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(testkit::json_of(&body)["error"]["code"], "not_found");
    }

    #[tokio::test]
    async fn a_route_that_the_server_does_not_ship_returns_a_json_404() {
        let (state, _engine) = one_engine_state();
        let app = router(state);
        for uri in ["/v1/engines", "/v1/sessions", "/metrics", "/nowhere"] {
            let (status, _headers, body) =
                testkit::send(&app, testkit::get(uri, Some(testkit::API_KEY))).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{uri}");
            assert_eq!(
                testkit::json_of(&body)["error"]["code"],
                "not_found",
                "{uri}"
            );
        }
    }

    #[tokio::test]
    async fn an_unknown_route_answers_404_before_it_asks_for_a_key() {
        let (state, _engine) = one_engine_state();
        let (status, _headers, _body) =
            testkit::send(&router(state), testkit::get("/nowhere", None)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_wrong_method_on_a_served_path_returns_the_json_404() {
        let (state, _engine) = one_engine_state();
        let app = router(state);
        for (uri, key) in [
            ("/health", None),
            ("/ready", None),
            ("/v1/models", Some(testkit::API_KEY)),
        ] {
            let (status, _headers, body) =
                testkit::send(&app, testkit::request("POST", uri, key)).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{uri}");
            let body = testkit::json_of(&body);
            assert_schema("Error", &body);
            assert_eq!(body["error"]["code"], "not_found", "{uri}");
        }
    }

    #[tokio::test]
    async fn ready_logs_an_engine_failure_once_even_when_another_engine_is_ready() {
        let broken = testkit::failing_engine();
        let healthy = testkit::engine(Vec::new());
        let entries = vec![
            testkit::entry("e1", "model-a", &broken, 4_096),
            testkit::entry("e2", "model-b", &healthy, 4_096),
        ];
        testkit::wait_for_state(&entries[0].handle, EngineState::Error);
        testkit::wait_for_state(&entries[1].handle, EngineState::Ready);
        let lines = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = lines.clone();
        let state = Arc::new(
            AppState::new(testkit::API_KEY, entries)
                .with_log(move |line| sink.lock().expect("log").push(line.to_string())),
        );
        let app = router(state);
        for _ in 0..2 {
            let (status, _headers, _body) = testkit::send(&app, testkit::get("/ready", None)).await;
            assert_eq!(status, StatusCode::OK);
        }
        let lines = lines.lock().expect("log");
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].starts_with("engine e1 failed: "), "{lines:?}");
        assert!(lines[0].contains("fixture load failure"), "{lines:?}");
    }
}

#[cfg(test)]
mod telemetry_tests {
    use super::parse_gpu_csv;

    #[test]
    fn gpu_csv_rows_parse_with_mib_converted_to_bytes() {
        let rows = parse_gpu_csv(
            "0, NVIDIA GeForce RTX 3090, 39, 1009, 24576, 55, 114.05\n1, Fake GPU, 0, 0, 8192, 30, 12.5\n",
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["name"], "NVIDIA GeForce RTX 3090");
        assert_eq!(rows[0]["util_percent"], 39.0);
        assert_eq!(rows[0]["mem_total_bytes"], 24576_u64 * 1024 * 1024);
        assert_eq!(rows[1]["temp_c"], 30.0);
    }

    #[test]
    fn gpu_csv_unparseable_fields_stay_null() {
        let rows = parse_gpu_csv("0, GPU, N/A, N/A, N/A, N/A, N/A\n\n");
        assert_eq!(rows.len(), 1);
        assert!(rows[0]["util_percent"].is_null());
        assert!(rows[0]["mem_used_bytes"].is_null());
    }
}
