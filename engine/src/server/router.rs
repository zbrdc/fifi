//! The router: the routes of the first server and the authentication layer.
//!
//! Routes: `GET /health`, `GET /ready`, `GET /v1/models`, `POST /v1/chat/completions`.
//! A route that the contract does not define, or that this server does not ship, gives a JSON 404.
//! A method that a served path does not accept counts as an undefined route, because the contract
//! defines no 405. `HEAD` on a `GET` route stays as axum serves it.

use super::auth::check_bearer;
use super::chat::chat;
use super::error::ServerError;
use super::state::AppState;
use super::wire::ModelList;
use axum::Json;
use axum::Router;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::json;
use std::sync::Arc;

/// Builds the router. `/health` and `/ready` are public, as the contract marks them with an
/// empty `security` array. Every other route needs the bearer key.
pub(crate) fn router(state: Arc<AppState>) -> Router {
    let protected = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_auth));
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .merge(protected)
        .fallback(unknown_route)
        .method_not_allowed_fallback(unknown_route)
        .with_state(state)
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

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

/// Only this handler builds the `not_ready` body (`docs/design/server.md`, "Error codes").
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

async fn models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    Json(state.models())
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
