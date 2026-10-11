//! Request failures and their HTTP form.
//!
//! The table "Error codes" in `docs/design/server.md` is normative.
//! `EngError` stays unchanged. This wrapper adds the three variants that only the server needs.

use crate::EngError;
use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

/// The only message that a `Backend` failure shows to a client.
pub(crate) const BACKEND_MESSAGE: &str = "Model execution failed.";

/// A failure that a route returns or that a stream reports after its headers.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ServerError {
    /// An engine failure.
    Eng(EngError),
    /// The bearer credential is missing or wrong.
    Unauthorized,
    /// The requested engine, model, session, or route does not exist.
    NotFound(String),
    /// The bounded queue or the engine state refuses the job.
    Overloaded(String),
}

impl From<EngError> for ServerError {
    fn from(error: EngError) -> Self {
        ServerError::Eng(error)
    }
}

impl ServerError {
    /// The HTTP status of the failure.
    pub(crate) fn status(&self) -> StatusCode {
        match self {
            ServerError::Eng(EngError::BadRequest(_)) => StatusCode::BAD_REQUEST,
            ServerError::Eng(EngError::MemoryPlan(_) | EngError::ContextLimit(_)) => {
                StatusCode::CONFLICT
            }
            ServerError::Eng(EngError::Checkpoint(_)) => StatusCode::UNPROCESSABLE_ENTITY,
            ServerError::Eng(EngError::Backend { .. }) => StatusCode::INTERNAL_SERVER_ERROR,
            ServerError::Unauthorized => StatusCode::UNAUTHORIZED,
            ServerError::NotFound(_) => StatusCode::NOT_FOUND,
            ServerError::Overloaded(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    /// The `Error.code` value of the failure.
    pub(crate) fn code(&self) -> &'static str {
        match self {
            ServerError::Eng(EngError::BadRequest(_)) => "bad_request",
            ServerError::Eng(EngError::MemoryPlan(_)) => "memory_plan",
            ServerError::Eng(EngError::ContextLimit(_)) => "context_limit",
            ServerError::Eng(EngError::Checkpoint(_)) => "checkpoint_failure",
            ServerError::Eng(EngError::Backend { .. }) => "backend_error",
            ServerError::Unauthorized => "unauthorized",
            ServerError::NotFound(_) => "not_found",
            ServerError::Overloaded(_) => "overloaded",
        }
    }

    /// The client message. A `Backend` failure never shows its native diagnostic text.
    pub(crate) fn message(&self) -> String {
        match self {
            ServerError::Eng(EngError::BadRequest(message))
            | ServerError::Eng(EngError::MemoryPlan(message))
            | ServerError::Eng(EngError::ContextLimit(message))
            | ServerError::Eng(EngError::Checkpoint(message))
            | ServerError::NotFound(message)
            | ServerError::Overloaded(message) => message.clone(),
            ServerError::Eng(EngError::Backend { .. }) => BACKEND_MESSAGE.to_string(),
            ServerError::Unauthorized => "missing or invalid bearer token".to_string(),
        }
    }

    /// The `Error` body of the contract. It also serves as the data of an `event: error`.
    pub(crate) fn body(&self) -> Value {
        json!({"error": {"message": self.message(), "code": self.code()}})
    }

    /// Writes the native diagnostic text of a `Backend` failure to the local log.
    pub(crate) fn log(&self) {
        if let ServerError::Eng(EngError::Backend { operation, message }) = self {
            eprintln!("backend failure in {operation}: {message}");
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        self.log();
        (self.status(), Json(self.body())).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::{BACKEND_MESSAGE, ServerError};
    use crate::EngError;
    use crate::server::contract::assert_schema;
    use axum::http::{StatusCode, header};
    use axum::response::IntoResponse;
    use serde_json::json;

    fn backend() -> ServerError {
        ServerError::Eng(EngError::Backend {
            operation: "qmm",
            message: "Metal command buffer 7 failed".to_string(),
        })
    }

    #[test]
    fn every_variant_maps_to_the_status_and_code_of_the_design_table() {
        let rows = [
            (EngError::BadRequest("m".into()).into(), 400, "bad_request"),
            (ServerError::Unauthorized, 401, "unauthorized"),
            (ServerError::NotFound("m".into()), 404, "not_found"),
            (EngError::MemoryPlan("m".into()).into(), 409, "memory_plan"),
            (
                EngError::ContextLimit("m".into()).into(),
                409,
                "context_limit",
            ),
            (
                EngError::Checkpoint("m".into()).into(),
                422,
                "checkpoint_failure",
            ),
            (ServerError::Overloaded("m".into()), 503, "overloaded"),
            (backend(), 500, "backend_error"),
        ];
        for (error, status, code) in rows {
            assert_eq!(error.status().as_u16(), status, "status of {code}");
            assert_eq!(error.code(), code, "code for status {status}");
        }
    }

    #[test]
    fn every_error_body_matches_the_error_schema() {
        let errors = [
            ServerError::from(EngError::BadRequest("max_tokens must be at least 1".into())),
            ServerError::Unauthorized,
            ServerError::NotFound("unknown engine".into()),
            ServerError::from(EngError::MemoryPlan("no room".into())),
            ServerError::from(EngError::ContextLimit("too long".into())),
            ServerError::from(EngError::Checkpoint("bad template".into())),
            ServerError::Overloaded("queue full".into()),
        ];
        for error in errors {
            assert_schema("Error", &error.body());
        }
    }

    #[test]
    fn the_backend_body_matches_the_backend_error_schema_and_hides_the_diagnostic() {
        let body = backend().body();
        assert_schema("BackendError", &body);
        assert_eq!(body["error"]["message"], BACKEND_MESSAGE);
        let text = body.to_string();
        assert!(!text.contains("qmm"), "operation name leaked: {text}");
        assert!(!text.contains("Metal"), "diagnostic text leaked: {text}");
    }

    #[test]
    fn memory_plan_and_context_limit_keep_their_own_messages_and_codes() {
        let memory = ServerError::from(EngError::MemoryPlan("budget 10 used 9".into()));
        let context = ServerError::from(EngError::ContextLimit("prompt 9 limit 8".into()));
        assert_eq!(memory.status(), context.status());
        assert_ne!(memory.code(), context.code());
        assert_eq!(memory.body()["error"]["message"], "budget 10 used 9");
        assert_eq!(context.body()["error"]["message"], "prompt 9 limit 8");
    }

    #[tokio::test]
    async fn a_failure_renders_as_a_json_response_with_its_status() {
        let response = ServerError::NotFound("no engine named e9".into()).into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let content_type = response.headers()[header::CONTENT_TYPE]
            .to_str()
            .expect("header");
        assert_eq!(content_type, "application/json");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        assert_eq!(
            parsed,
            json!({"error": {"message": "no engine named e9", "code": "not_found"}})
        );
    }
}
