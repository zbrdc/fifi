//! The HTTP server: authentication, validation, routing, and reply writers.
//!
//! Design: the server design. Contract: `api/openapi.yaml`.
//! Decision 0043 fixes the dependency set, the hand-written wire types, and the error wrapper.

mod auth;
#[cfg(test)]
mod cancel_tests;
mod chat;
mod config;
#[cfg(test)]
mod contract;
mod error;
mod fake;
mod messages;
mod pieces;
mod reply;
mod reqlog;
mod router;
mod run;
mod sse;
mod state;
#[cfg(test)]
mod testkit;
mod validate;
mod wire;

pub use config::StartupError;
pub use run::run_cli;
