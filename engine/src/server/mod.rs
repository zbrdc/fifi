//! The HTTP server: authentication, validation, routing, and reply writers.
//!
//! Design: the server design. Contract: `api/openapi.yaml`.
//! The server keeps a fixed dependency set, hand-written wire types, and one error wrapper.

pub(crate) mod anthropic;
mod auth;
#[cfg(test)]
mod cancel_tests;
mod chat;
mod config;
#[cfg(test)]
mod contract;
mod error;
mod fake;
#[cfg(test)]
mod family_registry_tests;
#[cfg(test)]
mod flint_tests;
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
