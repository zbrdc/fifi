//! Resident Flash-Next decode qualification (packet C5; decision 0039 gates 3, 5, and 6).
//!
//! `gates` parses the engine test reports and applies the gate rules. `main.rs` runs the ignored
//! engine tests, records identity, and writes the record.

pub mod gates;
