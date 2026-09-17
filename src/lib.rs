//! flakesim — grow a snow crystal from a time series of environmental
//! conditions, following Libbrecht's cellular-automaton scheme with his
//! attachment-kinetics model on a three-dimensional vapour field.
//!
//! The library holds the physics and the renderers; the `flakesim` binary
//! wraps them in a command line, and the `wasm` feature exposes a growth
//! session to the browser.

pub mod automaton;
pub mod config;
pub mod diagram;
pub mod par;
pub mod physics;
pub mod render;
#[cfg(feature = "wasm")]
pub mod wasm;
