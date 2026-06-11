//! Deterministic research helpers for searching PANC-like comparator settings.
//!
//! These modules are intentionally small and CPU-oriented. They are search
//! tooling for public-source experiments, not a claim to reproduce private PANC
//! internals.

pub mod genetic;
pub mod panc;

pub use genetic::{EvolutionConfig, ScoredGenome, evaluate_population_parallel};
pub use panc::{
    PancBinaryEvaluation, PancGenome, PancGenomeSpace, evaluate_panc_binary,
    predict_panc_binary_batch,
};
