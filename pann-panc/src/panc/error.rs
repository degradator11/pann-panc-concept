use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum PancError {
    #[error("expected vector length {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("reference vectors must not be empty")]
    EmptyReferenceVector,
}
