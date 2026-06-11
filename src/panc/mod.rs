mod binary;
mod dense;
mod error;
mod metrics;

#[cfg(test)]
mod tests;

pub use binary::{
    BinaryEncoder, BinaryExplanation, BinaryNeighbor, BinaryPancComparator, BinaryReference,
    BinaryVector,
};
pub use dense::{Explanation, Neighbor, PancComparator, Reference};
pub use error::PancError;
pub use metrics::{BinarySimilarityMetric, SimilarityMetric};
