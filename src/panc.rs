use std::collections::HashMap;
use std::hash::Hash;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    Hamming { threshold: f64 },
    Jaccard { threshold: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reference<L, M = ()> {
    pub vector: Vec<f64>,
    pub label: L,
    pub metadata: M,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Neighbor<L, M = ()> {
    pub index: usize,
    pub score: f64,
    pub label: L,
    pub metadata: M,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Explanation<L, M = ()> {
    pub neighbors: Vec<Neighbor<L, M>>,
    pub contributions: Vec<(usize, f64)>,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum PancError {
    #[error("expected vector length {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("reference vectors must not be empty")]
    EmptyReferenceVector,
}

#[derive(Debug, Clone)]
pub struct PancComparator<L, M = ()> {
    metric: SimilarityMetric,
    dimension: Option<usize>,
    references: Vec<Reference<L, M>>,
}

impl<L, M> PancComparator<L, M> {
    pub const fn new(metric: SimilarityMetric) -> Self {
        Self {
            metric,
            dimension: None,
            references: Vec::new(),
        }
    }

    pub const fn metric(&self) -> SimilarityMetric {
        self.metric
    }

    pub fn references(&self) -> &[Reference<L, M>] {
        &self.references
    }

    pub fn add_reference(
        &mut self,
        vector: Vec<f64>,
        label: L,
        metadata: M,
    ) -> Result<(), PancError> {
        self.validate_dimension(vector.len())?;
        self.references.push(Reference {
            vector,
            label,
            metadata,
        });
        Ok(())
    }

    pub fn top_k(&self, query: &[f64], k: usize) -> Result<Vec<Neighbor<L, M>>, PancError>
    where
        L: Clone,
        M: Clone,
    {
        self.validate_query(query)?;

        let mut scored = self
            .references
            .iter()
            .enumerate()
            .map(|(index, reference)| Neighbor {
                index,
                score: score(self.metric, query, &reference.vector),
                label: reference.label.clone(),
                metadata: reference.metadata.clone(),
            })
            .collect::<Vec<_>>();

        scored.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.index.cmp(&right.index))
        });
        scored.truncate(k);
        Ok(scored)
    }

    pub fn predict_label(&self, query: &[f64], k: usize) -> Result<Option<L>, PancError>
    where
        L: Clone + Eq + Hash,
        M: Clone,
    {
        let neighbors = self.top_k(query, k)?;
        if neighbors.is_empty() {
            return Ok(None);
        }

        let mut votes: HashMap<L, (f64, usize)> = HashMap::new();
        for neighbor in neighbors {
            let entry = votes.entry(neighbor.label).or_insert((0.0, 0));
            entry.0 += neighbor.score;
            entry.1 += 1;
        }

        Ok(votes
            .into_iter()
            .max_by(|(_, left), (_, right)| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            })
            .map(|(label, _)| label))
    }

    pub fn explain(
        &self,
        query: &[f64],
        k: usize,
        top_features: usize,
    ) -> Result<Explanation<L, M>, PancError>
    where
        L: Clone,
        M: Clone,
    {
        let neighbors = self.top_k(query, k)?;
        let mut contributions = if let Some(best) = neighbors.first() {
            self.feature_contributions(query, best.index)?
        } else {
            Vec::new()
        };
        contributions.truncate(top_features);
        Ok(Explanation {
            neighbors,
            contributions,
        })
    }

    pub fn feature_contributions(
        &self,
        query: &[f64],
        reference_index: usize,
    ) -> Result<Vec<(usize, f64)>, PancError> {
        self.validate_query(query)?;
        let Some(reference) = self.references.get(reference_index) else {
            return Ok(Vec::new());
        };

        let mut contributions = query
            .iter()
            .zip(&reference.vector)
            .enumerate()
            .map(|(index, (left, right))| {
                let contribution = match self.metric {
                    SimilarityMetric::Cosine => left * right,
                    SimilarityMetric::Euclidean => -(left - right).abs(),
                    SimilarityMetric::Hamming { threshold } => {
                        if (*left >= threshold) == (*right >= threshold) {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    SimilarityMetric::Jaccard { threshold } => {
                        if *left >= threshold && *right >= threshold {
                            1.0
                        } else {
                            0.0
                        }
                    }
                };
                (index, contribution)
            })
            .collect::<Vec<_>>();

        contributions.sort_by(|left, right| right.1.total_cmp(&left.1));
        Ok(contributions)
    }

    fn validate_dimension(&mut self, actual: usize) -> Result<(), PancError> {
        if actual == 0 {
            return Err(PancError::EmptyReferenceVector);
        }

        match self.dimension {
            Some(expected) if expected != actual => {
                Err(PancError::DimensionMismatch { expected, actual })
            }
            Some(_) => Ok(()),
            None => {
                self.dimension = Some(actual);
                Ok(())
            }
        }
    }

    fn validate_query(&self, query: &[f64]) -> Result<(), PancError> {
        match self.dimension {
            Some(expected) if expected != query.len() => Err(PancError::DimensionMismatch {
                expected,
                actual: query.len(),
            }),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryVector {
    len: usize,
    words: Vec<u64>,
}

impl BinaryVector {
    pub fn from_dense_threshold(values: &[f64], threshold: f64) -> Self {
        let mut words = vec![0; values.len().div_ceil(64)];
        for (index, value) in values.iter().copied().enumerate() {
            if value >= threshold {
                words[index / 64] |= 1_u64 << (index % 64);
            }
        }
        Self {
            len: values.len(),
            words,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn get(&self, index: usize) -> bool {
        if index >= self.len {
            false
        } else {
            (self.words[index / 64] & (1_u64 << (index % 64))) != 0
        }
    }

    pub fn hamming_similarity(&self, other: &Self) -> Result<f64, PancError> {
        self.validate_other(other)?;
        if self.len == 0 {
            return Ok(0.0);
        }

        let mismatches = self
            .words
            .iter()
            .zip(&other.words)
            .map(|(left, right)| (left ^ right).count_ones() as usize)
            .sum::<usize>();
        Ok((self.len - mismatches) as f64 / self.len as f64)
    }

    pub fn jaccard_similarity(&self, other: &Self) -> Result<f64, PancError> {
        self.validate_other(other)?;
        let mut intersection = 0usize;
        let mut union = 0usize;
        for (left, right) in self.words.iter().zip(&other.words) {
            intersection += (left & right).count_ones() as usize;
            union += (left | right).count_ones() as usize;
        }

        if union == 0 {
            Ok(1.0)
        } else {
            Ok(intersection as f64 / union as f64)
        }
    }

    pub fn contributing_bits(
        &self,
        other: &Self,
        top_bits: usize,
    ) -> Result<Vec<(usize, f64)>, PancError> {
        self.validate_other(other)?;
        let mut bits = (0..self.len)
            .map(|index| {
                let contribution: f64 = if self.get(index) && other.get(index) {
                    1.0
                } else if self.get(index) == other.get(index) {
                    0.5
                } else {
                    0.0
                };
                (index, contribution)
            })
            .collect::<Vec<_>>();
        bits.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        bits.truncate(top_bits);
        Ok(bits)
    }

    fn validate_other(&self, other: &Self) -> Result<(), PancError> {
        if self.len != other.len {
            Err(PancError::DimensionMismatch {
                expected: self.len,
                actual: other.len,
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinaryEncoder {
    pub threshold: f64,
}

impl BinaryEncoder {
    pub const fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    pub fn encode(&self, values: &[f64]) -> BinaryVector {
        BinaryVector::from_dense_threshold(values, self.threshold)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinarySimilarityMetric {
    Hamming,
    Jaccard,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryReference<L, M = ()> {
    pub vector: BinaryVector,
    pub label: L,
    pub metadata: M,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryNeighbor<L, M = ()> {
    pub index: usize,
    pub score: f64,
    pub label: L,
    pub metadata: M,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryExplanation<L, M = ()> {
    pub neighbors: Vec<BinaryNeighbor<L, M>>,
    pub contributions: Vec<(usize, f64)>,
}

#[derive(Debug, Clone)]
pub struct BinaryPancComparator<L, M = ()> {
    metric: BinarySimilarityMetric,
    dimension: Option<usize>,
    references: Vec<BinaryReference<L, M>>,
}

impl<L, M> BinaryPancComparator<L, M> {
    pub const fn new(metric: BinarySimilarityMetric) -> Self {
        Self {
            metric,
            dimension: None,
            references: Vec::new(),
        }
    }

    pub fn add_reference(
        &mut self,
        vector: BinaryVector,
        label: L,
        metadata: M,
    ) -> Result<(), PancError> {
        self.validate_dimension(vector.len())?;
        self.references.push(BinaryReference {
            vector,
            label,
            metadata,
        });
        Ok(())
    }

    pub fn top_k(
        &self,
        query: &BinaryVector,
        k: usize,
    ) -> Result<Vec<BinaryNeighbor<L, M>>, PancError>
    where
        L: Clone,
        M: Clone,
    {
        self.validate_query(query)?;
        let mut scored = self
            .references
            .iter()
            .enumerate()
            .map(|(index, reference)| {
                let score = match self.metric {
                    BinarySimilarityMetric::Hamming => query.hamming_similarity(&reference.vector),
                    BinarySimilarityMetric::Jaccard => query.jaccard_similarity(&reference.vector),
                }?;
                Ok(BinaryNeighbor {
                    index,
                    score,
                    label: reference.label.clone(),
                    metadata: reference.metadata.clone(),
                })
            })
            .collect::<Result<Vec<_>, PancError>>()?;
        scored.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.index.cmp(&right.index))
        });
        scored.truncate(k);
        Ok(scored)
    }

    pub fn predict_label(&self, query: &BinaryVector, k: usize) -> Result<Option<L>, PancError>
    where
        L: Clone + Eq + Hash,
        M: Clone,
    {
        let neighbors = self.top_k(query, k)?;
        if neighbors.is_empty() {
            return Ok(None);
        }

        let mut votes: HashMap<L, (f64, usize)> = HashMap::new();
        for neighbor in neighbors {
            let entry = votes.entry(neighbor.label).or_insert((0.0, 0));
            entry.0 += neighbor.score;
            entry.1 += 1;
        }
        Ok(votes
            .into_iter()
            .max_by(|(_, left), (_, right)| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            })
            .map(|(label, _)| label))
    }

    pub fn explain(
        &self,
        query: &BinaryVector,
        k: usize,
        top_bits: usize,
    ) -> Result<BinaryExplanation<L, M>, PancError>
    where
        L: Clone,
        M: Clone,
    {
        let neighbors = self.top_k(query, k)?;
        let contributions = if let Some(best) = neighbors.first() {
            query.contributing_bits(&self.references[best.index].vector, top_bits)?
        } else {
            Vec::new()
        };
        Ok(BinaryExplanation {
            neighbors,
            contributions,
        })
    }

    fn validate_dimension(&mut self, actual: usize) -> Result<(), PancError> {
        if actual == 0 {
            return Err(PancError::EmptyReferenceVector);
        }
        match self.dimension {
            Some(expected) if expected != actual => {
                Err(PancError::DimensionMismatch { expected, actual })
            }
            Some(_) => Ok(()),
            None => {
                self.dimension = Some(actual);
                Ok(())
            }
        }
    }

    fn validate_query(&self, query: &BinaryVector) -> Result<(), PancError> {
        match self.dimension {
            Some(expected) if expected != query.len() => Err(PancError::DimensionMismatch {
                expected,
                actual: query.len(),
            }),
            _ => Ok(()),
        }
    }
}

fn score(metric: SimilarityMetric, left: &[f64], right: &[f64]) -> f64 {
    match metric {
        SimilarityMetric::Cosine => cosine_similarity(left, right),
        SimilarityMetric::Euclidean => euclidean_similarity(left, right),
        SimilarityMetric::Hamming { threshold } => hamming_similarity(left, right, threshold),
        SimilarityMetric::Jaccard { threshold } => jaccard_similarity(left, right, threshold),
    }
}

fn cosine_similarity(left: &[f64], right: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;

    for (left, right) in left.iter().zip(right) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }

    if left_norm <= f64::EPSILON || right_norm <= f64::EPSILON {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

fn euclidean_similarity(left: &[f64], right: &[f64]) -> f64 {
    let squared_distance = left
        .iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = left - right;
            difference * difference
        })
        .sum::<f64>();

    1.0 / (1.0 + squared_distance.sqrt())
}

fn hamming_similarity(left: &[f64], right: &[f64], threshold: f64) -> f64 {
    if left.is_empty() {
        return 0.0;
    }

    let matches = left
        .iter()
        .zip(right)
        .filter(|(left, right)| (**left >= threshold) == (**right >= threshold))
        .count();

    matches as f64 / left.len() as f64
}

fn jaccard_similarity(left: &[f64], right: &[f64], threshold: f64) -> f64 {
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (left, right) in left.iter().zip(right) {
        let left_active = *left >= threshold;
        let right_active = *right >= threshold;
        if left_active && right_active {
            intersection += 1;
        }
        if left_active || right_active {
            union += 1;
        }
    }

    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_top_k_returns_best_analogue() {
        let mut comparator = PancComparator::new(SimilarityMetric::Cosine);
        comparator
            .add_reference(vec![1.0, 0.0], "left", "first")
            .unwrap();
        comparator
            .add_reference(vec![0.0, 1.0], "right", "second")
            .unwrap();

        let neighbors = comparator.top_k(&[0.9, 0.1], 1).unwrap();

        assert_eq!(neighbors[0].label, "left");
        assert_eq!(neighbors[0].metadata, "first");
    }

    #[test]
    fn predict_label_uses_top_k_score_weighted_vote() {
        let mut comparator = PancComparator::new(SimilarityMetric::Euclidean);
        comparator.add_reference(vec![0.0], "a", ()).unwrap();
        comparator.add_reference(vec![0.1], "b", ()).unwrap();
        comparator.add_reference(vec![0.2], "b", ()).unwrap();

        let prediction = comparator.predict_label(&[0.11], 3).unwrap();

        assert_eq!(prediction, Some("b"));
    }

    #[test]
    fn hamming_similarity_thresholds_dense_vectors() {
        let mut comparator = PancComparator::new(SimilarityMetric::Hamming { threshold: 0.5 });
        comparator
            .add_reference(vec![1.0, 0.0, 1.0], "a", ())
            .unwrap();
        comparator
            .add_reference(vec![0.0, 1.0, 0.0], "b", ())
            .unwrap();

        let prediction = comparator.predict_label(&[0.7, 0.2, 0.9], 1).unwrap();

        assert_eq!(prediction, Some("a"));
    }

    #[test]
    fn dense_jaccard_ranks_shared_active_features() {
        let mut comparator = PancComparator::new(SimilarityMetric::Jaccard { threshold: 0.5 });
        comparator
            .add_reference(vec![1.0, 0.0, 1.0], "a", ())
            .unwrap();
        comparator
            .add_reference(vec![1.0, 1.0, 0.0], "b", ())
            .unwrap();

        let prediction = comparator.predict_label(&[0.9, 0.1, 0.8], 1).unwrap();

        assert_eq!(prediction, Some("a"));
    }

    #[test]
    fn explanation_reports_best_neighbor_and_ordered_features() {
        let mut comparator = PancComparator::new(SimilarityMetric::Euclidean);
        comparator
            .add_reference(vec![0.0, 1.0], "a", "meta")
            .unwrap();
        comparator
            .add_reference(vec![1.0, 0.0], "b", "other")
            .unwrap();

        let explanation = comparator.explain(&[0.1, 0.9], 1, 1).unwrap();

        assert_eq!(explanation.neighbors[0].label, "a");
        assert_eq!(explanation.contributions.len(), 1);
    }

    #[test]
    fn binary_encoding_and_bit_similarities_work() {
        let encoder = BinaryEncoder::new(0.5);
        let left = encoder.encode(&[1.0, 0.0, 1.0, 0.0]);
        let right = encoder.encode(&[1.0, 1.0, 1.0, 0.0]);

        assert!(left.get(0));
        assert!(!left.get(1));
        assert_eq!(left.hamming_similarity(&right).unwrap(), 0.75);
        assert!((left.jaccard_similarity(&right).unwrap() - (2.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn binary_comparator_ranks_and_explains() {
        let encoder = BinaryEncoder::new(0.5);
        let mut comparator = BinaryPancComparator::new(BinarySimilarityMetric::Jaccard);
        comparator
            .add_reference(encoder.encode(&[1.0, 0.0, 1.0]), "a", ())
            .unwrap();
        comparator
            .add_reference(encoder.encode(&[0.0, 1.0, 0.0]), "b", ())
            .unwrap();

        let query = encoder.encode(&[0.8, 0.1, 0.9]);
        let explanation = comparator.explain(&query, 1, 2).unwrap();

        assert_eq!(explanation.neighbors[0].label, "a");
        assert_eq!(explanation.contributions[0], (0, 1.0));
    }
}
