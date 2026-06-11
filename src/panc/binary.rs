use std::collections::HashMap;
use std::hash::Hash;

use super::{BinarySimilarityMetric, PancError};

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
