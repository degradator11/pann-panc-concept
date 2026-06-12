use std::collections::HashMap;
use std::hash::Hash;

use super::{PancError, SimilarityMetric};
use crate::panc::metrics::score;

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
                        f64::from((*left >= threshold) == (*right >= threshold))
                    }
                    SimilarityMetric::Jaccard { threshold } => {
                        f64::from(*left >= threshold && *right >= threshold)
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
