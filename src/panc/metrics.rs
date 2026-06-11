#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    Hamming { threshold: f64 },
    Jaccard { threshold: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinarySimilarityMetric {
    Hamming,
    Jaccard,
}

pub(crate) fn score(metric: SimilarityMetric, left: &[f64], right: &[f64]) -> f64 {
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
