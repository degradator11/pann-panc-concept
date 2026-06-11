use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use crate::pann::FeatureRange;

#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    pub samples: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
    pub class_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SplitDataset {
    pub train_samples: Vec<Vec<f64>>,
    pub train_labels: Vec<usize>,
    pub test_samples: Vec<Vec<f64>>,
    pub test_labels: Vec<usize>,
}

pub fn one_hot_labels(labels: &[usize], output_count: usize) -> Vec<Vec<f64>> {
    labels
        .iter()
        .map(|label| crate::pann::one_hot(*label, output_count))
        .collect()
}

pub fn min_max_ranges(samples: &[Vec<f64>]) -> Vec<FeatureRange> {
    let Some(first) = samples.first() else {
        return Vec::new();
    };

    let mut ranges = vec![FeatureRange::new(f64::INFINITY, f64::NEG_INFINITY); first.len()];
    for sample in samples {
        for (index, value) in sample.iter().copied().enumerate() {
            if value.is_finite() {
                ranges[index].min = ranges[index].min.min(value);
                ranges[index].max = ranges[index].max.max(value);
            }
        }
    }

    for range in &mut ranges {
        if !range.min.is_finite() || !range.max.is_finite() {
            range.min = 0.0;
            range.max = 0.0;
        }
    }
    ranges
}

pub fn percentile_clip_ranges(samples: &[Vec<f64>], lower: f64, upper: f64) -> Vec<FeatureRange> {
    let Some(first) = samples.first() else {
        return Vec::new();
    };

    let mut ranges = Vec::with_capacity(first.len());
    for feature in 0..first.len() {
        let mut values = samples
            .iter()
            .map(|sample| sample[feature])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        ranges.push(FeatureRange::new(
            percentile_sorted(&values, lower),
            percentile_sorted(&values, upper),
        ));
    }
    ranges
}

pub fn min_max_scale(samples: &[Vec<f64>], ranges: &[FeatureRange]) -> Vec<Vec<f64>> {
    samples
        .iter()
        .map(|sample| {
            sample
                .iter()
                .copied()
                .enumerate()
                .map(|(index, value)| {
                    let range = ranges[index];
                    let width = range.max - range.min;
                    if width <= f64::EPSILON || !value.is_finite() {
                        0.0
                    } else {
                        ((value - range.min) / width).clamp(0.0, 1.0)
                    }
                })
                .collect()
        })
        .collect()
}

pub fn z_score_clip(samples: &[Vec<f64>], clip: f64) -> Vec<Vec<f64>> {
    let Some(first) = samples.first() else {
        return Vec::new();
    };
    let feature_count = first.len();
    let mut means = vec![0.0; feature_count];
    for sample in samples {
        for (index, value) in sample.iter().copied().enumerate() {
            means[index] += value;
        }
    }
    for mean in &mut means {
        *mean /= samples.len().max(1) as f64;
    }

    let mut variances = vec![0.0; feature_count];
    for sample in samples {
        for (index, value) in sample.iter().copied().enumerate() {
            let difference = value - means[index];
            variances[index] += difference * difference;
        }
    }
    let stddevs = variances
        .into_iter()
        .map(|variance| {
            (variance / samples.len().max(1) as f64)
                .sqrt()
                .max(f64::EPSILON)
        })
        .collect::<Vec<_>>();

    samples
        .iter()
        .map(|sample| {
            sample
                .iter()
                .copied()
                .enumerate()
                .map(|(index, value)| ((value - means[index]) / stddevs[index]).clamp(-clip, clip))
                .collect()
        })
        .collect()
}

pub fn train_test_split(
    samples: &[Vec<f64>],
    labels: &[usize],
    test_ratio: f64,
    seed: u64,
) -> SplitDataset {
    let (test_indexes, train_indexes) = train_test_split_indices(samples.len(), test_ratio, seed);

    let collect_samples = |indexes: &[usize]| {
        indexes
            .iter()
            .map(|index| samples[*index].clone())
            .collect::<Vec<_>>()
    };
    let collect_labels = |indexes: &[usize]| {
        indexes
            .iter()
            .map(|index| labels[*index])
            .collect::<Vec<_>>()
    };

    SplitDataset {
        train_samples: collect_samples(&train_indexes),
        train_labels: collect_labels(&train_indexes),
        test_samples: collect_samples(&test_indexes),
        test_labels: collect_labels(&test_indexes),
    }
}

pub fn train_test_split_indices(
    sample_count: usize,
    test_ratio: f64,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    let mut indexes = (0..sample_count).collect::<Vec<_>>();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    indexes.shuffle(&mut rng);

    let test_count = ((sample_count as f64) * test_ratio.clamp(0.0, 1.0)).round() as usize;
    let (test_indexes, train_indexes) = indexes.split_at(test_count.min(indexes.len()));
    (test_indexes.to_vec(), train_indexes.to_vec())
}

fn percentile_sorted(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let quantile = quantile.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_max_scaling_uses_supplied_ranges() {
        let samples = vec![vec![2.0, 10.0], vec![4.0, 20.0]];
        let ranges = min_max_ranges(&samples);
        let scaled = min_max_scale(&samples, &ranges);

        assert_eq!(scaled[0], vec![0.0, 0.0]);
        assert_eq!(scaled[1], vec![1.0, 1.0]);
    }

    #[test]
    fn split_is_deterministic_for_seed() {
        let samples = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let labels = vec![0, 1, 0, 1];

        let left = train_test_split(&samples, &labels, 0.5, 7);
        let right = train_test_split(&samples, &labels, 0.5, 7);

        assert_eq!(left, right);
    }
}
