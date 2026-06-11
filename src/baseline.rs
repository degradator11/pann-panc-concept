use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct NearestCentroidClassifier {
    centroids: Vec<Vec<f64>>,
    class_counts: Vec<usize>,
}

#[derive(Debug, Error, PartialEq)]
pub enum BaselineError {
    #[error("nearest-centroid classifier needs at least one sample")]
    NoSamples,
    #[error("nearest-centroid classifier needs at least one class")]
    NoClasses,
    #[error("sample and label counts differ: {sample_count} samples, {label_count} labels")]
    MismatchedLabelCount {
        sample_count: usize,
        label_count: usize,
    },
    #[error("sample {sample_index} has {actual} features; expected {expected}")]
    MismatchedFeatureCount {
        sample_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("label {label} is outside configured class count {class_count}")]
    InvalidLabel { label: usize, class_count: usize },
}

impl NearestCentroidClassifier {
    pub fn fit(
        samples: &[Vec<f64>],
        labels: &[usize],
        class_count: usize,
    ) -> Result<Self, BaselineError> {
        if samples.is_empty() {
            return Err(BaselineError::NoSamples);
        }
        if class_count == 0 {
            return Err(BaselineError::NoClasses);
        }
        if samples.len() != labels.len() {
            return Err(BaselineError::MismatchedLabelCount {
                sample_count: samples.len(),
                label_count: labels.len(),
            });
        }

        let feature_count = samples[0].len();
        let mut centroids = vec![vec![0.0; feature_count]; class_count];
        let mut class_counts = vec![0usize; class_count];

        for (sample_index, (sample, label)) in samples.iter().zip(labels).enumerate() {
            if sample.len() != feature_count {
                return Err(BaselineError::MismatchedFeatureCount {
                    sample_index,
                    expected: feature_count,
                    actual: sample.len(),
                });
            }
            if *label >= class_count {
                return Err(BaselineError::InvalidLabel {
                    label: *label,
                    class_count,
                });
            }

            class_counts[*label] += 1;
            for (feature, value) in sample.iter().copied().enumerate() {
                if value.is_finite() {
                    centroids[*label][feature] += value;
                }
            }
        }

        for (centroid, count) in centroids.iter_mut().zip(&class_counts) {
            if *count == 0 {
                continue;
            }
            for value in centroid {
                *value /= *count as f64;
            }
        }

        Ok(Self {
            centroids,
            class_counts,
        })
    }

    pub fn predict(&self, sample: &[f64]) -> Result<usize, BaselineError> {
        let feature_count = self.centroids.first().map_or(0, Vec::len);
        if sample.len() != feature_count {
            return Err(BaselineError::MismatchedFeatureCount {
                sample_index: 0,
                expected: feature_count,
                actual: sample.len(),
            });
        }

        self.centroids
            .iter()
            .enumerate()
            .filter(|(class, _)| self.class_counts[*class] > 0)
            .min_by(|(_, left), (_, right)| {
                squared_distance(sample, left).total_cmp(&squared_distance(sample, right))
            })
            .map(|(class, _)| class)
            .ok_or(BaselineError::NoSamples)
    }

    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Result<Vec<usize>, BaselineError> {
        samples
            .iter()
            .map(|sample| self.predict(sample))
            .collect::<Result<Vec<_>, _>>()
    }

    pub fn memory_bytes_estimate(&self) -> usize {
        self.centroids
            .iter()
            .map(|centroid| centroid.len() * std::mem::size_of::<f64>())
            .sum::<usize>()
            + self.class_counts.len() * std::mem::size_of::<usize>()
    }
}

fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = left - right;
            difference * difference
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_centroid_predicts_closest_class() {
        let samples = vec![
            vec![0.0, 0.0],
            vec![0.2, 0.0],
            vec![1.0, 1.0],
            vec![0.8, 1.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let classifier = NearestCentroidClassifier::fit(&samples, &labels, 2).unwrap();

        assert_eq!(classifier.predict(&[0.1, 0.1]).unwrap(), 0);
        assert_eq!(classifier.predict(&[0.9, 0.9]).unwrap(), 1);
    }

    #[test]
    fn nearest_centroid_rejects_mixed_feature_counts() {
        let samples = vec![vec![0.0], vec![1.0, 1.0]];
        let labels = vec![0, 1];

        assert_eq!(
            NearestCentroidClassifier::fit(&samples, &labels, 2).unwrap_err(),
            BaselineError::MismatchedFeatureCount {
                sample_index: 1,
                expected: 1,
                actual: 2
            }
        );
    }
}
