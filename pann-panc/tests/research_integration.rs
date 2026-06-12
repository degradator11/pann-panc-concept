use progress_ai::panc::{PancComparator, SimilarityMetric};
use progress_ai::pann::{Distributor, PannModel, one_hot};
use progress_ai::vision::{ImageVectorConfig, synthetic_image_dataset};

#[test]
fn pann_reaches_high_accuracy_on_synthetic_separable_data() {
    let samples = synthetic_samples();
    let labels = synthetic_labels();
    let targets = labels
        .iter()
        .map(|label| one_hot(*label, 3))
        .collect::<Vec<_>>();
    let mut model = PannModel::with_unit_ranges(2, 8, 3, Distributor::HardBin).unwrap();

    for _ in 0..3 {
        model.train_epoch_difference(&samples, &targets).unwrap();
    }

    assert!(model.accuracy(&samples, &labels).unwrap() >= 0.95);
}

#[test]
fn panc_reaches_high_accuracy_on_synthetic_separable_data() {
    let samples = synthetic_samples();
    let labels = synthetic_labels();
    let mut comparator = PancComparator::new(SimilarityMetric::Euclidean);
    for (sample, label) in samples.iter().zip(&labels) {
        comparator
            .add_reference(sample.clone(), *label, ())
            .unwrap();
    }

    let mut correct = 0usize;
    for (sample, label) in samples.iter().zip(&labels) {
        if comparator.predict_label(sample, 3).unwrap() == Some(*label) {
            correct += 1;
        }
    }

    assert!(correct as f64 / samples.len() as f64 >= 0.95);
}

#[test]
fn pann_reaches_high_accuracy_on_synthetic_image_patterns() {
    let dataset = synthetic_image_dataset(ImageVectorConfig::new(12, 12), 12, 123).unwrap();
    let targets = dataset
        .labels
        .iter()
        .map(|label| one_hot(*label, dataset.class_names.len()))
        .collect::<Vec<_>>();
    let mut model =
        PannModel::with_unit_ranges(12 * 12, 4, dataset.class_names.len(), Distributor::HardBin)
            .unwrap();

    for _ in 0..4 {
        model
            .train_epoch_difference(&dataset.samples, &targets)
            .unwrap();
    }

    assert!(model.accuracy(&dataset.samples, &dataset.labels).unwrap() >= 0.95);
}

#[test]
fn panc_reaches_high_accuracy_on_synthetic_image_patterns() {
    let dataset = synthetic_image_dataset(ImageVectorConfig::new(12, 12), 12, 321).unwrap();
    let mut comparator = PancComparator::new(SimilarityMetric::Euclidean);
    for (sample, label) in dataset.samples.iter().zip(&dataset.labels) {
        comparator
            .add_reference(sample.clone(), *label, ())
            .unwrap();
    }

    let mut correct = 0usize;
    for (sample, label) in dataset.samples.iter().zip(&dataset.labels) {
        if comparator.predict_label(sample, 3).unwrap() == Some(*label) {
            correct += 1;
        }
    }

    assert!(correct as f64 / dataset.samples.len() as f64 >= 0.95);
}

fn synthetic_samples() -> Vec<Vec<f64>> {
    vec![
        vec![0.1, 0.1],
        vec![0.12, 0.08],
        vec![0.08, 0.12],
        vec![0.85, 0.15],
        vec![0.88, 0.12],
        vec![0.82, 0.18],
        vec![0.45, 0.85],
        vec![0.50, 0.88],
        vec![0.55, 0.82],
    ]
}

fn synthetic_labels() -> Vec<usize> {
    vec![0, 0, 0, 1, 1, 1, 2, 2, 2]
}
