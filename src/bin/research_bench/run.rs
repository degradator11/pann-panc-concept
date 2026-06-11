use std::error::Error;
use std::time::Instant;

use progress_ai::panc::{PancComparator, SimilarityMetric};
use progress_ai::pann::{CorrectionMode, Distributor, IntervalStrategy, PannConfig, PannModel};
use progress_ai::preprocess::{
    Dataset, min_max_ranges, min_max_scale, one_hot_labels, train_test_split,
};
use progress_ai::vision::{load_image_folder, synthetic_image_dataset};

use super::datasets::{load_iris, synthetic_dataset};
use super::{Args, BenchMetrics, image_config, required_data_path};

pub fn run(args: &Args) -> Result<BenchMetrics, Box<dyn Error>> {
    match args.command.as_str() {
        "pann-iris" => run_pann(load_iris(args.data_path.as_deref())?, "iris", args),
        "pann-synthetic" => run_pann(synthetic_dataset(args.seed), "synthetic", args),
        "pann-image-synthetic" => run_pann(
            synthetic_image_dataset(image_config(args), args.samples_per_class, args.seed)?,
            "image-synthetic",
            args,
        ),
        "pann-image-folder" => run_pann(
            load_image_folder(required_data_path(args)?, image_config(args))?,
            "image-folder",
            args,
        ),
        "panc-iris" => run_panc(load_iris(args.data_path.as_deref())?, "iris", args),
        "panc-synthetic" => run_panc(synthetic_dataset(args.seed), "synthetic", args),
        "panc-image-synthetic" => run_panc(
            synthetic_image_dataset(image_config(args), args.samples_per_class, args.seed)?,
            "image-synthetic",
            args,
        ),
        "panc-image-folder" => run_panc(
            load_image_folder(required_data_path(args)?, image_config(args))?,
            "image-folder",
            args,
        ),
        command => Err(format!(
            "unknown command {command}; expected pann-iris, pann-synthetic, pann-image-synthetic, pann-image-folder, panc-iris, panc-synthetic, panc-image-synthetic, panc-image-folder"
        )
        .into()),
    }
}

fn run_pann(
    dataset: Dataset,
    dataset_name: &str,
    args: &Args,
) -> Result<BenchMetrics, Box<dyn Error>> {
    let split = train_test_split(&dataset.samples, &dataset.labels, 0.2, args.seed);
    let ranges = min_max_ranges(&split.train_samples);
    let train_samples = min_max_scale(&split.train_samples, &ranges);
    let test_samples = min_max_scale(&split.test_samples, &ranges);
    let output_count = dataset.class_names.len();
    let targets = one_hot_labels(&split.train_labels, output_count);

    let mut config = PannConfig::new(train_samples[0].len(), args.intervals, output_count);
    config.distributor = Distributor::Triangular;
    config.interval_strategy = IntervalStrategy::Uniform;
    config.correction_mode = CorrectionMode::DifferenceLeastSquares;
    let mut model = PannModel::from_training_data_with_config(&train_samples, config)?;

    let train_start = Instant::now();
    for _ in 0..args.epochs {
        model.train_epoch(&train_samples, &targets)?;
    }
    let train_ms = train_start.elapsed().as_millis();

    let inference_start = Instant::now();
    let train_accuracy = model.accuracy(&train_samples, &split.train_labels)?;
    let test_accuracy = model.accuracy(&test_samples, &split.test_labels)?;
    let inference_ms = inference_start.elapsed().as_millis();

    Ok(BenchMetrics {
        model: "pann".to_string(),
        dataset: dataset_name.to_string(),
        train_accuracy,
        test_accuracy,
        train_ms,
        inference_ms,
        memory_bytes: model.memory_bytes_estimate(),
        epochs: args.epochs,
        interval_count: args.intervals,
        distributor: "triangular".to_string(),
        correction_mode: "difference_least_squares".to_string(),
    })
}

fn run_panc(
    dataset: Dataset,
    dataset_name: &str,
    args: &Args,
) -> Result<BenchMetrics, Box<dyn Error>> {
    let split = train_test_split(&dataset.samples, &dataset.labels, 0.2, args.seed);
    let ranges = min_max_ranges(&split.train_samples);
    let train_samples = min_max_scale(&split.train_samples, &ranges);
    let test_samples = min_max_scale(&split.test_samples, &ranges);

    let train_start = Instant::now();
    let mut comparator = PancComparator::new(SimilarityMetric::Euclidean);
    for (sample, label) in train_samples.iter().zip(&split.train_labels) {
        comparator.add_reference(sample.clone(), *label, ())?;
    }
    let train_ms = train_start.elapsed().as_millis();

    let inference_start = Instant::now();
    let train_accuracy = panc_accuracy(&comparator, &train_samples, &split.train_labels, 3)?;
    let test_accuracy = panc_accuracy(&comparator, &test_samples, &split.test_labels, 3)?;
    let inference_ms = inference_start.elapsed().as_millis();

    Ok(BenchMetrics {
        model: "panc_like".to_string(),
        dataset: dataset_name.to_string(),
        train_accuracy,
        test_accuracy,
        train_ms,
        inference_ms,
        memory_bytes: train_samples.len() * train_samples[0].len() * std::mem::size_of::<f64>(),
        epochs: 0,
        interval_count: 0,
        distributor: "none".to_string(),
        correction_mode: "top_k_euclidean_vote".to_string(),
    })
}

fn panc_accuracy(
    comparator: &PancComparator<usize>,
    samples: &[Vec<f64>],
    labels: &[usize],
    k: usize,
) -> Result<f64, Box<dyn Error>> {
    if samples.is_empty() {
        return Ok(0.0);
    }

    let mut correct = 0usize;
    for (sample, label) in samples.iter().zip(labels) {
        if comparator.predict_label(sample, k)? == Some(*label) {
            correct += 1;
        }
    }
    Ok(correct as f64 / samples.len() as f64)
}
