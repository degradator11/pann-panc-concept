use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;

use progress_ai::panc::{PancComparator, SimilarityMetric};
use progress_ai::pann::{CorrectionMode, Distributor, IntervalStrategy, PannConfig, PannModel};
use progress_ai::preprocess::{
    Dataset, SplitDataset, min_max_ranges, min_max_scale, one_hot_labels, train_test_split,
};
use progress_ai::vision::{load_image_folder, synthetic_image_dataset};

use super::datasets::{load_iris, synthetic_dataset};
use super::{
    Args, BenchMetrics, CommandOutput, artifact_commands, image_config, matrix, required_data_path,
};

pub fn run(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    match args.command.as_str() {
        "train-pann-image-folder" => artifact_commands::train_pann_image_folder(args),
        "train-panc-image-folder" => artifact_commands::train_panc_image_folder(args),
        "eval-pann" => artifact_commands::eval_pann(args),
        "eval-panc" => artifact_commands::eval_panc(args),
        "predict-pann" => artifact_commands::predict_pann(args),
        "predict-panc" => artifact_commands::predict_panc(args),
        "image-matrix" => matrix::run_image_matrix(args),
        "pann-iris" => run_pann(load_iris(args.data_path.as_deref())?, "iris", args)
            .map(CommandOutput::Metrics),
        "pann-synthetic" => {
            run_pann(synthetic_dataset(args.seed), "synthetic", args).map(CommandOutput::Metrics)
        }
        "pann-image-synthetic" => run_pann(
            synthetic_image_dataset(image_config(args), args.samples_per_class, args.seed)?,
            "image-synthetic",
            args,
        )
        .map(CommandOutput::Metrics),
        "pann-image-folder" => run_pann(
            load_image_folder(required_data_path(args)?, image_config(args))?,
            "image-folder",
            args,
        )
        .map(CommandOutput::Metrics),
        "panc-iris" => run_panc(load_iris(args.data_path.as_deref())?, "iris", args)
            .map(CommandOutput::Metrics),
        "panc-synthetic" => {
            run_panc(synthetic_dataset(args.seed), "synthetic", args).map(CommandOutput::Metrics)
        }
        "panc-image-synthetic" => run_panc(
            synthetic_image_dataset(image_config(args), args.samples_per_class, args.seed)?,
            "image-synthetic",
            args,
        )
        .map(CommandOutput::Metrics),
        "panc-image-folder" => run_panc(
            load_image_folder(required_data_path(args)?, image_config(args))?,
            "image-folder",
            args,
        )
        .map(CommandOutput::Metrics),
        command => Err(format!(
            "unknown command {command}; expected pann-iris, pann-synthetic, pann-image-synthetic, pann-image-folder, panc-iris, panc-synthetic, panc-image-synthetic, panc-image-folder, train-pann-image-folder, train-panc-image-folder, eval-pann, eval-panc, predict-pann, predict-panc, or image-matrix"
        )
        .into()),
    }
}

pub(super) fn run_pann(
    dataset: Dataset,
    dataset_name: &str,
    args: &Args,
) -> Result<BenchMetrics, Box<dyn Error>> {
    let split = evaluation_split(&dataset, dataset_name, args)?;
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
        image_features: metrics_image_features(dataset_name, args),
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

pub(super) fn run_panc(
    dataset: Dataset,
    dataset_name: &str,
    args: &Args,
) -> Result<BenchMetrics, Box<dyn Error>> {
    let split = evaluation_split(&dataset, dataset_name, args)?;
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
        image_features: metrics_image_features(dataset_name, args),
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

fn evaluation_split(
    dataset: &Dataset,
    dataset_name: &str,
    args: &Args,
) -> Result<SplitDataset, Box<dyn Error>> {
    let Some(eval_path) = args.eval_data_path.as_deref() else {
        return Ok(train_test_split(
            &dataset.samples,
            &dataset.labels,
            0.2,
            args.seed,
        ));
    };

    if dataset_name != "image-folder" {
        return Err("--eval-data is only supported by image-folder benchmarks".into());
    }

    let eval_dataset = load_image_folder(eval_path, image_config(args))?;
    let train = train_test_split(&dataset.samples, &dataset.labels, 0.0, args.seed);
    let test_labels = remap_labels_by_class_name(&eval_dataset, &dataset.class_names)?;
    Ok(SplitDataset {
        train_samples: train.train_samples,
        train_labels: train.train_labels,
        test_samples: eval_dataset.samples,
        test_labels,
    })
}

fn remap_labels_by_class_name(
    source: &Dataset,
    target_class_names: &[String],
) -> Result<Vec<usize>, Box<dyn Error>> {
    let target_labels = target_class_names
        .iter()
        .enumerate()
        .map(|(label, name)| (name.as_str(), label))
        .collect::<HashMap<_, _>>();

    source
        .labels
        .iter()
        .map(|label| {
            let class_name = source
                .class_names
                .get(*label)
                .ok_or_else(|| format!("missing source class name for label {label}"))?;
            target_labels
                .get(class_name.as_str())
                .copied()
                .ok_or_else(|| {
                    format!("eval class {class_name:?} does not exist in training classes")
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn metrics_image_features(dataset_name: &str, args: &Args) -> String {
    if dataset_name.starts_with("image") {
        args.image_features.as_str().to_string()
    } else {
        "none".to_string()
    }
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
