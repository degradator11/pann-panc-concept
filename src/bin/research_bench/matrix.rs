use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::Path;

use progress_ai::vision::{ImageFeatureMode, load_image_folder};

use super::run::{run_panc, run_pann};
use super::{
    Args, CommandOutput, MatrixModel, MatrixReport, MatrixRow, MatrixSummary, OutputFormat,
    image_config, required_data_path,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SummaryKey {
    model: String,
    image_features: String,
    image_size: u32,
    interval_count: usize,
}

pub fn run_image_matrix(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    let data_path = required_data_path(args)?;
    let models = matrix_models(args);
    let features = matrix_features(args);
    let image_sizes = matrix_image_sizes(args);
    let intervals = matrix_intervals(args);
    let seeds = matrix_seeds(args);

    let mut rows = Vec::new();
    for model in models {
        for feature in &features {
            for image_size in &image_sizes {
                for seed in &seeds {
                    match model {
                        MatrixModel::Pann => {
                            for interval_count in &intervals {
                                let variant = variant_args(
                                    args,
                                    *feature,
                                    *image_size,
                                    *seed,
                                    *interval_count,
                                );
                                let dataset = load_image_folder(data_path, image_config(&variant))?;
                                let metrics = run_pann(dataset, "image-folder", &variant)?;
                                rows.push(row_from_metrics(&metrics, *image_size, *seed));
                            }
                        }
                        MatrixModel::Panc => {
                            let variant = variant_args(args, *feature, *image_size, *seed, 0);
                            let dataset = load_image_folder(data_path, image_config(&variant))?;
                            let metrics = run_panc(dataset, "image-folder", &variant)?;
                            rows.push(row_from_metrics(&metrics, *image_size, *seed));
                        }
                    }
                }
            }
        }
    }

    let mut report = MatrixReport {
        dataset: "image-folder".to_string(),
        report_path: args.out_path.clone(),
        summaries: summarize_rows(&rows),
        rows,
    };

    if let Some(out_path) = &args.out_path {
        save_matrix_report(out_path, &report, args.format)?;
        report.report_path = Some(out_path.clone());
    }

    Ok(CommandOutput::Matrix(report))
}

fn variant_args(
    args: &Args,
    feature: ImageFeatureMode,
    image_size: u32,
    seed: u64,
    interval_count: usize,
) -> Args {
    let mut variant = args.clone();
    variant.image_width = image_size;
    variant.image_height = image_size;
    variant.image_features = feature;
    variant.seed = seed;
    variant.intervals = interval_count.max(1);
    variant
}

fn row_from_metrics(metrics: &super::BenchMetrics, image_size: u32, seed: u64) -> MatrixRow {
    MatrixRow {
        model: metrics.model.clone(),
        image_features: metrics.image_features.clone(),
        image_size,
        seed,
        epochs: metrics.epochs,
        interval_count: metrics.interval_count,
        train_accuracy: metrics.train_accuracy,
        test_accuracy: metrics.test_accuracy,
        train_ms: metrics.train_ms,
        inference_ms: metrics.inference_ms,
        memory_bytes: metrics.memory_bytes,
    }
}

fn summarize_rows(rows: &[MatrixRow]) -> Vec<MatrixSummary> {
    let mut groups: HashMap<SummaryKey, Vec<&MatrixRow>> = HashMap::new();
    for row in rows {
        groups
            .entry(SummaryKey {
                model: row.model.clone(),
                image_features: row.image_features.clone(),
                image_size: row.image_size,
                interval_count: row.interval_count,
            })
            .or_default()
            .push(row);
    }

    let mut summaries = groups
        .into_iter()
        .map(|(key, rows)| summary_from_group(key, &rows))
        .collect::<Vec<_>>();
    summaries.sort_by(|left, right| {
        left.model
            .cmp(&right.model)
            .then_with(|| left.image_features.cmp(&right.image_features))
            .then_with(|| left.image_size.cmp(&right.image_size))
            .then_with(|| left.interval_count.cmp(&right.interval_count))
    });
    summaries
}

fn summary_from_group(key: SummaryKey, rows: &[&MatrixRow]) -> MatrixSummary {
    let runs = rows.len().max(1);
    let mean = |values: &[f64]| values.iter().sum::<f64>() / runs as f64;
    let test_values = rows.iter().map(|row| row.test_accuracy).collect::<Vec<_>>();
    let train_values = rows
        .iter()
        .map(|row| row.train_accuracy)
        .collect::<Vec<_>>();
    let train_ms = rows
        .iter()
        .map(|row| row.train_ms as f64)
        .collect::<Vec<_>>();
    let inference_ms = rows
        .iter()
        .map(|row| row.inference_ms as f64)
        .collect::<Vec<_>>();
    let memory_bytes = rows
        .iter()
        .map(|row| row.memory_bytes as f64)
        .collect::<Vec<_>>();

    MatrixSummary {
        model: key.model,
        image_features: key.image_features,
        image_size: key.image_size,
        interval_count: key.interval_count,
        runs: rows.len(),
        mean_test_accuracy: mean(&test_values),
        min_test_accuracy: test_values.iter().copied().fold(f64::INFINITY, f64::min),
        max_test_accuracy: test_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        mean_train_accuracy: mean(&train_values),
        mean_train_ms: mean(&train_ms),
        mean_inference_ms: mean(&inference_ms),
        mean_memory_bytes: mean(&memory_bytes),
    }
}

fn save_matrix_report(
    out_path: &str,
    report: &MatrixReport,
    format: OutputFormat,
) -> Result<(), Box<dyn Error>> {
    let path = Path::new(out_path);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    match format {
        OutputFormat::Json => fs::write(path, serde_json::to_string_pretty(report)?)?,
        OutputFormat::Csv => {
            let mut writer = csv::Writer::from_path(path)?;
            for row in &report.rows {
                writer.serialize(row)?;
            }
            writer.flush()?;
        }
    }
    Ok(())
}

fn matrix_models(args: &Args) -> Vec<MatrixModel> {
    if args.matrix_models.is_empty() {
        vec![MatrixModel::Pann, MatrixModel::Panc]
    } else {
        args.matrix_models.clone()
    }
}

fn matrix_features(args: &Args) -> Vec<ImageFeatureMode> {
    if args.matrix_features.is_empty() {
        vec![ImageFeatureMode::Pixels, ImageFeatureMode::Combined]
    } else {
        args.matrix_features.clone()
    }
}

fn matrix_image_sizes(args: &Args) -> Vec<u32> {
    if args.matrix_image_sizes.is_empty() {
        vec![args.image_width]
    } else {
        args.matrix_image_sizes.clone()
    }
}

fn matrix_intervals(args: &Args) -> Vec<usize> {
    if args.matrix_intervals.is_empty() {
        vec![args.intervals]
    } else {
        args.matrix_intervals.clone()
    }
}

fn matrix_seeds(args: &Args) -> Vec<u64> {
    if args.matrix_seeds.is_empty() {
        vec![args.seed]
    } else {
        args.matrix_seeds.clone()
    }
}
