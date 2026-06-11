use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use progress_ai::pann::CorrectionMode;
use progress_ai::vision::{ImageFeatureMode, ImageResizeMode, load_image_folder};

use super::run::{run_panc, run_pann};
use super::{
    Args, CommandOutput, MatrixModel, MatrixReport, MatrixRow, MatrixSummary, OutputFormat,
    PerClassAccuracy, image_config, most_common_confusion, required_data_path, worst_class,
    write_matrix_rows_csv, write_matrix_summaries_csv,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SummaryKey {
    model: String,
    image_features: String,
    image_resize: String,
    correction_mode: String,
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
    let resize_modes = matrix_resize_modes(args);
    let correction_modes = matrix_correction_modes(args);

    let mut rows = Vec::new();
    for model in models {
        for feature in &features {
            for resize_mode in &resize_modes {
                for image_size in &image_sizes {
                    for seed in &seeds {
                        match model {
                            MatrixModel::Pann => {
                                for correction_mode in &correction_modes {
                                    for interval_count in &intervals {
                                        let variant = variant_args(
                                            args,
                                            *feature,
                                            *resize_mode,
                                            *correction_mode,
                                            *image_size,
                                            *seed,
                                            *interval_count,
                                        );
                                        let dataset =
                                            load_image_folder(data_path, image_config(&variant))?;
                                        let metrics = run_pann(dataset, "image-folder", &variant)?;
                                        rows.push(row_from_metrics(&metrics, *image_size, *seed));
                                    }
                                }
                            }
                            MatrixModel::Panc => {
                                let variant = variant_args(
                                    args,
                                    *feature,
                                    *resize_mode,
                                    args.correction_mode,
                                    *image_size,
                                    *seed,
                                    0,
                                );
                                let dataset = load_image_folder(data_path, image_config(&variant))?;
                                let metrics = run_panc(dataset, "image-folder", &variant)?;
                                rows.push(row_from_metrics(&metrics, *image_size, *seed));
                            }
                        }
                    }
                }
            }
        }
    }

    let mut report = MatrixReport {
        dataset: "image-folder".to_string(),
        report_path: args.out_path.clone(),
        summary_report_path: None,
        top_report_path: None,
        summaries: summarize_rows(&rows),
        top_rows: top_rows(&rows, args.matrix_top),
        rows,
    };

    if let Some(out_path) = &args.out_path {
        let paths = save_matrix_report(out_path, &report, args.format)?;
        report.summary_report_path = paths.summary_report_path;
        report.top_report_path = paths.top_report_path;
        report.report_path = Some(out_path.clone());
    }

    Ok(CommandOutput::Matrix(report))
}

fn variant_args(
    args: &Args,
    feature: ImageFeatureMode,
    resize_mode: ImageResizeMode,
    correction_mode: CorrectionMode,
    image_size: u32,
    seed: u64,
    interval_count: usize,
) -> Args {
    let mut variant = args.clone();
    variant.image_width = image_size;
    variant.image_height = image_size;
    variant.image_features = feature;
    variant.image_resize = resize_mode;
    variant.correction_mode = correction_mode;
    variant.seed = seed;
    variant.intervals = interval_count.max(1);
    variant
}

fn row_from_metrics(metrics: &super::BenchMetrics, image_size: u32, seed: u64) -> MatrixRow {
    let class_names = metrics
        .per_class_accuracy
        .iter()
        .map(|class| class.class_name.clone())
        .collect::<Vec<_>>();
    let worst = worst_class(&metrics.per_class_accuracy);
    let (most_common_confusion, most_common_confusion_count) =
        most_common_confusion(&metrics.confusion_matrix, &class_names)
            .unwrap_or_else(|| ("none".to_string(), 0));

    MatrixRow {
        model: metrics.model.clone(),
        image_features: metrics.image_features.clone(),
        image_resize: metrics.image_resize.clone(),
        correction_mode: metrics.correction_mode.clone(),
        image_size,
        seed,
        epochs: metrics.epochs,
        interval_count: metrics.interval_count,
        train_accuracy: metrics.train_accuracy,
        test_accuracy: metrics.test_accuracy,
        overfit_gap: metrics.train_accuracy - metrics.test_accuracy,
        test_per_class_accuracy: metrics.per_class_accuracy.clone(),
        test_confusion_matrix: metrics.confusion_matrix.clone(),
        worst_class_name: worst
            .map(|class| class.class_name.clone())
            .unwrap_or_else(|| "none".to_string()),
        worst_class_accuracy: worst.map_or(0.0, |class| class.accuracy),
        most_common_confusion,
        most_common_confusion_count,
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
                image_resize: row.image_resize.clone(),
                correction_mode: row.correction_mode.clone(),
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
            .then_with(|| left.image_resize.cmp(&right.image_resize))
            .then_with(|| left.correction_mode.cmp(&right.correction_mode))
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
    let overfit_gaps = rows.iter().map(|row| row.overfit_gap).collect::<Vec<_>>();
    let best = rows
        .iter()
        .copied()
        .max_by(|left, right| left.test_accuracy.total_cmp(&right.test_accuracy))
        .expect("matrix summaries are only built from non-empty groups");
    let pooled_test_per_class_accuracy = pooled_per_class_accuracy(rows);
    let worst_mean = worst_class(&pooled_test_per_class_accuracy);
    let worst_mean_class_name = worst_mean
        .map(|class| class.class_name.clone())
        .unwrap_or_else(|| "none".to_string());
    let worst_mean_class_accuracy = worst_mean.map_or(0.0, |class| class.accuracy);

    MatrixSummary {
        model: key.model,
        image_features: key.image_features,
        image_resize: key.image_resize,
        correction_mode: key.correction_mode,
        image_size: key.image_size,
        interval_count: key.interval_count,
        runs: rows.len(),
        mean_test_accuracy: mean(&test_values),
        min_test_accuracy: test_values.iter().copied().fold(f64::INFINITY, f64::min),
        max_test_accuracy: test_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        best_seed: best.seed,
        best_test_accuracy: best.test_accuracy,
        best_train_accuracy: best.train_accuracy,
        mean_overfit_gap: mean(&overfit_gaps),
        pooled_test_per_class_accuracy,
        worst_mean_class_name,
        worst_mean_class_accuracy,
        mean_train_accuracy: mean(&train_values),
        mean_train_ms: mean(&train_ms),
        mean_inference_ms: mean(&inference_ms),
        mean_memory_bytes: mean(&memory_bytes),
    }
}

struct MatrixReportPaths {
    summary_report_path: Option<String>,
    top_report_path: Option<String>,
}

fn save_matrix_report(
    out_path: &str,
    report: &MatrixReport,
    format: OutputFormat,
) -> Result<MatrixReportPaths, Box<dyn Error>> {
    let path = Path::new(out_path);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    match format {
        OutputFormat::Json => fs::write(path, serde_json::to_string_pretty(report)?)?,
        OutputFormat::Csv => {
            let rows_file = fs::File::create(path)?;
            write_matrix_rows_csv(rows_file, &report.rows)?;
            let summary_path = summary_csv_path(path);
            let summary_file = fs::File::create(&summary_path)?;
            write_matrix_summaries_csv(summary_file, &report.summaries)?;
            let top_report_path = if report.top_rows.is_empty() {
                None
            } else {
                let top_path = top_csv_path(path);
                let top_file = fs::File::create(&top_path)?;
                write_matrix_rows_csv(top_file, &report.top_rows)?;
                Some(top_path.display().to_string())
            };
            return Ok(MatrixReportPaths {
                summary_report_path: Some(summary_path.display().to_string()),
                top_report_path,
            });
        }
    }
    Ok(MatrixReportPaths {
        summary_report_path: None,
        top_report_path: None,
    })
}

fn top_rows(rows: &[MatrixRow], limit: usize) -> Vec<MatrixRow> {
    if limit == 0 {
        return Vec::new();
    }

    let mut rows = rows.to_vec();
    rows.sort_by(|left, right| {
        right
            .test_accuracy
            .total_cmp(&left.test_accuracy)
            .then_with(|| {
                right
                    .worst_class_accuracy
                    .total_cmp(&left.worst_class_accuracy)
            })
            .then_with(|| left.overfit_gap.total_cmp(&right.overfit_gap))
            .then_with(|| left.model.cmp(&right.model))
            .then_with(|| left.image_features.cmp(&right.image_features))
            .then_with(|| left.image_resize.cmp(&right.image_resize))
            .then_with(|| left.correction_mode.cmp(&right.correction_mode))
            .then_with(|| left.interval_count.cmp(&right.interval_count))
            .then_with(|| left.seed.cmp(&right.seed))
    });
    rows.truncate(limit);
    rows
}

fn pooled_per_class_accuracy(rows: &[&MatrixRow]) -> Vec<PerClassAccuracy> {
    let Some(first) = rows.first() else {
        return Vec::new();
    };

    first
        .test_per_class_accuracy
        .iter()
        .map(|class| {
            let mut correct = 0usize;
            let mut total = 0usize;
            for row in rows {
                if let Some(value) = row
                    .test_per_class_accuracy
                    .iter()
                    .find(|value| value.class_index == class.class_index)
                {
                    correct += value.correct;
                    total += value.total;
                }
            }
            PerClassAccuracy {
                class_index: class.class_index,
                class_name: class.class_name.clone(),
                correct,
                total,
                accuracy: if total == 0 {
                    0.0
                } else {
                    correct as f64 / total as f64
                },
            }
        })
        .collect()
}

fn summary_csv_path(path: &Path) -> PathBuf {
    let mut summary_path = path.to_path_buf();
    summary_path.set_extension("summary.csv");
    summary_path
}

fn top_csv_path(path: &Path) -> PathBuf {
    let mut top_path = path.to_path_buf();
    top_path.set_extension("top.csv");
    top_path
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

fn matrix_resize_modes(args: &Args) -> Vec<ImageResizeMode> {
    if args.matrix_resize_modes.is_empty() {
        vec![args.image_resize]
    } else {
        args.matrix_resize_modes.clone()
    }
}

fn matrix_correction_modes(args: &Args) -> Vec<CorrectionMode> {
    if args.matrix_correction_modes.is_empty() {
        vec![args.correction_mode]
    } else {
        args.matrix_correction_modes.clone()
    }
}
