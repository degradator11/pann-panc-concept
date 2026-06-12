use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::OutputFormat;

#[derive(Debug)]
pub enum CommandOutput {
    Metrics(BenchMetrics),
    Artifact(ArtifactMetrics),
    Eval(EvalMetrics),
    Prediction(PredictionOutput),
    Matrix(MatrixReport),
    LearningCurve(LearningCurveReport),
    Evolution(EvolutionReport),
}

#[derive(Debug, Serialize)]
pub struct BenchMetrics {
    pub model: String,
    pub dataset: String,
    pub image_features: String,
    pub image_resize: String,
    pub train_accuracy: f64,
    pub test_accuracy: f64,
    pub train_ms: u128,
    pub inference_ms: u128,
    pub memory_bytes: usize,
    pub epochs: usize,
    pub interval_count: usize,
    pub distributor: String,
    pub correction_mode: String,
    pub per_class_accuracy: Vec<PerClassAccuracy>,
    pub confusion_matrix: Vec<ConfusionRow>,
}

#[derive(Debug, Serialize)]
pub struct ArtifactMetrics {
    pub model: String,
    pub dataset: String,
    pub image_features: String,
    pub image_resize: String,
    pub artifact_path: String,
    pub train_accuracy: f64,
    pub train_ms: u128,
    pub memory_bytes: usize,
    pub epochs: usize,
    pub interval_count: usize,
    pub correction_mode: String,
    pub reference_count: usize,
}

#[derive(Debug, Serialize)]
pub struct EvalMetrics {
    pub model: String,
    pub dataset: String,
    pub image_features: String,
    pub image_resize: String,
    pub model_path: String,
    pub accuracy: f64,
    pub inference_ms: u128,
    pub memory_bytes: usize,
    pub sample_count: usize,
    pub per_class_accuracy: Vec<PerClassAccuracy>,
    pub confusion_matrix: Vec<ConfusionRow>,
    pub misclassified_examples: Vec<MisclassifiedExample>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerClassAccuracy {
    pub class_index: usize,
    pub class_name: String,
    pub correct: usize,
    pub total: usize,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfusionRow {
    pub actual_index: usize,
    pub actual_name: String,
    pub predicted_counts: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MisclassifiedExample {
    pub path: String,
    pub expected_index: usize,
    pub expected_label: String,
    pub predicted_index: usize,
    pub predicted_label: String,
    pub score_margin: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClassScore {
    pub class_index: usize,
    pub class_name: String,
    pub score: f64,
}

#[derive(Debug, Serialize)]
pub struct PredictionNeighbor {
    pub index: usize,
    pub class_index: usize,
    pub class_name: String,
    pub score: f64,
}

#[derive(Debug, Serialize)]
pub struct PredictionOutput {
    pub model: String,
    pub image: String,
    pub predicted_index: usize,
    pub predicted_label: String,
    pub scores: Vec<ClassScore>,
    pub neighbors: Vec<PredictionNeighbor>,
}

#[derive(Debug, Serialize)]
pub struct MatrixReport {
    pub dataset: String,
    pub report_path: Option<String>,
    pub summary_report_path: Option<String>,
    pub top_report_path: Option<String>,
    pub rows: Vec<MatrixRow>,
    pub summaries: Vec<MatrixSummary>,
    pub top_rows: Vec<MatrixRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatrixRow {
    pub model: String,
    pub image_features: String,
    pub image_resize: String,
    pub correction_mode: String,
    pub image_size: u32,
    pub seed: u64,
    pub epochs: usize,
    pub interval_count: usize,
    pub train_accuracy: f64,
    pub test_accuracy: f64,
    pub overfit_gap: f64,
    pub test_per_class_accuracy: Vec<PerClassAccuracy>,
    pub test_confusion_matrix: Vec<ConfusionRow>,
    pub worst_class_name: String,
    pub worst_class_accuracy: f64,
    pub most_common_confusion: String,
    pub most_common_confusion_count: usize,
    pub train_ms: u128,
    pub inference_ms: u128,
    pub memory_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatrixSummary {
    pub model: String,
    pub image_features: String,
    pub image_resize: String,
    pub correction_mode: String,
    pub image_size: u32,
    pub interval_count: usize,
    pub runs: usize,
    pub mean_test_accuracy: f64,
    pub min_test_accuracy: f64,
    pub max_test_accuracy: f64,
    pub best_seed: u64,
    pub best_test_accuracy: f64,
    pub best_train_accuracy: f64,
    pub mean_overfit_gap: f64,
    pub pooled_test_per_class_accuracy: Vec<PerClassAccuracy>,
    pub worst_mean_class_name: String,
    pub worst_mean_class_accuracy: f64,
    pub mean_train_accuracy: f64,
    pub mean_train_ms: f64,
    pub mean_inference_ms: f64,
    pub mean_memory_bytes: f64,
}

#[derive(Debug, Serialize)]
pub struct LearningCurveReport {
    pub model: String,
    pub dataset: String,
    pub image_features: String,
    pub image_resize: String,
    pub correction_mode: String,
    pub report_path: Option<String>,
    pub target_mse: Option<f64>,
    pub epochs_requested: usize,
    pub epochs_completed: usize,
    pub final_train_mse: f64,
    pub rows: Vec<LearningCurveRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LearningCurveRow {
    pub epoch: usize,
    pub mean_mse_before: f64,
    pub mean_mse_after: f64,
    pub train_accuracy: f64,
    pub test_accuracy: f64,
    pub elapsed_ms: u128,
}

#[derive(Debug, Serialize)]
pub struct EvolutionReport {
    pub model: String,
    pub dataset: String,
    pub artifact_path: Option<String>,
    pub history_path: Option<String>,
    pub seed: u64,
    pub population_size: usize,
    pub generations: usize,
    pub elite_count: usize,
    pub mutation_rate: f64,
    pub validation_ratio: f64,
    pub threads: usize,
    pub hardware_note: String,
    pub best_genome: EvolvedPancGenomeReport,
    pub best_fitness: f64,
    pub validation_accuracy: f64,
    pub eval_accuracy: Option<f64>,
    pub eval_per_class_accuracy: Vec<PerClassAccuracy>,
    pub eval_confusion_matrix: Vec<ConfusionRow>,
    pub rows: Vec<EvolutionGenerationRow>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvolvedPancGenomeReport {
    pub image_size: u32,
    pub image_features: String,
    pub image_resize: String,
    pub threshold: f64,
    pub similarity: String,
    pub jaccard_weight: f64,
    pub top_k: usize,
    pub active_blocks: String,
    pub active_block_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvolutionGenerationRow {
    pub generation: usize,
    pub best_fitness: f64,
    pub validation_accuracy: f64,
    pub validation_ms: u128,
    pub memory_bytes: usize,
    pub image_size: u32,
    pub image_features: String,
    pub image_resize: String,
    pub threshold: f64,
    pub similarity: String,
    pub jaccard_weight: f64,
    pub top_k: usize,
    pub active_blocks: String,
    pub active_block_count: usize,
}

#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub per_class_accuracy: Vec<PerClassAccuracy>,
    pub confusion_matrix: Vec<ConfusionRow>,
}

pub fn classification_metrics(
    labels: &[usize],
    predictions: &[usize],
    class_names: &[String],
) -> ClassificationMetrics {
    let class_count = class_names.len();
    let mut confusion = vec![vec![0usize; class_count]; class_count];
    let mut totals = vec![0usize; class_count];
    let mut correct_by_class = vec![0usize; class_count];
    let mut correct = 0usize;
    let mut counted = 0usize;

    for (actual, predicted) in labels.iter().zip(predictions) {
        counted += 1;
        if actual == predicted {
            correct += 1;
        }
        if let Some(total) = totals.get_mut(*actual) {
            *total += 1;
        }
        if actual == predicted
            && let Some(class_correct) = correct_by_class.get_mut(*actual)
        {
            *class_correct += 1;
        }
        if let Some(row) = confusion.get_mut(*actual)
            && let Some(cell) = row.get_mut(*predicted)
        {
            *cell += 1;
        }
    }

    let per_class_accuracy = class_names
        .iter()
        .enumerate()
        .map(|(class_index, class_name)| {
            let total = totals[class_index];
            let class_correct = correct_by_class[class_index];
            PerClassAccuracy {
                class_index,
                class_name: class_name.clone(),
                correct: class_correct,
                total,
                accuracy: if total == 0 {
                    0.0
                } else {
                    class_correct as f64 / total as f64
                },
            }
        })
        .collect();

    let confusion_matrix = confusion
        .into_iter()
        .enumerate()
        .map(|(actual_index, predicted_counts)| ConfusionRow {
            actual_index,
            actual_name: class_names
                .get(actual_index)
                .cloned()
                .unwrap_or_else(|| actual_index.to_string()),
            predicted_counts,
        })
        .collect();

    ClassificationMetrics {
        accuracy: if counted == 0 {
            0.0
        } else {
            correct as f64 / counted as f64
        },
        per_class_accuracy,
        confusion_matrix,
    }
}

pub fn worst_class(values: &[PerClassAccuracy]) -> Option<&PerClassAccuracy> {
    values
        .iter()
        .filter(|value| value.total > 0)
        .min_by(|left, right| left.accuracy.total_cmp(&right.accuracy))
}

pub fn most_common_confusion(
    rows: &[ConfusionRow],
    class_names: &[String],
) -> Option<(String, usize)> {
    let mut best: Option<(String, usize)> = None;
    for row in rows {
        for (predicted_index, count) in row.predicted_counts.iter().copied().enumerate() {
            if predicted_index == row.actual_index || count == 0 {
                continue;
            }
            let predicted_name = class_names
                .get(predicted_index)
                .map(String::as_str)
                .unwrap_or("unknown");
            let label = format!("{} -> {}", row.actual_name, predicted_name);
            if best
                .as_ref()
                .is_none_or(|(_, best_count)| count > *best_count)
            {
                best = Some((label, count));
            }
        }
    }
    best
}

pub fn write_output(output: &CommandOutput, format: OutputFormat) -> Result<(), Box<dyn Error>> {
    match (output, format) {
        (CommandOutput::Metrics(metrics), OutputFormat::Json) => {
            println!("{}", serde_json::to_string_pretty(metrics)?);
        }
        (CommandOutput::Metrics(metrics), OutputFormat::Csv) => {
            let mut writer = csv::Writer::from_writer(std::io::stdout());
            writer.serialize(BenchMetricsCsv::from(metrics))?;
            writer.flush()?;
        }
        (CommandOutput::Artifact(metrics), OutputFormat::Json) => {
            println!("{}", serde_json::to_string_pretty(metrics)?);
        }
        (CommandOutput::Artifact(metrics), OutputFormat::Csv) => {
            let mut writer = csv::Writer::from_writer(std::io::stdout());
            writer.serialize(metrics)?;
            writer.flush()?;
        }
        (CommandOutput::Eval(metrics), OutputFormat::Json) => {
            println!("{}", serde_json::to_string_pretty(metrics)?);
        }
        (CommandOutput::Eval(metrics), OutputFormat::Csv) => {
            let mut writer = csv::Writer::from_writer(std::io::stdout());
            writer.serialize(EvalMetricsCsv::from(metrics))?;
            writer.flush()?;
        }
        (CommandOutput::Prediction(prediction), OutputFormat::Json) => {
            println!("{}", serde_json::to_string_pretty(prediction)?);
        }
        (CommandOutput::Prediction(_), OutputFormat::Csv) => {
            return Err("prediction output supports --format json only".into());
        }
        (CommandOutput::Matrix(report), OutputFormat::Json) => {
            println!("{}", serde_json::to_string_pretty(report)?);
        }
        (CommandOutput::Matrix(report), OutputFormat::Csv) => {
            write_matrix_rows_csv(std::io::stdout(), &report.rows)?;
        }
        (CommandOutput::LearningCurve(report), OutputFormat::Json) => {
            println!("{}", serde_json::to_string_pretty(report)?);
        }
        (CommandOutput::LearningCurve(report), OutputFormat::Csv) => {
            let mut writer = csv::Writer::from_writer(std::io::stdout());
            for row in &report.rows {
                writer.serialize(row)?;
            }
            writer.flush()?;
        }
        (CommandOutput::Evolution(report), OutputFormat::Json) => {
            println!("{}", serde_json::to_string_pretty(report)?);
        }
        (CommandOutput::Evolution(report), OutputFormat::Csv) => {
            write_evolution_history_csv(std::io::stdout(), &report.rows)?;
        }
    }
    Ok(())
}

pub fn save_output_json(
    path: impl AsRef<Path>,
    output: &CommandOutput,
) -> Result<(), Box<dyn Error>> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let json = match output {
        CommandOutput::Metrics(metrics) => serde_json::to_string_pretty(metrics)?,
        CommandOutput::Artifact(metrics) => serde_json::to_string_pretty(metrics)?,
        CommandOutput::Eval(metrics) => serde_json::to_string_pretty(metrics)?,
        CommandOutput::Prediction(prediction) => serde_json::to_string_pretty(prediction)?,
        CommandOutput::Matrix(report) => serde_json::to_string_pretty(report)?,
        CommandOutput::LearningCurve(report) => serde_json::to_string_pretty(report)?,
        CommandOutput::Evolution(report) => serde_json::to_string_pretty(report)?,
    };
    fs::write(path, json)?;
    Ok(())
}

pub fn write_matrix_rows_csv<W: Write>(
    writer: W,
    rows: &[MatrixRow],
) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_writer(writer);
    for row in rows {
        writer.serialize(MatrixRowCsv::from(row))?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_matrix_summaries_csv<W: Write>(
    writer: W,
    summaries: &[MatrixSummary],
) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_writer(writer);
    for summary in summaries {
        writer.serialize(MatrixSummaryCsv::from(summary))?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_evolution_history_csv<W: Write>(
    writer: W,
    rows: &[EvolutionGenerationRow],
) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_writer(writer);
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct BenchMetricsCsv<'a> {
    model: &'a str,
    dataset: &'a str,
    image_features: &'a str,
    image_resize: &'a str,
    train_accuracy: f64,
    test_accuracy: f64,
    train_ms: u128,
    inference_ms: u128,
    memory_bytes: usize,
    epochs: usize,
    interval_count: usize,
    distributor: &'a str,
    correction_mode: &'a str,
    per_class_accuracy: String,
    confusion_matrix: String,
}

impl<'a> From<&'a BenchMetrics> for BenchMetricsCsv<'a> {
    fn from(metrics: &'a BenchMetrics) -> Self {
        Self {
            model: &metrics.model,
            dataset: &metrics.dataset,
            image_features: &metrics.image_features,
            image_resize: &metrics.image_resize,
            train_accuracy: metrics.train_accuracy,
            test_accuracy: metrics.test_accuracy,
            train_ms: metrics.train_ms,
            inference_ms: metrics.inference_ms,
            memory_bytes: metrics.memory_bytes,
            epochs: metrics.epochs,
            interval_count: metrics.interval_count,
            distributor: &metrics.distributor,
            correction_mode: &metrics.correction_mode,
            per_class_accuracy: format_per_class_accuracy(&metrics.per_class_accuracy),
            confusion_matrix: format_confusion_matrix(&metrics.confusion_matrix),
        }
    }
}

#[derive(Debug, Serialize)]
struct MatrixRowCsv<'a> {
    model: &'a str,
    image_features: &'a str,
    image_resize: &'a str,
    correction_mode: &'a str,
    image_size: u32,
    seed: u64,
    epochs: usize,
    interval_count: usize,
    train_accuracy: f64,
    test_accuracy: f64,
    overfit_gap: f64,
    test_per_class_accuracy: String,
    test_confusion_matrix: String,
    worst_class_name: &'a str,
    worst_class_accuracy: f64,
    most_common_confusion: &'a str,
    most_common_confusion_count: usize,
    train_ms: u128,
    inference_ms: u128,
    memory_bytes: usize,
}

impl<'a> From<&'a MatrixRow> for MatrixRowCsv<'a> {
    fn from(row: &'a MatrixRow) -> Self {
        Self {
            model: &row.model,
            image_features: &row.image_features,
            image_resize: &row.image_resize,
            correction_mode: &row.correction_mode,
            image_size: row.image_size,
            seed: row.seed,
            epochs: row.epochs,
            interval_count: row.interval_count,
            train_accuracy: row.train_accuracy,
            test_accuracy: row.test_accuracy,
            overfit_gap: row.overfit_gap,
            test_per_class_accuracy: format_per_class_accuracy(&row.test_per_class_accuracy),
            test_confusion_matrix: format_confusion_matrix(&row.test_confusion_matrix),
            worst_class_name: &row.worst_class_name,
            worst_class_accuracy: row.worst_class_accuracy,
            most_common_confusion: &row.most_common_confusion,
            most_common_confusion_count: row.most_common_confusion_count,
            train_ms: row.train_ms,
            inference_ms: row.inference_ms,
            memory_bytes: row.memory_bytes,
        }
    }
}

#[derive(Debug, Serialize)]
struct MatrixSummaryCsv<'a> {
    model: &'a str,
    image_features: &'a str,
    image_resize: &'a str,
    correction_mode: &'a str,
    image_size: u32,
    interval_count: usize,
    runs: usize,
    mean_test_accuracy: f64,
    min_test_accuracy: f64,
    max_test_accuracy: f64,
    best_seed: u64,
    best_test_accuracy: f64,
    best_train_accuracy: f64,
    mean_overfit_gap: f64,
    pooled_test_per_class_accuracy: String,
    worst_mean_class_name: &'a str,
    worst_mean_class_accuracy: f64,
    mean_train_accuracy: f64,
    mean_train_ms: f64,
    mean_inference_ms: f64,
    mean_memory_bytes: f64,
}

impl<'a> From<&'a MatrixSummary> for MatrixSummaryCsv<'a> {
    fn from(summary: &'a MatrixSummary) -> Self {
        Self {
            model: &summary.model,
            image_features: &summary.image_features,
            image_resize: &summary.image_resize,
            correction_mode: &summary.correction_mode,
            image_size: summary.image_size,
            interval_count: summary.interval_count,
            runs: summary.runs,
            mean_test_accuracy: summary.mean_test_accuracy,
            min_test_accuracy: summary.min_test_accuracy,
            max_test_accuracy: summary.max_test_accuracy,
            best_seed: summary.best_seed,
            best_test_accuracy: summary.best_test_accuracy,
            best_train_accuracy: summary.best_train_accuracy,
            mean_overfit_gap: summary.mean_overfit_gap,
            pooled_test_per_class_accuracy: format_per_class_accuracy(
                &summary.pooled_test_per_class_accuracy,
            ),
            worst_mean_class_name: &summary.worst_mean_class_name,
            worst_mean_class_accuracy: summary.worst_mean_class_accuracy,
            mean_train_accuracy: summary.mean_train_accuracy,
            mean_train_ms: summary.mean_train_ms,
            mean_inference_ms: summary.mean_inference_ms,
            mean_memory_bytes: summary.mean_memory_bytes,
        }
    }
}

#[derive(Debug, Serialize)]
struct EvalMetricsCsv<'a> {
    model: &'a str,
    dataset: &'a str,
    image_features: &'a str,
    image_resize: &'a str,
    model_path: &'a str,
    accuracy: f64,
    inference_ms: u128,
    memory_bytes: usize,
    sample_count: usize,
    per_class_accuracy: String,
    confusion_matrix: String,
    misclassified_count: usize,
}

impl<'a> From<&'a EvalMetrics> for EvalMetricsCsv<'a> {
    fn from(metrics: &'a EvalMetrics) -> Self {
        Self {
            model: &metrics.model,
            dataset: &metrics.dataset,
            image_features: &metrics.image_features,
            image_resize: &metrics.image_resize,
            model_path: &metrics.model_path,
            accuracy: metrics.accuracy,
            inference_ms: metrics.inference_ms,
            memory_bytes: metrics.memory_bytes,
            sample_count: metrics.sample_count,
            per_class_accuracy: format_per_class_accuracy(&metrics.per_class_accuracy),
            confusion_matrix: format_confusion_matrix(&metrics.confusion_matrix),
            misclassified_count: metrics.misclassified_examples.len(),
        }
    }
}

pub fn format_per_class_accuracy(values: &[PerClassAccuracy]) -> String {
    values
        .iter()
        .map(|value| {
            format!(
                "{}={:.6}({}/{})",
                value.class_name, value.accuracy, value.correct, value.total
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

pub fn format_confusion_matrix(rows: &[ConfusionRow]) -> String {
    rows.iter()
        .map(|row| {
            format!(
                "{}=[{}]",
                row.actual_name,
                row.predicted_counts
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("|")
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}
