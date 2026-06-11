use std::error::Error;

use serde::Serialize;

use super::OutputFormat;

#[derive(Debug)]
pub enum CommandOutput {
    Metrics(BenchMetrics),
    Artifact(ArtifactMetrics),
    Eval(EvalMetrics),
    Prediction(PredictionOutput),
    Matrix(MatrixReport),
    LearningCurve(LearningCurveReport),
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

#[derive(Debug, Serialize)]
pub struct PerClassAccuracy {
    pub class_index: usize,
    pub class_name: String,
    pub correct: usize,
    pub total: usize,
    pub accuracy: f64,
}

#[derive(Debug, Serialize)]
pub struct ConfusionRow {
    pub actual_index: usize,
    pub actual_name: String,
    pub predicted_counts: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct MisclassifiedExample {
    pub path: String,
    pub expected_index: usize,
    pub expected_label: String,
    pub predicted_index: usize,
    pub predicted_label: String,
    pub score_margin: f64,
}

#[derive(Debug, Serialize)]
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
    pub rows: Vec<MatrixRow>,
    pub summaries: Vec<MatrixSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatrixRow {
    pub model: String,
    pub image_features: String,
    pub image_resize: String,
    pub image_size: u32,
    pub seed: u64,
    pub epochs: usize,
    pub interval_count: usize,
    pub train_accuracy: f64,
    pub test_accuracy: f64,
    pub train_ms: u128,
    pub inference_ms: u128,
    pub memory_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatrixSummary {
    pub model: String,
    pub image_features: String,
    pub image_resize: String,
    pub image_size: u32,
    pub interval_count: usize,
    pub runs: usize,
    pub mean_test_accuracy: f64,
    pub min_test_accuracy: f64,
    pub max_test_accuracy: f64,
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

pub fn write_output(output: &CommandOutput, format: OutputFormat) -> Result<(), Box<dyn Error>> {
    match (output, format) {
        (CommandOutput::Metrics(metrics), OutputFormat::Json) => {
            println!("{}", serde_json::to_string_pretty(metrics)?);
        }
        (CommandOutput::Metrics(metrics), OutputFormat::Csv) => {
            let mut writer = csv::Writer::from_writer(std::io::stdout());
            writer.serialize(metrics)?;
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
            let mut writer = csv::Writer::from_writer(std::io::stdout());
            for row in &report.rows {
                writer.serialize(row)?;
            }
            writer.flush()?;
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
    }
    Ok(())
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

fn format_per_class_accuracy(values: &[PerClassAccuracy]) -> String {
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

fn format_confusion_matrix(rows: &[ConfusionRow]) -> String {
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
