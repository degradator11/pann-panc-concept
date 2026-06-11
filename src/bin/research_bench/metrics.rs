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
}

#[derive(Debug, Serialize)]
pub struct BenchMetrics {
    pub model: String,
    pub dataset: String,
    pub image_features: String,
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
    pub model_path: String,
    pub accuracy: f64,
    pub inference_ms: u128,
    pub memory_bytes: usize,
    pub sample_count: usize,
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
            writer.serialize(metrics)?;
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
    }
    Ok(())
}
