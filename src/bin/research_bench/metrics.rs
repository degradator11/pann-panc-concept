use std::error::Error;

use serde::Serialize;

use super::OutputFormat;

#[derive(Debug, Serialize)]
pub struct BenchMetrics {
    pub model: String,
    pub dataset: String,
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

pub fn write_metrics(metrics: &BenchMetrics, format: OutputFormat) -> Result<(), Box<dyn Error>> {
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(metrics)?),
        OutputFormat::Csv => {
            let mut writer = csv::Writer::from_writer(std::io::stdout());
            writer.serialize(metrics)?;
            writer.flush()?;
        }
    }
    Ok(())
}
