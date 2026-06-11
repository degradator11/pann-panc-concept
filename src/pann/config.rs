use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FeatureRange {
    pub min: f64,
    pub max: f64,
}

impl FeatureRange {
    pub const fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Distributor {
    HardBin,
    Triangular,
    Gaussian { radius: usize, sigma: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum IntervalStrategy {
    Uniform,
    Quantile,
    ClippedPercentile { lower: f64, upper: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CorrectionMode {
    DifferenceLeastSquares,
    DifferencePatentProportional,
    Ratio { epsilon: f64, max_abs_factor: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PlasticitySchedule {
    None,
    PatentHalving { freeze_after: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PannConfig {
    pub input_count: usize,
    pub interval_count: usize,
    pub output_count: usize,
    pub distributor: Distributor,
    pub interval_strategy: IntervalStrategy,
    pub correction_mode: CorrectionMode,
    pub plasticity_schedule: PlasticitySchedule,
}

impl PannConfig {
    pub const fn new(input_count: usize, interval_count: usize, output_count: usize) -> Self {
        Self {
            input_count,
            interval_count,
            output_count,
            distributor: Distributor::HardBin,
            interval_strategy: IntervalStrategy::Uniform,
            correction_mode: CorrectionMode::DifferenceLeastSquares,
            plasticity_schedule: PlasticitySchedule::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub max_epochs: usize,
    pub target_mse: f64,
    pub batch: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 20,
            target_mse: 1e-6,
            batch: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Activation {
    pub input: usize,
    pub interval: usize,
    pub coefficient: f64,
}

impl Activation {
    pub fn column(self, interval_count: usize) -> usize {
        self.input * interval_count + self.interval
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingStep {
    pub output_before: Vec<f64>,
    pub output_after: Vec<f64>,
    pub error_before: Vec<f64>,
    pub mse_before: f64,
    pub active_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EpochMetrics {
    pub samples: usize,
    pub mean_mse_before: f64,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum PannError {
    #[error("at least one training sample is required")]
    EmptySamples,
    #[error("input_count must be greater than zero")]
    InvalidInputCount,
    #[error("interval_count must be greater than zero")]
    InvalidIntervalCount,
    #[error("output_count must be greater than zero")]
    InvalidOutputCount,
    #[error("invalid range at index {index}: min={min}, max={max}")]
    InvalidRange { index: usize, min: f64, max: f64 },
    #[error("expected {expected} feature ranges, got {actual}")]
    RangeCountMismatch { expected: usize, actual: usize },
    #[error("expected input length {expected}, got {actual}")]
    InputLengthMismatch { expected: usize, actual: usize },
    #[error("expected target length {expected}, got {actual}")]
    TargetLengthMismatch { expected: usize, actual: usize },
    #[error("sample count {samples} does not match target count {targets}")]
    SampleTargetCountMismatch { samples: usize, targets: usize },
    #[error("the distributor produced no active weights")]
    EmptyActiveSet,
    #[error("batch training does not support {mode:?}")]
    UnsupportedBatchCorrection { mode: CorrectionMode },
}
