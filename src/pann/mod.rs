mod activation;
mod config;
mod intervals;
mod matrix;
mod model;
mod training;

#[cfg(test)]
mod tests;

pub use config::{
    Activation, CorrectionMode, Distributor, EpochMetrics, FeatureRange, IntervalStrategy,
    PannConfig, PannError, PlasticitySchedule, TrainingConfig, TrainingStep,
};
pub use model::{PannModel, PannModelSnapshot};

pub fn one_hot(label: usize, output_count: usize) -> Vec<f64> {
    let mut target = vec![0.0; output_count];
    if let Some(value) = target.get_mut(label) {
        *value = 1.0;
    }
    target
}

pub fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

pub(crate) fn mean_squared(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64
    }
}

pub(crate) fn mean_or_zero(total: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}
