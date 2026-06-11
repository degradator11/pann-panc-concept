use ndarray::Array2;

use super::matrix::{column_usage, targets_as_matrix};
use super::{
    CorrectionMode, EpochMetrics, PannError, PannModel, PlasticitySchedule, TrainingStep,
    mean_or_zero, mean_squared,
};

impl PannModel {
    pub fn train_one(&mut self, input: &[f64], target: &[f64]) -> Result<TrainingStep, PannError> {
        self.train_one_with_mode(input, target, self.config.correction_mode)
    }

    pub fn train_one_difference(
        &mut self,
        input: &[f64],
        target: &[f64],
    ) -> Result<TrainingStep, PannError> {
        self.train_one_with_mode(input, target, CorrectionMode::DifferenceLeastSquares)
    }

    pub fn train_epoch(
        &mut self,
        samples: &[Vec<f64>],
        targets: &[Vec<f64>],
    ) -> Result<EpochMetrics, PannError> {
        match self.config.correction_mode {
            CorrectionMode::Ratio { .. } => {
                self.validate_samples_targets(samples, targets)?;
                let mut total_mse = 0.0;
                for (sample, target) in samples.iter().zip(targets) {
                    total_mse += self.train_one(sample, target)?.mse_before;
                }
                Ok(EpochMetrics {
                    samples: samples.len(),
                    mean_mse_before: mean_or_zero(total_mse, samples.len()),
                })
            }
            _ => self.train_epoch_difference(samples, targets),
        }
    }

    pub fn train_epoch_difference(
        &mut self,
        samples: &[Vec<f64>],
        targets: &[Vec<f64>],
    ) -> Result<EpochMetrics, PannError> {
        self.validate_samples_targets(samples, targets)?;

        let mut total_mse = 0.0;
        for (sample, target) in samples.iter().zip(targets) {
            let step = self.train_one_difference(sample, target)?;
            total_mse += step.mse_before;
        }

        Ok(EpochMetrics {
            samples: samples.len(),
            mean_mse_before: mean_or_zero(total_mse, samples.len()),
        })
    }

    pub fn train_epoch_matrix(
        &mut self,
        samples: &[Vec<f64>],
        targets: &[Vec<f64>],
    ) -> Result<EpochMetrics, PannError> {
        self.validate_samples_targets(samples, targets)?;
        if matches!(self.config.correction_mode, CorrectionMode::Ratio { .. }) {
            return Err(PannError::UnsupportedBatchCorrection {
                mode: self.config.correction_mode,
            });
        }

        let activations = self.activation_matrix(samples)?;
        let outputs = activations.dot(&self.weights_as_matrix());
        let targets = targets_as_matrix(targets, self.config.output_count)?;
        let errors = targets - outputs;
        let mse = errors.iter().map(|value| value * value).sum::<f64>()
            / (samples.len() * self.config.output_count).max(1) as f64;

        let mut normalized_errors = errors.clone();
        for row in 0..activations.nrows() {
            let denom = self
                .batch_error_denominator(&activations, row)
                .max(f64::EPSILON);
            for output in 0..self.config.output_count {
                normalized_errors[(row, output)] /= denom;
            }
        }

        let mut corrections = activations.t().dot(&normalized_errors);
        let usage = column_usage(&activations);
        for column in 0..corrections.nrows() {
            if usage[column] == 0 {
                continue;
            }
            let input = column / self.config.interval_count;
            let interval = column % self.config.interval_count;
            for output in 0..self.config.output_count {
                corrections[(column, output)] /= usage[column] as f64;
                let index = self.raw_weight_index(input, interval, output);
                let scale = self.plasticity_scale(index);
                self.weights[index] += corrections[(column, output)] * scale;
                self.access_counts[index] = self.access_counts[index].saturating_add(usage[column]);
            }
        }

        Ok(EpochMetrics {
            samples: samples.len(),
            mean_mse_before: mse,
        })
    }

    fn train_one_with_mode(
        &mut self,
        input: &[f64],
        target: &[f64],
        mode: CorrectionMode,
    ) -> Result<TrainingStep, PannError> {
        self.validate_target(target)?;

        let active = self.encode(input)?;
        let output_before = self.forward_from_activations(&active);
        let error_before = target
            .iter()
            .zip(&output_before)
            .map(|(target, actual)| target - actual)
            .collect::<Vec<_>>();
        let mse_before = mean_squared(&error_before);

        match mode {
            CorrectionMode::DifferenceLeastSquares => {
                let denom = active
                    .iter()
                    .map(|activation| activation.coefficient * activation.coefficient)
                    .sum::<f64>()
                    .max(f64::EPSILON);
                for activation in &active {
                    for (output_index, error) in error_before.iter().copied().enumerate() {
                        let weight_index = self.raw_weight_index(
                            activation.input,
                            activation.interval,
                            output_index,
                        );
                        self.apply_delta(weight_index, error * activation.coefficient / denom);
                    }
                }
            }
            CorrectionMode::DifferencePatentProportional => {
                let denom = (active.len() as f64).max(1.0);
                for activation in &active {
                    for (output_index, error) in error_before.iter().copied().enumerate() {
                        let weight_index = self.raw_weight_index(
                            activation.input,
                            activation.interval,
                            output_index,
                        );
                        self.apply_delta(weight_index, error * activation.coefficient / denom);
                    }
                }
            }
            CorrectionMode::Ratio {
                epsilon,
                max_abs_factor,
            } => self.apply_ratio_update(target, &active, &output_before, epsilon, max_abs_factor),
        }

        let output_after = self.forward_from_activations(&active);
        Ok(TrainingStep {
            output_before,
            output_after,
            error_before,
            mse_before,
            active_count: active.len(),
        })
    }

    fn apply_ratio_update(
        &mut self,
        target: &[f64],
        active: &[super::Activation],
        output_before: &[f64],
        epsilon: f64,
        max_abs_factor: f64,
    ) {
        let denom = active
            .iter()
            .map(|activation| activation.coefficient * activation.coefficient)
            .sum::<f64>()
            .max(f64::EPSILON);

        for output_index in 0..self.config.output_count {
            let actual = output_before[output_index];
            if actual.abs() <= epsilon {
                for activation in active {
                    let weight_index =
                        self.raw_weight_index(activation.input, activation.interval, output_index);
                    let delta = target[output_index] * activation.coefficient / denom;
                    self.apply_delta(weight_index, delta);
                }
                continue;
            }

            let factor =
                (target[output_index] / actual).clamp(-max_abs_factor.abs(), max_abs_factor.abs());
            for activation in active {
                let weight_index =
                    self.raw_weight_index(activation.input, activation.interval, output_index);
                let current = self.weights[weight_index];
                self.apply_delta(weight_index, current * factor - current);
            }
        }
    }

    fn batch_error_denominator(&self, activations: &Array2<f64>, row: usize) -> f64 {
        match self.config.correction_mode {
            CorrectionMode::DifferenceLeastSquares => activations
                .row(row)
                .iter()
                .map(|value| value * value)
                .sum::<f64>(),
            CorrectionMode::DifferencePatentProportional => activations
                .row(row)
                .iter()
                .filter(|value| value.abs() > f64::EPSILON)
                .count() as f64,
            CorrectionMode::Ratio { .. } => unreachable!(),
        }
    }

    fn apply_delta(&mut self, weight_index: usize, delta: f64) {
        let scale = self.plasticity_scale(weight_index);
        self.weights[weight_index] += delta * scale;
        self.access_counts[weight_index] = self.access_counts[weight_index].saturating_add(1);
    }

    fn plasticity_scale(&self, weight_index: usize) -> f64 {
        match self.config.plasticity_schedule {
            PlasticitySchedule::None => 1.0,
            PlasticitySchedule::PatentHalving { freeze_after } => {
                let access_count = self.access_counts[weight_index];
                if access_count >= freeze_after {
                    0.0
                } else {
                    0.5_f64.powi(access_count as i32)
                }
            }
        }
    }

    fn validate_target(&self, target: &[f64]) -> Result<(), PannError> {
        if target.len() != self.config.output_count {
            return Err(PannError::TargetLengthMismatch {
                expected: self.config.output_count,
                actual: target.len(),
            });
        }
        Ok(())
    }

    fn validate_samples_targets(
        &self,
        samples: &[Vec<f64>],
        targets: &[Vec<f64>],
    ) -> Result<(), PannError> {
        if samples.len() != targets.len() {
            return Err(PannError::SampleTargetCountMismatch {
                samples: samples.len(),
                targets: targets.len(),
            });
        }
        for sample in samples {
            self.validate_input(sample)?;
        }
        for target in targets {
            self.validate_target(target)?;
        }
        Ok(())
    }

    fn weights_as_matrix(&self) -> Array2<f64> {
        let rows = self.config.input_count * self.config.interval_count;
        let mut matrix = Array2::zeros((rows, self.config.output_count));
        for input in 0..self.config.input_count {
            for interval in 0..self.config.interval_count {
                let row = input * self.config.interval_count + interval;
                for output in 0..self.config.output_count {
                    matrix[(row, output)] =
                        self.weights[self.raw_weight_index(input, interval, output)];
                }
            }
        }
        matrix
    }
}
