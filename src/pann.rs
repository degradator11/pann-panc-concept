use ndarray::Array2;
use serde::{Deserialize, Serialize};
use thiserror::Error;

type QuantileEdges = Vec<Vec<f64>>;
type FittedIntervals = (Vec<FeatureRange>, Option<QuantileEdges>);

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

#[derive(Debug, Clone)]
pub struct PannModel {
    config: PannConfig,
    ranges: Vec<FeatureRange>,
    quantile_edges: Option<QuantileEdges>,
    weights: Vec<f64>,
    access_counts: Vec<u32>,
}

impl PannModel {
    pub fn new(
        input_count: usize,
        interval_count: usize,
        output_count: usize,
        ranges: Vec<FeatureRange>,
        distributor: Distributor,
    ) -> Result<Self, PannError> {
        let mut config = PannConfig::new(input_count, interval_count, output_count);
        config.distributor = distributor;
        Self::with_config_and_ranges(config, ranges, None)
    }

    pub fn with_unit_ranges(
        input_count: usize,
        interval_count: usize,
        output_count: usize,
        distributor: Distributor,
    ) -> Result<Self, PannError> {
        Self::new(
            input_count,
            interval_count,
            output_count,
            vec![FeatureRange::new(0.0, 1.0); input_count],
            distributor,
        )
    }

    pub fn with_config_and_ranges(
        config: PannConfig,
        ranges: Vec<FeatureRange>,
        quantile_edges: Option<Vec<Vec<f64>>>,
    ) -> Result<Self, PannError> {
        validate_config(&config)?;
        validate_ranges(config.input_count, &ranges)?;

        if let Some(edges) = &quantile_edges
            && edges.len() != config.input_count
        {
            return Err(PannError::RangeCountMismatch {
                expected: config.input_count,
                actual: edges.len(),
            });
        }

        let weight_count = config.input_count * config.interval_count * config.output_count;
        Ok(Self {
            config,
            ranges,
            quantile_edges,
            weights: vec![0.0; weight_count],
            access_counts: vec![0; weight_count],
        })
    }

    pub fn from_training_data(
        samples: &[Vec<f64>],
        interval_count: usize,
        output_count: usize,
        distributor: Distributor,
    ) -> Result<Self, PannError> {
        let input_count = validate_samples(samples)?;
        let mut config = PannConfig::new(input_count, interval_count, output_count);
        config.distributor = distributor;
        Self::from_training_data_with_config(samples, config)
    }

    pub fn from_training_data_with_config(
        samples: &[Vec<f64>],
        config: PannConfig,
    ) -> Result<Self, PannError> {
        validate_config(&config)?;
        let input_count = validate_samples(samples)?;
        if input_count != config.input_count {
            return Err(PannError::InputLengthMismatch {
                expected: config.input_count,
                actual: input_count,
            });
        }

        let (ranges, quantile_edges) =
            fit_intervals(samples, config.interval_count, config.interval_strategy)?;
        Self::with_config_and_ranges(config, ranges, quantile_edges)
    }

    pub const fn config(&self) -> &PannConfig {
        &self.config
    }

    pub const fn input_count(&self) -> usize {
        self.config.input_count
    }

    pub const fn interval_count(&self) -> usize {
        self.config.interval_count
    }

    pub const fn output_count(&self) -> usize {
        self.config.output_count
    }

    pub const fn distributor(&self) -> Distributor {
        self.config.distributor
    }

    pub fn ranges(&self) -> &[FeatureRange] {
        &self.ranges
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn access_counts(&self) -> &[u32] {
        &self.access_counts
    }

    pub fn memory_bytes_estimate(&self) -> usize {
        self.weights.len() * std::mem::size_of::<f64>()
            + self.access_counts.len() * std::mem::size_of::<u32>()
    }

    pub fn weight(&self, input: usize, interval: usize, output: usize) -> Option<f64> {
        self.weight_index(input, interval, output)
            .map(|index| self.weights[index])
    }

    pub fn access_count(&self, input: usize, interval: usize, output: usize) -> Option<u32> {
        self.weight_index(input, interval, output)
            .map(|index| self.access_counts[index])
    }

    pub fn encode(&self, input: &[f64]) -> Result<Vec<Activation>, PannError> {
        self.validate_input(input)?;

        let mut active = Vec::with_capacity(self.config.input_count);
        for (input_index, value) in input.iter().copied().enumerate() {
            match self.config.distributor {
                Distributor::HardBin => active.push(Activation {
                    input: input_index,
                    interval: self.hard_interval(input_index, value),
                    coefficient: 1.0,
                }),
                Distributor::Triangular => {
                    self.push_triangular_activations(input_index, value, &mut active);
                }
                Distributor::Gaussian { radius, sigma } => {
                    self.push_gaussian_activations(input_index, value, radius, sigma, &mut active);
                }
            }
        }

        if active.is_empty() {
            Err(PannError::EmptyActiveSet)
        } else {
            Ok(active)
        }
    }

    pub fn activation_matrix(&self, samples: &[Vec<f64>]) -> Result<Array2<f64>, PannError> {
        for sample in samples {
            self.validate_input(sample)?;
        }

        let column_count = self.config.input_count * self.config.interval_count;
        let mut matrix = Array2::zeros((samples.len(), column_count));
        for (row, sample) in samples.iter().enumerate() {
            for activation in self.encode(sample)? {
                let column = activation.column(self.config.interval_count);
                matrix[(row, column)] += activation.coefficient;
            }
        }
        Ok(matrix)
    }

    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>, PannError> {
        let active = self.encode(input)?;
        Ok(self.forward_from_activations(&active))
    }

    pub fn forward_from_activations(&self, active: &[Activation]) -> Vec<f64> {
        let mut output = vec![0.0; self.config.output_count];
        for activation in active {
            for (output_index, output_value) in output.iter_mut().enumerate() {
                let weight_index =
                    self.raw_weight_index(activation.input, activation.interval, output_index);
                *output_value += activation.coefficient * self.weights[weight_index];
            }
        }
        output
    }

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
                let mut total_mse = 0.0;
                self.validate_samples_targets(samples, targets)?;
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
        let weights = self.weights_as_matrix();
        let outputs = activations.dot(&weights);
        let targets = targets_as_matrix(targets, self.config.output_count)?;
        let errors = targets - outputs;
        let mse = errors.iter().map(|value| value * value).sum::<f64>()
            / (samples.len() * self.config.output_count).max(1) as f64;

        let mut normalized_errors = errors.clone();
        for row in 0..activations.nrows() {
            let denom = match self.config.correction_mode {
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
            };
            let denom = denom.max(f64::EPSILON);
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

    pub fn predict(&self, input: &[f64]) -> Result<usize, PannError> {
        let output = self.forward(input)?;
        Ok(argmax(&output))
    }

    pub fn accuracy(&self, samples: &[Vec<f64>], labels: &[usize]) -> Result<f64, PannError> {
        if samples.len() != labels.len() {
            return Err(PannError::SampleTargetCountMismatch {
                samples: samples.len(),
                targets: labels.len(),
            });
        }

        if samples.is_empty() {
            return Ok(0.0);
        }

        let mut correct = 0usize;
        for (sample, expected) in samples.iter().zip(labels) {
            if self.predict(sample)? == *expected {
                correct += 1;
            }
        }

        Ok(correct as f64 / samples.len() as f64)
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
                        let delta = error * activation.coefficient / denom;
                        self.apply_delta(weight_index, delta);
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
                        let delta = error * activation.coefficient / denom;
                        self.apply_delta(weight_index, delta);
                    }
                }
            }
            CorrectionMode::Ratio {
                epsilon,
                max_abs_factor,
            } => {
                let denom = active
                    .iter()
                    .map(|activation| activation.coefficient * activation.coefficient)
                    .sum::<f64>()
                    .max(f64::EPSILON);
                for output_index in 0..self.config.output_count {
                    let actual = output_before[output_index];
                    if actual.abs() <= epsilon {
                        for activation in &active {
                            let weight_index = self.raw_weight_index(
                                activation.input,
                                activation.interval,
                                output_index,
                            );
                            let delta = target[output_index] * activation.coefficient / denom;
                            self.apply_delta(weight_index, delta);
                        }
                        continue;
                    }

                    let factor = (target[output_index] / actual)
                        .clamp(-max_abs_factor.abs(), max_abs_factor.abs());
                    for activation in &active {
                        let weight_index = self.raw_weight_index(
                            activation.input,
                            activation.interval,
                            output_index,
                        );
                        let current = self.weights[weight_index];
                        let desired = current * factor;
                        self.apply_delta(weight_index, desired - current);
                    }
                }
            }
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

    fn validate_input(&self, input: &[f64]) -> Result<(), PannError> {
        if input.len() != self.config.input_count {
            return Err(PannError::InputLengthMismatch {
                expected: self.config.input_count,
                actual: input.len(),
            });
        }
        Ok(())
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

    fn hard_interval(&self, input_index: usize, value: f64) -> usize {
        if self.config.interval_count == 1 {
            return 0;
        }

        if let Some(edges) = &self.quantile_edges {
            return interval_from_edges(&edges[input_index], value, self.config.interval_count);
        }

        let position = self.normalized_position(input_index, value);
        let raw = (position * self.config.interval_count as f64).floor() as usize;
        raw.min(self.config.interval_count - 1)
    }

    fn push_triangular_activations(
        &self,
        input_index: usize,
        value: f64,
        active: &mut Vec<Activation>,
    ) {
        if self.config.interval_count == 1 {
            active.push(Activation {
                input: input_index,
                interval: 0,
                coefficient: 1.0,
            });
            return;
        }

        let continuous = self.continuous_interval_position(input_index, value);
        let left = continuous.floor() as usize;
        let right = continuous.ceil() as usize;

        for interval in [left, right] {
            if interval >= self.config.interval_count {
                continue;
            }
            let coefficient = 1.0 - (continuous - interval as f64).abs();
            if coefficient > 0.0 {
                active.push(Activation {
                    input: input_index,
                    interval,
                    coefficient,
                });
            }
        }
    }

    fn push_gaussian_activations(
        &self,
        input_index: usize,
        value: f64,
        radius: usize,
        sigma: f64,
        active: &mut Vec<Activation>,
    ) {
        let center = self.hard_interval(input_index, value);
        let sigma = sigma.abs().max(f64::EPSILON);
        let start = center.saturating_sub(radius);
        let end = (center + radius).min(self.config.interval_count - 1);

        for interval in start..=end {
            let distance = interval.abs_diff(center) as f64;
            let coefficient = (-(distance * distance) / (2.0 * sigma * sigma)).exp();
            if coefficient > 0.0 {
                active.push(Activation {
                    input: input_index,
                    interval,
                    coefficient,
                });
            }
        }
    }

    fn continuous_interval_position(&self, input_index: usize, value: f64) -> f64 {
        if self.quantile_edges.is_some() {
            return self.hard_interval(input_index, value) as f64;
        }

        self.normalized_position(input_index, value) * (self.config.interval_count - 1) as f64
    }

    fn normalized_position(&self, input_index: usize, value: f64) -> f64 {
        let range = self.ranges[input_index];
        let width = range.max - range.min;
        if width <= f64::EPSILON || !value.is_finite() {
            return 0.0;
        }

        ((value - range.min) / width).clamp(0.0, 1.0)
    }

    fn weight_index(&self, input: usize, interval: usize, output: usize) -> Option<usize> {
        if input >= self.config.input_count
            || interval >= self.config.interval_count
            || output >= self.config.output_count
        {
            None
        } else {
            Some(self.raw_weight_index(input, interval, output))
        }
    }

    fn raw_weight_index(&self, input: usize, interval: usize, output: usize) -> usize {
        (input * self.config.interval_count + interval) * self.config.output_count + output
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

fn validate_config(config: &PannConfig) -> Result<(), PannError> {
    if config.input_count == 0 {
        return Err(PannError::InvalidInputCount);
    }
    if config.interval_count == 0 {
        return Err(PannError::InvalidIntervalCount);
    }
    if config.output_count == 0 {
        return Err(PannError::InvalidOutputCount);
    }
    Ok(())
}

fn validate_ranges(input_count: usize, ranges: &[FeatureRange]) -> Result<(), PannError> {
    if ranges.len() != input_count {
        return Err(PannError::RangeCountMismatch {
            expected: input_count,
            actual: ranges.len(),
        });
    }

    for (index, range) in ranges.iter().enumerate() {
        let valid = range.min.is_finite() && range.max.is_finite() && range.min <= range.max;
        if !valid {
            return Err(PannError::InvalidRange {
                index,
                min: range.min,
                max: range.max,
            });
        }
    }
    Ok(())
}

fn validate_samples(samples: &[Vec<f64>]) -> Result<usize, PannError> {
    let first = samples.first().ok_or(PannError::EmptySamples)?;
    let input_count = first.len();
    if input_count == 0 {
        return Err(PannError::InvalidInputCount);
    }
    for sample in samples {
        if sample.len() != input_count {
            return Err(PannError::InputLengthMismatch {
                expected: input_count,
                actual: sample.len(),
            });
        }
    }
    Ok(input_count)
}

fn fit_intervals(
    samples: &[Vec<f64>],
    interval_count: usize,
    strategy: IntervalStrategy,
) -> Result<FittedIntervals, PannError> {
    let input_count = validate_samples(samples)?;
    let mut ranges = Vec::with_capacity(input_count);
    let mut quantile_edges = if matches!(strategy, IntervalStrategy::Quantile) {
        Some(Vec::with_capacity(input_count))
    } else {
        None
    };

    for input_index in 0..input_count {
        let mut values = samples
            .iter()
            .map(|sample| sample[input_index])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);

        let (min, max) = match strategy {
            IntervalStrategy::Uniform | IntervalStrategy::Quantile => (
                *values.first().unwrap_or(&0.0),
                *values.last().unwrap_or(&0.0),
            ),
            IntervalStrategy::ClippedPercentile { lower, upper } => (
                percentile_sorted(&values, lower),
                percentile_sorted(&values, upper),
            ),
        };
        ranges.push(FeatureRange::new(min, max));

        if let Some(edges) = &mut quantile_edges {
            let mut feature_edges = Vec::with_capacity(interval_count + 1);
            for index in 0..=interval_count {
                feature_edges.push(percentile_sorted(
                    &values,
                    index as f64 / interval_count as f64,
                ));
            }
            edges.push(feature_edges);
        }
    }

    validate_ranges(input_count, &ranges)?;
    Ok((ranges, quantile_edges))
}

fn interval_from_edges(edges: &[f64], value: f64, interval_count: usize) -> usize {
    if interval_count == 1 || edges.len() < 2 || !value.is_finite() {
        return 0;
    }

    for interval in 0..interval_count {
        if value <= edges[interval + 1] {
            return interval;
        }
    }
    interval_count - 1
}

fn percentile_sorted(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let quantile = quantile.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values[index]
}

fn targets_as_matrix(targets: &[Vec<f64>], output_count: usize) -> Result<Array2<f64>, PannError> {
    let mut matrix = Array2::zeros((targets.len(), output_count));
    for (row, target) in targets.iter().enumerate() {
        if target.len() != output_count {
            return Err(PannError::TargetLengthMismatch {
                expected: output_count,
                actual: target.len(),
            });
        }
        for (column, value) in target.iter().copied().enumerate() {
            matrix[(row, column)] = value;
        }
    }
    Ok(matrix)
}

fn column_usage(activations: &Array2<f64>) -> Vec<u32> {
    let mut usage = vec![0; activations.ncols()];
    for row in 0..activations.nrows() {
        for column in 0..activations.ncols() {
            if activations[(row, column)].abs() > f64::EPSILON {
                usage[column] += 1;
            }
        }
    }
    usage
}

fn mean_squared(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64
    }
}

fn mean_or_zero(total: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-10, "left={left}, right={right}");
    }

    #[test]
    fn hard_bin_one_step_update_moves_current_sample_to_target() {
        let mut model = PannModel::with_unit_ranges(2, 4, 2, Distributor::HardBin).unwrap();
        let input = [0.2, 0.8];
        let target = [1.0, -1.0];

        let step = model.train_one_difference(&input, &target).unwrap();

        assert_eq!(step.active_count, 2);
        assert_close(step.output_after[0], target[0]);
        assert_close(step.output_after[1], target[1]);
    }

    #[test]
    fn hard_bin_update_changes_only_active_weights() {
        let mut model = PannModel::with_unit_ranges(2, 4, 2, Distributor::HardBin).unwrap();
        let input = [0.2, 0.8];
        let target = [1.0, -1.0];

        model.train_one_difference(&input, &target).unwrap();

        for input_index in 0..model.input_count() {
            for interval_index in 0..model.interval_count() {
                for output_index in 0..model.output_count() {
                    let weight = model
                        .weight(input_index, interval_index, output_index)
                        .unwrap();
                    let is_active = (input_index == 0 && interval_index == 0)
                        || (input_index == 1 && interval_index == 3);

                    if is_active && output_index == 0 {
                        assert_close(weight, 0.5);
                    } else if is_active && output_index == 1 {
                        assert_close(weight, -0.5);
                    } else {
                        assert_close(weight, 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn triangular_update_is_coefficient_aware() {
        let mut model = PannModel::with_unit_ranges(1, 4, 1, Distributor::Triangular).unwrap();
        let input = [0.5];
        let target = [0.75];

        let step = model.train_one_difference(&input, &target).unwrap();

        assert_eq!(step.active_count, 2);
        assert_close(step.output_after[0], target[0]);
    }

    #[test]
    fn gaussian_activates_center_and_neighbors() {
        let model = PannModel::with_unit_ranges(
            1,
            5,
            1,
            Distributor::Gaussian {
                radius: 1,
                sigma: 1.0,
            },
        )
        .unwrap();

        let active = model.encode(&[0.5]).unwrap();

        assert_eq!(active.len(), 3);
        assert!(active.iter().any(|activation| activation.interval == 2));
    }

    #[test]
    fn ratio_update_uses_additive_zero_fallback() {
        let mut config = PannConfig::new(1, 2, 1);
        config.correction_mode = CorrectionMode::Ratio {
            epsilon: 1e-9,
            max_abs_factor: 10.0,
        };
        let mut model =
            PannModel::with_config_and_ranges(config, vec![FeatureRange::new(0.0, 1.0)], None)
                .unwrap();

        let step = model.train_one(&[0.25], &[1.0]).unwrap();

        assert_close(step.output_after[0], 1.0);
    }

    #[test]
    fn ratio_update_clips_large_factor() {
        let mut config = PannConfig::new(1, 1, 1);
        config.correction_mode = CorrectionMode::Ratio {
            epsilon: 1e-9,
            max_abs_factor: 2.0,
        };
        let mut model =
            PannModel::with_config_and_ranges(config, vec![FeatureRange::new(0.0, 1.0)], None)
                .unwrap();

        model.train_one(&[0.5], &[1.0]).unwrap();
        model.train_one(&[0.5], &[100.0]).unwrap();

        assert_close(model.forward(&[0.5]).unwrap()[0], 2.0);
    }

    #[test]
    fn access_count_decay_reduces_later_weight_changes() {
        let mut config = PannConfig::new(1, 1, 1);
        config.plasticity_schedule = PlasticitySchedule::PatentHalving { freeze_after: 5 };
        let mut model =
            PannModel::with_config_and_ranges(config, vec![FeatureRange::new(0.0, 1.0)], None)
                .unwrap();

        model.train_one_difference(&[0.5], &[1.0]).unwrap();
        model.train_one_difference(&[0.5], &[0.0]).unwrap();

        assert_close(model.weight(0, 0, 0).unwrap(), 0.5);
        assert_eq!(model.access_count(0, 0, 0), Some(2));
    }

    #[test]
    fn quantile_intervals_split_skewed_values() {
        let samples = vec![vec![0.0], vec![1.0], vec![10.0], vec![100.0]];
        let mut config = PannConfig::new(1, 2, 1);
        config.interval_strategy = IntervalStrategy::Quantile;
        let model = PannModel::from_training_data_with_config(&samples, config).unwrap();

        assert_eq!(model.encode(&[0.0]).unwrap()[0].interval, 0);
        assert_eq!(model.encode(&[100.0]).unwrap()[0].interval, 1);
    }

    #[test]
    fn matrix_training_matches_online_for_non_overlapping_samples() {
        let samples = vec![vec![0.1], vec![0.9]];
        let targets = vec![one_hot(0, 2), one_hot(1, 2)];
        let mut online = PannModel::with_unit_ranges(1, 2, 2, Distributor::HardBin).unwrap();
        let mut matrix = PannModel::with_unit_ranges(1, 2, 2, Distributor::HardBin).unwrap();

        online.train_epoch_difference(&samples, &targets).unwrap();
        matrix.train_epoch_matrix(&samples, &targets).unwrap();

        assert_eq!(online.weights(), matrix.weights());
    }

    #[test]
    fn learns_tiny_separable_dataset_when_bins_do_not_overlap() {
        let samples = vec![vec![0.1, 0.1], vec![0.9, 0.9]];
        let targets = vec![one_hot(0, 2), one_hot(1, 2)];
        let labels = vec![0, 1];
        let mut model = PannModel::with_unit_ranges(2, 4, 2, Distributor::HardBin).unwrap();

        model.train_epoch_difference(&samples, &targets).unwrap();

        assert_close(model.accuracy(&samples, &labels).unwrap(), 1.0);
    }
}
