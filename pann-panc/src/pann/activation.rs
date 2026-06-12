use ndarray::Array2;

use super::intervals::interval_from_edges;
use super::{Activation, Distributor, PannError, PannModel, argmax};

impl PannModel {
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

    pub(super) fn validate_input(&self, input: &[f64]) -> Result<(), PannError> {
        if input.len() != self.config.input_count {
            return Err(PannError::InputLengthMismatch {
                expected: self.config.input_count,
                actual: input.len(),
            });
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
}
