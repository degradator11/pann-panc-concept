use super::intervals::{
    QuantileEdges, fit_intervals, validate_config, validate_ranges, validate_samples,
};
use super::{Distributor, FeatureRange, PannConfig, PannError};

#[derive(Debug, Clone)]
pub struct PannModel {
    pub(super) config: PannConfig,
    pub(super) ranges: Vec<FeatureRange>,
    pub(super) quantile_edges: Option<QuantileEdges>,
    pub(super) weights: Vec<f64>,
    pub(super) access_counts: Vec<u32>,
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
        quantile_edges: Option<QuantileEdges>,
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

    pub(super) fn weight_index(
        &self,
        input: usize,
        interval: usize,
        output: usize,
    ) -> Option<usize> {
        if input >= self.config.input_count
            || interval >= self.config.interval_count
            || output >= self.config.output_count
        {
            None
        } else {
            Some(self.raw_weight_index(input, interval, output))
        }
    }

    pub(super) fn raw_weight_index(&self, input: usize, interval: usize, output: usize) -> usize {
        (input * self.config.interval_count + interval) * self.config.output_count + output
    }
}
