use super::{FeatureRange, IntervalStrategy, PannConfig, PannError};

pub(crate) type QuantileEdges = Vec<Vec<f64>>;
pub(crate) type FittedIntervals = (Vec<FeatureRange>, Option<QuantileEdges>);

pub(crate) fn validate_config(config: &PannConfig) -> Result<(), PannError> {
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

pub(crate) fn validate_ranges(
    input_count: usize,
    ranges: &[FeatureRange],
) -> Result<(), PannError> {
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

pub(crate) fn validate_samples(samples: &[Vec<f64>]) -> Result<usize, PannError> {
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

pub(crate) fn fit_intervals(
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

pub(crate) fn interval_from_edges(edges: &[f64], value: f64, interval_count: usize) -> usize {
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
