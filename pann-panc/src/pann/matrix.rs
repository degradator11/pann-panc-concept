use ndarray::Array2;

use super::PannError;

pub(crate) fn targets_as_matrix(
    targets: &[Vec<f64>],
    output_count: usize,
) -> Result<Array2<f64>, PannError> {
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

pub(crate) fn column_usage(activations: &Array2<f64>) -> Vec<u32> {
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
