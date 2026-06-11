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
        PannModel::with_config_and_ranges(config, vec![FeatureRange::new(0.0, 1.0)], None).unwrap();

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
        PannModel::with_config_and_ranges(config, vec![FeatureRange::new(0.0, 1.0)], None).unwrap();

    model.train_one(&[0.5], &[1.0]).unwrap();
    model.train_one(&[0.5], &[100.0]).unwrap();

    assert_close(model.forward(&[0.5]).unwrap()[0], 2.0);
}

#[test]
fn access_count_decay_reduces_later_weight_changes() {
    let mut config = PannConfig::new(1, 1, 1);
    config.plasticity_schedule = PlasticitySchedule::PatentHalving { freeze_after: 5 };
    let mut model =
        PannModel::with_config_and_ranges(config, vec![FeatureRange::new(0.0, 1.0)], None).unwrap();

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

#[test]
fn snapshot_round_trip_preserves_predictions() {
    let samples = vec![vec![0.1], vec![0.9]];
    let targets = vec![one_hot(0, 2), one_hot(1, 2)];
    let mut model = PannModel::with_unit_ranges(1, 2, 2, Distributor::HardBin).unwrap();

    model.train_epoch_difference(&samples, &targets).unwrap();
    let restored = PannModel::from_snapshot(model.snapshot()).unwrap();

    assert_eq!(
        restored.predict(&[0.1]).unwrap(),
        model.predict(&[0.1]).unwrap()
    );
    assert_eq!(
        restored.predict(&[0.9]).unwrap(),
        model.predict(&[0.9]).unwrap()
    );
    assert_eq!(restored.weights(), model.weights());
    assert_eq!(restored.access_counts(), model.access_counts());
}
