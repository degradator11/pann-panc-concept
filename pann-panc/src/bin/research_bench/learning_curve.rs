use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::Instant;

use progress_ai::pann::{Distributor, IntervalStrategy, PannConfig, PannModel};
use progress_ai::preprocess::{min_max_ranges, min_max_scale, one_hot_labels};
use progress_ai::vision::load_image_folder;

use super::run::evaluation_split;
use super::{
    Args, CommandOutput, LearningCurveReport, LearningCurveRow, OutputFormat, correction_mode_name,
    image_config, required_data_path,
};

pub fn run_pann_learning_curve(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    let data_path = required_data_path(args)?;
    let dataset = load_image_folder(data_path, image_config(args))?;
    let split = evaluation_split(&dataset, "image-folder", args)?;
    let ranges = min_max_ranges(&split.train_samples);
    let train_samples = min_max_scale(&split.train_samples, &ranges);
    let test_samples = min_max_scale(&split.test_samples, &ranges);
    let output_count = dataset.class_names.len();
    let targets = one_hot_labels(&split.train_labels, output_count);

    let mut config = PannConfig::new(train_samples[0].len(), args.intervals, output_count);
    config.distributor = Distributor::Triangular;
    config.interval_strategy = IntervalStrategy::Uniform;
    config.correction_mode = args.correction_mode;
    let mut model = PannModel::from_training_data_with_config(&train_samples, config)?;

    let start = Instant::now();
    let mut rows = Vec::with_capacity(args.epochs);
    for epoch in 1..=args.epochs {
        let epoch_metrics = model.train_epoch(&train_samples, &targets)?;
        let mean_mse_after = mean_model_mse(&model, &train_samples, &targets)?;
        rows.push(LearningCurveRow {
            epoch,
            mean_mse_before: epoch_metrics.mean_mse_before,
            mean_mse_after,
            train_accuracy: model.accuracy(&train_samples, &split.train_labels)?,
            test_accuracy: model.accuracy(&test_samples, &split.test_labels)?,
            elapsed_ms: start.elapsed().as_millis(),
        });

        if let Some(target_mse) = args.target_mse
            && mean_mse_after <= target_mse
        {
            break;
        }
    }

    let mut report = LearningCurveReport {
        model: "pann".to_string(),
        dataset: "image-folder".to_string(),
        image_features: args.image_features.as_str().to_string(),
        image_resize: args.image_resize.as_str().to_string(),
        correction_mode: correction_mode_name(args.correction_mode).to_string(),
        report_path: args.out_path.clone(),
        target_mse: args.target_mse,
        epochs_requested: args.epochs,
        epochs_completed: rows.len(),
        final_train_mse: rows.last().map(|row| row.mean_mse_after).unwrap_or(0.0),
        rows,
    };

    if let Some(out_path) = &args.out_path {
        save_learning_curve(out_path, &report, args.format)?;
        report.report_path = Some(out_path.clone());
    }

    Ok(CommandOutput::LearningCurve(report))
}

fn mean_model_mse(
    model: &PannModel,
    samples: &[Vec<f64>],
    targets: &[Vec<f64>],
) -> Result<f64, Box<dyn Error>> {
    let mut total = 0.0;
    let mut count = 0usize;
    for (sample, target) in samples.iter().zip(targets) {
        let output = model.forward(sample)?;
        for (actual, expected) in output.iter().zip(target) {
            let error = expected - actual;
            total += error * error;
            count += 1;
        }
    }
    Ok(if count == 0 {
        0.0
    } else {
        total / count as f64
    })
}

fn save_learning_curve(
    out_path: &str,
    report: &LearningCurveReport,
    format: OutputFormat,
) -> Result<(), Box<dyn Error>> {
    let path = Path::new(out_path);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    match format {
        OutputFormat::Json => fs::write(path, serde_json::to_string_pretty(report)?)?,
        OutputFormat::Csv => {
            let mut writer = csv::Writer::from_path(path)?;
            for row in &report.rows {
                writer.serialize(row)?;
            }
            writer.flush()?;
        }
    }
    Ok(())
}
