use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use progress_ai::panc::{PancComparator, SimilarityMetric};
use progress_ai::pann::{CorrectionMode, Distributor, IntervalStrategy, PannConfig, PannModel};
use progress_ai::preprocess::{
    Dataset, min_max_ranges, min_max_scale, one_hot_labels, train_test_split,
};

#[derive(Debug)]
struct Args {
    command: String,
    format: OutputFormat,
    data_path: Option<String>,
    epochs: usize,
    intervals: usize,
    seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Csv,
}

#[derive(Debug, Serialize)]
struct BenchMetrics {
    model: String,
    dataset: String,
    train_accuracy: f64,
    test_accuracy: f64,
    train_ms: u128,
    inference_ms: u128,
    memory_bytes: usize,
    epochs: usize,
    interval_count: usize,
    distributor: String,
    correction_mode: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let metrics = match args.command.as_str() {
        "pann-iris" => run_pann(load_iris(args.data_path.as_deref())?, "iris", &args)?,
        "pann-synthetic" => run_pann(synthetic_dataset(args.seed), "synthetic", &args)?,
        "panc-iris" => run_panc(load_iris(args.data_path.as_deref())?, "iris", &args)?,
        "panc-synthetic" => run_panc(synthetic_dataset(args.seed), "synthetic", &args)?,
        command => {
            return Err(format!(
                "unknown command {command}; expected pann-iris, pann-synthetic, panc-iris, panc-synthetic"
            )
            .into());
        }
    };
    write_metrics(&metrics, args.format)?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut raw = env::args().skip(1);
    let command = raw
        .next()
        .ok_or("usage: research-bench <pann-iris|pann-synthetic|panc-iris|panc-synthetic> [--format json|csv] [--data path] [--epochs n] [--intervals n] [--seed n]")?;

    let mut args = Args {
        command,
        format: OutputFormat::Json,
        data_path: None,
        epochs: 12,
        intervals: 8,
        seed: 42,
    };

    while let Some(flag) = raw.next() {
        match flag.as_str() {
            "--format" => {
                args.format = match raw.next().as_deref() {
                    Some("json") => OutputFormat::Json,
                    Some("csv") => OutputFormat::Csv,
                    other => return Err(format!("invalid --format value: {other:?}").into()),
                };
            }
            "--data" => args.data_path = raw.next(),
            "--epochs" => {
                args.epochs = raw
                    .next()
                    .ok_or("--epochs requires a value")?
                    .parse::<usize>()?;
            }
            "--intervals" => {
                args.intervals = raw
                    .next()
                    .ok_or("--intervals requires a value")?
                    .parse::<usize>()?;
            }
            "--seed" => {
                args.seed = raw
                    .next()
                    .ok_or("--seed requires a value")?
                    .parse::<u64>()?;
            }
            other => return Err(format!("unknown option {other}").into()),
        }
    }

    Ok(args)
}

fn run_pann(
    dataset: Dataset,
    dataset_name: &str,
    args: &Args,
) -> Result<BenchMetrics, Box<dyn Error>> {
    let split = train_test_split(&dataset.samples, &dataset.labels, 0.2, args.seed);
    let ranges = min_max_ranges(&split.train_samples);
    let train_samples = min_max_scale(&split.train_samples, &ranges);
    let test_samples = min_max_scale(&split.test_samples, &ranges);
    let output_count = dataset.class_names.len();
    let targets = one_hot_labels(&split.train_labels, output_count);

    let mut config = PannConfig::new(train_samples[0].len(), args.intervals, output_count);
    config.distributor = Distributor::Triangular;
    config.interval_strategy = IntervalStrategy::Uniform;
    config.correction_mode = CorrectionMode::DifferenceLeastSquares;
    let mut model = PannModel::from_training_data_with_config(&train_samples, config)?;

    let train_start = Instant::now();
    for _ in 0..args.epochs {
        model.train_epoch(&train_samples, &targets)?;
    }
    let train_ms = train_start.elapsed().as_millis();

    let inference_start = Instant::now();
    let train_accuracy = model.accuracy(&train_samples, &split.train_labels)?;
    let test_accuracy = model.accuracy(&test_samples, &split.test_labels)?;
    let inference_ms = inference_start.elapsed().as_millis();

    Ok(BenchMetrics {
        model: "pann".to_string(),
        dataset: dataset_name.to_string(),
        train_accuracy,
        test_accuracy,
        train_ms,
        inference_ms,
        memory_bytes: model.memory_bytes_estimate(),
        epochs: args.epochs,
        interval_count: args.intervals,
        distributor: "triangular".to_string(),
        correction_mode: "difference_least_squares".to_string(),
    })
}

fn run_panc(
    dataset: Dataset,
    dataset_name: &str,
    args: &Args,
) -> Result<BenchMetrics, Box<dyn Error>> {
    let split = train_test_split(&dataset.samples, &dataset.labels, 0.2, args.seed);
    let ranges = min_max_ranges(&split.train_samples);
    let train_samples = min_max_scale(&split.train_samples, &ranges);
    let test_samples = min_max_scale(&split.test_samples, &ranges);

    let train_start = Instant::now();
    let mut comparator = PancComparator::new(SimilarityMetric::Euclidean);
    for (sample, label) in train_samples.iter().zip(&split.train_labels) {
        comparator.add_reference(sample.clone(), *label, ())?;
    }
    let train_ms = train_start.elapsed().as_millis();

    let inference_start = Instant::now();
    let train_accuracy = panc_accuracy(&comparator, &train_samples, &split.train_labels, 3)?;
    let test_accuracy = panc_accuracy(&comparator, &test_samples, &split.test_labels, 3)?;
    let inference_ms = inference_start.elapsed().as_millis();

    Ok(BenchMetrics {
        model: "panc_like".to_string(),
        dataset: dataset_name.to_string(),
        train_accuracy,
        test_accuracy,
        train_ms,
        inference_ms,
        memory_bytes: train_samples.len() * train_samples[0].len() * std::mem::size_of::<f64>(),
        epochs: 0,
        interval_count: 0,
        distributor: "none".to_string(),
        correction_mode: "top_k_euclidean_vote".to_string(),
    })
}

fn panc_accuracy(
    comparator: &PancComparator<usize>,
    samples: &[Vec<f64>],
    labels: &[usize],
    k: usize,
) -> Result<f64, Box<dyn Error>> {
    if samples.is_empty() {
        return Ok(0.0);
    }

    let mut correct = 0usize;
    for (sample, label) in samples.iter().zip(labels) {
        if comparator.predict_label(sample, k)? == Some(*label) {
            correct += 1;
        }
    }
    Ok(correct as f64 / samples.len() as f64)
}

fn load_iris(path: Option<&str>) -> Result<Dataset, Box<dyn Error>> {
    let contents = if let Some(path) = path {
        fs::read_to_string(path)?
    } else {
        include_str!("../../data/iris.csv").to_string()
    };

    let mut rdr = csv::Reader::from_reader(contents.as_bytes());
    let mut samples = Vec::new();
    let mut labels = Vec::new();
    let mut class_to_label = HashMap::<String, usize>::new();
    let mut class_names = Vec::<String>::new();

    for record in rdr.records() {
        let record = record?;
        let sample = (0..4)
            .map(|index| record[index].parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        let species = record[4].to_string();
        let label = if let Some(label) = class_to_label.get(&species) {
            *label
        } else {
            let label = class_names.len();
            class_to_label.insert(species.clone(), label);
            class_names.push(species);
            label
        };
        samples.push(sample);
        labels.push(label);
    }

    Ok(Dataset {
        samples,
        labels,
        class_names,
    })
}

fn synthetic_dataset(seed: u64) -> Dataset {
    let centers: [[f64; 2]; 3] = [[0.15, 0.15], [0.85, 0.2], [0.5, 0.85]];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut samples = Vec::new();
    let mut labels = Vec::new();

    for (label, center) in centers.iter().enumerate() {
        for _ in 0..80 {
            let x = (center[0] + rng.gen_range(-0.05_f64..0.05_f64)).clamp(0.0, 1.0);
            let y = (center[1] + rng.gen_range(-0.05_f64..0.05_f64)).clamp(0.0, 1.0);
            samples.push(vec![x, y]);
            labels.push(label);
        }
    }

    Dataset {
        samples,
        labels,
        class_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
    }
}

fn write_metrics(metrics: &BenchMetrics, format: OutputFormat) -> Result<(), Box<dyn Error>> {
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(metrics)?),
        OutputFormat::Csv => {
            let mut writer = csv::Writer::from_writer(std::io::stdout());
            writer.serialize(metrics)?;
            writer.flush()?;
        }
    }
    Ok(())
}
