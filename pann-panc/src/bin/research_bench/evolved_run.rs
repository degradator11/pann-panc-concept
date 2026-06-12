use std::error::Error;
use std::time::Instant;

use progress_ai::evolution::panc::{format_block_mask, normalize_block_mask};
use progress_ai::evolution::{PancGenome, evaluate_panc_binary};
use progress_ai::preprocess::{Dataset, min_max_ranges, min_max_scale, train_test_split_indices};
use progress_ai::vision::{ImageVectorConfig, load_image_folder};

use super::{Args, BenchMetrics, CommandOutput, classification_metrics, required_data_path};

pub fn run_evolved_panc_image_folder(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    let data_path = required_data_path(args)?;
    let genome = genome_from_args(args);
    let image_config = image_config_from_genome(genome);
    let dataset = load_image_folder(data_path, image_config)?;
    let split = split_dataset(&dataset, args)?;

    let ranges = min_max_ranges(&split.train_samples);
    let train_samples = min_max_scale(&split.train_samples, &ranges);
    let test_samples = min_max_scale(&split.test_samples, &ranges);
    let class_count = dataset.class_names.len();

    let train_start = Instant::now();
    let train_evaluation = evaluate_panc_binary(
        &train_samples,
        &split.train_labels,
        &train_samples,
        &split.train_labels,
        class_count,
        genome,
    );
    let train_ms = train_start.elapsed().as_millis();

    let inference_start = Instant::now();
    let test_evaluation = evaluate_panc_binary(
        &train_samples,
        &split.train_labels,
        &test_samples,
        &split.test_labels,
        class_count,
        genome,
    );
    let inference_ms = inference_start.elapsed().as_millis();
    let test_diagnostics = classification_metrics(
        &split.test_labels,
        &test_evaluation.predictions,
        &dataset.class_names,
    );

    Ok(CommandOutput::Metrics(BenchMetrics {
        model: "evolved_panc_like".to_string(),
        dataset: "image-folder".to_string(),
        image_features: genome.feature_mode.as_str().to_string(),
        image_resize: genome.resize_mode.as_str().to_string(),
        train_accuracy: train_evaluation.accuracy,
        test_accuracy: test_diagnostics.accuracy,
        train_ms,
        inference_ms,
        memory_bytes: test_evaluation.memory_bytes,
        epochs: 0,
        interval_count: 0,
        distributor: "binary_packed_analogue_library".to_string(),
        correction_mode: format!(
            "{}_threshold_{:.3}_blocks_{}",
            genome.similarity_name(),
            genome.threshold,
            format_block_mask(genome.active_blocks)
        ),
        per_class_accuracy: test_diagnostics.per_class_accuracy,
        confusion_matrix: test_diagnostics.confusion_matrix,
    }))
}

fn genome_from_args(args: &Args) -> PancGenome {
    PancGenome {
        image_size: args.image_width.max(args.image_height).max(1),
        feature_mode: args.image_features,
        resize_mode: args.image_resize,
        threshold: args.panc_threshold.unwrap_or(0.5).clamp(0.0, 1.0),
        jaccard_weight: args.panc_jaccard_weight.unwrap_or(0.0).clamp(0.0, 1.0),
        top_k: args.top_k.max(1),
        active_blocks: normalize_block_mask(args.panc_active_blocks.unwrap_or(u32::MAX)),
    }
}

fn image_config_from_genome(genome: PancGenome) -> ImageVectorConfig {
    ImageVectorConfig::new(genome.image_size, genome.image_size)
        .with_feature_mode(genome.feature_mode)
        .with_resize_mode(genome.resize_mode)
}

fn split_dataset(dataset: &Dataset, args: &Args) -> Result<FolderSplit, Box<dyn Error>> {
    if let Some(eval_path) = args.eval_data_path.as_deref() {
        let eval_dataset = load_image_folder(eval_path, image_config_from_args(args))?;
        let eval_labels = remap_labels_by_class_name(&eval_dataset, &dataset.class_names)?;
        return Ok(FolderSplit {
            train_samples: dataset.samples.clone(),
            train_labels: dataset.labels.clone(),
            test_samples: eval_dataset.samples,
            test_labels: eval_labels,
        });
    }

    let (test_indexes, train_indexes) =
        train_test_split_indices(dataset.samples.len(), 0.2, args.seed);
    Ok(FolderSplit {
        train_samples: collect_samples(&dataset.samples, &train_indexes),
        train_labels: collect_labels(&dataset.labels, &train_indexes),
        test_samples: collect_samples(&dataset.samples, &test_indexes),
        test_labels: collect_labels(&dataset.labels, &test_indexes),
    })
}

fn image_config_from_args(args: &Args) -> ImageVectorConfig {
    image_config_from_genome(genome_from_args(args))
}

fn remap_labels_by_class_name(
    source: &Dataset,
    target_class_names: &[String],
) -> Result<Vec<usize>, Box<dyn Error>> {
    source
        .labels
        .iter()
        .map(|label| {
            let class_name = source
                .class_names
                .get(*label)
                .ok_or_else(|| format!("missing source class name for label {label}"))?;
            target_class_names
                .iter()
                .position(|target| target == class_name)
                .ok_or_else(|| {
                    format!("eval class {class_name:?} does not exist in training classes")
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn collect_samples(samples: &[Vec<f64>], indexes: &[usize]) -> Vec<Vec<f64>> {
    indexes
        .iter()
        .map(|index| samples[*index].clone())
        .collect()
}

fn collect_labels(labels: &[usize], indexes: &[usize]) -> Vec<usize> {
    indexes.iter().map(|index| labels[*index]).collect()
}

struct FolderSplit {
    train_samples: Vec<Vec<f64>>,
    train_labels: Vec<usize>,
    test_samples: Vec<Vec<f64>>,
    test_labels: Vec<usize>,
}
