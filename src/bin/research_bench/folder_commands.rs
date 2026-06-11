use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use progress_ai::panc::{PancComparator, SimilarityMetric};
use progress_ai::pann::{Distributor, IntervalStrategy, PannConfig, PannModel, argmax};
use progress_ai::preprocess::{
    Dataset, min_max_ranges, min_max_scale, one_hot_labels, train_test_split_indices,
};
use progress_ai::vision::{ImageResizeMode, load_image_as_vector, load_image_folder_with_paths};

use super::{
    Args, BenchMetrics, ClassScore, CommandOutput, DebugReference, ImageEvalDebugData,
    ImageEvalPrediction, ResizePrediction, SampleResizeComparison, classification_metrics,
    correction_mode_name, image_config, required_data_path, selected_prediction_indices,
    write_image_eval_debug_report,
};

struct PathSplit {
    train_samples: Vec<Vec<f64>>,
    train_labels: Vec<usize>,
    train_paths: Vec<PathBuf>,
    test_samples: Vec<Vec<f64>>,
    test_labels: Vec<usize>,
    test_paths: Vec<PathBuf>,
}

struct PannTrained {
    model: PannModel,
    train_ms: u128,
    train_accuracy: f64,
}

struct PancTrained {
    comparator: PancComparator<usize>,
    train_ms: u128,
    train_accuracy: f64,
    memory_bytes: usize,
}

struct PredictedClass {
    index: usize,
    score_margin: f64,
    scores: Vec<ClassScore>,
}

pub fn run_pann_image_folder(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    let data_path = required_data_path(args)?;
    let loaded = load_image_folder_with_paths(data_path, image_config(args))?;
    let split = folder_split(&loaded.dataset, loaded.image_paths, data_path, args)?;
    let ranges = min_max_ranges(&split.train_samples);
    let train_samples = min_max_scale(&split.train_samples, &ranges);
    let test_samples = min_max_scale(&split.test_samples, &ranges);
    let targets = one_hot_labels(&split.train_labels, loaded.dataset.class_names.len());

    let trained = train_pann(args, &train_samples, &split.train_labels, &targets)?;

    let inference_start = Instant::now();
    let test_predictions =
        pann_prediction_details(&trained.model, &test_samples, &loaded.dataset.class_names)?;
    let inference_ms = inference_start.elapsed().as_millis();
    let test_prediction_indexes = test_predictions
        .iter()
        .map(|prediction| prediction.index)
        .collect::<Vec<_>>();
    let test_diagnostics = classification_metrics(
        &split.test_labels,
        &test_prediction_indexes,
        &loaded.dataset.class_names,
    );

    if let Some(debug_out_path) = args.debug_out_path.as_deref() {
        let prediction_records = image_eval_predictions(
            &split.test_labels,
            &test_predictions,
            &split.test_paths,
            &loaded.dataset.class_names,
        );
        let selected =
            selected_prediction_indices(&prediction_records, args.debug_samples, args.debug_limit);
        let resize_comparisons = pann_resize_comparisons(
            &selected,
            &split.test_paths,
            &ranges,
            &trained.model,
            &loaded.dataset.class_names,
            args,
        )?;
        let references =
            debug_references(&split, &train_samples, &loaded.dataset.class_names, args)?;
        write_image_eval_debug_report(&ImageEvalDebugData {
            out_path: debug_out_path,
            model: "pann",
            data_path: eval_data_path(args).unwrap_or(data_path),
            model_path: "in-memory:pann-image-folder",
            image_config: image_config(args),
            image_features: args.image_features.as_str(),
            image_resize: args.image_resize.as_str(),
            accuracy: test_diagnostics.accuracy,
            inference_ms,
            memory_bytes: trained.model.memory_bytes_estimate(),
            per_class_accuracy: &test_diagnostics.per_class_accuracy,
            confusion_matrix: &test_diagnostics.confusion_matrix,
            predictions: &prediction_records,
            scaled_samples: &test_samples,
            resize_comparisons: &resize_comparisons,
            references: &references,
            debug_limit: args.debug_limit,
            debug_samples: args.debug_samples,
            debug_neighbors: args.debug_neighbors,
        })?;
    }

    Ok(CommandOutput::Metrics(BenchMetrics {
        model: "pann".to_string(),
        dataset: "image-folder".to_string(),
        image_features: args.image_features.as_str().to_string(),
        image_resize: args.image_resize.as_str().to_string(),
        train_accuracy: trained.train_accuracy,
        test_accuracy: test_diagnostics.accuracy,
        train_ms: trained.train_ms,
        inference_ms,
        memory_bytes: trained.model.memory_bytes_estimate(),
        epochs: args.epochs,
        interval_count: args.intervals,
        distributor: "triangular".to_string(),
        correction_mode: correction_mode_name(args.correction_mode).to_string(),
        per_class_accuracy: test_diagnostics.per_class_accuracy,
        confusion_matrix: test_diagnostics.confusion_matrix,
    }))
}

pub fn run_panc_image_folder(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    let data_path = required_data_path(args)?;
    let loaded = load_image_folder_with_paths(data_path, image_config(args))?;
    let split = folder_split(&loaded.dataset, loaded.image_paths, data_path, args)?;
    let ranges = min_max_ranges(&split.train_samples);
    let train_samples = min_max_scale(&split.train_samples, &ranges);
    let test_samples = min_max_scale(&split.test_samples, &ranges);

    let trained = train_panc(args, &train_samples, &split.train_labels)?;

    let inference_start = Instant::now();
    let test_predictions = panc_prediction_details(
        &trained.comparator,
        &test_samples,
        args.top_k,
        &loaded.dataset.class_names,
    )?;
    let inference_ms = inference_start.elapsed().as_millis();
    let test_prediction_indexes = test_predictions
        .iter()
        .map(|prediction| prediction.index)
        .collect::<Vec<_>>();
    let test_diagnostics = classification_metrics(
        &split.test_labels,
        &test_prediction_indexes,
        &loaded.dataset.class_names,
    );

    if let Some(debug_out_path) = args.debug_out_path.as_deref() {
        let prediction_records = image_eval_predictions(
            &split.test_labels,
            &test_predictions,
            &split.test_paths,
            &loaded.dataset.class_names,
        );
        let selected =
            selected_prediction_indices(&prediction_records, args.debug_samples, args.debug_limit);
        let resize_comparisons = panc_resize_comparisons(
            &selected,
            &split.test_paths,
            &ranges,
            &trained.comparator,
            args.top_k,
            &loaded.dataset.class_names,
            args,
        )?;
        let references =
            debug_references(&split, &train_samples, &loaded.dataset.class_names, args)?;
        write_image_eval_debug_report(&ImageEvalDebugData {
            out_path: debug_out_path,
            model: "panc_like",
            data_path: eval_data_path(args).unwrap_or(data_path),
            model_path: "in-memory:panc-image-folder",
            image_config: image_config(args),
            image_features: args.image_features.as_str(),
            image_resize: args.image_resize.as_str(),
            accuracy: test_diagnostics.accuracy,
            inference_ms,
            memory_bytes: trained.memory_bytes,
            per_class_accuracy: &test_diagnostics.per_class_accuracy,
            confusion_matrix: &test_diagnostics.confusion_matrix,
            predictions: &prediction_records,
            scaled_samples: &test_samples,
            resize_comparisons: &resize_comparisons,
            references: &references,
            debug_limit: args.debug_limit,
            debug_samples: args.debug_samples,
            debug_neighbors: args.debug_neighbors,
        })?;
    }

    Ok(CommandOutput::Metrics(BenchMetrics {
        model: "panc_like".to_string(),
        dataset: "image-folder".to_string(),
        image_features: args.image_features.as_str().to_string(),
        image_resize: args.image_resize.as_str().to_string(),
        train_accuracy: trained.train_accuracy,
        test_accuracy: test_diagnostics.accuracy,
        train_ms: trained.train_ms,
        inference_ms,
        memory_bytes: trained.memory_bytes,
        epochs: 0,
        interval_count: 0,
        distributor: "none".to_string(),
        correction_mode: "top_k_euclidean_vote".to_string(),
        per_class_accuracy: test_diagnostics.per_class_accuracy,
        confusion_matrix: test_diagnostics.confusion_matrix,
    }))
}

fn train_pann(
    args: &Args,
    train_samples: &[Vec<f64>],
    train_labels: &[usize],
    targets: &[Vec<f64>],
) -> Result<PannTrained, Box<dyn Error>> {
    let output_count = targets.first().map(Vec::len).unwrap_or(0);
    let mut config = PannConfig::new(train_samples[0].len(), args.intervals, output_count);
    config.distributor = Distributor::Triangular;
    config.interval_strategy = IntervalStrategy::Uniform;
    config.correction_mode = args.correction_mode;
    let mut model = PannModel::from_training_data_with_config(train_samples, config)?;

    let train_start = Instant::now();
    for _ in 0..args.epochs {
        model.train_epoch(train_samples, targets)?;
    }
    let train_ms = train_start.elapsed().as_millis();
    let train_predictions = pann_prediction_indexes(&model, train_samples)?;
    let train_diagnostics = classification_metrics(train_labels, &train_predictions, &Vec::new());

    Ok(PannTrained {
        model,
        train_ms,
        train_accuracy: train_diagnostics.accuracy,
    })
}

fn train_panc(
    args: &Args,
    train_samples: &[Vec<f64>],
    train_labels: &[usize],
) -> Result<PancTrained, Box<dyn Error>> {
    let train_start = Instant::now();
    let mut comparator = PancComparator::new(SimilarityMetric::Euclidean);
    for (sample, label) in train_samples.iter().zip(train_labels) {
        comparator.add_reference(sample.clone(), *label, ())?;
    }
    let train_ms = train_start.elapsed().as_millis();
    let train_predictions = panc_prediction_indexes(&comparator, train_samples, args.top_k)?;
    let train_diagnostics = classification_metrics(train_labels, &train_predictions, &Vec::new());
    let memory_bytes = train_samples.len() * train_samples[0].len() * std::mem::size_of::<f64>();

    Ok(PancTrained {
        comparator,
        train_ms,
        train_accuracy: train_diagnostics.accuracy,
        memory_bytes,
    })
}

fn folder_split(
    dataset: &Dataset,
    paths: Vec<PathBuf>,
    data_path: &str,
    args: &Args,
) -> Result<PathSplit, Box<dyn Error>> {
    if let Some(eval_path) = eval_data_path(args) {
        let eval_dataset = load_image_folder_with_paths(eval_path, image_config(args))?;
        let eval_labels = remap_labels_by_class_name(&eval_dataset.dataset, &dataset.class_names)?;
        let (_, train_indexes) = train_test_split_indices(dataset.samples.len(), 0.0, args.seed);
        return Ok(PathSplit {
            train_samples: collect_samples(&dataset.samples, &train_indexes),
            train_labels: collect_labels(&dataset.labels, &train_indexes),
            train_paths: collect_paths(&paths, &train_indexes),
            test_samples: eval_dataset.dataset.samples,
            test_labels: eval_labels,
            test_paths: eval_dataset.image_paths,
        });
    }

    let _ = data_path;
    let (test_indexes, train_indexes) =
        train_test_split_indices(dataset.samples.len(), 0.2, args.seed);
    Ok(PathSplit {
        train_samples: collect_samples(&dataset.samples, &train_indexes),
        train_labels: collect_labels(&dataset.labels, &train_indexes),
        train_paths: collect_paths(&paths, &train_indexes),
        test_samples: collect_samples(&dataset.samples, &test_indexes),
        test_labels: collect_labels(&dataset.labels, &test_indexes),
        test_paths: collect_paths(&paths, &test_indexes),
    })
}

fn debug_references(
    split: &PathSplit,
    train_samples: &[Vec<f64>],
    class_names: &[String],
    args: &Args,
) -> Result<Vec<DebugReference>, Box<dyn Error>> {
    let Some(debug_train_path) = args.debug_train_data_path.as_deref() else {
        return Ok(train_references_from_split(
            split,
            train_samples,
            class_names,
        ));
    };

    let debug_dataset = load_image_folder_with_paths(debug_train_path, image_config(args))?;
    let labels = remap_labels_by_class_name(&debug_dataset.dataset, class_names)?;
    let ranges = min_max_ranges(&split.train_samples);
    let samples = min_max_scale(&debug_dataset.dataset.samples, &ranges);
    Ok(samples
        .into_iter()
        .zip(labels)
        .zip(debug_dataset.image_paths)
        .map(|((scaled_sample, label_index), path)| DebugReference {
            path: path.display().to_string(),
            label_index,
            label: label_name(class_names, label_index),
            scaled_sample,
        })
        .collect())
}

fn train_references_from_split(
    split: &PathSplit,
    train_samples: &[Vec<f64>],
    class_names: &[String],
) -> Vec<DebugReference> {
    train_samples
        .iter()
        .cloned()
        .zip(&split.train_labels)
        .zip(&split.train_paths)
        .map(|((scaled_sample, label_index), path)| DebugReference {
            path: path.display().to_string(),
            label_index: *label_index,
            label: label_name(class_names, *label_index),
            scaled_sample,
        })
        .collect()
}

fn pann_resize_comparisons(
    selected_prediction_indices: &[usize],
    test_paths: &[PathBuf],
    ranges: &[progress_ai::pann::FeatureRange],
    model: &PannModel,
    class_names: &[String],
    args: &Args,
) -> Result<Vec<SampleResizeComparison>, Box<dyn Error>> {
    selected_prediction_indices
        .iter()
        .copied()
        .map(|index| {
            let path = test_paths
                .get(index)
                .ok_or_else(|| format!("missing test path for sample {index}"))?;
            let mut predictions = Vec::new();
            for resize_mode in all_resize_modes() {
                let sample = load_scaled_image_with_resize(path, ranges, resize_mode, args)?;
                let outputs = model.forward(&sample)?;
                let predicted_index = argmax(&outputs);
                predictions.push(ResizePrediction {
                    resize_mode: resize_mode.as_str().to_string(),
                    predicted_index,
                    predicted_label: label_name(class_names, predicted_index),
                    score_margin: score_margin(&outputs),
                    differs_from_artifact_prediction: false,
                    scores: class_scores(&outputs, class_names),
                });
            }
            mark_resize_differences(&mut predictions, args.image_resize);
            Ok(SampleResizeComparison {
                sample_index: index,
                predictions,
            })
        })
        .collect()
}

fn panc_resize_comparisons(
    selected_prediction_indices: &[usize],
    test_paths: &[PathBuf],
    ranges: &[progress_ai::pann::FeatureRange],
    comparator: &PancComparator<usize>,
    top_k: usize,
    class_names: &[String],
    args: &Args,
) -> Result<Vec<SampleResizeComparison>, Box<dyn Error>> {
    selected_prediction_indices
        .iter()
        .copied()
        .map(|index| {
            let path = test_paths
                .get(index)
                .ok_or_else(|| format!("missing test path for sample {index}"))?;
            let mut predictions = Vec::new();
            for resize_mode in all_resize_modes() {
                let sample = load_scaled_image_with_resize(path, ranges, resize_mode, args)?;
                let predicted = panc_predict_with_margin(comparator, &sample, top_k, class_names)?;
                predictions.push(ResizePrediction {
                    resize_mode: resize_mode.as_str().to_string(),
                    predicted_index: predicted.index,
                    predicted_label: label_name(class_names, predicted.index),
                    score_margin: predicted.score_margin,
                    differs_from_artifact_prediction: false,
                    scores: predicted.scores,
                });
            }
            mark_resize_differences(&mut predictions, args.image_resize);
            Ok(SampleResizeComparison {
                sample_index: index,
                predictions,
            })
        })
        .collect()
}

fn load_scaled_image_with_resize(
    path: &Path,
    ranges: &[progress_ai::pann::FeatureRange],
    resize_mode: ImageResizeMode,
    args: &Args,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let sample = load_image_as_vector(path, image_config(args).with_resize_mode(resize_mode))?;
    let mut scaled = min_max_scale(&[sample], ranges);
    Ok(scaled.remove(0))
}

fn image_eval_predictions(
    labels: &[usize],
    predictions: &[PredictedClass],
    paths: &[PathBuf],
    class_names: &[String],
) -> Vec<ImageEvalPrediction> {
    labels
        .iter()
        .zip(predictions)
        .enumerate()
        .map(|(sample_index, (actual, prediction))| ImageEvalPrediction {
            sample_index,
            path: paths
                .get(sample_index)
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| format!("sample-{sample_index}")),
            expected_index: *actual,
            expected_label: label_name(class_names, *actual),
            predicted_index: prediction.index,
            predicted_label: label_name(class_names, prediction.index),
            correct: *actual == prediction.index,
            score_margin: prediction.score_margin,
            scores: prediction.scores.clone(),
        })
        .collect()
}

fn pann_prediction_details(
    model: &PannModel,
    samples: &[Vec<f64>],
    class_names: &[String],
) -> Result<Vec<PredictedClass>, Box<dyn Error>> {
    samples
        .iter()
        .map(|sample| {
            let outputs = model.forward(sample)?;
            Ok(PredictedClass {
                index: argmax(&outputs),
                score_margin: score_margin(&outputs),
                scores: class_scores(&outputs, class_names),
            })
        })
        .collect()
}

fn pann_prediction_indexes(
    model: &PannModel,
    samples: &[Vec<f64>],
) -> Result<Vec<usize>, Box<dyn Error>> {
    samples
        .iter()
        .map(|sample| model.predict(sample).map_err(Into::into))
        .collect()
}

fn panc_prediction_details(
    comparator: &PancComparator<usize>,
    samples: &[Vec<f64>],
    top_k: usize,
    class_names: &[String],
) -> Result<Vec<PredictedClass>, Box<dyn Error>> {
    samples
        .iter()
        .map(|sample| panc_predict_with_margin(comparator, sample, top_k, class_names))
        .collect()
}

fn panc_prediction_indexes(
    comparator: &PancComparator<usize>,
    samples: &[Vec<f64>],
    top_k: usize,
) -> Result<Vec<usize>, Box<dyn Error>> {
    samples
        .iter()
        .map(|sample| {
            comparator
                .predict_label(sample, top_k.max(1))?
                .ok_or_else(|| "PANC comparator returned no prediction".into())
        })
        .collect()
}

fn panc_predict_with_margin(
    comparator: &PancComparator<usize>,
    sample: &[f64],
    top_k: usize,
    class_names: &[String],
) -> Result<PredictedClass, Box<dyn Error>> {
    let neighbors = comparator.top_k(sample, top_k.max(1))?;
    let mut scores = vec![0.0; class_names.len()];
    for neighbor in neighbors {
        if let Some(score) = scores.get_mut(neighbor.label) {
            *score += neighbor.score;
        }
    }
    Ok(PredictedClass {
        index: argmax(&scores),
        score_margin: score_margin(&scores),
        scores: class_scores(&scores, class_names),
    })
}

fn score_margin(scores: &[f64]) -> f64 {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    sorted.first().copied().unwrap_or(0.0) - sorted.get(1).copied().unwrap_or(0.0)
}

fn class_scores(outputs: &[f64], class_names: &[String]) -> Vec<ClassScore> {
    let mut scores = outputs
        .iter()
        .copied()
        .enumerate()
        .map(|(class_index, score)| ClassScore {
            class_index,
            class_name: label_name(class_names, class_index),
            score,
        })
        .collect::<Vec<_>>();
    scores.sort_by(|left, right| right.score.total_cmp(&left.score));
    scores
}

fn mark_resize_differences(predictions: &mut [ResizePrediction], configured_mode: ImageResizeMode) {
    let configured_prediction = predictions
        .iter()
        .find(|prediction| prediction.resize_mode == configured_mode.as_str())
        .map(|prediction| prediction.predicted_index)
        .unwrap_or_else(|| {
            predictions
                .first()
                .map(|prediction| prediction.predicted_index)
                .unwrap_or(0)
        });
    for prediction in predictions {
        prediction.differs_from_artifact_prediction =
            prediction.predicted_index != configured_prediction;
    }
}

fn all_resize_modes() -> [ImageResizeMode; 3] {
    [
        ImageResizeMode::Stretch,
        ImageResizeMode::CenterCrop,
        ImageResizeMode::Letterbox,
    ]
}

fn remap_labels_by_class_name(
    source: &Dataset,
    target_class_names: &[String],
) -> Result<Vec<usize>, Box<dyn Error>> {
    let target_labels = target_class_names
        .iter()
        .enumerate()
        .map(|(label, name)| (name.as_str(), label))
        .collect::<std::collections::HashMap<_, _>>();

    source
        .labels
        .iter()
        .map(|label| {
            let class_name = source
                .class_names
                .get(*label)
                .ok_or_else(|| format!("missing source class name for label {label}"))?;
            target_labels
                .get(class_name.as_str())
                .copied()
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

fn collect_paths(paths: &[PathBuf], indexes: &[usize]) -> Vec<PathBuf> {
    indexes.iter().map(|index| paths[*index].clone()).collect()
}

fn eval_data_path(args: &Args) -> Option<&str> {
    args.eval_data_path.as_deref()
}

fn label_name(class_names: &[String], label: usize) -> String {
    class_names
        .get(label)
        .cloned()
        .unwrap_or_else(|| label.to_string())
}
