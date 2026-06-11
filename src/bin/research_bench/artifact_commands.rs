use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use progress_ai::panc::{PancComparator, SimilarityMetric};
use progress_ai::pann::{
    CorrectionMode, Distributor, IntervalStrategy, PannConfig, PannModel, argmax,
};
use progress_ai::preprocess::{
    Dataset, min_max_ranges, min_max_scale, one_hot_labels, train_test_split,
};
use progress_ai::vision::{
    ImageResizeMode, load_image_as_vector, load_image_folder, load_image_folder_with_paths,
};

use super::artifacts::{
    ARTIFACT_VERSION, ImageArtifact, ModelArtifact, PancImageArtifact, PancReferenceArtifact,
    PannImageArtifact, PreprocessingArtifact, load_artifact, save_artifact,
};
use super::{
    Args, ArtifactMetrics, ClassScore, CommandOutput, ConfusionRow, DebugReference, EvalMetrics,
    ImageEvalDebugData, ImageEvalPrediction, MisclassifiedExample, PerClassAccuracy,
    PredictionNeighbor, PredictionOutput, ResizePrediction, SampleResizeComparison, image_config,
    required_data_path, required_image_path, required_model_path, required_out_path,
    selected_prediction_indices, write_image_eval_debug_report,
};

const MAX_MISCLASSIFIED_EXAMPLES: usize = 25;

struct ScaledEvalDataset {
    samples: Vec<Vec<f64>>,
    labels: Vec<usize>,
    paths: Vec<PathBuf>,
}

struct PredictedClass {
    index: usize,
    score_margin: f64,
    scores: Vec<ClassScore>,
}

struct ClassificationDiagnostics {
    accuracy: f64,
    per_class_accuracy: Vec<PerClassAccuracy>,
    confusion_matrix: Vec<ConfusionRow>,
    misclassified_examples: Vec<MisclassifiedExample>,
}

pub fn train_pann_image_folder(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    let data_path = required_data_path(args)?;
    let out_path = required_out_path(args)?;
    let dataset = load_image_folder(data_path, image_config(args))?;
    let split = train_test_split(&dataset.samples, &dataset.labels, 0.0, args.seed);
    let ranges = min_max_ranges(&split.train_samples);
    let train_samples = min_max_scale(&split.train_samples, &ranges);
    let targets = one_hot_labels(&split.train_labels, dataset.class_names.len());

    let mut config = PannConfig::new(
        train_samples[0].len(),
        args.intervals,
        dataset.class_names.len(),
    );
    config.distributor = Distributor::Triangular;
    config.interval_strategy = IntervalStrategy::Uniform;
    config.correction_mode = CorrectionMode::DifferenceLeastSquares;
    let mut model = PannModel::from_training_data_with_config(&train_samples, config)?;

    let train_start = Instant::now();
    for _ in 0..args.epochs {
        model.train_epoch(&train_samples, &targets)?;
    }
    let train_ms = train_start.elapsed().as_millis();
    let train_accuracy = model.accuracy(&train_samples, &split.train_labels)?;

    let artifact = ModelArtifact::PannImage(PannImageArtifact {
        version: ARTIFACT_VERSION,
        class_names: dataset.class_names,
        image: ImageArtifact::from_config(image_config(args)),
        preprocessing: PreprocessingArtifact {
            min_max_ranges: ranges,
        },
        model: model.snapshot(),
        epochs_trained: args.epochs,
    });
    save_artifact(out_path, &artifact)?;

    Ok(CommandOutput::Artifact(ArtifactMetrics {
        model: "pann".to_string(),
        dataset: "image-folder".to_string(),
        image_features: args.image_features.as_str().to_string(),
        image_resize: args.image_resize.as_str().to_string(),
        artifact_path: out_path.to_string(),
        train_accuracy,
        train_ms,
        memory_bytes: model.memory_bytes_estimate(),
        epochs: args.epochs,
        interval_count: args.intervals,
        reference_count: 0,
    }))
}

pub fn train_panc_image_folder(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    let data_path = required_data_path(args)?;
    let out_path = required_out_path(args)?;
    let dataset = load_image_folder(data_path, image_config(args))?;
    let split = train_test_split(&dataset.samples, &dataset.labels, 0.0, args.seed);
    let ranges = min_max_ranges(&split.train_samples);
    let train_samples = min_max_scale(&split.train_samples, &ranges);

    let train_start = Instant::now();
    let comparator = build_panc_comparator_from_samples(&train_samples, &split.train_labels)?;
    let train_ms = train_start.elapsed().as_millis();
    let train_accuracy =
        panc_accuracy(&comparator, &train_samples, &split.train_labels, args.top_k)?;

    let references = train_samples
        .iter()
        .zip(&split.train_labels)
        .map(|(vector, label)| PancReferenceArtifact {
            vector: vector.clone(),
            label: *label,
        })
        .collect::<Vec<_>>();
    let memory_bytes = panc_memory_bytes(&references);
    let reference_count = references.len();
    let artifact = ModelArtifact::PancImage(PancImageArtifact {
        version: ARTIFACT_VERSION,
        class_names: dataset.class_names,
        image: ImageArtifact::from_config(image_config(args)),
        preprocessing: PreprocessingArtifact {
            min_max_ranges: ranges,
        },
        metric: SimilarityMetric::Euclidean,
        references,
    });
    save_artifact(out_path, &artifact)?;

    Ok(CommandOutput::Artifact(ArtifactMetrics {
        model: "panc_like".to_string(),
        dataset: "image-folder".to_string(),
        image_features: args.image_features.as_str().to_string(),
        image_resize: args.image_resize.as_str().to_string(),
        artifact_path: out_path.to_string(),
        train_accuracy,
        train_ms,
        memory_bytes,
        epochs: 0,
        interval_count: 0,
        reference_count,
    }))
}

pub fn eval_pann(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    let model_path = required_model_path(args)?;
    let data_path = required_data_path(args)?;
    let ModelArtifact::PannImage(artifact) = load_artifact(model_path)? else {
        return Err("artifact is not a PANN image artifact".into());
    };
    validate_version(artifact.version)?;

    let model = PannModel::from_snapshot(artifact.model.clone())?;
    let eval_dataset = load_scaled_eval_dataset(data_path, &artifact)?;

    let inference_start = Instant::now();
    let predictions = eval_dataset
        .samples
        .iter()
        .map(|sample| {
            let outputs = model.forward(sample)?;
            Ok(PredictedClass {
                index: argmax(&outputs),
                score_margin: score_margin(&outputs),
                scores: class_scores(&outputs, &artifact.class_names),
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let inference_ms = inference_start.elapsed().as_millis();
    let prediction_records = image_eval_predictions(
        &eval_dataset.labels,
        &predictions,
        &eval_dataset.paths,
        &artifact.class_names,
    );
    let diagnostics = classification_diagnostics(&prediction_records, &artifact.class_names);
    if let Some(debug_out_path) = args.debug_out_path.as_deref() {
        let selected =
            selected_prediction_indices(&prediction_records, args.debug_samples, args.debug_limit);
        let resize_comparisons =
            pann_resize_comparisons(&selected, &eval_dataset, &artifact, &model)?;
        let debug_references = load_debug_references(args, data_path, &artifact)?;
        write_image_eval_debug_report(&ImageEvalDebugData {
            out_path: debug_out_path,
            model: "pann",
            data_path,
            model_path,
            image_config: artifact.image.to_config()?,
            image_features: &artifact.image.feature_mode,
            image_resize: &artifact.image.resize_mode,
            accuracy: diagnostics.accuracy,
            inference_ms,
            memory_bytes: model.memory_bytes_estimate(),
            per_class_accuracy: &diagnostics.per_class_accuracy,
            confusion_matrix: &diagnostics.confusion_matrix,
            predictions: &prediction_records,
            scaled_samples: &eval_dataset.samples,
            resize_comparisons: &resize_comparisons,
            references: &debug_references,
            debug_limit: args.debug_limit,
            debug_samples: args.debug_samples,
            debug_neighbors: args.debug_neighbors,
        })?;
    }

    Ok(CommandOutput::Eval(EvalMetrics {
        model: "pann".to_string(),
        dataset: "image-folder".to_string(),
        image_features: artifact.image.feature_mode,
        image_resize: artifact.image.resize_mode,
        model_path: model_path.to_string(),
        accuracy: diagnostics.accuracy,
        inference_ms,
        memory_bytes: model.memory_bytes_estimate(),
        sample_count: eval_dataset.samples.len(),
        per_class_accuracy: diagnostics.per_class_accuracy,
        confusion_matrix: diagnostics.confusion_matrix,
        misclassified_examples: diagnostics.misclassified_examples,
    }))
}

pub fn eval_panc(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    let model_path = required_model_path(args)?;
    let data_path = required_data_path(args)?;
    let ModelArtifact::PancImage(artifact) = load_artifact(model_path)? else {
        return Err("artifact is not a PANC image artifact".into());
    };
    validate_version(artifact.version)?;

    let eval_dataset = load_scaled_eval_dataset(data_path, &artifact)?;
    let comparator = build_panc_comparator_from_artifact(&artifact)?;

    let inference_start = Instant::now();
    let predictions = eval_dataset
        .samples
        .iter()
        .map(|sample| {
            panc_predict_with_margin(&comparator, sample, args.top_k, &artifact.class_names)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let inference_ms = inference_start.elapsed().as_millis();
    let memory_bytes = panc_memory_bytes(&artifact.references);
    let prediction_records = image_eval_predictions(
        &eval_dataset.labels,
        &predictions,
        &eval_dataset.paths,
        &artifact.class_names,
    );
    let diagnostics = classification_diagnostics(&prediction_records, &artifact.class_names);
    if let Some(debug_out_path) = args.debug_out_path.as_deref() {
        let selected =
            selected_prediction_indices(&prediction_records, args.debug_samples, args.debug_limit);
        let resize_comparisons =
            panc_resize_comparisons(&selected, &eval_dataset, &artifact, &comparator, args.top_k)?;
        let debug_references = load_debug_references(args, data_path, &artifact)?;
        write_image_eval_debug_report(&ImageEvalDebugData {
            out_path: debug_out_path,
            model: "panc_like",
            data_path,
            model_path,
            image_config: artifact.image.to_config()?,
            image_features: &artifact.image.feature_mode,
            image_resize: &artifact.image.resize_mode,
            accuracy: diagnostics.accuracy,
            inference_ms,
            memory_bytes,
            per_class_accuracy: &diagnostics.per_class_accuracy,
            confusion_matrix: &diagnostics.confusion_matrix,
            predictions: &prediction_records,
            scaled_samples: &eval_dataset.samples,
            resize_comparisons: &resize_comparisons,
            references: &debug_references,
            debug_limit: args.debug_limit,
            debug_samples: args.debug_samples,
            debug_neighbors: args.debug_neighbors,
        })?;
    }

    Ok(CommandOutput::Eval(EvalMetrics {
        model: "panc_like".to_string(),
        dataset: "image-folder".to_string(),
        image_features: artifact.image.feature_mode,
        image_resize: artifact.image.resize_mode,
        model_path: model_path.to_string(),
        accuracy: diagnostics.accuracy,
        inference_ms,
        memory_bytes,
        sample_count: eval_dataset.samples.len(),
        per_class_accuracy: diagnostics.per_class_accuracy,
        confusion_matrix: diagnostics.confusion_matrix,
        misclassified_examples: diagnostics.misclassified_examples,
    }))
}

pub fn predict_pann(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    let model_path = required_model_path(args)?;
    let image_path = required_image_path(args)?;
    let ModelArtifact::PannImage(artifact) = load_artifact(model_path)? else {
        return Err("artifact is not a PANN image artifact".into());
    };
    validate_version(artifact.version)?;

    let model = PannModel::from_snapshot(artifact.model.clone())?;
    let sample = load_scaled_image(image_path, &artifact.image, &artifact.preprocessing)?;
    let outputs = model.forward(&sample)?;
    let predicted_index = argmax(&outputs);
    let scores = class_scores(&outputs, &artifact.class_names);

    Ok(CommandOutput::Prediction(PredictionOutput {
        model: "pann".to_string(),
        image: image_path.to_string(),
        predicted_index,
        predicted_label: label_name(&artifact.class_names, predicted_index),
        scores,
        neighbors: Vec::new(),
    }))
}

pub fn predict_panc(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    let model_path = required_model_path(args)?;
    let image_path = required_image_path(args)?;
    let ModelArtifact::PancImage(artifact) = load_artifact(model_path)? else {
        return Err("artifact is not a PANC image artifact".into());
    };
    validate_version(artifact.version)?;

    let sample = load_scaled_image(image_path, &artifact.image, &artifact.preprocessing)?;
    let comparator = build_panc_comparator_from_artifact(&artifact)?;
    let neighbors = comparator.top_k(&sample, args.top_k)?;
    let predicted_index = comparator.predict_label(&sample, args.top_k)?.unwrap_or(0);
    let scores = panc_class_scores(&neighbors, &artifact.class_names);
    let neighbors = neighbors
        .into_iter()
        .map(|neighbor| PredictionNeighbor {
            index: neighbor.index,
            class_index: neighbor.label,
            class_name: label_name(&artifact.class_names, neighbor.label),
            score: neighbor.score,
        })
        .collect();

    Ok(CommandOutput::Prediction(PredictionOutput {
        model: "panc_like".to_string(),
        image: image_path.to_string(),
        predicted_index,
        predicted_label: label_name(&artifact.class_names, predicted_index),
        scores,
        neighbors,
    }))
}

fn load_scaled_eval_dataset(
    data_path: &str,
    artifact: &impl ImageClassifierArtifact,
) -> Result<ScaledEvalDataset, Box<dyn Error>> {
    let image_dataset = load_image_folder_with_paths(data_path, artifact.image().to_config()?)?;
    let labels = remap_labels_by_class_name(&image_dataset.dataset, artifact.class_names())?;
    let samples = min_max_scale(
        &image_dataset.dataset.samples,
        &artifact.preprocessing().min_max_ranges,
    );
    Ok(ScaledEvalDataset {
        samples,
        labels,
        paths: image_dataset.image_paths,
    })
}

fn load_debug_references(
    args: &Args,
    eval_data_path: &str,
    artifact: &impl ImageClassifierArtifact,
) -> Result<Vec<DebugReference>, Box<dyn Error>> {
    let Some(train_path) = debug_train_data_path(args, eval_data_path) else {
        return Ok(Vec::new());
    };
    let dataset = match load_image_folder_with_paths(&train_path, artifact.image().to_config()?) {
        Ok(dataset) => dataset,
        Err(error) if args.debug_train_data_path.is_none() => {
            eprintln!("warning: skipped inferred debug train data {train_path}: {error}");
            return Ok(Vec::new());
        }
        Err(error) => return Err(error.into()),
    };
    let labels = remap_labels_by_class_name(&dataset.dataset, artifact.class_names())?;
    let samples = min_max_scale(
        &dataset.dataset.samples,
        &artifact.preprocessing().min_max_ranges,
    );

    Ok(samples
        .into_iter()
        .zip(labels)
        .zip(dataset.image_paths)
        .map(|((scaled_sample, label_index), path)| DebugReference {
            path: path.display().to_string(),
            label_index,
            label: label_name(artifact.class_names(), label_index),
            scaled_sample,
        })
        .collect())
}

fn debug_train_data_path(args: &Args, eval_data_path: &str) -> Option<String> {
    if let Some(path) = &args.debug_train_data_path {
        return Some(path.clone());
    }

    let eval_path = Path::new(eval_data_path);
    let folder_name = eval_path.file_name()?.to_string_lossy();
    if folder_name.eq_ignore_ascii_case("eval") {
        eval_path.parent().map(|path| path.display().to_string())
    } else {
        None
    }
}

fn load_scaled_image(
    image_path: &str,
    image: &ImageArtifact,
    preprocessing: &PreprocessingArtifact,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let sample = load_image_as_vector(image_path, image.to_config()?)?;
    let mut scaled = min_max_scale(&[sample], &preprocessing.min_max_ranges);
    Ok(scaled.remove(0))
}

fn pann_resize_comparisons(
    selected_prediction_indices: &[usize],
    eval_dataset: &ScaledEvalDataset,
    artifact: &PannImageArtifact,
    model: &PannModel,
) -> Result<Vec<SampleResizeComparison>, Box<dyn Error>> {
    selected_prediction_indices
        .iter()
        .copied()
        .map(|index| {
            let path = eval_dataset
                .paths
                .get(index)
                .ok_or_else(|| format!("missing eval path for sample {index}"))?;
            let mut predictions = Vec::new();
            for resize_mode in all_resize_modes() {
                let sample = load_scaled_image_with_resize(
                    path,
                    &artifact.image,
                    &artifact.preprocessing,
                    resize_mode,
                )?;
                let outputs = model.forward(&sample)?;
                let predicted_index = argmax(&outputs);
                predictions.push(ResizePrediction {
                    resize_mode: resize_mode.as_str().to_string(),
                    predicted_index,
                    predicted_label: label_name(&artifact.class_names, predicted_index),
                    score_margin: score_margin(&outputs),
                    differs_from_artifact_prediction: false,
                    scores: class_scores(&outputs, &artifact.class_names),
                });
            }
            mark_resize_differences(&mut predictions, artifact.image.to_config()?.resize_mode);
            Ok(SampleResizeComparison {
                sample_index: index,
                predictions,
            })
        })
        .collect()
}

fn panc_resize_comparisons(
    selected_prediction_indices: &[usize],
    eval_dataset: &ScaledEvalDataset,
    artifact: &PancImageArtifact,
    comparator: &PancComparator<usize>,
    top_k: usize,
) -> Result<Vec<SampleResizeComparison>, Box<dyn Error>> {
    selected_prediction_indices
        .iter()
        .copied()
        .map(|index| {
            let path = eval_dataset
                .paths
                .get(index)
                .ok_or_else(|| format!("missing eval path for sample {index}"))?;
            let mut predictions = Vec::new();
            for resize_mode in all_resize_modes() {
                let sample = load_scaled_image_with_resize(
                    path,
                    &artifact.image,
                    &artifact.preprocessing,
                    resize_mode,
                )?;
                let predicted =
                    panc_predict_with_margin(comparator, &sample, top_k, &artifact.class_names)?;
                predictions.push(ResizePrediction {
                    resize_mode: resize_mode.as_str().to_string(),
                    predicted_index: predicted.index,
                    predicted_label: label_name(&artifact.class_names, predicted.index),
                    score_margin: predicted.score_margin,
                    differs_from_artifact_prediction: false,
                    scores: predicted.scores,
                });
            }
            mark_resize_differences(&mut predictions, artifact.image.to_config()?.resize_mode);
            Ok(SampleResizeComparison {
                sample_index: index,
                predictions,
            })
        })
        .collect()
}

fn load_scaled_image_with_resize(
    image_path: &Path,
    image: &ImageArtifact,
    preprocessing: &PreprocessingArtifact,
    resize_mode: ImageResizeMode,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let config = image.to_config()?.with_resize_mode(resize_mode);
    let sample = load_image_as_vector(image_path, config)?;
    let mut scaled = min_max_scale(&[sample], &preprocessing.min_max_ranges);
    Ok(scaled.remove(0))
}

fn mark_resize_differences(predictions: &mut [ResizePrediction], artifact_mode: ImageResizeMode) {
    let artifact_prediction = predictions
        .iter()
        .find(|prediction| prediction.resize_mode == artifact_mode.as_str())
        .map(|prediction| prediction.predicted_index)
        .unwrap_or_else(|| {
            predictions
                .first()
                .map(|prediction| prediction.predicted_index)
                .unwrap_or(0)
        });
    for prediction in predictions {
        prediction.differs_from_artifact_prediction =
            prediction.predicted_index != artifact_prediction;
    }
}

fn all_resize_modes() -> [ImageResizeMode; 3] {
    [
        ImageResizeMode::Stretch,
        ImageResizeMode::CenterCrop,
        ImageResizeMode::Letterbox,
    ]
}

fn build_panc_comparator_from_samples(
    samples: &[Vec<f64>],
    labels: &[usize],
) -> Result<PancComparator<usize>, Box<dyn Error>> {
    let mut comparator = PancComparator::new(SimilarityMetric::Euclidean);
    for (sample, label) in samples.iter().zip(labels) {
        comparator.add_reference(sample.clone(), *label, ())?;
    }
    Ok(comparator)
}

fn build_panc_comparator_from_artifact(
    artifact: &PancImageArtifact,
) -> Result<PancComparator<usize>, Box<dyn Error>> {
    let mut comparator = PancComparator::new(artifact.metric);
    for reference in &artifact.references {
        comparator.add_reference(reference.vector.clone(), reference.label, ())?;
    }
    Ok(comparator)
}

fn remap_labels_by_class_name(
    source: &Dataset,
    target_class_names: &[String],
) -> Result<Vec<usize>, Box<dyn Error>> {
    let target_labels = target_class_names
        .iter()
        .enumerate()
        .map(|(label, name)| (name.as_str(), label))
        .collect::<HashMap<_, _>>();

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
        if comparator.predict_label(sample, k.max(1))? == Some(*label) {
            correct += 1;
        }
    }
    Ok(correct as f64 / samples.len() as f64)
}

fn panc_predict_with_margin(
    comparator: &PancComparator<usize>,
    sample: &[f64],
    k: usize,
    class_names: &[String],
) -> Result<PredictedClass, Box<dyn Error>> {
    let neighbors = comparator.top_k(sample, k.max(1))?;
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

fn classification_diagnostics(
    predictions: &[ImageEvalPrediction],
    class_names: &[String],
) -> ClassificationDiagnostics {
    let class_count = class_names.len();
    let mut confusion = vec![vec![0usize; class_count]; class_count];
    let mut totals = vec![0usize; class_count];
    let mut correct_by_class = vec![0usize; class_count];
    let mut correct = 0usize;
    let mut misclassified_examples = Vec::new();

    for prediction in predictions {
        if let Some(total) = totals.get_mut(prediction.expected_index) {
            *total += 1;
        }
        if let Some(row) = confusion.get_mut(prediction.expected_index)
            && let Some(cell) = row.get_mut(prediction.predicted_index)
        {
            *cell += 1;
        }

        if prediction.correct {
            correct += 1;
            if let Some(class_correct) = correct_by_class.get_mut(prediction.expected_index) {
                *class_correct += 1;
            }
        } else if misclassified_examples.len() < MAX_MISCLASSIFIED_EXAMPLES {
            misclassified_examples.push(MisclassifiedExample {
                path: prediction.path.clone(),
                expected_index: prediction.expected_index,
                expected_label: prediction.expected_label.clone(),
                predicted_index: prediction.predicted_index,
                predicted_label: prediction.predicted_label.clone(),
                score_margin: prediction.score_margin,
            });
        }
    }

    let per_class_accuracy = class_names
        .iter()
        .enumerate()
        .map(|(class_index, class_name)| {
            let total = totals[class_index];
            let class_correct = correct_by_class[class_index];
            PerClassAccuracy {
                class_index,
                class_name: class_name.clone(),
                correct: class_correct,
                total,
                accuracy: if total == 0 {
                    0.0
                } else {
                    class_correct as f64 / total as f64
                },
            }
        })
        .collect::<Vec<_>>();

    let confusion_matrix = confusion
        .into_iter()
        .enumerate()
        .map(|(actual_index, predicted_counts)| ConfusionRow {
            actual_index,
            actual_name: label_name(class_names, actual_index),
            predicted_counts,
        })
        .collect::<Vec<_>>();

    ClassificationDiagnostics {
        accuracy: if predictions.is_empty() {
            0.0
        } else {
            correct as f64 / predictions.len() as f64
        },
        per_class_accuracy,
        confusion_matrix,
        misclassified_examples,
    }
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

fn panc_class_scores(
    neighbors: &[progress_ai::panc::Neighbor<usize>],
    class_names: &[String],
) -> Vec<ClassScore> {
    let mut scores_by_class = vec![0.0; class_names.len()];
    for neighbor in neighbors {
        if let Some(score) = scores_by_class.get_mut(neighbor.label) {
            *score += neighbor.score;
        }
    }
    class_scores(&scores_by_class, class_names)
}

fn label_name(class_names: &[String], label: usize) -> String {
    class_names
        .get(label)
        .cloned()
        .unwrap_or_else(|| format!("class-{label}"))
}

fn panc_memory_bytes(references: &[PancReferenceArtifact]) -> usize {
    references
        .first()
        .map(|reference| references.len() * reference.vector.len() * std::mem::size_of::<f64>())
        .unwrap_or(0)
}

fn validate_version(version: u32) -> Result<(), Box<dyn Error>> {
    if version == ARTIFACT_VERSION {
        Ok(())
    } else {
        Err(format!("unsupported artifact version {version}").into())
    }
}

trait ImageClassifierArtifact {
    fn class_names(&self) -> &[String];
    fn image(&self) -> &ImageArtifact;
    fn preprocessing(&self) -> &PreprocessingArtifact;
}

impl ImageClassifierArtifact for PannImageArtifact {
    fn class_names(&self) -> &[String] {
        &self.class_names
    }

    fn image(&self) -> &ImageArtifact {
        &self.image
    }

    fn preprocessing(&self) -> &PreprocessingArtifact {
        &self.preprocessing
    }
}

impl ImageClassifierArtifact for PancImageArtifact {
    fn class_names(&self) -> &[String] {
        &self.class_names
    }

    fn image(&self) -> &ImageArtifact {
        &self.image
    }

    fn preprocessing(&self) -> &PreprocessingArtifact {
        &self.preprocessing
    }
}
