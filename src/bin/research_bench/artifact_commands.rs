use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use progress_ai::panc::{PancComparator, SimilarityMetric};
use progress_ai::pann::{
    CorrectionMode, Distributor, IntervalStrategy, PannConfig, PannModel, argmax,
};
use progress_ai::preprocess::{
    Dataset, min_max_ranges, min_max_scale, one_hot_labels, train_test_split,
};
use progress_ai::vision::{load_image_as_vector, load_image_folder, load_image_folder_with_paths};

use super::artifacts::{
    ARTIFACT_VERSION, ImageArtifact, ModelArtifact, PancImageArtifact, PancReferenceArtifact,
    PannImageArtifact, PreprocessingArtifact, load_artifact, save_artifact,
};
use super::{
    Args, ArtifactMetrics, ClassScore, CommandOutput, ConfusionRow, EvalMetrics,
    MisclassifiedExample, PerClassAccuracy, PredictionNeighbor, PredictionOutput, image_config,
    required_data_path, required_image_path, required_model_path, required_out_path,
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
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let inference_ms = inference_start.elapsed().as_millis();
    let diagnostics = classification_diagnostics(
        &eval_dataset.labels,
        &predictions,
        &eval_dataset.paths,
        &artifact.class_names,
    );

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
            panc_predict_with_margin(&comparator, sample, args.top_k, artifact.class_names.len())
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let inference_ms = inference_start.elapsed().as_millis();
    let memory_bytes = panc_memory_bytes(&artifact.references);
    let diagnostics = classification_diagnostics(
        &eval_dataset.labels,
        &predictions,
        &eval_dataset.paths,
        &artifact.class_names,
    );

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

fn load_scaled_image(
    image_path: &str,
    image: &ImageArtifact,
    preprocessing: &PreprocessingArtifact,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let sample = load_image_as_vector(image_path, image.to_config()?)?;
    let mut scaled = min_max_scale(&[sample], &preprocessing.min_max_ranges);
    Ok(scaled.remove(0))
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
    class_count: usize,
) -> Result<PredictedClass, Box<dyn Error>> {
    let neighbors = comparator.top_k(sample, k.max(1))?;
    let mut scores = vec![0.0; class_count];
    for neighbor in neighbors {
        if let Some(score) = scores.get_mut(neighbor.label) {
            *score += neighbor.score;
        }
    }
    Ok(PredictedClass {
        index: argmax(&scores),
        score_margin: score_margin(&scores),
    })
}

fn classification_diagnostics(
    labels: &[usize],
    predictions: &[PredictedClass],
    paths: &[PathBuf],
    class_names: &[String],
) -> ClassificationDiagnostics {
    let class_count = class_names.len();
    let mut confusion = vec![vec![0usize; class_count]; class_count];
    let mut totals = vec![0usize; class_count];
    let mut correct_by_class = vec![0usize; class_count];
    let mut correct = 0usize;
    let mut misclassified_examples = Vec::new();

    for (index, (actual, prediction)) in labels.iter().zip(predictions).enumerate() {
        if let Some(total) = totals.get_mut(*actual) {
            *total += 1;
        }
        if let Some(row) = confusion.get_mut(*actual)
            && let Some(cell) = row.get_mut(prediction.index)
        {
            *cell += 1;
        }

        if *actual == prediction.index {
            correct += 1;
            if let Some(class_correct) = correct_by_class.get_mut(*actual) {
                *class_correct += 1;
            }
        } else if misclassified_examples.len() < MAX_MISCLASSIFIED_EXAMPLES {
            misclassified_examples.push(MisclassifiedExample {
                path: paths
                    .get(index)
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| format!("sample-{index}")),
                expected_index: *actual,
                expected_label: label_name(class_names, *actual),
                predicted_index: prediction.index,
                predicted_label: label_name(class_names, prediction.index),
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
        accuracy: if labels.is_empty() {
            0.0
        } else {
            correct as f64 / labels.len() as f64
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
