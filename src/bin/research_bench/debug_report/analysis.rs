use std::collections::HashMap;
use std::error::Error;

use serde::Serialize;

use super::super::{ConfusionRow, PerClassAccuracy};
use super::{ImageEvalPrediction, SampleResizeComparison};

#[derive(Debug, Clone, Serialize)]
pub struct ImageStats {
    pub width: u32,
    pub height: u32,
    pub aspect_ratio: f64,
    pub orientation: String,
    pub brightness: f64,
    pub contrast: f64,
    pub brightness_bucket: String,
    pub contrast_bucket: String,
    pub center_crop_loss: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RankedPrediction {
    pub sample_index: usize,
    pub path: String,
    pub expected_label: String,
    pub predicted_label: String,
    pub score_margin: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClassFinding {
    pub class_name: String,
    pub correct: usize,
    pub total: usize,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfusionFinding {
    pub actual_name: String,
    pub predicted_name: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BucketSummary {
    pub bucket_group: String,
    pub bucket: String,
    pub total: usize,
    pub wrong: usize,
    pub wrong_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResizeDisagreement {
    pub sample_index: usize,
    pub path: String,
    pub expected_label: String,
    pub artifact_prediction: String,
    pub alternate_predictions: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfidenceSummary {
    pub wrong_total: usize,
    pub high_confidence_wrong: usize,
    pub ambiguous_wrong: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FailureAnalysis {
    pub summary: Vec<String>,
    pub worst_class: Option<ClassFinding>,
    pub most_common_confusion: Option<ConfusionFinding>,
    pub confidence: ConfidenceSummary,
    pub buckets: Vec<BucketSummary>,
    pub high_confidence_wrong: Vec<RankedPrediction>,
    pub ambiguous_wrong: Vec<RankedPrediction>,
    pub resize_disagreements: Vec<ResizeDisagreement>,
}

pub fn build_failure_analysis(
    predictions: &[ImageEvalPrediction],
    per_class_accuracy: &[PerClassAccuracy],
    confusion_matrix: &[ConfusionRow],
    resize_comparisons: &[SampleResizeComparison],
) -> FailureAnalysis {
    let wrong = predictions
        .iter()
        .filter(|prediction| !prediction.correct)
        .collect::<Vec<_>>();
    let high_confidence_wrong = wrong
        .iter()
        .filter(|prediction| prediction.score_margin >= 0.5)
        .count();
    let ambiguous_wrong = wrong
        .iter()
        .filter(|prediction| prediction.score_margin <= 0.1)
        .count();
    let worst_class = worst_class(per_class_accuracy);
    let most_common_confusion = most_common_confusion(confusion_matrix);
    let resize_disagreements = resize_disagreements(predictions, resize_comparisons);

    let mut summary = Vec::new();
    if let Some(class) = &worst_class {
        summary.push(format!(
            "Weakest class is {}: {:.2}% accuracy ({} / {}).",
            class.class_name,
            class.accuracy * 100.0,
            class.correct,
            class.total
        ));
    }
    if let Some(confusion) = &most_common_confusion {
        summary.push(format!(
            "Most common mistake is {} predicted as {} ({} images).",
            confusion.actual_name, confusion.predicted_name, confusion.count
        ));
    }
    summary.push(format!(
        "{} wrong predictions are high-confidence; {} wrong predictions are ambiguous.",
        high_confidence_wrong, ambiguous_wrong
    ));
    if !resize_disagreements.is_empty() {
        summary.push(format!(
            "Resize mode changes the predicted class for {} selected samples.",
            resize_disagreements.len()
        ));
    }

    FailureAnalysis {
        summary,
        worst_class,
        most_common_confusion,
        confidence: ConfidenceSummary {
            wrong_total: wrong.len(),
            high_confidence_wrong,
            ambiguous_wrong,
        },
        buckets: Vec::new(),
        high_confidence_wrong: ranked_wrong(predictions, SortMode::HighConfidence, 24),
        ambiguous_wrong: ranked_wrong(predictions, SortMode::Ambiguous, 24),
        resize_disagreements,
    }
}

pub fn add_image_buckets(
    analysis: &mut FailureAnalysis,
    predictions: &[ImageEvalPrediction],
    stats_by_sample: &HashMap<usize, ImageStats>,
) {
    let mut buckets = Vec::new();
    for group in [
        BucketGroup::Brightness,
        BucketGroup::Contrast,
        BucketGroup::Orientation,
        BucketGroup::CropLoss,
    ] {
        buckets.extend(bucket_group(group, predictions, stats_by_sample));
    }
    analysis.buckets = buckets;
}

pub fn image_stats(path: &str) -> Result<ImageStats, Box<dyn Error>> {
    let image = image::open(path)?;
    let gray = image.to_luma8();
    let width = gray.width();
    let height = gray.height();
    let count = f64::from(width * height).max(1.0);
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    for pixel in gray.pixels() {
        let value = f64::from(pixel.0[0]) / 255.0;
        sum += value;
        sum_squares += value * value;
    }
    let brightness = sum / count;
    let variance = (sum_squares / count - brightness * brightness).max(0.0);
    let contrast = variance.sqrt();
    let aspect_ratio = f64::from(width) / f64::from(height.max(1));
    let short = width.min(height);
    let crop_loss = if width == 0 || height == 0 {
        0.0
    } else {
        1.0 - f64::from(short * short) / f64::from(width * height)
    };

    Ok(ImageStats {
        width,
        height,
        aspect_ratio,
        orientation: orientation_bucket(aspect_ratio).to_string(),
        brightness,
        contrast,
        brightness_bucket: brightness_bucket(brightness).to_string(),
        contrast_bucket: contrast_bucket(contrast).to_string(),
        center_crop_loss: crop_loss.clamp(0.0, 1.0),
    })
}

fn worst_class(values: &[PerClassAccuracy]) -> Option<ClassFinding> {
    values
        .iter()
        .filter(|value| value.total > 0)
        .min_by(|left, right| left.accuracy.total_cmp(&right.accuracy))
        .map(|value| ClassFinding {
            class_name: value.class_name.clone(),
            correct: value.correct,
            total: value.total,
            accuracy: value.accuracy,
        })
}

fn most_common_confusion(rows: &[ConfusionRow]) -> Option<ConfusionFinding> {
    let class_names = rows
        .iter()
        .map(|row| row.actual_name.clone())
        .collect::<Vec<_>>();
    rows.iter()
        .flat_map(|row| {
            row.predicted_counts
                .iter()
                .copied()
                .enumerate()
                .filter(move |(predicted_index, _)| *predicted_index != row.actual_index)
                .map({
                    let class_names = class_names.clone();
                    move |(predicted_index, count)| ConfusionFinding {
                        actual_name: row.actual_name.clone(),
                        predicted_name: class_names
                            .get(predicted_index)
                            .cloned()
                            .unwrap_or_else(|| format!("class-{predicted_index}")),
                        count,
                    }
                })
        })
        .max_by_key(|finding| finding.count)
}

enum SortMode {
    HighConfidence,
    Ambiguous,
}

fn ranked_wrong(
    predictions: &[ImageEvalPrediction],
    mode: SortMode,
    limit: usize,
) -> Vec<RankedPrediction> {
    let mut ranked = predictions
        .iter()
        .filter(|prediction| !prediction.correct)
        .collect::<Vec<_>>();
    match mode {
        SortMode::HighConfidence => {
            ranked.sort_by(|left, right| right.score_margin.total_cmp(&left.score_margin))
        }
        SortMode::Ambiguous => {
            ranked.sort_by(|left, right| left.score_margin.total_cmp(&right.score_margin))
        }
    }
    ranked
        .into_iter()
        .take(limit)
        .map(|prediction| RankedPrediction {
            sample_index: prediction.sample_index,
            path: prediction.path.clone(),
            expected_label: prediction.expected_label.clone(),
            predicted_label: prediction.predicted_label.clone(),
            score_margin: prediction.score_margin,
        })
        .collect()
}

fn resize_disagreements(
    predictions: &[ImageEvalPrediction],
    comparisons: &[SampleResizeComparison],
) -> Vec<ResizeDisagreement> {
    let by_index = predictions
        .iter()
        .map(|prediction| (prediction.sample_index, prediction))
        .collect::<HashMap<_, _>>();
    comparisons
        .iter()
        .filter_map(|comparison| {
            let prediction = by_index.get(&comparison.sample_index)?;
            if !comparison
                .predictions
                .iter()
                .any(|resize| resize.differs_from_artifact_prediction)
            {
                return None;
            }
            Some(ResizeDisagreement {
                sample_index: prediction.sample_index,
                path: prediction.path.clone(),
                expected_label: prediction.expected_label.clone(),
                artifact_prediction: prediction.predicted_label.clone(),
                alternate_predictions: comparison
                    .predictions
                    .iter()
                    .map(|resize| format!("{}={}", resize.resize_mode, resize.predicted_label))
                    .collect::<Vec<_>>()
                    .join(";"),
            })
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
enum BucketGroup {
    Brightness,
    Contrast,
    Orientation,
    CropLoss,
}

fn bucket_group(
    group: BucketGroup,
    predictions: &[ImageEvalPrediction],
    stats_by_sample: &HashMap<usize, ImageStats>,
) -> Vec<BucketSummary> {
    let mut grouped: HashMap<String, (usize, usize)> = HashMap::new();
    for prediction in predictions {
        let Some(stats) = stats_by_sample.get(&prediction.sample_index) else {
            continue;
        };
        let bucket = match group {
            BucketGroup::Brightness => stats.brightness_bucket.clone(),
            BucketGroup::Contrast => stats.contrast_bucket.clone(),
            BucketGroup::Orientation => stats.orientation.clone(),
            BucketGroup::CropLoss => crop_loss_bucket(stats.center_crop_loss).to_string(),
        };
        let entry = grouped.entry(bucket).or_insert((0, 0));
        entry.0 += 1;
        if !prediction.correct {
            entry.1 += 1;
        }
    }

    let group_name = match group {
        BucketGroup::Brightness => "brightness",
        BucketGroup::Contrast => "contrast",
        BucketGroup::Orientation => "orientation",
        BucketGroup::CropLoss => "center_crop_loss",
    };
    let mut rows = grouped
        .into_iter()
        .map(|(bucket, (total, wrong))| BucketSummary {
            bucket_group: group_name.to_string(),
            bucket,
            total,
            wrong,
            wrong_rate: if total == 0 {
                0.0
            } else {
                wrong as f64 / total as f64
            },
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.bucket_group
            .cmp(&right.bucket_group)
            .then_with(|| right.wrong_rate.total_cmp(&left.wrong_rate))
            .then_with(|| left.bucket.cmp(&right.bucket))
    });
    rows
}

fn brightness_bucket(value: f64) -> &'static str {
    if value < 0.33 {
        "dark"
    } else if value > 0.66 {
        "bright"
    } else {
        "medium"
    }
}

fn contrast_bucket(value: f64) -> &'static str {
    if value < 0.12 {
        "low"
    } else if value > 0.28 {
        "high"
    } else {
        "medium"
    }
}

fn orientation_bucket(aspect_ratio: f64) -> &'static str {
    if aspect_ratio > 1.2 {
        "landscape"
    } else if aspect_ratio < 0.83 {
        "portrait"
    } else {
        "square-ish"
    }
}

fn crop_loss_bucket(value: f64) -> &'static str {
    if value < 0.10 {
        "low"
    } else if value > 0.35 {
        "high"
    } else {
        "medium"
    }
}
