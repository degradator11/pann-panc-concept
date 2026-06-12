use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use progress_ai::vision::{ImageVectorConfig, load_image_processing_steps};
use serde::Serialize;

use super::{ClassScore, ConfusionRow, DebugSamples, PerClassAccuracy};

mod analysis;

#[derive(Debug, Clone, Serialize)]
pub struct ImageEvalPrediction {
    pub sample_index: usize,
    pub path: String,
    pub expected_index: usize,
    pub expected_label: String,
    pub predicted_index: usize,
    pub predicted_label: String,
    pub correct: bool,
    pub score_margin: f64,
    pub scores: Vec<ClassScore>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResizePrediction {
    pub resize_mode: String,
    pub predicted_index: usize,
    pub predicted_label: String,
    pub score_margin: f64,
    pub differs_from_artifact_prediction: bool,
    pub scores: Vec<ClassScore>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleResizeComparison {
    pub sample_index: usize,
    pub predictions: Vec<ResizePrediction>,
}

#[derive(Debug, Clone)]
pub struct DebugReference {
    pub path: String,
    pub label_index: usize,
    pub label: String,
    pub scaled_sample: Vec<f64>,
}

pub struct ImageEvalDebugData<'a> {
    pub out_path: &'a str,
    pub model: &'a str,
    pub data_path: &'a str,
    pub model_path: &'a str,
    pub image_config: ImageVectorConfig,
    pub image_features: &'a str,
    pub image_resize: &'a str,
    pub accuracy: f64,
    pub inference_ms: u128,
    pub memory_bytes: usize,
    pub per_class_accuracy: &'a [PerClassAccuracy],
    pub confusion_matrix: &'a [ConfusionRow],
    pub predictions: &'a [ImageEvalPrediction],
    pub scaled_samples: &'a [Vec<f64>],
    pub resize_comparisons: &'a [SampleResizeComparison],
    pub references: &'a [DebugReference],
    pub debug_limit: usize,
    pub debug_samples: DebugSamples,
    pub debug_neighbors: usize,
}

#[derive(Debug, Serialize)]
struct DebugConfig<'a> {
    model: &'a str,
    data_path: &'a str,
    model_path: &'a str,
    image_width: u32,
    image_height: u32,
    image_features: &'a str,
    image_resize: &'a str,
    debug_limit: usize,
    debug_samples: &'a str,
    debug_neighbors: usize,
    reference_count: usize,
}

#[derive(Debug, Serialize)]
struct DebugMetrics<'a> {
    model: &'a str,
    accuracy: f64,
    inference_ms: u128,
    memory_bytes: usize,
    sample_count: usize,
    per_class_accuracy: &'a [PerClassAccuracy],
    confusion_matrix: &'a [ConfusionRow],
}

#[derive(Debug, Serialize)]
struct PredictionCsvRow<'a> {
    sample_index: usize,
    path: &'a str,
    expected_label: &'a str,
    predicted_label: &'a str,
    correct: bool,
    score_margin: f64,
    scores: String,
}

#[derive(Debug, Serialize)]
struct ConfusionCsvRow<'a> {
    actual_index: usize,
    actual_name: &'a str,
    predicted_counts: String,
}

#[derive(Debug, Serialize)]
struct SampleAsset {
    name: String,
    path: String,
}

#[derive(Debug, Serialize)]
struct SampleSummary<'a> {
    prediction: &'a ImageEvalPrediction,
    assets: Vec<SampleAsset>,
    scaled_feature_vector_path: String,
    image_stats: analysis::ImageStats,
    resize_predictions: Vec<ResizePrediction>,
    nearest_neighbors: Vec<NeighborSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct NeighborSummary {
    path: String,
    label_index: usize,
    label: String,
    distance: f64,
    asset_path: Option<String>,
}

struct HtmlSample {
    title: String,
    original_path: String,
    processed_path: String,
    expected_label: String,
    predicted_label: String,
    score_margin: f64,
    correct: bool,
    image_stats: analysis::ImageStats,
    resize_predictions: Vec<ResizePrediction>,
    nearest_neighbors: Vec<NeighborSummary>,
}

pub fn write_image_eval_debug_report(data: &ImageEvalDebugData<'_>) -> Result<(), Box<dyn Error>> {
    let root = Path::new(data.out_path);
    fs::create_dir_all(root)?;
    fs::create_dir_all(root.join("samples"))?;

    let stats_by_sample = image_stats_by_sample(data.predictions);
    let resize_by_sample = resize_comparisons_by_sample(data.resize_comparisons);
    let mut failure_analysis = analysis::build_failure_analysis(
        data.predictions,
        data.per_class_accuracy,
        data.confusion_matrix,
        data.resize_comparisons,
    );
    analysis::add_image_buckets(&mut failure_analysis, data.predictions, &stats_by_sample);

    write_json(root.join("config.json"), &debug_config(data))?;
    write_json(root.join("metrics.json"), &debug_metrics(data))?;
    write_json(root.join("failure_analysis.json"), &failure_analysis)?;
    write_json(root.join("predictions.json"), data.predictions)?;
    write_predictions_csv(root.join("predictions.csv"), data.predictions)?;
    write_per_class_csv(root.join("per_class_accuracy.csv"), data.per_class_accuracy)?;
    write_confusion_csv(root.join("confusion_matrix.csv"), data.confusion_matrix)?;
    write_failure_buckets_csv(root.join("failure_buckets.csv"), &failure_analysis.buckets)?;

    let selected =
        selected_prediction_indices(data.predictions, data.debug_samples, data.debug_limit);
    let mut html_samples = Vec::with_capacity(selected.len());
    for (position, prediction_index) in selected.into_iter().enumerate() {
        let prediction = &data.predictions[prediction_index];
        html_samples.push(write_sample_debug_directory(
            root,
            data,
            position,
            prediction,
            &stats_by_sample,
            &resize_by_sample,
        )?);
    }
    write_index_html(
        root.join("index.html"),
        data,
        &failure_analysis,
        &html_samples,
    )?;

    Ok(())
}

fn debug_config<'a>(data: &'a ImageEvalDebugData<'a>) -> DebugConfig<'a> {
    DebugConfig {
        model: data.model,
        data_path: data.data_path,
        model_path: data.model_path,
        image_width: data.image_config.width,
        image_height: data.image_config.height,
        image_features: data.image_features,
        image_resize: data.image_resize,
        debug_limit: data.debug_limit,
        debug_samples: data.debug_samples.as_str(),
        debug_neighbors: data.debug_neighbors,
        reference_count: data.references.len(),
    }
}

fn debug_metrics<'a>(data: &'a ImageEvalDebugData<'a>) -> DebugMetrics<'a> {
    DebugMetrics {
        model: data.model,
        accuracy: data.accuracy,
        inference_ms: data.inference_ms,
        memory_bytes: data.memory_bytes,
        sample_count: data.predictions.len(),
        per_class_accuracy: data.per_class_accuracy,
        confusion_matrix: data.confusion_matrix,
    }
}

pub fn selected_prediction_indices(
    predictions: &[ImageEvalPrediction],
    debug_samples: DebugSamples,
    limit: usize,
) -> Vec<usize> {
    if limit == 0 {
        return Vec::new();
    }

    match debug_samples {
        DebugSamples::All => (0..predictions.len()).take(limit).collect(),
        DebugSamples::Correct => predictions
            .iter()
            .enumerate()
            .filter(|(_, prediction)| prediction.correct)
            .take(limit)
            .map(|(index, _)| index)
            .collect(),
        DebugSamples::Misclassified => select_misclassified_indices(predictions, limit),
    }
}

fn select_misclassified_indices(predictions: &[ImageEvalPrediction], limit: usize) -> Vec<usize> {
    let mut high_confidence = predictions
        .iter()
        .enumerate()
        .filter(|(_, prediction)| !prediction.correct)
        .collect::<Vec<_>>();
    high_confidence
        .sort_by(|(_, left), (_, right)| right.score_margin.total_cmp(&left.score_margin));

    let mut ambiguous = high_confidence.clone();
    ambiguous.sort_by(|(_, left), (_, right)| left.score_margin.total_cmp(&right.score_margin));

    let mut selected = Vec::new();
    let high_quota = (limit / 2).max(1);
    push_unique_indices(
        &mut selected,
        high_confidence
            .into_iter()
            .take(high_quota)
            .map(|(index, _)| index),
    );
    let remaining = limit.saturating_sub(selected.len());
    push_unique_indices(
        &mut selected,
        ambiguous
            .into_iter()
            .take(remaining)
            .map(|(index, _)| index),
    );
    push_unique_indices(
        &mut selected,
        predictions
            .iter()
            .enumerate()
            .filter(|(_, prediction)| !prediction.correct)
            .map(|(index, _)| index),
    );
    selected.truncate(limit);
    selected
}

fn push_unique_indices(target: &mut Vec<usize>, source: impl Iterator<Item = usize>) {
    for index in source {
        if !target.contains(&index) {
            target.push(index);
        }
    }
}

fn write_sample_debug_directory(
    root: &Path,
    data: &ImageEvalDebugData<'_>,
    position: usize,
    prediction: &ImageEvalPrediction,
    stats_by_sample: &std::collections::HashMap<usize, analysis::ImageStats>,
    resize_by_sample: &std::collections::HashMap<usize, Vec<ResizePrediction>>,
) -> Result<HtmlSample, Box<dyn Error>> {
    let sample_dir_name = sample_dir_name(position, prediction);
    let sample_dir = root.join("samples").join(&sample_dir_name);
    fs::create_dir_all(&sample_dir)?;

    let mut assets = Vec::new();
    for (step_index, step) in load_image_processing_steps(&prediction.path, data.image_config)?
        .into_iter()
        .enumerate()
    {
        let file_name = format!("step_{step_index}_{}.png", sanitize_file_part(&step.name));
        let path = sample_dir.join(&file_name);
        step.image.save(&path)?;
        assets.push(SampleAsset {
            name: step.name,
            path: sample_rel_path(&sample_dir_name, &file_name),
        });
    }

    let feature_file_name = format!("step_{}_scaled_feature_vector.csv", assets.len());
    write_feature_vector_csv(
        sample_dir.join(&feature_file_name),
        data.scaled_samples
            .get(prediction.sample_index)
            .map(Vec::as_slice)
            .unwrap_or(&[]),
    )?;

    let feature_rel_path = sample_rel_path(&sample_dir_name, &feature_file_name);
    let image_stats = stats_by_sample
        .get(&prediction.sample_index)
        .cloned()
        .unwrap_or_else(fallback_image_stats);
    let resize_predictions = resize_by_sample
        .get(&prediction.sample_index)
        .cloned()
        .unwrap_or_default();
    let nearest_neighbors =
        write_nearest_neighbors(&sample_dir, &sample_dir_name, data, prediction)?;
    let summary = SampleSummary {
        prediction,
        assets,
        scaled_feature_vector_path: feature_rel_path,
        image_stats: image_stats.clone(),
        resize_predictions: resize_predictions.clone(),
        nearest_neighbors: nearest_neighbors.clone(),
    };
    write_json(sample_dir.join("summary.json"), &summary)?;

    let original_path = sample_rel_path(&sample_dir_name, "step_0_original.png");
    let processed_path = summary
        .assets
        .last()
        .map(|asset| asset.path.clone())
        .unwrap_or_else(|| original_path.clone());

    Ok(HtmlSample {
        title: sample_dir_name,
        original_path,
        processed_path,
        expected_label: prediction.expected_label.clone(),
        predicted_label: prediction.predicted_label.clone(),
        score_margin: prediction.score_margin,
        correct: prediction.correct,
        image_stats,
        resize_predictions,
        nearest_neighbors,
    })
}

fn image_stats_by_sample(
    predictions: &[ImageEvalPrediction],
) -> std::collections::HashMap<usize, analysis::ImageStats> {
    predictions
        .iter()
        .filter_map(|prediction| {
            analysis::image_stats(&prediction.path)
                .ok()
                .map(|stats| (prediction.sample_index, stats))
        })
        .collect()
}

fn resize_comparisons_by_sample(
    comparisons: &[SampleResizeComparison],
) -> std::collections::HashMap<usize, Vec<ResizePrediction>> {
    comparisons
        .iter()
        .map(|comparison| (comparison.sample_index, comparison.predictions.clone()))
        .collect()
}

fn write_nearest_neighbors(
    sample_dir: &Path,
    sample_dir_name: &str,
    data: &ImageEvalDebugData<'_>,
    prediction: &ImageEvalPrediction,
) -> Result<Vec<NeighborSummary>, Box<dyn Error>> {
    if data.references.is_empty() || data.debug_neighbors == 0 {
        return Ok(Vec::new());
    }
    let Some(query) = data.scaled_samples.get(prediction.sample_index) else {
        return Ok(Vec::new());
    };

    let mut neighbors = data
        .references
        .iter()
        .map(|reference| {
            (
                reference,
                euclidean_distance(query, &reference.scaled_sample),
            )
        })
        .collect::<Vec<_>>();
    neighbors.sort_by(|(_, left), (_, right)| left.total_cmp(right));

    let mut summaries = Vec::new();
    for (neighbor_index, (reference, distance)) in
        neighbors.into_iter().take(data.debug_neighbors).enumerate()
    {
        let asset_path = write_neighbor_asset(
            sample_dir,
            sample_dir_name,
            data.image_config,
            neighbor_index,
            reference,
        )
        .ok();
        summaries.push(NeighborSummary {
            path: reference.path.clone(),
            label_index: reference.label_index,
            label: reference.label.clone(),
            distance,
            asset_path,
        });
    }
    Ok(summaries)
}

fn write_neighbor_asset(
    sample_dir: &Path,
    sample_dir_name: &str,
    config: ImageVectorConfig,
    neighbor_index: usize,
    reference: &DebugReference,
) -> Result<String, Box<dyn Error>> {
    let stem = Path::new(&reference.path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("neighbor");
    let file_name = format!(
        "neighbor_{neighbor_index}_{}_{}.png",
        sanitize_file_part(&reference.label),
        sanitize_file_part(stem)
    );
    let Some(step) = load_image_processing_steps(&reference.path, config)?.pop() else {
        return Err("neighbor preprocessing produced no image steps".into());
    };
    step.image.save(sample_dir.join(&file_name))?;
    Ok(sample_rel_path(sample_dir_name, &file_name))
}

fn euclidean_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = left - right;
            difference * difference
        })
        .sum::<f64>()
        .sqrt()
}

fn fallback_image_stats() -> analysis::ImageStats {
    analysis::ImageStats {
        width: 0,
        height: 0,
        aspect_ratio: 0.0,
        orientation: "unknown".to_string(),
        brightness: 0.0,
        contrast: 0.0,
        brightness_bucket: "unknown".to_string(),
        contrast_bucket: "unknown".to_string(),
        center_crop_loss: 0.0,
    }
}

fn write_json(
    path: impl AsRef<Path>,
    value: &(impl Serialize + ?Sized),
) -> Result<(), Box<dyn Error>> {
    fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn write_predictions_csv(
    path: impl AsRef<Path>,
    predictions: &[ImageEvalPrediction],
) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_path(path)?;
    for prediction in predictions {
        writer.serialize(PredictionCsvRow {
            sample_index: prediction.sample_index,
            path: &prediction.path,
            expected_label: &prediction.expected_label,
            predicted_label: &prediction.predicted_label,
            correct: prediction.correct,
            score_margin: prediction.score_margin,
            scores: format_scores(&prediction.scores),
        })?;
    }
    writer.flush()?;
    Ok(())
}

fn write_per_class_csv(
    path: impl AsRef<Path>,
    values: &[PerClassAccuracy],
) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_path(path)?;
    for value in values {
        writer.serialize(value)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_confusion_csv(
    path: impl AsRef<Path>,
    values: &[ConfusionRow],
) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_path(path)?;
    for value in values {
        writer.serialize(ConfusionCsvRow {
            actual_index: value.actual_index,
            actual_name: &value.actual_name,
            predicted_counts: value
                .predicted_counts
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("|"),
        })?;
    }
    writer.flush()?;
    Ok(())
}

fn write_failure_buckets_csv(
    path: impl AsRef<Path>,
    values: &[analysis::BucketSummary],
) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_path(path)?;
    for value in values {
        writer.serialize(value)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_feature_vector_csv(path: impl AsRef<Path>, values: &[f64]) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["feature_index", "scaled_value"])?;
    for (index, value) in values.iter().enumerate() {
        writer.write_record([index.to_string(), value.to_string()])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_index_html(
    path: impl AsRef<Path>,
    data: &ImageEvalDebugData<'_>,
    failure_analysis: &analysis::FailureAnalysis,
    samples: &[HtmlSample],
) -> Result<(), Box<dyn Error>> {
    let mut html = String::new();
    html.push_str("<!doctype html><html><head><meta charset=\"utf-8\">");
    html.push_str("<title>PANN/PANC Image Debug Report</title>");
    html.push_str("<style>");
    html.push_str(
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;background:#f7f7f5;color:#222}\
        h1,h2,h3{margin:0 0 12px}section{margin:24px 0}table{border-collapse:collapse;background:#fff}\
        th,td{border:1px solid #ddd;padding:8px 10px;text-align:left;vertical-align:top}th{background:#eee}\
        .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}\
        .card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px}\
        .images,.neighbors{display:grid;grid-template-columns:1fr 1fr;gap:8px}.images img,.neighbors img{width:100%;height:auto;border:1px solid #ddd;background:#888}\
        .ok{color:#126b2f}.bad{color:#9b1c1c}.muted{color:#666}.metric{font-size:22px;font-weight:700}\
        .pill{display:inline-block;background:#eee;border-radius:999px;padding:2px 8px;margin:2px}.summary{background:#fff;border-left:4px solid #444;padding:12px}",
    );
    html.push_str("</style></head><body>");
    html.push_str("<h1>PANN/PANC Image Debug Report</h1>");
    html.push_str(&format!(
        "<p class=\"muted\">model={} features={} resize={} samples={}</p>",
        escape_html(data.model),
        escape_html(data.image_features),
        escape_html(data.image_resize),
        data.predictions.len()
    ));
    html.push_str(&format!(
        "<p class=\"metric\">Accuracy: {:.2}%</p>",
        data.accuracy * 100.0
    ));

    html.push_str("<section class=\"summary\"><h2>What This Says</h2><ul>");
    for line in &failure_analysis.summary {
        html.push_str(&format!("<li>{}</li>", escape_html(line)));
    }
    html.push_str("</ul></section>");

    html.push_str("<section><h2>Files</h2><table><tr><th>File</th><th>Purpose</th></tr>");
    for (file, purpose) in [
        ("config.json", "run configuration"),
        ("metrics.json", "summary metrics"),
        ("failure_analysis.json", "interpreted failure analysis"),
        (
            "failure_buckets.csv",
            "failure rates by brightness/contrast/aspect/crop-loss bucket",
        ),
        ("predictions.csv", "all prediction rows"),
        ("predictions.json", "all prediction rows with scores"),
        ("per_class_accuracy.csv", "per-class accuracy"),
        ("confusion_matrix.csv", "confusion matrix rows"),
    ] {
        html.push_str(&format!(
            "<tr><td><a href=\"{}\">{}</a></td><td>{}</td></tr>",
            file, file, purpose
        ));
    }
    html.push_str("</table></section>");

    html.push_str("<section><h2>Failure Buckets</h2><table><tr><th>Group</th><th>Bucket</th><th>Total</th><th>Wrong</th><th>Wrong Rate</th></tr>");
    for row in &failure_analysis.buckets {
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.2}%</td></tr>",
            escape_html(&row.bucket_group),
            escape_html(&row.bucket),
            row.total,
            row.wrong,
            row.wrong_rate * 100.0
        ));
    }
    html.push_str("</table></section>");

    html.push_str("<section><h2>High-Confidence Wrong</h2>");
    write_ranked_prediction_table(&mut html, &failure_analysis.high_confidence_wrong);
    html.push_str("</section>");

    html.push_str("<section><h2>Ambiguous Wrong</h2>");
    write_ranked_prediction_table(&mut html, &failure_analysis.ambiguous_wrong);
    html.push_str("</section>");

    if !failure_analysis.resize_disagreements.is_empty() {
        html.push_str("<section><h2>Resize Sensitivity</h2><table><tr><th>Sample</th><th>Expected</th><th>Artifact Prediction</th><th>Resize Predictions</th></tr>");
        for row in failure_analysis.resize_disagreements.iter().take(50) {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                escape_html(&row.path),
                escape_html(&row.expected_label),
                escape_html(&row.artifact_prediction),
                escape_html(&row.alternate_predictions)
            ));
        }
        html.push_str("</table></section>");
    }

    html.push_str("<section><h2>Per-Class Accuracy</h2><table><tr><th>Class</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>");
    for row in data.per_class_accuracy {
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2}%</td></tr>",
            escape_html(&row.class_name),
            row.correct,
            row.total,
            row.accuracy * 100.0
        ));
    }
    html.push_str("</table></section>");

    html.push_str("<section><h2>Confusion Matrix</h2><table><tr><th>Actual</th><th>Predicted Counts</th></tr>");
    for row in data.confusion_matrix {
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td></tr>",
            escape_html(&row.actual_name),
            row.predicted_counts
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(" | ")
        ));
    }
    html.push_str("</table></section>");

    html.push_str("<section><h2>Selected Samples</h2><div class=\"cards\">");
    for sample in samples {
        let status_class = if sample.correct { "ok" } else { "bad" };
        let status = if sample.correct { "correct" } else { "wrong" };
        html.push_str("<article class=\"card\">");
        html.push_str(&format!("<h3>{}</h3>", escape_html(&sample.title)));
        html.push_str(&format!(
            "<p class=\"{}\">{}: expected {}, predicted {}, margin {:.4}</p>",
            status_class,
            status,
            escape_html(&sample.expected_label),
            escape_html(&sample.predicted_label),
            sample.score_margin
        ));
        html.push_str(&format!(
            "<p><span class=\"pill\">{}</span><span class=\"pill\">{}</span><span class=\"pill\">{}x{}</span><span class=\"pill\">crop loss {:.1}%</span></p>",
            escape_html(&sample.image_stats.brightness_bucket),
            escape_html(&sample.image_stats.contrast_bucket),
            sample.image_stats.width,
            sample.image_stats.height,
            sample.image_stats.center_crop_loss * 100.0
        ));
        if !sample.resize_predictions.is_empty() {
            html.push_str("<table><tr><th>Resize</th><th>Prediction</th><th>Margin</th></tr>");
            for resize in &sample.resize_predictions {
                let class = if resize.differs_from_artifact_prediction {
                    "bad"
                } else {
                    "muted"
                };
                html.push_str(&format!(
                    "<tr><td>{}</td><td class=\"{}\">{}</td><td>{:.4}</td></tr>",
                    escape_html(&resize.resize_mode),
                    class,
                    escape_html(&resize.predicted_label),
                    resize.score_margin
                ));
            }
            html.push_str("</table>");
        }
        html.push_str("<div class=\"images\">");
        html.push_str(&format!(
            "<figure><img src=\"{}\"><figcaption>original</figcaption></figure>",
            escape_html(&sample.original_path)
        ));
        html.push_str(&format!(
            "<figure><img src=\"{}\"><figcaption>processed</figcaption></figure>",
            escape_html(&sample.processed_path)
        ));
        html.push_str("</div>");
        if !sample.nearest_neighbors.is_empty() {
            html.push_str("<h4>Nearest Training Examples</h4><div class=\"neighbors\">");
            for neighbor in &sample.nearest_neighbors {
                if let Some(asset_path) = &neighbor.asset_path {
                    html.push_str(&format!(
                        "<figure><img src=\"{}\"><figcaption>{} distance {:.4}</figcaption></figure>",
                        escape_html(asset_path),
                        escape_html(&neighbor.label),
                        neighbor.distance
                    ));
                }
            }
            html.push_str("</div>");
        }
        html.push_str("</article>");
    }
    html.push_str("</div></section></body></html>");

    fs::write(path, html)?;
    Ok(())
}

fn sample_dir_name(position: usize, prediction: &ImageEvalPrediction) -> String {
    let stem = Path::new(&prediction.path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("sample");
    sanitize_file_part(&format!(
        "{position:04}_expected_{}_predicted_{}_{}",
        prediction.expected_label, prediction.predicted_label, stem
    ))
}

fn sample_rel_path(sample_dir_name: &str, file_name: &str) -> String {
    let mut path = PathBuf::from("samples");
    path.push(sample_dir_name);
    path.push(file_name);
    path.to_string_lossy().replace('\\', "/")
}

fn write_ranked_prediction_table(html: &mut String, rows: &[analysis::RankedPrediction]) {
    html.push_str(
        "<table><tr><th>Sample</th><th>Expected</th><th>Predicted</th><th>Margin</th></tr>",
    );
    for row in rows {
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.4}</td></tr>",
            escape_html(&row.path),
            escape_html(&row.expected_label),
            escape_html(&row.predicted_label),
            row.score_margin
        ));
    }
    html.push_str("</table>");
}

fn sanitize_file_part(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    sanitized.trim_matches('_').chars().take(96).collect()
}

fn format_scores(scores: &[ClassScore]) -> String {
    scores
        .iter()
        .map(|score| format!("{}={:.6}", score.class_name, score.score))
        .collect::<Vec<_>>()
        .join(";")
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
