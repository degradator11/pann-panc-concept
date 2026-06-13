use std::error::Error;
use std::fs;
use std::path::Path;

use image::{GrayImage, Rgba, RgbaImage};
use serde::Serialize;

use super::patch_scan::{ScannedImage, ScoredPatch, predicted_mask};
use super::{Args, DebugSamples, PatchScanImageResult};

#[derive(Debug, Clone)]
pub(super) struct PatchMaskMetrics {
    pub(super) patch_count: usize,
    pub(super) positive_patch_count: usize,
    pub(super) accuracy: f64,
    pub(super) f1: f64,
    pub(super) auroc: f64,
}

#[derive(Debug, Serialize)]
struct PatchDebugRow {
    image_path: String,
    label: String,
    source: String,
    expected_anomaly: bool,
    predicted_anomaly: bool,
    image_score: f64,
    mask_path: Option<String>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    patch_score: f64,
    predicted_patch_anomaly: bool,
    mask_overlap_ratio: Option<f64>,
    mask_positive: Option<bool>,
}

pub(super) fn write_patch_debug(
    debug_out: &str,
    scans: &[ScannedImage],
    threshold: f64,
    args: &Args,
) -> Result<(), Box<dyn Error>> {
    let out = Path::new(debug_out);
    fs::create_dir_all(out)?;
    let mut writer = csv::Writer::from_path(out.join("patch_scan_predictions.csv"))?;
    for scan in scans {
        writer.serialize(&scan.result)?;
    }
    writer.flush()?;

    let mut patch_writer = csv::Writer::from_path(out.join("patch_scan_patches.csv"))?;
    for scan in scans {
        for row in patch_debug_rows(scan, threshold)? {
            patch_writer.serialize(row)?;
        }
    }
    patch_writer.flush()?;

    write_patch_debug_samples(out, scans, threshold, args)?;
    Ok(())
}

pub(super) fn patch_mask_metrics(
    scans: &[ScannedImage],
    threshold: f64,
) -> Result<Option<PatchMaskMetrics>, Box<dyn Error>> {
    let mut scores = Vec::new();
    let mut labels = Vec::new();
    for scan in scans {
        for row in patch_debug_rows(scan, threshold)? {
            let Some(mask_positive) = row.mask_positive else {
                continue;
            };
            scores.push(row.patch_score);
            labels.push(mask_positive);
        }
    }
    if labels.is_empty() {
        return Ok(None);
    }

    let positive_patch_count = labels.iter().filter(|label| **label).count();
    let correct = scores
        .iter()
        .zip(&labels)
        .filter(|(score, label)| (**score >= threshold) == **label)
        .count();
    let true_positive = scores
        .iter()
        .zip(&labels)
        .filter(|(score, label)| **score >= threshold && **label)
        .count();
    let false_positive = scores
        .iter()
        .zip(&labels)
        .filter(|(score, label)| **score >= threshold && !**label)
        .count();
    let false_negative = scores
        .iter()
        .zip(&labels)
        .filter(|(score, label)| **score < threshold && **label)
        .count();
    let precision = if true_positive + false_positive == 0 {
        0.0
    } else {
        true_positive as f64 / (true_positive + false_positive) as f64
    };
    let recall = if true_positive + false_negative == 0 {
        0.0
    } else {
        true_positive as f64 / (true_positive + false_negative) as f64
    };
    let f1 = if precision + recall <= f64::EPSILON {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    Ok(Some(PatchMaskMetrics {
        patch_count: labels.len(),
        positive_patch_count,
        accuracy: correct as f64 / labels.len() as f64,
        f1,
        auroc: binary_scores_auroc(&scores, &labels),
    }))
}

fn patch_debug_rows(
    scan: &ScannedImage,
    threshold: f64,
) -> Result<Vec<PatchDebugRow>, Box<dyn Error>> {
    let mask = match scan.result.mask_path.as_deref() {
        Some(path) => Some(image::open(path)?.to_luma8()),
        None => None,
    };
    let mut rows = Vec::with_capacity(scan.patches.len());
    for patch in &scan.patches {
        let mask_overlap_ratio = if scan.result.expected_anomaly {
            mask.as_ref()
                .map(|mask| patch_mask_overlap_ratio(mask, patch))
        } else {
            Some(0.0)
        };
        let mask_positive = mask_overlap_ratio.map(|ratio| ratio > 0.0);
        rows.push(PatchDebugRow {
            image_path: scan.result.path.clone(),
            label: scan.result.label.clone(),
            source: scan.result.source.clone(),
            expected_anomaly: scan.result.expected_anomaly,
            predicted_anomaly: scan.result.predicted_anomaly,
            image_score: scan.result.score,
            mask_path: scan.result.mask_path.clone(),
            x: patch.x,
            y: patch.y,
            width: patch.width,
            height: patch.height,
            patch_score: patch.anomaly_score,
            predicted_patch_anomaly: patch.anomaly_score >= threshold,
            mask_overlap_ratio,
            mask_positive,
        });
    }
    Ok(rows)
}

fn patch_mask_overlap_ratio(mask: &GrayImage, patch: &ScoredPatch) -> f64 {
    let x_end = (patch.x + patch.width).min(mask.width());
    let y_end = (patch.y + patch.height).min(mask.height());
    if patch.x >= x_end || patch.y >= y_end {
        return 0.0;
    }
    let mut active = 0usize;
    let mut total = 0usize;
    for y in patch.y..y_end {
        for x in patch.x..x_end {
            total += 1;
            if mask.get_pixel(x, y).0[0] > 0 {
                active += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        active as f64 / total as f64
    }
}

fn binary_scores_auroc(scores: &[f64], labels: &[bool]) -> f64 {
    let positives = scores
        .iter()
        .zip(labels)
        .filter_map(|(score, label)| (*label).then_some(*score))
        .collect::<Vec<_>>();
    let negatives = scores
        .iter()
        .zip(labels)
        .filter_map(|(score, label)| (!*label).then_some(*score))
        .collect::<Vec<_>>();
    if positives.is_empty() || negatives.is_empty() {
        return 0.0;
    }

    let mut wins = 0.0;
    for positive in &positives {
        for negative in &negatives {
            if positive > negative {
                wins += 1.0;
            } else if (positive - negative).abs() <= f64::EPSILON {
                wins += 0.5;
            }
        }
    }
    wins / (positives.len() * negatives.len()) as f64
}

fn write_patch_debug_samples(
    out: &Path,
    scans: &[ScannedImage],
    threshold: f64,
    args: &Args,
) -> Result<(), Box<dyn Error>> {
    let samples_dir = out.join("samples");
    fs::create_dir_all(&samples_dir)?;
    let selected = scans
        .iter()
        .filter(|scan| match args.debug_samples {
            DebugSamples::All => true,
            DebugSamples::Correct => scan.result.expected_anomaly == scan.result.predicted_anomaly,
            DebugSamples::Misclassified => {
                scan.result.expected_anomaly != scan.result.predicted_anomaly
            }
        })
        .take(args.debug_limit)
        .collect::<Vec<_>>();

    let mut html = String::from(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Patch Scan Debug</title><style>body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7;color:#222}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ddd;padding:8px;vertical-align:top}img{max-width:220px;height:auto}code{font-size:12px}</style></head><body><h1>Patch Scan Debug</h1><p>Images show original, anomaly heatmap, overlay, predicted mask, and ground-truth mask when available.</p><table><thead><tr><th>Sample</th><th>Original</th><th>Heatmap</th><th>Overlay</th><th>Predicted Mask</th><th>Ground Truth</th></tr></thead><tbody>",
    );

    for (index, scan) in selected.iter().enumerate() {
        let sample_name = sample_debug_name(index, &scan.result);
        let sample_dir = samples_dir.join(&sample_name);
        fs::create_dir_all(&sample_dir)?;
        write_sample_images(&sample_dir, scan, threshold)?;
        let mask_cell = if scan.result.mask_path.is_some() {
            format!("<img src=\"samples/{sample_name}/mask.png\" alt=\"mask\">")
        } else {
            String::from("none")
        };
        html.push_str(&format!(
            "<tr><td><strong>{}</strong><br><code>{}</code><br>expected={} predicted={} score={:.6}</td><td><img src=\"samples/{}/original.png\" alt=\"original\"></td><td><img src=\"samples/{}/heatmap.png\" alt=\"heatmap\"></td><td><img src=\"samples/{}/overlay.png\" alt=\"overlay\"></td><td><img src=\"samples/{}/predicted_mask.png\" alt=\"predicted\"></td><td>{}</td></tr>",
            html_escape(&scan.result.label),
            html_escape(&scan.result.path),
            scan.result.expected_anomaly,
            scan.result.predicted_anomaly,
            scan.result.score,
            sample_name,
            sample_name,
            sample_name,
            sample_name,
            mask_cell
        ));
    }
    html.push_str("</tbody></table></body></html>");
    fs::write(out.join("index.html"), html)?;
    Ok(())
}

fn sample_debug_name(index: usize, result: &PatchScanImageResult) -> String {
    format!(
        "{index:04}_{}_{}_score_{:.3}",
        sanitize_path_part(&result.label),
        sanitize_path_part(&result.source),
        result.score
    )
}

fn sanitize_path_part(value: &str) -> String {
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
    if sanitized.is_empty() {
        "sample".to_string()
    } else {
        sanitized
    }
}

fn write_sample_images(
    sample_dir: &Path,
    scan: &ScannedImage,
    threshold: f64,
) -> Result<(), Box<dyn Error>> {
    let original = image::open(Path::new(&scan.result.path))?.to_rgba8();
    original.save(sample_dir.join("original.png"))?;
    let heatmap = heatmap_image(original.width(), original.height(), &scan.patches);
    heatmap.save(sample_dir.join("heatmap.png"))?;
    overlay_heatmap(&original, &heatmap).save(sample_dir.join("overlay.png"))?;
    predicted_mask(
        original.width(),
        original.height(),
        &scan.patches,
        threshold,
    )
    .save(sample_dir.join("predicted_mask.png"))?;
    if let Some(mask_path) = scan.result.mask_path.as_deref() {
        image::open(mask_path)?.save(sample_dir.join("mask.png"))?;
    }
    Ok(())
}

fn heatmap_image(width: u32, height: u32, patches: &[ScoredPatch]) -> RgbaImage {
    let mut values = vec![0.0f64; (width * height) as usize];
    let min_score = patches
        .iter()
        .map(|patch| patch.anomaly_score)
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let max_score = patches
        .iter()
        .map(|patch| patch.anomaly_score)
        .max_by(f64::total_cmp)
        .unwrap_or(1.0);
    let score_width = (max_score - min_score).max(f64::EPSILON);

    for patch in patches {
        let normalized = ((patch.anomaly_score - min_score) / score_width).clamp(0.0, 1.0);
        let x_end = (patch.x + patch.width).min(width);
        let y_end = (patch.y + patch.height).min(height);
        for y in patch.y..y_end {
            for x in patch.x..x_end {
                let index = (y * width + x) as usize;
                values[index] = values[index].max(normalized);
            }
        }
    }

    let mut image = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            image.put_pixel(x, y, heat_color(values[(y * width + x) as usize]));
        }
    }
    image
}

fn heat_color(value: f64) -> Rgba<u8> {
    let value = value.clamp(0.0, 1.0);
    let red = (255.0 * value).round() as u8;
    let green = (255.0 * (value - 0.45).max(0.0) / 0.55).min(255.0) as u8;
    let blue = (180.0 * (1.0 - value)).round() as u8;
    let alpha = if value <= f64::EPSILON {
        0
    } else {
        (80.0 + 175.0 * value).round() as u8
    };
    Rgba([red, green, blue, alpha])
}

fn overlay_heatmap(original: &RgbaImage, heatmap: &RgbaImage) -> RgbaImage {
    let mut output = original.clone();
    for (pixel, heat) in output.pixels_mut().zip(heatmap.pixels()) {
        let alpha = (heat.0[3] as f32 / 255.0) * 0.55;
        for channel in 0..3 {
            pixel.0[channel] = ((pixel.0[channel] as f32 * (1.0 - alpha))
                + (heat.0[channel] as f32 * alpha))
                .round() as u8;
        }
    }
    output
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
