use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::GrayImage;
use progress_ai::panc::{PancComparator, SimilarityMetric};
use progress_ai::preprocess::{min_max_ranges, min_max_scale, train_test_split_indices};
use progress_ai::vision::{
    ImageDatasetEntry, ImageDatasetManifest, ImageManifestDataset, ManifestSources,
    dynamic_image_to_vector, load_image_manifest, load_image_manifest_from_value,
};

use super::{
    Args, CommandOutput, PatchScanImageResult, PatchScanReport, image_config, required_data_path,
};

#[derive(Debug, Clone)]
struct PatchVector {
    vector: Vec<f64>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Debug, Clone)]
struct ScoredPatch {
    anomaly_score: f64,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Debug, Clone)]
struct ScannedImage {
    result: PatchScanImageResult,
    patches: Vec<ScoredPatch>,
}

struct PatchDatasetInput {
    loaded: ImageManifestDataset,
    dataset_config_path: Option<String>,
    data_path: Option<String>,
}

pub fn run_panc_patch_scan(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    validate_patch_args(args)?;
    let PatchDatasetInput {
        loaded,
        dataset_config_path,
        data_path,
    } = load_patch_dataset(args)?;
    let normal_class = normal_class_name(&loaded)?;
    let normal_entries = loaded
        .train
        .entries
        .iter()
        .filter(|entry| entry.class_name == normal_class)
        .cloned()
        .collect::<Vec<_>>();
    if normal_entries.is_empty() {
        return Err(format!("manifest train split has no normal class {normal_class:?}").into());
    }

    let (calibration_indexes, reference_indexes) =
        train_test_split_indices(normal_entries.len(), 0.2, args.seed);
    let reference_entries = if reference_indexes.is_empty() {
        normal_entries.clone()
    } else {
        collect_entries(&normal_entries, &reference_indexes)
    };
    let calibration_entries = if calibration_indexes.is_empty() {
        reference_entries.clone()
    } else {
        collect_entries(&normal_entries, &calibration_indexes)
    };

    let train_start = Instant::now();
    let mut reference_vectors = extract_entries_patches(&reference_entries, args)?;
    reference_vectors = cap_reference_patches(reference_vectors, args.max_train_patches);
    if reference_vectors.is_empty() {
        return Err("patch scanner found no reference patches".into());
    }
    let ranges = min_max_ranges(&reference_vectors);
    let scaled_references = min_max_scale(&reference_vectors, &ranges);
    let mut comparator = PancComparator::new(SimilarityMetric::Euclidean);
    for vector in scaled_references {
        comparator.add_reference(vector, 0usize, ())?;
    }
    let train_ms = train_start.elapsed().as_millis();

    let calibration_scores = calibration_entries
        .iter()
        .map(|entry| scan_entry(entry, false, None, &comparator, &ranges, args))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|scan| scan.result.score)
        .collect::<Vec<_>>();
    let anomaly_threshold = percentile(&calibration_scores, args.anomaly_threshold_quantile);

    let eval_entries = loaded
        .eval
        .as_ref()
        .ok_or("patch scan requires an eval split in the manifest or MVTec category")?
        .entries
        .clone();
    let inference_start = Instant::now();
    let scans = eval_entries
        .iter()
        .map(|entry| {
            let expected_anomaly = entry.class_name != normal_class;
            let mask_path = expected_anomaly
                .then(|| mask_path_for_entry(entry, &loaded.masks))
                .flatten();
            scan_entry(
                entry,
                expected_anomaly,
                mask_path.as_deref(),
                &comparator,
                &ranges,
                args,
            )
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let inference_ms = inference_start.elapsed().as_millis();

    let mut results = scans
        .into_iter()
        .map(|mut scan| {
            scan.result.predicted_anomaly = scan.result.score >= anomaly_threshold;
            if let Some(mask_path) = scan.result.mask_path.clone() {
                scan.result.mask_iou =
                    mask_iou(Path::new(&mask_path), &scan.patches, anomaly_threshold).ok();
            }
            scan.result
        })
        .collect::<Vec<_>>();
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.path.cmp(&right.path))
    });

    if let Some(debug_out) = args.debug_out_path.as_deref() {
        write_patch_debug(debug_out, &results)?;
    }

    let image_accuracy = binary_accuracy(&results);
    let normal_accuracy = class_binary_accuracy(&results, false);
    let anomaly_recall = class_binary_accuracy(&results, true);
    let image_auroc = binary_auroc(&results);
    let mask_iou_mean = mean_option(results.iter().map(|result| result.mask_iou));
    let normal_eval_images = results
        .iter()
        .filter(|result| !result.expected_anomaly)
        .count();
    let anomaly_eval_images = results.len().saturating_sub(normal_eval_images);
    let reference_patches = comparator.references().len();
    let feature_len = comparator
        .references()
        .first()
        .map(|reference| reference.vector.len())
        .unwrap_or(0);

    Ok(CommandOutput::PatchScan(PatchScanReport {
        model: "panc_like_patch_scan".to_string(),
        dataset: loaded
            .name
            .clone()
            .unwrap_or_else(|| "image-manifest".to_string()),
        dataset_config_path,
        data_path,
        image_features: args.image_features.as_str().to_string(),
        image_resize: args.image_resize.as_str().to_string(),
        patch_size: args.patch_size,
        patch_stride: args.patch_stride,
        image_size: args.image_width,
        max_train_patches: args.max_train_patches,
        reference_images: reference_entries.len(),
        calibration_images: calibration_entries.len(),
        reference_patches,
        eval_images: results.len(),
        normal_eval_images,
        anomaly_eval_images,
        threshold_quantile: args.anomaly_threshold_quantile,
        anomaly_threshold,
        image_accuracy,
        normal_accuracy,
        anomaly_recall,
        image_auroc,
        train_ms,
        inference_ms,
        memory_bytes: reference_patches * feature_len * std::mem::size_of::<f64>(),
        mask_iou_mean,
        results,
    }))
}

fn load_patch_dataset(args: &Args) -> Result<PatchDatasetInput, Box<dyn Error>> {
    if let Some(path) = args.dataset_config_path.as_deref() {
        return Ok(PatchDatasetInput {
            loaded: load_image_manifest(path, image_config(args))?,
            dataset_config_path: Some(path.to_string()),
            data_path: None,
        });
    }

    let data_path = required_data_path(args)?;
    let category_root = Path::new(data_path);
    let manifest = mvtec_manifest(category_root)?;
    let manifest_path = category_root.join("mvtec.auto-manifest.json");
    Ok(PatchDatasetInput {
        loaded: load_image_manifest_from_value(&manifest_path, manifest, image_config(args))?,
        dataset_config_path: None,
        data_path: Some(data_path.to_string()),
    })
}

fn mvtec_manifest(category_root: &Path) -> Result<ImageDatasetManifest, Box<dyn Error>> {
    if !category_root.join("train").join("good").is_dir() || !category_root.join("test").is_dir() {
        return Err(format!(
            "panc-patch-scan --data expects one MVTec category root with train/good and test/* folders, got {}",
            category_root.display()
        )
        .into());
    }

    let mut train = BTreeMap::new();
    train.insert(
        "normal".to_string(),
        ManifestSources::One(PathBuf::from("train/good")),
    );

    let mut anomaly_dirs = fs::read_dir(category_root.join("test"))?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|entry| entry.path().is_dir() && entry.file_name() != "good")
        .map(|entry| PathBuf::from("test").join(entry.file_name()))
        .collect::<Vec<_>>();
    anomaly_dirs.sort();

    let mut eval = BTreeMap::new();
    eval.insert(
        "normal".to_string(),
        ManifestSources::One(PathBuf::from("test/good")),
    );
    eval.insert(
        "anomaly".to_string(),
        ManifestSources::Many(anomaly_dirs.clone()),
    );

    let mut masks = BTreeMap::new();
    for defect_dir in anomaly_dirs {
        let Some(defect_name) = defect_dir.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let mask_dir = PathBuf::from("ground_truth").join(defect_name);
        if category_root.join(&mask_dir).is_dir() {
            masks.insert(defect_name.to_string(), ManifestSources::One(mask_dir));
        }
    }

    Ok(ImageDatasetManifest {
        version: Some(1),
        name: category_root
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| format!("mvtec-{name}")),
        root: Some(category_root.to_path_buf()),
        train: Some(train),
        eval: Some(eval),
        masks: Some(masks),
    })
}

fn normal_class_name(loaded: &ImageManifestDataset) -> Result<String, Box<dyn Error>> {
    for preferred in ["normal", "good"] {
        if loaded
            .train
            .dataset
            .class_names
            .iter()
            .any(|class_name| class_name == preferred)
        {
            return Ok(preferred.to_string());
        }
    }
    loaded
        .train
        .dataset
        .class_names
        .first()
        .cloned()
        .ok_or_else(|| "manifest train split has no classes".into())
}

fn extract_entries_patches(
    entries: &[ImageDatasetEntry],
    args: &Args,
) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut vectors = Vec::new();
    for entry in entries {
        vectors.extend(
            extract_patch_vectors(&entry.path, args)?
                .into_iter()
                .map(|patch| patch.vector),
        );
    }
    Ok(vectors)
}

fn extract_patch_vectors(path: &Path, args: &Args) -> Result<Vec<PatchVector>, Box<dyn Error>> {
    let image = image::open(path)?;
    let width = image.width();
    let height = image.height();
    let patch_width = args.patch_size.min(width).max(1);
    let patch_height = args.patch_size.min(height).max(1);
    let mut patches = Vec::new();
    for y in sliding_positions(height, patch_height, args.patch_stride) {
        for x in sliding_positions(width, patch_width, args.patch_stride) {
            let patch = image.crop_imm(x, y, patch_width, patch_height);
            patches.push(PatchVector {
                vector: dynamic_image_to_vector(patch, image_config(args))?,
                x,
                y,
                width: patch_width,
                height: patch_height,
            });
        }
    }
    Ok(patches)
}

fn scan_entry(
    entry: &ImageDatasetEntry,
    expected_anomaly: bool,
    mask_path: Option<&Path>,
    comparator: &PancComparator<usize>,
    ranges: &[progress_ai::pann::FeatureRange],
    args: &Args,
) -> Result<ScannedImage, Box<dyn Error>> {
    let patches = extract_patch_vectors(&entry.path, args)?;
    let raw_vectors = patches
        .iter()
        .map(|patch| patch.vector.clone())
        .collect::<Vec<_>>();
    let scaled = min_max_scale(&raw_vectors, ranges);
    let mut scored_patches = Vec::with_capacity(patches.len());
    let mut scores = Vec::with_capacity(patches.len());

    for (patch, vector) in patches.iter().zip(&scaled) {
        let best_similarity = comparator
            .top_k(vector, args.top_k.max(1))?
            .first()
            .map(|neighbor| neighbor.score)
            .unwrap_or(0.0);
        let anomaly_score = (1.0 - best_similarity).clamp(0.0, 1.0);
        scores.push(anomaly_score);
        scored_patches.push(ScoredPatch {
            anomaly_score,
            x: patch.x,
            y: patch.y,
            width: patch.width,
            height: patch.height,
        });
    }

    let max_patch_score = scores.iter().copied().max_by(f64::total_cmp).unwrap_or(0.0);
    let mean_top_patch_score = mean_top_fraction(&scores, 0.10);
    let score = mean_top_patch_score;

    Ok(ScannedImage {
        result: PatchScanImageResult {
            path: entry.path.display().to_string(),
            label: entry.class_name.clone(),
            source: entry.source_name.clone(),
            expected_anomaly,
            predicted_anomaly: false,
            score,
            max_patch_score,
            mean_top_patch_score,
            patch_count: scores.len(),
            mask_path: mask_path.map(|path| path.display().to_string()),
            mask_iou: None,
        },
        patches: scored_patches,
    })
}

fn sliding_positions(size: u32, patch: u32, stride: u32) -> Vec<u32> {
    if size <= patch {
        return vec![0];
    }
    let stride = stride.max(1);
    let last = size - patch;
    let mut positions = Vec::new();
    let mut value = 0;
    while value < last {
        positions.push(value);
        value = value.saturating_add(stride);
    }
    positions.push(last);
    positions.sort_unstable();
    positions.dedup();
    positions
}

fn cap_reference_patches(mut vectors: Vec<Vec<f64>>, limit: usize) -> Vec<Vec<f64>> {
    if limit == 0 || vectors.len() <= limit {
        return vectors;
    }
    let step = vectors.len() as f64 / limit as f64;
    let last_index = vectors.len() - 1;
    (0..limit)
        .map(|index| {
            let source_index = (index as f64 * step).floor() as usize;
            std::mem::take(&mut vectors[source_index.min(last_index)])
        })
        .collect()
}

fn collect_entries(entries: &[ImageDatasetEntry], indexes: &[usize]) -> Vec<ImageDatasetEntry> {
    indexes
        .iter()
        .filter_map(|index| entries.get(*index).cloned())
        .collect()
}

fn mask_path_for_entry(
    entry: &ImageDatasetEntry,
    masks: &BTreeMap<String, Vec<PathBuf>>,
) -> Option<PathBuf> {
    let mask_roots = masks.get(&entry.source_name)?;
    let stem = entry.path.file_stem()?.to_str()?;
    let candidates = [
        format!("{stem}_mask.png"),
        format!("{stem}.png"),
        format!("{stem}_mask.jpg"),
        format!("{stem}.jpg"),
    ];
    for root in mask_roots {
        for candidate in &candidates {
            let path = root.join(candidate);
            if path.exists() {
                return Some(path);
            }
        }
    }
    None
}

fn mask_iou(
    mask_path: &Path,
    patches: &[ScoredPatch],
    threshold: f64,
) -> Result<f64, Box<dyn Error>> {
    let mask = image::open(mask_path)?.to_luma8();
    let predicted = predicted_mask(mask.width(), mask.height(), patches, threshold);
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (truth, predicted) in mask.pixels().zip(predicted.pixels()) {
        let truth_active = truth.0[0] > 0;
        let predicted_active = predicted.0[0] > 0;
        if truth_active && predicted_active {
            intersection += 1;
        }
        if truth_active || predicted_active {
            union += 1;
        }
    }
    Ok(if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    })
}

fn predicted_mask(width: u32, height: u32, patches: &[ScoredPatch], threshold: f64) -> GrayImage {
    let mut mask = GrayImage::new(width, height);
    for patch in patches {
        if patch.anomaly_score < threshold {
            continue;
        }
        let x_end = (patch.x + patch.width).min(width);
        let y_end = (patch.y + patch.height).min(height);
        for y in patch.y..y_end {
            for x in patch.x..x_end {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }
    }
    mask
}

fn mean_top_fraction(values: &[f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    let count = ((sorted.len() as f64) * fraction.clamp(0.0, 1.0))
        .ceil()
        .max(1.0) as usize;
    sorted.iter().take(count).sum::<f64>() / count as f64
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

fn binary_accuracy(results: &[PatchScanImageResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let correct = results
        .iter()
        .filter(|result| result.expected_anomaly == result.predicted_anomaly)
        .count();
    correct as f64 / results.len() as f64
}

fn class_binary_accuracy(results: &[PatchScanImageResult], anomaly: bool) -> f64 {
    let mut total = 0usize;
    let mut correct = 0usize;
    for result in results
        .iter()
        .filter(|result| result.expected_anomaly == anomaly)
    {
        total += 1;
        if result.predicted_anomaly == anomaly {
            correct += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    }
}

fn binary_auroc(results: &[PatchScanImageResult]) -> f64 {
    let positives = results
        .iter()
        .filter(|result| result.expected_anomaly)
        .map(|result| result.score)
        .collect::<Vec<_>>();
    let negatives = results
        .iter()
        .filter(|result| !result.expected_anomaly)
        .map(|result| result.score)
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

fn mean_option(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let values = values.flatten().collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn write_patch_debug(
    debug_out: &str,
    results: &[PatchScanImageResult],
) -> Result<(), Box<dyn Error>> {
    let out = Path::new(debug_out);
    fs::create_dir_all(out)?;
    let mut writer = csv::Writer::from_path(out.join("patch_scan_predictions.csv"))?;
    for result in results {
        writer.serialize(result)?;
    }
    writer.flush()?;
    Ok(())
}

fn validate_patch_args(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.patch_size == 0 {
        return Err("--patch-size must be greater than zero".into());
    }
    if args.patch_stride == 0 {
        return Err("--patch-stride must be greater than zero".into());
    }
    if args.image_width == 0 || args.image_height == 0 {
        return Err("--image-size must be greater than zero".into());
    }
    if !(0.0..=1.0).contains(&args.anomaly_threshold_quantile) {
        return Err("--anomaly-threshold-quantile must be in 0..1".into());
    }
    Ok(())
}
