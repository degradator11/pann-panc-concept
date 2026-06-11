use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

use crate::preprocess::Dataset;

mod features;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageVectorConfig {
    pub width: u32,
    pub height: u32,
    pub invert: bool,
    pub feature_mode: ImageFeatureMode,
    pub resize_mode: ImageResizeMode,
}

impl ImageVectorConfig {
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            invert: false,
            feature_mode: ImageFeatureMode::Pixels,
            resize_mode: ImageResizeMode::Stretch,
        }
    }

    pub const fn pixel_count(self) -> usize {
        (self.width * self.height) as usize
    }

    pub const fn with_feature_mode(mut self, feature_mode: ImageFeatureMode) -> Self {
        self.feature_mode = feature_mode;
        self
    }

    pub const fn with_resize_mode(mut self, resize_mode: ImageResizeMode) -> Self {
        self.resize_mode = resize_mode;
        self
    }

    pub const fn vector_len(self) -> usize {
        match self.feature_mode {
            ImageFeatureMode::Pixels => self.pixel_count(),
            ImageFeatureMode::ColorHistogram => features::COLOR_HISTOGRAM_LEN,
            ImageFeatureMode::Hog => features::HOG_LEN,
            ImageFeatureMode::Combined => features::COMBINED_LEN,
            ImageFeatureMode::Rich => features::RICH_LEN,
            ImageFeatureMode::RichSpatial => features::RICH_SPATIAL_LEN,
            ImageFeatureMode::RichNormalized => features::RICH_NORMALIZED_LEN,
            ImageFeatureMode::RichHog => features::RICH_HOG_LEN,
            ImageFeatureMode::RichTexture => features::RICH_TEXTURE_LEN,
            ImageFeatureMode::RichEdge => features::RICH_EDGE_LEN,
            ImageFeatureMode::RichLayout => features::RICH_LAYOUT_LEN,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFeatureMode {
    Pixels,
    ColorHistogram,
    Hog,
    Combined,
    Rich,
    RichSpatial,
    RichNormalized,
    RichHog,
    RichTexture,
    RichEdge,
    RichLayout,
}

impl ImageFeatureMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pixels => "pixels",
            Self::ColorHistogram => "color",
            Self::Hog => "hog",
            Self::Combined => "combined",
            Self::Rich => "rich",
            Self::RichSpatial => "rich-spatial",
            Self::RichNormalized => "rich-normalized",
            Self::RichHog => "rich-hog",
            Self::RichTexture => "rich-texture",
            Self::RichEdge => "rich-edge",
            Self::RichLayout => "rich-layout",
        }
    }
}

impl FromStr for ImageFeatureMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "pixels" | "raw" => Ok(Self::Pixels),
            "color" | "histogram" | "color-histogram" => Ok(Self::ColorHistogram),
            "hog" | "edges" => Ok(Self::Hog),
            "combined" => Ok(Self::Combined),
            "rich" | "enhanced" => Ok(Self::Rich),
            "rich-spatial" | "spatial-rich" | "rich_spatial" => Ok(Self::RichSpatial),
            "rich-normalized" | "rich-normalised" | "rich-norm" | "rich_normalized" => {
                Ok(Self::RichNormalized)
            }
            "rich-hog" | "rich_hog" | "rich-block-hog" | "rich-blockhog" => Ok(Self::RichHog),
            "rich-texture" | "rich_texture" | "rich-lbp" | "rich_lbp" => Ok(Self::RichTexture),
            "rich-edge" | "rich_edges" | "rich-edge-density" | "rich_edge" => Ok(Self::RichEdge),
            "rich-layout" | "rich_layout" | "rich-symmetry" | "rich_symmetry" => {
                Ok(Self::RichLayout)
            }
            other => Err(format!(
                "invalid image feature mode {other:?}; expected pixels, color, hog, combined, rich, rich-spatial, rich-normalized, rich-hog, rich-texture, rich-edge, or rich-layout"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageResizeMode {
    Stretch,
    CenterCrop,
    Letterbox,
    ForegroundCrop,
}

impl ImageResizeMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stretch => "stretch",
            Self::CenterCrop => "center-crop",
            Self::Letterbox => "letterbox",
            Self::ForegroundCrop => "foreground-crop",
        }
    }
}

impl FromStr for ImageResizeMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "stretch" | "resize" | "resize-exact" => Ok(Self::Stretch),
            "center-crop" | "centercrop" | "crop" => Ok(Self::CenterCrop),
            "letterbox" | "contain" | "pad" | "padding" => Ok(Self::Letterbox),
            "foreground-crop" | "foreground" | "object-crop" | "object" => Ok(Self::ForegroundCrop),
            other => Err(format!(
                "invalid image resize mode {other:?}; expected stretch, center-crop, letterbox, or foreground-crop"
            )),
        }
    }
}

#[derive(Debug, Error)]
pub enum VisionError {
    #[error("image width and height must be greater than zero")]
    InvalidImageSize,
    #[error("image dataset root does not exist: {0}")]
    MissingRoot(PathBuf),
    #[error("image dataset root must contain one subdirectory per class")]
    NoClassDirectories,
    #[error("image dataset contains no supported image files")]
    NoImages,
    #[error("failed to read directory {path}: {source}")]
    ReadDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to decode image {path}: {source}")]
    DecodeImage {
        path: PathBuf,
        source: image::ImageError,
    },
}

#[derive(Debug, Clone)]
pub struct ImageProcessingStep {
    pub name: String,
    pub image: image::DynamicImage,
}

pub fn load_image_as_vector(
    path: impl AsRef<Path>,
    config: ImageVectorConfig,
) -> Result<Vec<f64>, VisionError> {
    validate_config(config)?;
    let path = path.as_ref();
    let image = image::open(path).map_err(|source| VisionError::DecodeImage {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(features::image_to_vector(image, config))
}

pub fn load_image_processing_steps(
    path: impl AsRef<Path>,
    config: ImageVectorConfig,
) -> Result<Vec<ImageProcessingStep>, VisionError> {
    validate_config(config)?;
    let path = path.as_ref();
    let image = image::open(path).map_err(|source| VisionError::DecodeImage {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(features::image_processing_steps(image, config)
        .into_iter()
        .map(|step| ImageProcessingStep {
            name: step.name.to_string(),
            image: step.image,
        })
        .collect())
}

pub fn load_image_folder(
    root: impl AsRef<Path>,
    config: ImageVectorConfig,
) -> Result<Dataset, VisionError> {
    Ok(load_image_folder_with_paths(root, config)?.dataset)
}

pub fn load_image_folder_with_paths(
    root: impl AsRef<Path>,
    config: ImageVectorConfig,
) -> Result<ImageFolderDataset, VisionError> {
    validate_config(config)?;
    let root = root.as_ref();
    if !root.exists() {
        return Err(VisionError::MissingRoot(root.to_path_buf()));
    }

    let mut class_dirs = fs::read_dir(root)
        .map_err(|source| VisionError::ReadDir {
            path: root.to_path_buf(),
            source,
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| VisionError::ReadDir {
            path: root.to_path_buf(),
            source,
        })?
        .into_iter()
        .filter(|entry| entry.path().is_dir())
        .collect::<Vec<_>>();
    class_dirs.sort_by_key(|entry| entry.file_name());

    if class_dirs.is_empty() {
        return Err(VisionError::NoClassDirectories);
    }

    let mut samples = Vec::new();
    let mut labels = Vec::new();
    let mut class_names = Vec::new();
    let mut image_paths = Vec::new();

    for class_dir in class_dirs {
        let class_name = class_dir.file_name().to_string_lossy().to_string();
        let mut image_files = fs::read_dir(class_dir.path())
            .map_err(|source| VisionError::ReadDir {
                path: class_dir.path(),
                source,
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| VisionError::ReadDir {
                path: class_dir.path(),
                source,
            })?
            .into_iter()
            .filter(|entry| supported_image_path(&entry.path()))
            .collect::<Vec<_>>();
        image_files.sort_by_key(|entry| entry.file_name());
        if image_files.is_empty() {
            continue;
        }

        let label = class_names.len();
        let sample_count_before = samples.len();
        let mut skipped_images = 0usize;
        for image_file in image_files {
            let image_path = image_file.path();
            match load_image_as_vector(&image_path, config) {
                Ok(sample) => {
                    samples.push(sample);
                    labels.push(label);
                    image_paths.push(image_path);
                }
                Err(VisionError::DecodeImage { .. }) => skipped_images += 1,
                Err(error) => return Err(error),
            }
        }

        if samples.len() > sample_count_before {
            class_names.push(class_name.clone());
        }

        if skipped_images > 0 {
            eprintln!("warning: skipped {skipped_images} unreadable images in class {class_name}");
        }
    }

    if samples.is_empty() {
        return Err(VisionError::NoImages);
    }

    Ok(ImageFolderDataset {
        dataset: Dataset {
            samples,
            labels,
            class_names,
        },
        image_paths,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageFolderDataset {
    pub dataset: Dataset,
    pub image_paths: Vec<PathBuf>,
}

pub fn synthetic_image_dataset(
    config: ImageVectorConfig,
    samples_per_class: usize,
    seed: u64,
) -> Result<Dataset, VisionError> {
    validate_config(config)?;
    let mut samples = Vec::with_capacity(samples_per_class * 3);
    let mut labels = Vec::with_capacity(samples_per_class * 3);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for label in 0..3 {
        for _ in 0..samples_per_class {
            samples.push(synthetic_pattern(label, config, &mut rng));
            labels.push(label);
        }
    }

    Ok(Dataset {
        samples,
        labels,
        class_names: vec![
            "vertical".to_string(),
            "horizontal".to_string(),
            "diagonal".to_string(),
        ],
    })
}

pub fn class_counts(dataset: &Dataset) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for label in &dataset.labels {
        let class_name = dataset
            .class_names
            .get(*label)
            .cloned()
            .unwrap_or_else(|| format!("class-{label}"));
        *counts.entry(class_name).or_insert(0) += 1;
    }
    counts
}

fn synthetic_pattern(label: usize, config: ImageVectorConfig, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let width = config.width as usize;
    let height = config.height as usize;
    let mut values = vec![0.0; width * height];
    let jitter_x = centered_jitter(width, rng);
    let jitter_y = centered_jitter(height, rng);

    for y in 0..height {
        for x in 0..width {
            let signal = match label {
                0 => x.abs_diff(jitter_x) <= 1,
                1 => y.abs_diff(jitter_y) <= 1,
                _ => x.abs_diff(y) <= 1 || x.abs_diff(width.saturating_sub(1) - y) <= 1,
            };
            let noise = rng.gen_range(0.0_f64..0.12_f64);
            let value = if signal {
                0.85 + rng.gen_range(0.0_f64..0.15_f64)
            } else {
                noise
            };
            values[y * width + x] = value.clamp(0.0, 1.0);
        }
    }

    if config.invert {
        for value in &mut values {
            *value = 1.0 - *value;
        }
    }

    features::vectorize_grayscale_values(&values, config)
}

fn centered_jitter(size: usize, rng: &mut ChaCha8Rng) -> usize {
    if size <= 2 {
        return size / 2;
    }

    let center = size / 2;
    let offset = rng.gen_range(-1_i32..=1_i32);
    (center as i32 + offset).clamp(0, size.saturating_sub(1) as i32) as usize
}

fn validate_config(config: ImageVectorConfig) -> Result<(), VisionError> {
    if config.width == 0 || config.height == 0 {
        Err(VisionError::InvalidImageSize)
    } else {
        Ok(())
    }
}

fn supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgb, RgbImage};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn synthetic_image_dataset_has_expected_dimensions_and_classes() {
        let config = ImageVectorConfig::new(8, 8);
        let dataset = synthetic_image_dataset(config, 4, 9).unwrap();

        assert_eq!(dataset.samples.len(), 12);
        assert_eq!(dataset.labels.len(), 12);
        assert_eq!(
            dataset.class_names,
            vec!["vertical", "horizontal", "diagonal"]
        );
        assert!(
            dataset
                .samples
                .iter()
                .all(|sample| sample.len() == config.vector_len())
        );
    }

    #[test]
    fn combined_feature_mode_has_expected_length() {
        let config = ImageVectorConfig::new(8, 8).with_feature_mode(ImageFeatureMode::Combined);
        let dataset = synthetic_image_dataset(config, 1, 9).unwrap();

        assert_eq!(dataset.samples[0].len(), config.vector_len());
        assert_eq!(config.vector_len(), 184);
    }

    #[test]
    fn rich_feature_mode_has_expected_length() {
        let config = ImageVectorConfig::new(8, 8).with_feature_mode(ImageFeatureMode::Rich);
        let dataset = synthetic_image_dataset(config, 1, 9).unwrap();

        assert_eq!(dataset.samples[0].len(), config.vector_len());
        assert_eq!(config.vector_len(), 476);
    }

    #[test]
    fn rich_spatial_feature_mode_has_expected_length() {
        let config = ImageVectorConfig::new(8, 8).with_feature_mode(ImageFeatureMode::RichSpatial);
        let dataset = synthetic_image_dataset(config, 1, 9).unwrap();

        assert_eq!(dataset.samples[0].len(), config.vector_len());
        assert_eq!(config.vector_len(), 860);
    }

    #[test]
    fn rich_normalized_feature_mode_has_expected_length() {
        let config =
            ImageVectorConfig::new(8, 8).with_feature_mode(ImageFeatureMode::RichNormalized);
        let dataset = synthetic_image_dataset(config, 1, 9).unwrap();

        assert_eq!(dataset.samples[0].len(), config.vector_len());
        assert_eq!(config.vector_len(), 866);
    }

    #[test]
    fn rich_hog_feature_mode_has_expected_length() {
        let config = ImageVectorConfig::new(8, 8).with_feature_mode(ImageFeatureMode::RichHog);
        let dataset = synthetic_image_dataset(config, 1, 9).unwrap();

        assert_eq!(dataset.samples[0].len(), config.vector_len());
        assert_eq!(config.vector_len(), 1154);
    }

    #[test]
    fn rich_texture_feature_mode_has_expected_length() {
        let config = ImageVectorConfig::new(8, 8).with_feature_mode(ImageFeatureMode::RichTexture);
        let dataset = synthetic_image_dataset(config, 1, 9).unwrap();

        assert_eq!(dataset.samples[0].len(), config.vector_len());
        assert_eq!(config.vector_len(), 1410);
    }

    #[test]
    fn rich_edge_feature_mode_has_expected_length() {
        let config = ImageVectorConfig::new(8, 8).with_feature_mode(ImageFeatureMode::RichEdge);
        let dataset = synthetic_image_dataset(config, 1, 9).unwrap();

        assert_eq!(dataset.samples[0].len(), config.vector_len());
        assert_eq!(config.vector_len(), 1458);
    }

    #[test]
    fn rich_layout_feature_mode_has_expected_length() {
        let config = ImageVectorConfig::new(8, 8).with_feature_mode(ImageFeatureMode::RichLayout);
        let dataset = synthetic_image_dataset(config, 1, 9).unwrap();

        assert_eq!(dataset.samples[0].len(), config.vector_len());
        assert_eq!(config.vector_len(), 1426);
    }

    #[test]
    fn resize_mode_parses_expected_aliases() {
        assert_eq!(
            "stretch".parse::<ImageResizeMode>().unwrap(),
            ImageResizeMode::Stretch
        );
        assert_eq!(
            "crop".parse::<ImageResizeMode>().unwrap(),
            ImageResizeMode::CenterCrop
        );
        assert_eq!(
            "contain".parse::<ImageResizeMode>().unwrap(),
            ImageResizeMode::Letterbox
        );
        assert_eq!(
            "object-crop".parse::<ImageResizeMode>().unwrap(),
            ImageResizeMode::ForegroundCrop
        );
    }

    #[test]
    fn letterbox_resize_preserves_aspect_with_neutral_padding() {
        let root = std::env::temp_dir().join(format!(
            "progress_ai_letterbox_test_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let image_path = root.join("wide.png");
        let image = RgbImage::from_pixel(4, 2, Rgb([255, 255, 255]));
        image.save(&image_path).unwrap();

        let stretch = load_image_as_vector(&image_path, ImageVectorConfig::new(4, 4)).unwrap();
        let letterbox = load_image_as_vector(
            &image_path,
            ImageVectorConfig::new(4, 4).with_resize_mode(ImageResizeMode::Letterbox),
        )
        .unwrap();

        assert!(stretch.iter().all(|value| *value > 0.99));
        assert!(letterbox[0] > 0.49 && letterbox[0] < 0.51);
        assert!(letterbox[5] > 0.99);
        assert!(letterbox[10] > 0.99);
        assert!(letterbox[15] > 0.49 && letterbox[15] < 0.51);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn foreground_crop_removes_simple_border_before_resize() {
        let root = std::env::temp_dir().join(format!(
            "progress_ai_foreground_crop_test_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let image_path = root.join("object.png");
        let mut image = RgbImage::from_pixel(8, 8, Rgb([255, 255, 255]));
        for y in 2..6 {
            for x in 2..6 {
                image.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        image.save(&image_path).unwrap();

        let steps = load_image_processing_steps(
            &image_path,
            ImageVectorConfig::new(4, 4).with_resize_mode(ImageResizeMode::ForegroundCrop),
        )
        .unwrap();
        let names = steps
            .iter()
            .map(|step| step.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["original", "foreground_crop", "resize_exact"]);
        assert_eq!(steps[1].image.width(), 6);
        assert_eq!(steps[1].image.height(), 6);
        assert_eq!(steps.last().unwrap().image.width(), 4);
        assert_eq!(steps.last().unwrap().image.height(), 4);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn processing_steps_include_original_and_resize_steps() {
        let root = std::env::temp_dir().join(format!(
            "progress_ai_steps_test_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let image_path = root.join("wide.png");
        let image = RgbImage::from_pixel(4, 2, Rgb([255, 255, 255]));
        image.save(&image_path).unwrap();

        let steps = load_image_processing_steps(
            &image_path,
            ImageVectorConfig::new(4, 4).with_resize_mode(ImageResizeMode::CenterCrop),
        )
        .unwrap();
        let names = steps
            .iter()
            .map(|step| step.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["original", "center_crop", "resize_exact"]);
        assert_eq!(steps.last().unwrap().image.width(), 4);
        assert_eq!(steps.last().unwrap().image.height(), 4);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn class_counts_groups_labels_by_name() {
        let config = ImageVectorConfig::new(4, 4);
        let dataset = synthetic_image_dataset(config, 2, 1).unwrap();
        let counts = class_counts(&dataset);

        assert_eq!(counts["vertical"], 2);
        assert_eq!(counts["horizontal"], 2);
        assert_eq!(counts["diagonal"], 2);
    }

    #[test]
    fn image_folder_loader_reads_class_subdirectories() {
        let root = std::env::temp_dir().join(format!(
            "progress_ai_image_test_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let dark_dir = root.join("dark");
        let light_dir = root.join("light");
        let eval_dir = root.join("eval").join("dark");
        fs::create_dir_all(&dark_dir).unwrap();
        fs::create_dir_all(&light_dir).unwrap();
        fs::create_dir_all(&eval_dir).unwrap();

        let dark = GrayImage::from_pixel(4, 4, Luma([0]));
        let light = GrayImage::from_pixel(4, 4, Luma([255]));
        dark.save(dark_dir.join("sample.png")).unwrap();
        light.save(light_dir.join("sample.png")).unwrap();
        dark.save(eval_dir.join("sample.png")).unwrap();

        let dataset = load_image_folder(&root, ImageVectorConfig::new(2, 2)).unwrap();

        assert_eq!(dataset.samples.len(), 2);
        assert_eq!(dataset.class_names, vec!["dark", "light"]);
        assert_eq!(dataset.samples[0].len(), 4);
        fs::remove_dir_all(root).unwrap();
    }
}
