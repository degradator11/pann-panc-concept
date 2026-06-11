use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use image::imageops::FilterType;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

use crate::preprocess::Dataset;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageVectorConfig {
    pub width: u32,
    pub height: u32,
    pub invert: bool,
}

impl ImageVectorConfig {
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            invert: false,
        }
    }

    pub const fn pixel_count(self) -> usize {
        (self.width * self.height) as usize
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
    Ok(image_to_vector(image, config))
}

pub fn load_image_folder(
    root: impl AsRef<Path>,
    config: ImageVectorConfig,
) -> Result<Dataset, VisionError> {
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

    for (label, class_dir) in class_dirs.into_iter().enumerate() {
        let class_name = class_dir.file_name().to_string_lossy().to_string();
        class_names.push(class_name.clone());

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

        let mut skipped_images = 0usize;
        for image_file in image_files {
            match load_image_as_vector(image_file.path(), config) {
                Ok(sample) => {
                    samples.push(sample);
                    labels.push(label);
                }
                Err(VisionError::DecodeImage { .. }) => skipped_images += 1,
                Err(error) => return Err(error),
            }
        }

        if skipped_images > 0 {
            eprintln!("warning: skipped {skipped_images} unreadable images in class {class_name}");
        }
    }

    if samples.is_empty() {
        return Err(VisionError::NoImages);
    }

    Ok(Dataset {
        samples,
        labels,
        class_names,
    })
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

fn image_to_vector(image: image::DynamicImage, config: ImageVectorConfig) -> Vec<f64> {
    let grayscale = image
        .resize_exact(config.width, config.height, FilterType::Triangle)
        .to_luma8();
    grayscale
        .pixels()
        .map(|pixel| {
            let value = f64::from(pixel.0[0]) / 255.0;
            if config.invert { 1.0 - value } else { value }
        })
        .collect()
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

    values
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
    use image::{GrayImage, Luma};
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
                .all(|sample| sample.len() == config.pixel_count())
        );
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
        fs::create_dir_all(&dark_dir).unwrap();
        fs::create_dir_all(&light_dir).unwrap();

        let dark = GrayImage::from_pixel(4, 4, Luma([0]));
        let light = GrayImage::from_pixel(4, 4, Luma([255]));
        dark.save(dark_dir.join("sample.png")).unwrap();
        light.save(light_dir.join("sample.png")).unwrap();

        let dataset = load_image_folder(&root, ImageVectorConfig::new(2, 2)).unwrap();

        assert_eq!(dataset.samples.len(), 2);
        assert_eq!(dataset.class_names, vec!["dark", "light"]);
        assert_eq!(dataset.samples[0].len(), 4);
        fs::remove_dir_all(root).unwrap();
    }
}
