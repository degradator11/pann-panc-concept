use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, RgbImage};

use super::{ImageFeatureMode, ImageVectorConfig};

const COLOR_BINS: usize = 8;
const SPATIAL_GRID: usize = 4;
const HOG_BINS: usize = 8;

pub(super) const COLOR_HISTOGRAM_LEN: usize = COLOR_BINS * 3;
pub(super) const SPATIAL_STATS_LEN: usize = SPATIAL_GRID * SPATIAL_GRID * 2;
pub(super) const HOG_LEN: usize = SPATIAL_GRID * SPATIAL_GRID * HOG_BINS;
pub(super) const COMBINED_LEN: usize = COLOR_HISTOGRAM_LEN + SPATIAL_STATS_LEN + HOG_LEN;

pub(super) fn image_to_vector(image: DynamicImage, config: ImageVectorConfig) -> Vec<f64> {
    let resized = image.resize_exact(config.width, config.height, FilterType::Triangle);
    match config.feature_mode {
        ImageFeatureMode::Pixels => grayscale_pixels(&resized.to_luma8(), config.invert),
        ImageFeatureMode::ColorHistogram => color_histogram(&resized.to_rgb8(), config.invert),
        ImageFeatureMode::Hog => {
            let gray_values = grayscale_pixels(&resized.to_luma8(), config.invert);
            hog_features(&gray_values, config.width as usize, config.height as usize)
        }
        ImageFeatureMode::Combined => {
            let gray_values = grayscale_pixels(&resized.to_luma8(), config.invert);
            let mut features = color_histogram(&resized.to_rgb8(), config.invert);
            features.extend(spatial_intensity_stats(
                &gray_values,
                config.width as usize,
                config.height as usize,
            ));
            features.extend(hog_features(
                &gray_values,
                config.width as usize,
                config.height as usize,
            ));
            features
        }
    }
}

pub(super) fn vectorize_grayscale_values(values: &[f64], config: ImageVectorConfig) -> Vec<f64> {
    let width = config.width as usize;
    let height = config.height as usize;
    let values = if config.invert {
        values.iter().map(|value| 1.0 - value).collect::<Vec<_>>()
    } else {
        values.to_vec()
    };

    match config.feature_mode {
        ImageFeatureMode::Pixels => values,
        ImageFeatureMode::ColorHistogram => color_histogram_from_gray(&values),
        ImageFeatureMode::Hog => hog_features(&values, width, height),
        ImageFeatureMode::Combined => {
            let mut features = color_histogram_from_gray(&values);
            features.extend(spatial_intensity_stats(&values, width, height));
            features.extend(hog_features(&values, width, height));
            features
        }
    }
}

fn grayscale_pixels(image: &GrayImage, invert: bool) -> Vec<f64> {
    image
        .pixels()
        .map(|pixel| {
            let value = f64::from(pixel.0[0]) / 255.0;
            if invert { 1.0 - value } else { value }
        })
        .collect()
}

fn color_histogram(image: &RgbImage, invert: bool) -> Vec<f64> {
    let mut features = vec![0.0; COLOR_HISTOGRAM_LEN];
    let pixel_count = f64::from(image.width() * image.height()).max(1.0);

    for pixel in image.pixels() {
        for channel in 0..3 {
            let raw = if invert {
                255 - pixel.0[channel]
            } else {
                pixel.0[channel]
            };
            let bin = (usize::from(raw) * COLOR_BINS / 256).min(COLOR_BINS - 1);
            features[channel * COLOR_BINS + bin] += 1.0 / pixel_count;
        }
    }

    features
}

fn color_histogram_from_gray(values: &[f64]) -> Vec<f64> {
    let mut features = vec![0.0; COLOR_HISTOGRAM_LEN];
    let pixel_count = values.len().max(1) as f64;

    for value in values {
        let bin = ((*value).clamp(0.0, 1.0) * COLOR_BINS as f64).floor() as usize;
        let bin = bin.min(COLOR_BINS - 1);
        for channel in 0..3 {
            features[channel * COLOR_BINS + bin] += 1.0 / pixel_count;
        }
    }

    features
}

fn spatial_intensity_stats(values: &[f64], width: usize, height: usize) -> Vec<f64> {
    let cell_count = SPATIAL_GRID * SPATIAL_GRID;
    let mut sums = vec![0.0; cell_count];
    let mut squares = vec![0.0; cell_count];
    let mut counts = vec![0usize; cell_count];

    for y in 0..height {
        for x in 0..width {
            let Some(value) = values.get(y * width + x).copied() else {
                continue;
            };
            let cell = spatial_cell(x, y, width, height);
            sums[cell] += value;
            squares[cell] += value * value;
            counts[cell] += 1;
        }
    }

    let mut features = Vec::with_capacity(SPATIAL_STATS_LEN);
    for cell in 0..cell_count {
        let count = counts[cell].max(1) as f64;
        let mean = sums[cell] / count;
        let variance = (squares[cell] / count - mean * mean).max(0.0);
        features.push(mean);
        features.push(variance.sqrt());
    }
    features
}

fn hog_features(values: &[f64], width: usize, height: usize) -> Vec<f64> {
    let mut features = vec![0.0; HOG_LEN];
    if width < 3 || height < 3 {
        return features;
    }

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = y * width + x;
            if center >= values.len() {
                continue;
            }

            let gx = values[y * width + x + 1] - values[y * width + x - 1];
            let gy = values[(y + 1) * width + x] - values[(y - 1) * width + x];
            let magnitude = (gx * gx + gy * gy).sqrt();
            if magnitude <= f64::EPSILON {
                continue;
            }

            let mut angle = gy.atan2(gx);
            while angle < 0.0 {
                angle += std::f64::consts::PI;
            }
            while angle >= std::f64::consts::PI {
                angle -= std::f64::consts::PI;
            }

            let bin = ((angle / std::f64::consts::PI) * HOG_BINS as f64).floor() as usize;
            let bin = bin.min(HOG_BINS - 1);
            let cell = spatial_cell(x, y, width, height);
            features[cell * HOG_BINS + bin] += magnitude;
        }
    }

    for cell in 0..SPATIAL_GRID * SPATIAL_GRID {
        let start = cell * HOG_BINS;
        let end = start + HOG_BINS;
        let sum = features[start..end].iter().sum::<f64>();
        if sum > f64::EPSILON {
            for value in &mut features[start..end] {
                *value /= sum;
            }
        }
    }

    features
}

fn spatial_cell(x: usize, y: usize, width: usize, height: usize) -> usize {
    let cell_x = (x * SPATIAL_GRID / width.max(1)).min(SPATIAL_GRID - 1);
    let cell_y = (y * SPATIAL_GRID / height.max(1)).min(SPATIAL_GRID - 1);
    cell_y * SPATIAL_GRID + cell_x
}
