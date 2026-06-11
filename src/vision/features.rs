use image::imageops::{FilterType, overlay};
use image::{DynamicImage, GrayImage, Rgb, RgbImage};

use super::{ImageFeatureMode, ImageResizeMode, ImageVectorConfig};

const COLOR_BINS: usize = 8;
const HSV_BINS: usize = 8;
const SPATIAL_GRID: usize = 4;
const HOG_BINS: usize = 8;
const LBP_BINS: usize = 16;

pub(super) const COLOR_HISTOGRAM_LEN: usize = COLOR_BINS * 3;
pub(super) const HSV_HISTOGRAM_LEN: usize = HSV_BINS * 3;
pub(super) const COLOR_MOMENTS_LEN: usize = 12;
pub(super) const SPATIAL_STATS_LEN: usize = SPATIAL_GRID * SPATIAL_GRID * 2;
pub(super) const HOG_LEN: usize = SPATIAL_GRID * SPATIAL_GRID * HOG_BINS;
pub(super) const LBP_LEN: usize = SPATIAL_GRID * SPATIAL_GRID * LBP_BINS;
pub(super) const COMBINED_LEN: usize = COLOR_HISTOGRAM_LEN + SPATIAL_STATS_LEN + HOG_LEN;
pub(super) const RICH_LEN: usize = COMBINED_LEN + HSV_HISTOGRAM_LEN + COLOR_MOMENTS_LEN + LBP_LEN;

pub(super) struct ImageProcessingStep {
    pub name: &'static str,
    pub image: DynamicImage,
}

pub(super) fn image_to_vector(image: DynamicImage, config: ImageVectorConfig) -> Vec<f64> {
    let resized = prepare_image(image, config);
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
        ImageFeatureMode::Rich => {
            let rgb = resized.to_rgb8();
            let gray_values = grayscale_pixels(&resized.to_luma8(), config.invert);
            let mut features = color_histogram(&rgb, config.invert);
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
            features.extend(hsv_histogram(&rgb, config.invert));
            features.extend(color_moments(&rgb, config.invert));
            features.extend(lbp_features(
                &gray_values,
                config.width as usize,
                config.height as usize,
            ));
            features
        }
    }
}

fn prepare_image(image: DynamicImage, config: ImageVectorConfig) -> DynamicImage {
    match config.resize_mode {
        ImageResizeMode::Stretch => {
            image.resize_exact(config.width, config.height, FilterType::Triangle)
        }
        ImageResizeMode::CenterCrop => center_crop(image, config),
        ImageResizeMode::Letterbox => letterbox(image, config),
    }
}

pub(super) fn image_processing_steps(
    image: DynamicImage,
    config: ImageVectorConfig,
) -> Vec<ImageProcessingStep> {
    let mut steps = vec![ImageProcessingStep {
        name: "original",
        image: image.clone(),
    }];

    match config.resize_mode {
        ImageResizeMode::Stretch => {
            steps.push(ImageProcessingStep {
                name: "resize_exact",
                image: image.resize_exact(config.width, config.height, FilterType::Triangle),
            });
        }
        ImageResizeMode::CenterCrop => {
            let cropped = center_crop_only(&image);
            steps.push(ImageProcessingStep {
                name: "center_crop",
                image: cropped.clone(),
            });
            steps.push(ImageProcessingStep {
                name: "resize_exact",
                image: cropped.resize_exact(config.width, config.height, FilterType::Triangle),
            });
        }
        ImageResizeMode::Letterbox => {
            let contained = image.resize(config.width, config.height, FilterType::Triangle);
            steps.push(ImageProcessingStep {
                name: "resize_contain",
                image: contained.clone(),
            });
            steps.push(ImageProcessingStep {
                name: "letterbox",
                image: letterbox_from_resized(contained, config),
            });
        }
    }

    steps
}

fn center_crop(image: DynamicImage, config: ImageVectorConfig) -> DynamicImage {
    center_crop_only(&image).resize_exact(config.width, config.height, FilterType::Triangle)
}

fn letterbox(image: DynamicImage, config: ImageVectorConfig) -> DynamicImage {
    let resized = image.resize(config.width, config.height, FilterType::Triangle);
    letterbox_from_resized(resized, config)
}

fn center_crop_only(image: &DynamicImage) -> DynamicImage {
    let crop_size = image.width().min(image.height()).max(1);
    let left = (image.width().saturating_sub(crop_size)) / 2;
    let top = (image.height().saturating_sub(crop_size)) / 2;
    image.crop_imm(left, top, crop_size, crop_size)
}

fn letterbox_from_resized(resized: DynamicImage, config: ImageVectorConfig) -> DynamicImage {
    let mut canvas = RgbImage::from_pixel(config.width, config.height, Rgb([127, 127, 127]));
    let x = (config.width.saturating_sub(resized.width()) / 2) as i64;
    let y = (config.height.saturating_sub(resized.height()) / 2) as i64;
    overlay(&mut canvas, &resized.to_rgb8(), x, y);
    DynamicImage::ImageRgb8(canvas)
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
        ImageFeatureMode::Rich => {
            let mut features = color_histogram_from_gray(&values);
            features.extend(spatial_intensity_stats(&values, width, height));
            features.extend(hog_features(&values, width, height));
            features.extend(hsv_histogram_from_gray(&values));
            features.extend(gray_color_moments(&values));
            features.extend(lbp_features(&values, width, height));
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

fn hsv_histogram(image: &RgbImage, invert: bool) -> Vec<f64> {
    let mut features = vec![0.0; HSV_HISTOGRAM_LEN];
    let pixel_count = f64::from(image.width() * image.height()).max(1.0);

    for pixel in image.pixels() {
        let r = channel_value(pixel.0[0], invert);
        let g = channel_value(pixel.0[1], invert);
        let b = channel_value(pixel.0[2], invert);
        let [hue, saturation, value] = rgb_to_hsv(r, g, b);
        for (channel, value) in [hue, saturation, value].into_iter().enumerate() {
            let bin = (value.clamp(0.0, 1.0) * HSV_BINS as f64).floor() as usize;
            let bin = bin.min(HSV_BINS - 1);
            features[channel * HSV_BINS + bin] += 1.0 / pixel_count;
        }
    }

    features
}

fn hsv_histogram_from_gray(values: &[f64]) -> Vec<f64> {
    let mut features = vec![0.0; HSV_HISTOGRAM_LEN];
    let pixel_count = values.len().max(1) as f64;

    for value in values {
        let value = value.clamp(0.0, 1.0);
        let bin = (value * HSV_BINS as f64).floor() as usize;
        let bin = bin.min(HSV_BINS - 1);
        features[bin] += 1.0 / pixel_count;
        features[HSV_BINS + bin] += 1.0 / pixel_count;
        features[2 * HSV_BINS + bin] += 1.0 / pixel_count;
    }

    features
}

fn color_moments(image: &RgbImage, invert: bool) -> Vec<f64> {
    let mut rgb_values = Vec::with_capacity((image.width() * image.height()) as usize);
    let mut hsv_values = Vec::with_capacity(rgb_values.capacity());

    for pixel in image.pixels() {
        let r = channel_value(pixel.0[0], invert);
        let g = channel_value(pixel.0[1], invert);
        let b = channel_value(pixel.0[2], invert);
        rgb_values.push([r, g, b]);
        hsv_values.push(rgb_to_hsv(r, g, b));
    }

    let mut features = channel_mean_std(&rgb_values);
    features.extend(channel_mean_std(&hsv_values));
    features
}

fn gray_color_moments(values: &[f64]) -> Vec<f64> {
    let triples = values
        .iter()
        .map(|value| {
            let value = value.clamp(0.0, 1.0);
            [value, value, value]
        })
        .collect::<Vec<_>>();
    let mut features = channel_mean_std(&triples);
    features.extend(channel_mean_std(&triples));
    features
}

fn channel_mean_std(values: &[[f64; 3]]) -> Vec<f64> {
    let count = values.len().max(1) as f64;
    let mut means = [0.0; 3];
    let mut squares = [0.0; 3];

    for value in values {
        for channel in 0..3 {
            means[channel] += value[channel];
            squares[channel] += value[channel] * value[channel];
        }
    }

    let mut features = Vec::with_capacity(6);
    for channel in 0..3 {
        let mean = means[channel] / count;
        let variance = (squares[channel] / count - mean * mean).max(0.0);
        features.push(mean);
        features.push(variance.sqrt());
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

fn lbp_features(values: &[f64], width: usize, height: usize) -> Vec<f64> {
    let mut features = vec![0.0; LBP_LEN];
    if width < 3 || height < 3 {
        return features;
    }

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center_index = y * width + x;
            let Some(center) = values.get(center_index).copied() else {
                continue;
            };

            let mut code = 0u8;
            let neighbors = [
                (x - 1, y - 1),
                (x, y - 1),
                (x + 1, y - 1),
                (x + 1, y),
                (x + 1, y + 1),
                (x, y + 1),
                (x - 1, y + 1),
                (x - 1, y),
            ];
            for (bit, (neighbor_x, neighbor_y)) in neighbors.into_iter().enumerate() {
                let neighbor_index = neighbor_y * width + neighbor_x;
                if values.get(neighbor_index).copied().unwrap_or(0.0) >= center {
                    code |= 1 << bit;
                }
            }

            let cell = spatial_cell(x, y, width, height);
            let bin = usize::from(code) * LBP_BINS / 256;
            features[cell * LBP_BINS + bin.min(LBP_BINS - 1)] += 1.0;
        }
    }

    for cell in 0..SPATIAL_GRID * SPATIAL_GRID {
        let start = cell * LBP_BINS;
        let end = start + LBP_BINS;
        let sum = features[start..end].iter().sum::<f64>();
        if sum > f64::EPSILON {
            for value in &mut features[start..end] {
                *value /= sum;
            }
        }
    }

    features
}

fn channel_value(value: u8, invert: bool) -> f64 {
    let value = f64::from(value) / 255.0;
    if invert { 1.0 - value } else { value }
}

fn rgb_to_hsv(r: f64, g: f64, b: f64) -> [f64; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let hue = if delta <= f64::EPSILON {
        0.0
    } else if (max - r).abs() <= f64::EPSILON {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if (max - g).abs() <= f64::EPSILON {
        (((b - r) / delta) + 2.0) / 6.0
    } else {
        (((r - g) / delta) + 4.0) / 6.0
    };
    let saturation = if max <= f64::EPSILON {
        0.0
    } else {
        delta / max
    };

    [hue.clamp(0.0, 1.0), saturation.clamp(0.0, 1.0), max]
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
