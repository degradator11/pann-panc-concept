use image::imageops::{FilterType, overlay};
use image::{DynamicImage, GrayImage, Rgb, RgbImage};

use super::{ImageFeatureMode, ImageResizeMode, ImageVectorConfig};

const COLOR_BINS: usize = 8;
const HSV_BINS: usize = 8;
const SPATIAL_GRID: usize = 4;
const HOG_BINS: usize = 8;
const LBP_BINS: usize = 16;
const HOG_BLOCK_GRID: usize = SPATIAL_GRID - 1;

pub(super) const COLOR_HISTOGRAM_LEN: usize = COLOR_BINS * 3;
pub(super) const HSV_HISTOGRAM_LEN: usize = HSV_BINS * 3;
pub(super) const COLOR_MOMENTS_LEN: usize = 12;
pub(super) const NORMALIZED_COLOR_MOMENTS_LEN: usize = 6;
pub(super) const SPATIAL_STATS_LEN: usize = SPATIAL_GRID * SPATIAL_GRID * 2;
pub(super) const HOG_LEN: usize = SPATIAL_GRID * SPATIAL_GRID * HOG_BINS;
pub(super) const HOG_BLOCK_LEN: usize = HOG_BLOCK_GRID * HOG_BLOCK_GRID * 4 * HOG_BINS;
pub(super) const LBP_LEN: usize = SPATIAL_GRID * SPATIAL_GRID * LBP_BINS;
pub(super) const EDGE_DENSITY_LEN: usize = SPATIAL_GRID * SPATIAL_GRID * 3;
pub(super) const LAYOUT_LEN: usize = 16;
pub(super) const SPATIAL_HSV_LEN: usize = SPATIAL_GRID * SPATIAL_GRID * HSV_HISTOGRAM_LEN;
pub(super) const COMBINED_LEN: usize = COLOR_HISTOGRAM_LEN + SPATIAL_STATS_LEN + HOG_LEN;
pub(super) const RICH_LEN: usize = COMBINED_LEN + HSV_HISTOGRAM_LEN + COLOR_MOMENTS_LEN + LBP_LEN;
pub(super) const RICH_SPATIAL_LEN: usize = RICH_LEN + SPATIAL_HSV_LEN;
pub(super) const RICH_NORMALIZED_LEN: usize = RICH_SPATIAL_LEN + NORMALIZED_COLOR_MOMENTS_LEN;
pub(super) const RICH_HOG_LEN: usize = RICH_NORMALIZED_LEN + HOG_BLOCK_LEN;
pub(super) const RICH_TEXTURE_LEN: usize = RICH_HOG_LEN + LBP_LEN;
pub(super) const RICH_EDGE_LEN: usize = RICH_TEXTURE_LEN + EDGE_DENSITY_LEN;
pub(super) const RICH_LAYOUT_LEN: usize = RICH_TEXTURE_LEN + LAYOUT_LEN;

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
        ImageFeatureMode::Rich
        | ImageFeatureMode::RichSpatial
        | ImageFeatureMode::RichNormalized
        | ImageFeatureMode::RichHog
        | ImageFeatureMode::RichTexture
        | ImageFeatureMode::RichEdge
        | ImageFeatureMode::RichLayout => {
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
            if matches!(
                config.feature_mode,
                ImageFeatureMode::RichSpatial
                    | ImageFeatureMode::RichNormalized
                    | ImageFeatureMode::RichHog
                    | ImageFeatureMode::RichTexture
                    | ImageFeatureMode::RichEdge
                    | ImageFeatureMode::RichLayout
            ) {
                features.extend(spatial_hsv_histogram(&rgb, config.invert));
            }
            if matches!(
                config.feature_mode,
                ImageFeatureMode::RichNormalized
                    | ImageFeatureMode::RichHog
                    | ImageFeatureMode::RichTexture
                    | ImageFeatureMode::RichEdge
                    | ImageFeatureMode::RichLayout
            ) {
                features.extend(normalized_color_moments(&rgb, config.invert));
            }
            if matches!(
                config.feature_mode,
                ImageFeatureMode::RichHog
                    | ImageFeatureMode::RichTexture
                    | ImageFeatureMode::RichEdge
                    | ImageFeatureMode::RichLayout
            ) {
                features.extend(hog_block_features(
                    &gray_values,
                    config.width as usize,
                    config.height as usize,
                ));
            }
            if matches!(
                config.feature_mode,
                ImageFeatureMode::RichTexture
                    | ImageFeatureMode::RichEdge
                    | ImageFeatureMode::RichLayout
            ) {
                features.extend(lbp_radius_features(
                    &gray_values,
                    config.width as usize,
                    config.height as usize,
                    2,
                ));
            }
            if config.feature_mode == ImageFeatureMode::RichEdge {
                features.extend(edge_density_features(
                    &gray_values,
                    config.width as usize,
                    config.height as usize,
                ));
            }
            if config.feature_mode == ImageFeatureMode::RichLayout {
                features.extend(layout_features(
                    &gray_values,
                    config.width as usize,
                    config.height as usize,
                ));
            }
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
        ImageResizeMode::ForegroundCrop => foreground_crop(image, config),
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
        ImageResizeMode::ForegroundCrop => {
            let cropped = foreground_crop_only(&image);
            steps.push(ImageProcessingStep {
                name: "foreground_crop",
                image: cropped.clone(),
            });
            steps.push(ImageProcessingStep {
                name: "resize_exact",
                image: cropped.resize_exact(config.width, config.height, FilterType::Triangle),
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

fn foreground_crop(image: DynamicImage, config: ImageVectorConfig) -> DynamicImage {
    foreground_crop_only(&image).resize_exact(config.width, config.height, FilterType::Triangle)
}

fn center_crop_only(image: &DynamicImage) -> DynamicImage {
    let crop_size = image.width().min(image.height()).max(1);
    let left = (image.width().saturating_sub(crop_size)) / 2;
    let top = (image.height().saturating_sub(crop_size)) / 2;
    image.crop_imm(left, top, crop_size, crop_size)
}

fn foreground_crop_only(image: &DynamicImage) -> DynamicImage {
    let rgb = image.to_rgb8();
    let width = rgb.width();
    let height = rgb.height();
    if width < 3 || height < 3 {
        return image.clone();
    }

    let (background, threshold) = estimate_border_background(&rgb);
    let mut left = width;
    let mut top = height;
    let mut right = 0;
    let mut bottom = 0;
    let mut foreground_pixels = 0u32;

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            if normalized_rgb_distance(pixel, background) > threshold {
                left = left.min(x);
                top = top.min(y);
                right = right.max(x);
                bottom = bottom.max(y);
                foreground_pixels += 1;
            }
        }
    }

    let pixel_count = width * height;
    if foreground_pixels < (pixel_count / 100).max(4) {
        return center_crop_only(image);
    }

    let bbox_width = right - left + 1;
    let bbox_height = bottom - top + 1;
    let bbox_area = bbox_width * bbox_height;
    if bbox_area as f64 / f64::from(pixel_count) > 0.95 {
        return center_crop_only(image);
    }

    let padding_x = ((f64::from(bbox_width) * 0.08).ceil() as u32).max(1);
    let padding_y = ((f64::from(bbox_height) * 0.08).ceil() as u32).max(1);
    left = left.saturating_sub(padding_x);
    top = top.saturating_sub(padding_y);
    right = (right + padding_x).min(width - 1);
    bottom = (bottom + padding_y).min(height - 1);

    image.crop_imm(left, top, right - left + 1, bottom - top + 1)
}

fn letterbox_from_resized(resized: DynamicImage, config: ImageVectorConfig) -> DynamicImage {
    let mut canvas = RgbImage::from_pixel(config.width, config.height, Rgb([127, 127, 127]));
    let x = (config.width.saturating_sub(resized.width()) / 2) as i64;
    let y = (config.height.saturating_sub(resized.height()) / 2) as i64;
    overlay(&mut canvas, &resized.to_rgb8(), x, y);
    DynamicImage::ImageRgb8(canvas)
}

fn estimate_border_background(image: &RgbImage) -> ([f64; 3], f64) {
    let width = image.width();
    let height = image.height();
    let mut sums = [0.0; 3];
    let mut count = 0.0;

    for y in 0..height {
        for x in 0..width {
            if x != 0 && y != 0 && x != width - 1 && y != height - 1 {
                continue;
            }
            let pixel = image.get_pixel(x, y);
            for (channel, sum) in sums.iter_mut().enumerate() {
                *sum += f64::from(pixel.0[channel]) / 255.0;
            }
            count += 1.0;
        }
    }

    let background = if count > 0.0 {
        [sums[0] / count, sums[1] / count, sums[2] / count]
    } else {
        [0.5, 0.5, 0.5]
    };

    let mut distances = Vec::with_capacity((width * 2 + height * 2) as usize);
    for y in 0..height {
        for x in 0..width {
            if x != 0 && y != 0 && x != width - 1 && y != height - 1 {
                continue;
            }
            distances.push(normalized_rgb_distance(image.get_pixel(x, y), background));
        }
    }
    let distance_count = distances.len().max(1) as f64;
    let mean = distances.iter().sum::<f64>() / distance_count;
    let variance = distances
        .iter()
        .map(|distance| {
            let delta = distance - mean;
            delta * delta
        })
        .sum::<f64>()
        / distance_count;
    let threshold = (mean + 2.0 * variance.sqrt()).clamp(0.08, 0.35);

    (background, threshold)
}

fn normalized_rgb_distance(pixel: &Rgb<u8>, background: [f64; 3]) -> f64 {
    let mut sum = 0.0;
    for (channel, background_value) in background.into_iter().enumerate() {
        let value = f64::from(pixel.0[channel]) / 255.0;
        let delta = value - background_value;
        sum += delta * delta;
    }
    (sum / 3.0).sqrt().clamp(0.0, 1.0)
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
        ImageFeatureMode::Rich
        | ImageFeatureMode::RichSpatial
        | ImageFeatureMode::RichNormalized
        | ImageFeatureMode::RichHog
        | ImageFeatureMode::RichTexture
        | ImageFeatureMode::RichEdge
        | ImageFeatureMode::RichLayout => {
            let mut features = color_histogram_from_gray(&values);
            features.extend(spatial_intensity_stats(&values, width, height));
            features.extend(hog_features(&values, width, height));
            features.extend(hsv_histogram_from_gray(&values));
            features.extend(gray_color_moments(&values));
            features.extend(lbp_features(&values, width, height));
            if matches!(
                config.feature_mode,
                ImageFeatureMode::RichSpatial
                    | ImageFeatureMode::RichNormalized
                    | ImageFeatureMode::RichHog
                    | ImageFeatureMode::RichTexture
                    | ImageFeatureMode::RichEdge
                    | ImageFeatureMode::RichLayout
            ) {
                features.extend(spatial_hsv_histogram_from_gray(&values, width, height));
            }
            if matches!(
                config.feature_mode,
                ImageFeatureMode::RichNormalized
                    | ImageFeatureMode::RichHog
                    | ImageFeatureMode::RichTexture
                    | ImageFeatureMode::RichEdge
                    | ImageFeatureMode::RichLayout
            ) {
                features.extend(gray_normalized_color_moments(&values));
            }
            if matches!(
                config.feature_mode,
                ImageFeatureMode::RichHog
                    | ImageFeatureMode::RichTexture
                    | ImageFeatureMode::RichEdge
                    | ImageFeatureMode::RichLayout
            ) {
                features.extend(hog_block_features(&values, width, height));
            }
            if matches!(
                config.feature_mode,
                ImageFeatureMode::RichTexture
                    | ImageFeatureMode::RichEdge
                    | ImageFeatureMode::RichLayout
            ) {
                features.extend(lbp_radius_features(&values, width, height, 2));
            }
            if config.feature_mode == ImageFeatureMode::RichEdge {
                features.extend(edge_density_features(&values, width, height));
            }
            if config.feature_mode == ImageFeatureMode::RichLayout {
                features.extend(layout_features(&values, width, height));
            }
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

fn spatial_hsv_histogram(image: &RgbImage, invert: bool) -> Vec<f64> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let cell_count = SPATIAL_GRID * SPATIAL_GRID;
    let mut features = vec![0.0; SPATIAL_HSV_LEN];
    let mut counts = vec![0usize; cell_count];

    for (x, y, pixel) in image.enumerate_pixels() {
        let cell = spatial_cell(x as usize, y as usize, width, height);
        counts[cell] += 1;
        let r = channel_value(pixel.0[0], invert);
        let g = channel_value(pixel.0[1], invert);
        let b = channel_value(pixel.0[2], invert);
        let hsv = rgb_to_hsv(r, g, b);
        add_spatial_hsv_bins(&mut features, cell, hsv);
    }

    normalize_spatial_histogram(&mut features, &counts, HSV_HISTOGRAM_LEN);
    features
}

fn spatial_hsv_histogram_from_gray(values: &[f64], width: usize, height: usize) -> Vec<f64> {
    let cell_count = SPATIAL_GRID * SPATIAL_GRID;
    let mut features = vec![0.0; SPATIAL_HSV_LEN];
    let mut counts = vec![0usize; cell_count];

    for y in 0..height {
        for x in 0..width {
            let Some(value) = values.get(y * width + x).copied() else {
                continue;
            };
            let cell = spatial_cell(x, y, width, height);
            counts[cell] += 1;
            let value = value.clamp(0.0, 1.0);
            add_spatial_hsv_bins(&mut features, cell, [value, value, value]);
        }
    }

    normalize_spatial_histogram(&mut features, &counts, HSV_HISTOGRAM_LEN);
    features
}

fn add_spatial_hsv_bins(features: &mut [f64], cell: usize, hsv: [f64; 3]) {
    let cell_offset = cell * HSV_HISTOGRAM_LEN;
    for (channel, value) in hsv.into_iter().enumerate() {
        let bin = (value.clamp(0.0, 1.0) * HSV_BINS as f64).floor() as usize;
        let bin = bin.min(HSV_BINS - 1);
        features[cell_offset + channel * HSV_BINS + bin] += 1.0;
    }
}

fn normalize_spatial_histogram(features: &mut [f64], counts: &[usize], histogram_len: usize) {
    for (cell, count) in counts.iter().copied().enumerate() {
        if count == 0 {
            continue;
        }
        let start = cell * histogram_len;
        let end = start + histogram_len;
        for value in &mut features[start..end] {
            *value /= count as f64;
        }
    }
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

fn normalized_color_moments(image: &RgbImage, invert: bool) -> Vec<f64> {
    let chromaticities = image
        .pixels()
        .map(|pixel| {
            let r = channel_value(pixel.0[0], invert);
            let g = channel_value(pixel.0[1], invert);
            let b = channel_value(pixel.0[2], invert);
            normalized_rgb([r, g, b])
        })
        .collect::<Vec<_>>();
    channel_mean_std(&chromaticities)
}

fn gray_normalized_color_moments(values: &[f64]) -> Vec<f64> {
    let chromaticities = values
        .iter()
        .map(|_| [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        .collect::<Vec<_>>();
    channel_mean_std(&chromaticities)
}

fn normalized_rgb(rgb: [f64; 3]) -> [f64; 3] {
    let sum = (rgb[0] + rgb[1] + rgb[2]).max(f64::EPSILON);
    [rgb[0] / sum, rgb[1] / sum, rgb[2] / sum]
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
    lbp_radius_features(values, width, height, 1)
}

fn lbp_radius_features(values: &[f64], width: usize, height: usize, radius: usize) -> Vec<f64> {
    let mut features = vec![0.0; LBP_LEN];
    if radius == 0 || width <= radius * 2 || height <= radius * 2 {
        return features;
    }

    for y in radius..height - radius {
        for x in radius..width - radius {
            let center_index = y * width + x;
            let Some(center) = values.get(center_index).copied() else {
                continue;
            };

            let mut code = 0u8;
            let neighbors = [
                (x - radius, y - radius),
                (x, y - radius),
                (x + radius, y - radius),
                (x + radius, y),
                (x + radius, y + radius),
                (x, y + radius),
                (x - radius, y + radius),
                (x - radius, y),
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

fn edge_density_features(values: &[f64], width: usize, height: usize) -> Vec<f64> {
    let cell_count = SPATIAL_GRID * SPATIAL_GRID;
    let mut magnitudes = vec![0.0; cell_count];
    let mut horizontal_edges = vec![0.0; cell_count];
    let mut vertical_edges = vec![0.0; cell_count];
    let mut counts = vec![0usize; cell_count];

    if width < 3 || height < 3 {
        return vec![0.0; EDGE_DENSITY_LEN];
    }

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = y * width + x;
            if center >= values.len() {
                continue;
            }

            let gx = values[y * width + x + 1] - values[y * width + x - 1];
            let gy = values[(y + 1) * width + x] - values[(y - 1) * width + x];
            let cell = spatial_cell(x, y, width, height);
            magnitudes[cell] += (gx * gx + gy * gy).sqrt();
            horizontal_edges[cell] += gx.abs();
            vertical_edges[cell] += gy.abs();
            counts[cell] += 1;
        }
    }

    let mut features = Vec::with_capacity(EDGE_DENSITY_LEN);
    for cell in 0..cell_count {
        let count = counts[cell].max(1) as f64;
        features.push((magnitudes[cell] / count).clamp(0.0, 1.0));
        features.push((horizontal_edges[cell] / count).clamp(0.0, 1.0));
        features.push((vertical_edges[cell] / count).clamp(0.0, 1.0));
    }

    features
}

fn layout_features(values: &[f64], width: usize, height: usize) -> Vec<f64> {
    if width == 0 || height == 0 || values.is_empty() {
        return vec![0.0; LAYOUT_LEN];
    }

    let count = (width * height).min(values.len()).max(1);
    let mean = values.iter().take(count).sum::<f64>() / count as f64;
    let mut total_weight = 0.0;
    let mut weighted_x = 0.0;
    let mut weighted_y = 0.0;
    let mut center_weight = 0.0;
    let mut left_weight = 0.0;
    let mut right_weight = 0.0;
    let mut top_weight = 0.0;
    let mut bottom_weight = 0.0;
    let mut quadrants = [0.0; 4];

    for y in 0..height {
        for x in 0..width {
            let Some(value) = values.get(y * width + x).copied() else {
                continue;
            };
            let weight = (value - mean).abs();
            total_weight += weight;
            weighted_x += weight * normalized_position(x, width);
            weighted_y += weight * normalized_position(y, height);

            if x >= width / 4 && x < width - width / 4 && y >= height / 4 && y < height - height / 4
            {
                center_weight += weight;
            }
            if x < width / 2 {
                left_weight += weight;
            } else {
                right_weight += weight;
            }
            if y < height / 2 {
                top_weight += weight;
            } else {
                bottom_weight += weight;
            }

            let quadrant_x = usize::from(x >= width / 2);
            let quadrant_y = usize::from(y >= height / 2);
            quadrants[quadrant_y * 2 + quadrant_x] += weight;
        }
    }

    let total_weight = total_weight.max(f64::EPSILON);
    let centroid_x = weighted_x / total_weight;
    let centroid_y = weighted_y / total_weight;
    let mut spread_x = 0.0;
    let mut spread_y = 0.0;
    for y in 0..height {
        for x in 0..width {
            let Some(value) = values.get(y * width + x).copied() else {
                continue;
            };
            let weight = (value - mean).abs();
            spread_x += weight * (normalized_position(x, width) - centroid_x).powi(2);
            spread_y += weight * (normalized_position(y, height) - centroid_y).powi(2);
        }
    }
    spread_x = (spread_x / total_weight).sqrt();
    spread_y = (spread_y / total_weight).sqrt();

    let center_ratio = center_weight / total_weight;
    let border_ratio = 1.0 - center_ratio;
    let left_right_balance = (left_weight - right_weight).abs() / total_weight;
    let top_bottom_balance = (top_weight - bottom_weight).abs() / total_weight;

    let mut features = Vec::with_capacity(LAYOUT_LEN);
    features.push(horizontal_mirror_difference(values, width, height));
    features.push(vertical_mirror_difference(values, width, height));
    features.push(diagonal_difference(values, width, height, false));
    features.push(diagonal_difference(values, width, height, true));
    features.push(centroid_x.clamp(0.0, 1.0));
    features.push(centroid_y.clamp(0.0, 1.0));
    features.push(spread_x.clamp(0.0, 1.0));
    features.push(spread_y.clamp(0.0, 1.0));
    features.push(center_ratio.clamp(0.0, 1.0));
    features.push(border_ratio.clamp(0.0, 1.0));
    features.push(left_right_balance.clamp(0.0, 1.0));
    features.push(top_bottom_balance.clamp(0.0, 1.0));
    features.extend(
        quadrants
            .into_iter()
            .map(|value| (value / total_weight).clamp(0.0, 1.0)),
    );
    features
}

fn normalized_position(index: usize, size: usize) -> f64 {
    if size <= 1 {
        0.5
    } else {
        index as f64 / (size - 1) as f64
    }
}

fn horizontal_mirror_difference(values: &[f64], width: usize, height: usize) -> f64 {
    let pair_count = (width / 2) * height;
    if pair_count == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    for y in 0..height {
        for x in 0..width / 2 {
            let left = values.get(y * width + x).copied().unwrap_or(0.0);
            let right = values
                .get(y * width + width.saturating_sub(1) - x)
                .copied()
                .unwrap_or(0.0);
            total += (left - right).abs();
        }
    }
    (total / pair_count as f64).clamp(0.0, 1.0)
}

fn vertical_mirror_difference(values: &[f64], width: usize, height: usize) -> f64 {
    let pair_count = width * (height / 2);
    if pair_count == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    for y in 0..height / 2 {
        for x in 0..width {
            let top = values.get(y * width + x).copied().unwrap_or(0.0);
            let bottom = values
                .get((height.saturating_sub(1) - y) * width + x)
                .copied()
                .unwrap_or(0.0);
            total += (top - bottom).abs();
        }
    }
    (total / pair_count as f64).clamp(0.0, 1.0)
}

fn diagonal_difference(values: &[f64], width: usize, height: usize, anti: bool) -> f64 {
    let size = width.min(height);
    if size <= 1 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut pairs = 0usize;
    for y in 0..size {
        for x in 0..size {
            if x >= y {
                continue;
            }
            let first_x = if anti { width - 1 - x } else { x };
            let first_y = y;
            let second_x = if anti { width - 1 - y } else { y };
            let second_y = x;
            let first = values
                .get(first_y * width + first_x)
                .copied()
                .unwrap_or(0.0);
            let second = values
                .get(second_y * width + second_x)
                .copied()
                .unwrap_or(0.0);
            total += (first - second).abs();
            pairs += 1;
        }
    }
    (total / pairs.max(1) as f64).clamp(0.0, 1.0)
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
    let mut features = hog_cell_features(values, width, height);
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

fn hog_block_features(values: &[f64], width: usize, height: usize) -> Vec<f64> {
    let cells = hog_cell_features(values, width, height);
    let mut features = Vec::with_capacity(HOG_BLOCK_LEN);

    for block_y in 0..HOG_BLOCK_GRID {
        for block_x in 0..HOG_BLOCK_GRID {
            let mut block = Vec::with_capacity(4 * HOG_BINS);
            for cell_y in [block_y, block_y + 1] {
                for cell_x in [block_x, block_x + 1] {
                    let cell = cell_y * SPATIAL_GRID + cell_x;
                    let start = cell * HOG_BINS;
                    block.extend_from_slice(&cells[start..start + HOG_BINS]);
                }
            }

            let norm = block
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
                .max(f64::EPSILON);
            features.extend(block.into_iter().map(|value| value / norm));
        }
    }

    features
}

fn hog_cell_features(values: &[f64], width: usize, height: usize) -> Vec<f64> {
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

    features
}

fn spatial_cell(x: usize, y: usize, width: usize, height: usize) -> usize {
    let cell_x = (x * SPATIAL_GRID / width.max(1)).min(SPATIAL_GRID - 1);
    let cell_y = (y * SPATIAL_GRID / height.max(1)).min(SPATIAL_GRID - 1);
    cell_y * SPATIAL_GRID + cell_x
}
