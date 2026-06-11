use std::env;
use std::error::Error;

use progress_ai::pann::CorrectionMode;
use progress_ai::vision::{ImageFeatureMode, ImageResizeMode, ImageVectorConfig};

#[derive(Debug, Clone)]
pub struct Args {
    pub command: String,
    pub format: OutputFormat,
    pub data_path: Option<String>,
    pub eval_data_path: Option<String>,
    pub out_path: Option<String>,
    pub model_path: Option<String>,
    pub image_path: Option<String>,
    pub epochs: usize,
    pub intervals: usize,
    pub correction_mode: CorrectionMode,
    pub seed: u64,
    pub image_width: u32,
    pub image_height: u32,
    pub image_features: ImageFeatureMode,
    pub image_resize: ImageResizeMode,
    pub samples_per_class: usize,
    pub top_k: usize,
    pub target_mse: Option<f64>,
    pub matrix_models: Vec<MatrixModel>,
    pub matrix_features: Vec<ImageFeatureMode>,
    pub matrix_image_sizes: Vec<u32>,
    pub matrix_intervals: Vec<usize>,
    pub matrix_seeds: Vec<u64>,
    pub matrix_resize_modes: Vec<ImageResizeMode>,
    pub matrix_correction_modes: Vec<CorrectionMode>,
    pub matrix_top: usize,
    pub debug_out_path: Option<String>,
    pub debug_train_data_path: Option<String>,
    pub debug_limit: usize,
    pub debug_samples: DebugSamples,
    pub debug_neighbors: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
    Csv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixModel {
    Pann,
    Panc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugSamples {
    Misclassified,
    All,
    Correct,
}

impl DebugSamples {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Misclassified => "misclassified",
            Self::All => "all",
            Self::Correct => "correct",
        }
    }
}

pub fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut raw = env::args().skip(1);
    let command = raw.next().ok_or(
        "usage: research-bench <pann-iris|pann-synthetic|pann-image-synthetic|pann-image-folder|panc-iris|panc-synthetic|panc-image-synthetic|panc-image-folder|train-pann-image-folder|train-panc-image-folder|eval-pann|eval-panc|predict-pann|predict-panc|image-matrix|pann-learning-curve> [--format json|csv] [--data path] [--eval-data path] [--out path] [--model path] [--image path] [--epochs n] [--intervals n] [--correction-mode difference-ls|patent-proportional|ratio] [--seed n] [--target-mse f] [--image-size n] [--image-features pixels|color|hog|combined|rich|rich-spatial] [--image-resize stretch|center-crop|letterbox] [--samples-per-class n] [--top-k n] [--matrix-models pann,panc] [--matrix-features pixels,combined,rich,rich-spatial] [--matrix-image-sizes 16,32] [--matrix-intervals 4,8] [--matrix-seeds 1,2,3] [--matrix-resize-modes stretch,letterbox] [--matrix-correction-modes difference-ls,patent-proportional,ratio] [--matrix-top n] [--debug-out path] [--debug-train-data path] [--debug-limit n] [--debug-samples misclassified|all|correct] [--debug-neighbors n]",
    )?;

    let mut args = Args {
        command,
        format: OutputFormat::Json,
        data_path: None,
        eval_data_path: None,
        out_path: None,
        model_path: None,
        image_path: None,
        epochs: 12,
        intervals: 8,
        correction_mode: CorrectionMode::DifferenceLeastSquares,
        seed: 42,
        image_width: 16,
        image_height: 16,
        image_features: ImageFeatureMode::Pixels,
        image_resize: ImageResizeMode::Stretch,
        samples_per_class: 80,
        top_k: 3,
        target_mse: None,
        matrix_models: Vec::new(),
        matrix_features: Vec::new(),
        matrix_image_sizes: Vec::new(),
        matrix_intervals: Vec::new(),
        matrix_seeds: Vec::new(),
        matrix_resize_modes: Vec::new(),
        matrix_correction_modes: Vec::new(),
        matrix_top: 0,
        debug_out_path: None,
        debug_train_data_path: None,
        debug_limit: 50,
        debug_samples: DebugSamples::Misclassified,
        debug_neighbors: 5,
    };

    while let Some(flag) = raw.next() {
        match flag.as_str() {
            "--format" => {
                args.format = match raw.next().as_deref() {
                    Some("json") => OutputFormat::Json,
                    Some("csv") => OutputFormat::Csv,
                    other => return Err(format!("invalid --format value: {other:?}").into()),
                };
            }
            "--data" => args.data_path = Some(raw.next().ok_or("--data requires a value")?),
            "--eval-data" => {
                args.eval_data_path = Some(raw.next().ok_or("--eval-data requires a value")?);
            }
            "--out" => args.out_path = Some(raw.next().ok_or("--out requires a value")?),
            "--model" => args.model_path = Some(raw.next().ok_or("--model requires a value")?),
            "--image" => args.image_path = Some(raw.next().ok_or("--image requires a value")?),
            "--epochs" => {
                args.epochs = raw
                    .next()
                    .ok_or("--epochs requires a value")?
                    .parse::<usize>()?;
            }
            "--intervals" => {
                args.intervals = raw
                    .next()
                    .ok_or("--intervals requires a value")?
                    .parse::<usize>()?;
            }
            "--correction-mode" => {
                args.correction_mode = parse_correction_mode(
                    &raw.next().ok_or("--correction-mode requires a value")?,
                )?;
            }
            "--seed" => {
                args.seed = raw
                    .next()
                    .ok_or("--seed requires a value")?
                    .parse::<u64>()?;
            }
            "--image-size" => {
                let size = raw
                    .next()
                    .ok_or("--image-size requires a value")?
                    .parse::<u32>()?;
                args.image_width = size;
                args.image_height = size;
            }
            "--image-width" => {
                args.image_width = raw
                    .next()
                    .ok_or("--image-width requires a value")?
                    .parse::<u32>()?;
            }
            "--image-height" => {
                args.image_height = raw
                    .next()
                    .ok_or("--image-height requires a value")?
                    .parse::<u32>()?;
            }
            "--image-features" => {
                args.image_features = raw
                    .next()
                    .ok_or("--image-features requires a value")?
                    .parse::<ImageFeatureMode>()?;
            }
            "--image-resize" => {
                args.image_resize = raw
                    .next()
                    .ok_or("--image-resize requires a value")?
                    .parse::<ImageResizeMode>()?;
            }
            "--samples-per-class" => {
                args.samples_per_class = raw
                    .next()
                    .ok_or("--samples-per-class requires a value")?
                    .parse::<usize>()?;
            }
            "--top-k" => {
                args.top_k = raw
                    .next()
                    .ok_or("--top-k requires a value")?
                    .parse::<usize>()?;
            }
            "--target-mse" => {
                args.target_mse = Some(
                    raw.next()
                        .ok_or("--target-mse requires a value")?
                        .parse::<f64>()?,
                );
            }
            "--matrix-models" => {
                args.matrix_models =
                    parse_matrix_models(&raw.next().ok_or("--matrix-models requires a value")?)?;
            }
            "--matrix-features" => {
                args.matrix_features = parse_image_features_list(
                    &raw.next().ok_or("--matrix-features requires a value")?,
                )?;
            }
            "--matrix-image-sizes" => {
                args.matrix_image_sizes =
                    parse_number_list(&raw.next().ok_or("--matrix-image-sizes requires a value")?)?;
            }
            "--matrix-intervals" => {
                args.matrix_intervals =
                    parse_number_list(&raw.next().ok_or("--matrix-intervals requires a value")?)?;
            }
            "--matrix-seeds" => {
                args.matrix_seeds =
                    parse_number_list(&raw.next().ok_or("--matrix-seeds requires a value")?)?;
            }
            "--matrix-resize-modes" => {
                args.matrix_resize_modes = parse_image_resize_modes_list(
                    &raw.next().ok_or("--matrix-resize-modes requires a value")?,
                )?;
            }
            "--matrix-correction-modes" => {
                args.matrix_correction_modes = parse_correction_modes_list(
                    &raw.next()
                        .ok_or("--matrix-correction-modes requires a value")?,
                )?;
            }
            "--matrix-top" => {
                args.matrix_top = raw
                    .next()
                    .ok_or("--matrix-top requires a value")?
                    .parse::<usize>()?;
            }
            "--debug" | "--debug-out" => {
                args.debug_out_path = Some(raw.next().ok_or("--debug-out requires a value")?);
            }
            "--debug-train-data" => {
                args.debug_train_data_path =
                    Some(raw.next().ok_or("--debug-train-data requires a value")?);
            }
            "--debug-limit" => {
                args.debug_limit = raw
                    .next()
                    .ok_or("--debug-limit requires a value")?
                    .parse::<usize>()?;
            }
            "--debug-samples" => {
                args.debug_samples =
                    parse_debug_samples(&raw.next().ok_or("--debug-samples requires a value")?)?;
            }
            "--debug-neighbors" => {
                args.debug_neighbors = raw
                    .next()
                    .ok_or("--debug-neighbors requires a value")?
                    .parse::<usize>()?;
            }
            other => return Err(format!("unknown option {other}").into()),
        }
    }

    Ok(args)
}

fn parse_matrix_models(value: &str) -> Result<Vec<MatrixModel>, Box<dyn Error>> {
    split_csv_values(value)
        .into_iter()
        .map(|model| match model {
            "pann" => Ok(MatrixModel::Pann),
            "panc" | "panc_like" | "panc-like" => Ok(MatrixModel::Panc),
            other => Err(format!("invalid matrix model {other:?}; expected pann or panc").into()),
        })
        .collect()
}

fn parse_image_features_list(value: &str) -> Result<Vec<ImageFeatureMode>, Box<dyn Error>> {
    split_csv_values(value)
        .into_iter()
        .map(|feature| feature.parse::<ImageFeatureMode>().map_err(Into::into))
        .collect()
}

fn parse_image_resize_modes_list(value: &str) -> Result<Vec<ImageResizeMode>, Box<dyn Error>> {
    split_csv_values(value)
        .into_iter()
        .map(|mode| mode.parse::<ImageResizeMode>().map_err(Into::into))
        .collect()
}

fn parse_correction_modes_list(value: &str) -> Result<Vec<CorrectionMode>, Box<dyn Error>> {
    split_csv_values(value)
        .into_iter()
        .map(parse_correction_mode)
        .collect()
}

fn parse_correction_mode(value: &str) -> Result<CorrectionMode, Box<dyn Error>> {
    match value {
        "difference-ls" | "difference_ls" | "least-squares" | "least_squares"
        | "difference-least-squares" | "difference_least_squares" => {
            Ok(CorrectionMode::DifferenceLeastSquares)
        }
        "patent-proportional" | "patent_proportional" | "difference-patent-proportional"
        | "difference_patent_proportional" => Ok(CorrectionMode::DifferencePatentProportional),
        "ratio" => Ok(CorrectionMode::Ratio {
            epsilon: 1e-9,
            max_abs_factor: 100.0,
        }),
        other => Err(format!(
            "invalid correction mode {other:?}; expected difference-ls, patent-proportional, or ratio"
        )
        .into()),
    }
}

pub const fn correction_mode_name(mode: CorrectionMode) -> &'static str {
    match mode {
        CorrectionMode::DifferenceLeastSquares => "difference_least_squares",
        CorrectionMode::DifferencePatentProportional => "difference_patent_proportional",
        CorrectionMode::Ratio { .. } => "ratio",
    }
}

fn parse_debug_samples(value: &str) -> Result<DebugSamples, Box<dyn Error>> {
    match value {
        "misclassified" | "wrong" | "errors" => Ok(DebugSamples::Misclassified),
        "all" => Ok(DebugSamples::All),
        "correct" => Ok(DebugSamples::Correct),
        other => Err(format!(
            "invalid --debug-samples value {other:?}; expected misclassified, all, or correct"
        )
        .into()),
    }
}

fn parse_number_list<T>(value: &str) -> Result<Vec<T>, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    split_csv_values(value)
        .into_iter()
        .map(str::parse::<T>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn split_csv_values(value: &str) -> Vec<&str> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect()
}

pub fn image_config(args: &Args) -> ImageVectorConfig {
    ImageVectorConfig::new(args.image_width, args.image_height)
        .with_feature_mode(args.image_features)
        .with_resize_mode(args.image_resize)
}

pub fn required_data_path(args: &Args) -> Result<&str, Box<dyn Error>> {
    args.data_path
        .as_deref()
        .ok_or_else(|| "--data path is required for image-folder benchmarks".into())
}

pub fn required_out_path(args: &Args) -> Result<&str, Box<dyn Error>> {
    args.out_path
        .as_deref()
        .ok_or_else(|| "--out path is required for training artifact commands".into())
}

pub fn required_model_path(args: &Args) -> Result<&str, Box<dyn Error>> {
    args.model_path
        .as_deref()
        .ok_or_else(|| "--model path is required for artifact eval/predict commands".into())
}

pub fn required_image_path(args: &Args) -> Result<&str, Box<dyn Error>> {
    args.image_path
        .as_deref()
        .ok_or_else(|| "--image path is required for predict commands".into())
}
