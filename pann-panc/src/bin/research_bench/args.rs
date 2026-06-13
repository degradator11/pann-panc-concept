use std::env;
use std::error::Error;

use progress_ai::evolution::panc::normalize_block_mask;
use progress_ai::pann::CorrectionMode;
use progress_ai::vision::{ImageFeatureMode, ImageResizeMode, ImageVectorConfig};

use super::config::{
    ConfigReport, ConfigSource, SourceMap, apply_config, apply_search_artifact,
    build_config_report, default_sources, find_config_path, find_search_artifact_path, load_config,
    set_source, skip_config_flag,
};

#[derive(Debug, Clone)]
pub struct Args {
    pub command: String,
    pub format: OutputFormat,
    pub data_path: Option<String>,
    pub eval_data_path: Option<String>,
    pub dataset_config_path: Option<String>,
    pub out_path: Option<String>,
    pub report_out_path: Option<String>,
    pub model_path: Option<String>,
    pub image_path: Option<String>,
    pub search_artifact_path: Option<String>,
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
    pub panc_threshold: Option<f64>,
    pub panc_jaccard_weight: Option<f64>,
    pub panc_active_blocks: Option<u32>,
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
    pub patch_size: u32,
    pub patch_stride: u32,
    pub max_train_patches: usize,
    pub anomaly_threshold_quantile: f64,
    pub evolution_population: usize,
    pub evolution_generations: usize,
    pub evolution_elite_count: usize,
    pub evolution_mutation_rate: f64,
    pub evolution_validation_ratio: f64,
    pub evolution_threads: usize,
    pub evolution_feature_modes: Vec<ImageFeatureMode>,
    pub evolution_image_sizes: Vec<u32>,
    pub evolution_resize_modes: Vec<ImageResizeMode>,
    pub evolution_top_k_values: Vec<usize>,
    pub evolution_memory_penalty_per_mb: f64,
    pub evolution_inference_penalty_per_ms: f64,
    pub config_report: Option<ConfigReport>,
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
    Centroid,
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

const USAGE: &str = "usage: research-bench <pann-iris|pann-synthetic|pann-image-synthetic|pann-image-folder|pann-image-manifest|pann-embedding-csv|panc-iris|panc-synthetic|panc-image-synthetic|panc-image-folder|panc-image-manifest|panc-patch-scan|panc-embedding-csv|centroid-iris|centroid-synthetic|centroid-image-synthetic|centroid-image-folder|centroid-image-manifest|centroid-embedding-csv|train-pann-image-folder|train-panc-image-folder|eval-pann|eval-panc|predict-pann|predict-panc|image-matrix|pann-learning-curve|evolve-panc-image-folder|evolved-panc-image-folder> [--config path|--config.location=path] [--format json|csv] [--data path] [--eval-data path] [--dataset-config path] [--out path] [--report-out path] [--model path] [--image path] [--search-artifact path] [--epochs n] [--intervals n] [--correction-mode difference-ls|patent-proportional|ratio] [--seed n] [--target-mse f] [--image-size n] [--image-features pixels|color|hog|combined|rich|rich-spatial|rich-normalized|rich-hog|rich-texture|rich-edge|rich-layout] [--image-resize stretch|center-crop|letterbox|foreground-crop] [--samples-per-class n] [--top-k n] [--panc-threshold f] [--panc-jaccard-weight f] [--panc-active-blocks 0xffff] [--patch-size n] [--patch-stride n] [--max-train-patches n] [--anomaly-threshold-quantile f] [--matrix-models pann,panc,centroid] [--matrix-features pixels,combined,rich,rich-spatial,rich-normalized,rich-hog,rich-texture,rich-edge,rich-layout] [--matrix-image-sizes 16,32] [--matrix-intervals 4,8] [--matrix-seeds 1,2,3] [--matrix-resize-modes stretch,letterbox,foreground-crop] [--matrix-correction-modes difference-ls,patent-proportional,ratio] [--matrix-top n] [--debug-out path] [--debug-train-data path] [--debug-limit n] [--debug-samples misclassified|all|correct] [--debug-neighbors n] [--population n] [--generations n] [--elite-count n] [--mutation-rate f] [--validation-ratio f] [--threads n] [--evolve-features rich,rich-texture] [--evolve-image-sizes 64,128] [--evolve-resize-modes center-crop,foreground-crop] [--evolve-top-k 1,3,5]";

pub fn parse_args() -> Result<Args, Box<dyn Error>> {
    parse_args_from(env::args().skip(1))
}

pub fn parse_args_from<I, S>(raw: I) -> Result<Args, Box<dyn Error>>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let raw = raw.into_iter().map(Into::into).collect::<Vec<_>>();
    let config_path = find_config_path(&raw)?;
    let cli_search_artifact_path = find_search_artifact_path(&raw)?;
    let config = config_path.as_deref().map(load_config).transpose()?;
    let positional_command = raw
        .first()
        .filter(|value| !value.starts_with("--"))
        .cloned();
    let command = positional_command
        .clone()
        .or_else(|| config.as_ref().and_then(|config| config.command.clone()))
        .ok_or(USAGE)?;

    let mut args = default_args(command);
    let mut sources = default_sources();
    if positional_command.is_some() {
        set_source(&mut sources, "command", ConfigSource::Cli);
    } else if config
        .as_ref()
        .and_then(|config| config.command.as_ref())
        .is_some()
    {
        set_source(&mut sources, "command", ConfigSource::Config);
    }

    if let Some(config) = config.as_ref() {
        apply_config(&mut args, config, &mut sources)?;
    }
    if let Some(path) = cli_search_artifact_path {
        args.search_artifact_path = Some(path);
        set_source(&mut sources, "search_artifact_path", ConfigSource::Cli);
    }
    if let Some(path) = args.search_artifact_path.clone() {
        apply_search_artifact(&mut args, &path, &mut sources)?;
    }
    apply_cli_flags(&mut args, &raw, positional_command.is_some(), &mut sources)?;

    if let Some(path) = config_path.or_else(|| {
        args.search_artifact_path
            .as_ref()
            .map(|path| format!("search artifact: {path}"))
    }) {
        args.config_report = Some(build_config_report(path, &args, &sources));
    }

    Ok(args)
}

fn default_args(command: String) -> Args {
    Args {
        command,
        format: OutputFormat::Json,
        data_path: None,
        eval_data_path: None,
        dataset_config_path: None,
        out_path: None,
        report_out_path: None,
        model_path: None,
        image_path: None,
        search_artifact_path: None,
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
        panc_threshold: None,
        panc_jaccard_weight: None,
        panc_active_blocks: None,
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
        patch_size: 96,
        patch_stride: 48,
        max_train_patches: 4096,
        anomaly_threshold_quantile: 0.95,
        evolution_population: 32,
        evolution_generations: 20,
        evolution_elite_count: 4,
        evolution_mutation_rate: 0.2,
        evolution_validation_ratio: 0.2,
        evolution_threads: 0,
        evolution_feature_modes: Vec::new(),
        evolution_image_sizes: Vec::new(),
        evolution_resize_modes: Vec::new(),
        evolution_top_k_values: Vec::new(),
        evolution_memory_penalty_per_mb: 0.0001,
        evolution_inference_penalty_per_ms: 0.00001,
        config_report: None,
    }
}

fn apply_cli_flags(
    args: &mut Args,
    raw: &[String],
    has_positional_command: bool,
    sources: &mut SourceMap,
) -> Result<(), Box<dyn Error>> {
    let mut index = usize::from(has_positional_command);
    while index < raw.len() {
        let flag = raw[index].as_str();
        if skip_config_flag(raw, &mut index) {
            continue;
        }
        match flag {
            "--format" => {
                args.format = parse_format(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "format", ConfigSource::Cli);
            }
            "--data" => {
                args.data_path = Some(next_value(raw, &mut index, flag)?);
                set_source(sources, "data_path", ConfigSource::Cli);
            }
            "--eval-data" => {
                args.eval_data_path = Some(next_value(raw, &mut index, flag)?);
                set_source(sources, "eval_data_path", ConfigSource::Cli);
            }
            "--dataset-config" | "--dataset-config.location" | "--dataset-config-location" => {
                args.dataset_config_path = Some(next_value(raw, &mut index, flag)?);
                set_source(sources, "dataset_config_path", ConfigSource::Cli);
            }
            other
                if other.starts_with("--dataset-config=")
                    || other.starts_with("--dataset-config.location=") =>
            {
                args.dataset_config_path = Some(
                    other
                        .split_once('=')
                        .map(|(_, value)| value)
                        .unwrap_or_default()
                        .to_string(),
                );
                set_source(sources, "dataset_config_path", ConfigSource::Cli);
            }
            "--out" => {
                args.out_path = Some(next_value(raw, &mut index, flag)?);
                set_source(sources, "out_path", ConfigSource::Cli);
            }
            "--report-out" => {
                args.report_out_path = Some(next_value(raw, &mut index, flag)?);
                set_source(sources, "report_out_path", ConfigSource::Cli);
            }
            other if other.starts_with("--report-out=") => {
                args.report_out_path = Some(
                    other
                        .strip_prefix("--report-out=")
                        .unwrap_or_default()
                        .to_string(),
                );
                set_source(sources, "report_out_path", ConfigSource::Cli);
            }
            "--model" => {
                args.model_path = Some(next_value(raw, &mut index, flag)?);
                set_source(sources, "model_path", ConfigSource::Cli);
            }
            "--image" => {
                args.image_path = Some(next_value(raw, &mut index, flag)?);
                set_source(sources, "image_path", ConfigSource::Cli);
            }
            "--search-artifact" | "--search-artifact.location" | "--search-artifact-location" => {
                args.search_artifact_path = Some(next_value(raw, &mut index, flag)?);
                set_source(sources, "search_artifact_path", ConfigSource::Cli);
            }
            "--epochs" => {
                args.epochs = next_value(raw, &mut index, flag)?.parse::<usize>()?;
                set_source(sources, "epochs", ConfigSource::Cli);
            }
            "--intervals" => {
                args.intervals = next_value(raw, &mut index, flag)?.parse::<usize>()?;
                set_source(sources, "intervals", ConfigSource::Cli);
            }
            "--correction-mode" => {
                args.correction_mode = parse_correction_mode(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "correction_mode", ConfigSource::Cli);
            }
            "--seed" => {
                args.seed = next_value(raw, &mut index, flag)?.parse::<u64>()?;
                set_source(sources, "seed", ConfigSource::Cli);
            }
            "--image-size" => {
                let size = next_value(raw, &mut index, flag)?.parse::<u32>()?;
                args.image_width = size;
                args.image_height = size;
                set_source(sources, "image_width", ConfigSource::Cli);
                set_source(sources, "image_height", ConfigSource::Cli);
            }
            "--image-width" => {
                args.image_width = next_value(raw, &mut index, flag)?.parse::<u32>()?;
                set_source(sources, "image_width", ConfigSource::Cli);
            }
            "--image-height" => {
                args.image_height = next_value(raw, &mut index, flag)?.parse::<u32>()?;
                set_source(sources, "image_height", ConfigSource::Cli);
            }
            "--image-features" => {
                args.image_features =
                    next_value(raw, &mut index, flag)?.parse::<ImageFeatureMode>()?;
                set_source(sources, "image_features", ConfigSource::Cli);
            }
            "--image-resize" => {
                args.image_resize =
                    next_value(raw, &mut index, flag)?.parse::<ImageResizeMode>()?;
                set_source(sources, "image_resize", ConfigSource::Cli);
            }
            "--samples-per-class" => {
                args.samples_per_class = next_value(raw, &mut index, flag)?.parse::<usize>()?;
                set_source(sources, "samples_per_class", ConfigSource::Cli);
            }
            "--top-k" => {
                args.top_k = next_value(raw, &mut index, flag)?.parse::<usize>()?;
                set_source(sources, "top_k", ConfigSource::Cli);
            }
            "--panc-threshold" => {
                args.panc_threshold = Some(next_value(raw, &mut index, flag)?.parse::<f64>()?);
                set_source(sources, "panc_threshold", ConfigSource::Cli);
            }
            "--panc-jaccard-weight" => {
                args.panc_jaccard_weight = Some(next_value(raw, &mut index, flag)?.parse::<f64>()?);
                set_source(sources, "panc_jaccard_weight", ConfigSource::Cli);
            }
            "--panc-active-blocks" => {
                args.panc_active_blocks =
                    Some(parse_active_blocks(&next_value(raw, &mut index, flag)?)?);
                set_source(sources, "panc_active_blocks", ConfigSource::Cli);
            }
            "--target-mse" => {
                args.target_mse = Some(next_value(raw, &mut index, flag)?.parse::<f64>()?);
                set_source(sources, "target_mse", ConfigSource::Cli);
            }
            "--matrix-models" => {
                args.matrix_models = parse_matrix_models(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "matrix_models", ConfigSource::Cli);
            }
            "--matrix-features" => {
                args.matrix_features =
                    parse_image_features_list(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "matrix_features", ConfigSource::Cli);
            }
            "--matrix-image-sizes" => {
                args.matrix_image_sizes = parse_number_list(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "matrix_image_sizes", ConfigSource::Cli);
            }
            "--matrix-intervals" => {
                args.matrix_intervals = parse_number_list(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "matrix_intervals", ConfigSource::Cli);
            }
            "--matrix-seeds" => {
                args.matrix_seeds = parse_number_list(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "matrix_seeds", ConfigSource::Cli);
            }
            "--matrix-resize-modes" => {
                args.matrix_resize_modes =
                    parse_image_resize_modes_list(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "matrix_resize_modes", ConfigSource::Cli);
            }
            "--matrix-correction-modes" => {
                args.matrix_correction_modes =
                    parse_correction_modes_list(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "matrix_correction_modes", ConfigSource::Cli);
            }
            "--matrix-top" => {
                args.matrix_top = next_value(raw, &mut index, flag)?.parse::<usize>()?;
                set_source(sources, "matrix_top", ConfigSource::Cli);
            }
            "--debug" | "--debug-out" => {
                args.debug_out_path = Some(next_value(raw, &mut index, flag)?);
                set_source(sources, "debug_out_path", ConfigSource::Cli);
            }
            "--debug-train-data" => {
                args.debug_train_data_path = Some(next_value(raw, &mut index, flag)?);
                set_source(sources, "debug_train_data_path", ConfigSource::Cli);
            }
            "--debug-limit" => {
                args.debug_limit = next_value(raw, &mut index, flag)?.parse::<usize>()?;
                set_source(sources, "debug_limit", ConfigSource::Cli);
            }
            "--debug-samples" => {
                args.debug_samples = parse_debug_samples(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "debug_samples", ConfigSource::Cli);
            }
            "--debug-neighbors" => {
                args.debug_neighbors = next_value(raw, &mut index, flag)?.parse::<usize>()?;
                set_source(sources, "debug_neighbors", ConfigSource::Cli);
            }
            "--patch-size" => {
                args.patch_size = next_value(raw, &mut index, flag)?.parse::<u32>()?;
                set_source(sources, "patch_size", ConfigSource::Cli);
            }
            "--patch-stride" => {
                args.patch_stride = next_value(raw, &mut index, flag)?.parse::<u32>()?;
                set_source(sources, "patch_stride", ConfigSource::Cli);
            }
            "--max-train-patches" => {
                args.max_train_patches = next_value(raw, &mut index, flag)?.parse::<usize>()?;
                set_source(sources, "max_train_patches", ConfigSource::Cli);
            }
            "--anomaly-threshold-quantile" => {
                args.anomaly_threshold_quantile =
                    next_value(raw, &mut index, flag)?.parse::<f64>()?;
                set_source(sources, "anomaly_threshold_quantile", ConfigSource::Cli);
            }
            "--population" => {
                args.evolution_population = next_value(raw, &mut index, flag)?.parse::<usize>()?;
                set_source(sources, "evolution_population", ConfigSource::Cli);
            }
            "--generations" => {
                args.evolution_generations = next_value(raw, &mut index, flag)?.parse::<usize>()?;
                set_source(sources, "evolution_generations", ConfigSource::Cli);
            }
            "--elite" | "--elite-count" => {
                args.evolution_elite_count = next_value(raw, &mut index, flag)?.parse::<usize>()?;
                set_source(sources, "evolution_elite_count", ConfigSource::Cli);
            }
            "--mutation-rate" => {
                args.evolution_mutation_rate = next_value(raw, &mut index, flag)?.parse::<f64>()?;
                set_source(sources, "evolution_mutation_rate", ConfigSource::Cli);
            }
            "--validation-ratio" => {
                args.evolution_validation_ratio =
                    next_value(raw, &mut index, flag)?.parse::<f64>()?;
                set_source(sources, "evolution_validation_ratio", ConfigSource::Cli);
            }
            "--threads" => {
                args.evolution_threads = next_value(raw, &mut index, flag)?.parse::<usize>()?;
                set_source(sources, "evolution_threads", ConfigSource::Cli);
            }
            "--evolve-features" => {
                args.evolution_feature_modes =
                    parse_image_features_list(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "evolution_feature_modes", ConfigSource::Cli);
            }
            "--evolve-image-sizes" => {
                args.evolution_image_sizes =
                    parse_number_list(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "evolution_image_sizes", ConfigSource::Cli);
            }
            "--evolve-resize-modes" => {
                args.evolution_resize_modes =
                    parse_image_resize_modes_list(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "evolution_resize_modes", ConfigSource::Cli);
            }
            "--evolve-top-k" => {
                args.evolution_top_k_values =
                    parse_number_list(&next_value(raw, &mut index, flag)?)?;
                set_source(sources, "evolution_top_k_values", ConfigSource::Cli);
            }
            "--memory-penalty-per-mb" => {
                args.evolution_memory_penalty_per_mb =
                    next_value(raw, &mut index, flag)?.parse::<f64>()?;
                set_source(
                    sources,
                    "evolution_memory_penalty_per_mb",
                    ConfigSource::Cli,
                );
            }
            "--inference-penalty-per-ms" => {
                args.evolution_inference_penalty_per_ms =
                    next_value(raw, &mut index, flag)?.parse::<f64>()?;
                set_source(
                    sources,
                    "evolution_inference_penalty_per_ms",
                    ConfigSource::Cli,
                );
            }
            other
                if other.starts_with("--config=")
                    || other.starts_with("--config.location=")
                    || other.starts_with("--dataset-config=")
                    || other.starts_with("--dataset-config.location=")
                    || other.starts_with("--search-artifact=")
                    || other.starts_with("--search-artifact.location=") => {}
            other if !other.starts_with("--") => {
                return Err(format!("unexpected positional argument {other:?}").into());
            }
            other => return Err(format!("unknown option {other}").into()),
        }
        index += 1;
    }

    Ok(())
}

pub(super) fn parse_matrix_models(value: &str) -> Result<Vec<MatrixModel>, Box<dyn Error>> {
    split_csv_values(value)
        .into_iter()
        .map(|model| match model {
            "pann" => Ok(MatrixModel::Pann),
            "panc" | "panc_like" | "panc-like" => Ok(MatrixModel::Panc),
            "centroid" | "nearest-centroid" | "nearest_centroid" | "baseline" => {
                Ok(MatrixModel::Centroid)
            }
            other => Err(format!(
                "invalid matrix model {other:?}; expected pann, panc, or centroid"
            )
            .into()),
        })
        .collect()
}

pub(super) fn parse_format(value: &str) -> Result<OutputFormat, Box<dyn Error>> {
    match value {
        "json" => Ok(OutputFormat::Json),
        "csv" => Ok(OutputFormat::Csv),
        other => Err(format!("invalid --format value: {other:?}; expected json or csv").into()),
    }
}

pub(super) fn output_format_name(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Json => "json",
        OutputFormat::Csv => "csv",
    }
}

pub(super) fn parse_image_features_list(
    value: &str,
) -> Result<Vec<ImageFeatureMode>, Box<dyn Error>> {
    split_csv_values(value)
        .into_iter()
        .map(|feature| feature.parse::<ImageFeatureMode>().map_err(Into::into))
        .collect()
}

pub(super) fn parse_image_resize_modes_list(
    value: &str,
) -> Result<Vec<ImageResizeMode>, Box<dyn Error>> {
    split_csv_values(value)
        .into_iter()
        .map(|mode| mode.parse::<ImageResizeMode>().map_err(Into::into))
        .collect()
}

pub(super) fn parse_correction_modes_list(
    value: &str,
) -> Result<Vec<CorrectionMode>, Box<dyn Error>> {
    split_csv_values(value)
        .into_iter()
        .map(parse_correction_mode)
        .collect()
}

pub(super) fn parse_correction_mode(value: &str) -> Result<CorrectionMode, Box<dyn Error>> {
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

pub(super) fn parse_debug_samples(value: &str) -> Result<DebugSamples, Box<dyn Error>> {
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

pub(super) fn parse_number_list<T>(value: &str) -> Result<Vec<T>, Box<dyn Error>>
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

pub(super) fn parse_active_blocks(value: &str) -> Result<u32, Box<dyn Error>> {
    let trimmed = value.trim();
    let parsed = if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u32::from_str_radix(hex, 16)?
    } else {
        trimmed.parse::<u32>()?
    };
    Ok(normalize_block_mask(parsed))
}

fn split_csv_values(value: &str) -> Vec<&str> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect()
}

fn next_value(raw: &[String], index: &mut usize, flag: &str) -> Result<String, Box<dyn Error>> {
    *index += 1;
    raw.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value").into())
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

pub fn required_dataset_config_path(args: &Args) -> Result<&str, Box<dyn Error>> {
    args.dataset_config_path
        .as_deref()
        .or(args.data_path.as_deref())
        .ok_or_else(|| "--dataset-config path is required".into())
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
