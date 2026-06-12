use std::collections::HashMap;
use std::error::Error;
use std::fs;

use serde::Deserialize;
use serde_json::Value;

use super::args::{
    Args, MatrixModel, correction_mode_name, output_format_name, parse_correction_mode,
    parse_correction_modes_list, parse_debug_samples, parse_format, parse_image_features_list,
    parse_image_resize_modes_list, parse_matrix_models, parse_number_list,
};

pub type SourceMap = HashMap<&'static str, ConfigSource>;

#[derive(Debug, Clone)]
pub struct ConfigReport {
    pub path: String,
    pub entries: Vec<ConfigReportEntry>,
}

#[derive(Debug, Clone)]
pub struct ConfigReportEntry {
    pub name: &'static str,
    pub value: String,
    pub source: ConfigSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigSource {
    Default,
    Config,
    Cli,
}

impl ConfigSource {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Config => "config",
            Self::Cli => "cli",
        }
    }
}

const REPORT_FIELDS: &[&str] = &[
    "command",
    "format",
    "data_path",
    "eval_data_path",
    "out_path",
    "model_path",
    "image_path",
    "epochs",
    "intervals",
    "correction_mode",
    "seed",
    "image_width",
    "image_height",
    "image_features",
    "image_resize",
    "samples_per_class",
    "top_k",
    "target_mse",
    "matrix_models",
    "matrix_features",
    "matrix_image_sizes",
    "matrix_intervals",
    "matrix_seeds",
    "matrix_resize_modes",
    "matrix_correction_modes",
    "matrix_top",
    "debug_out_path",
    "debug_train_data_path",
    "debug_limit",
    "debug_samples",
    "debug_neighbors",
    "evolution_population",
    "evolution_generations",
    "evolution_elite_count",
    "evolution_mutation_rate",
    "evolution_validation_ratio",
    "evolution_threads",
    "evolution_feature_modes",
    "evolution_image_sizes",
    "evolution_resize_modes",
    "evolution_top_k_values",
    "evolution_memory_penalty_per_mb",
    "evolution_inference_penalty_per_ms",
];

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchConfig {
    pub command: Option<String>,
    format: Option<String>,
    #[serde(alias = "data_path", alias = "data-path")]
    data: Option<String>,
    #[serde(alias = "eval_data_path", alias = "eval-data")]
    eval_data: Option<String>,
    #[serde(alias = "out_path")]
    out: Option<String>,
    #[serde(alias = "model_path")]
    model: Option<String>,
    #[serde(alias = "image_path")]
    image: Option<String>,
    epochs: Option<usize>,
    intervals: Option<usize>,
    #[serde(alias = "correction-mode")]
    correction_mode: Option<String>,
    seed: Option<u64>,
    image_size: Option<u32>,
    image_width: Option<u32>,
    image_height: Option<u32>,
    #[serde(alias = "image-features")]
    image_features: Option<String>,
    #[serde(alias = "image-resize")]
    image_resize: Option<String>,
    samples_per_class: Option<usize>,
    #[serde(alias = "top-k")]
    top_k: Option<usize>,
    target_mse: Option<f64>,
    matrix_models: Option<Value>,
    matrix_features: Option<Value>,
    matrix_image_sizes: Option<Value>,
    matrix_intervals: Option<Value>,
    matrix_seeds: Option<Value>,
    matrix_resize_modes: Option<Value>,
    matrix_correction_modes: Option<Value>,
    matrix_top: Option<usize>,
    #[serde(alias = "debug-out", alias = "debug")]
    debug_out: Option<String>,
    debug_train_data: Option<String>,
    debug_limit: Option<usize>,
    debug_samples: Option<String>,
    debug_neighbors: Option<usize>,
    #[serde(alias = "evolution_population")]
    population: Option<usize>,
    #[serde(alias = "evolution_generations")]
    generations: Option<usize>,
    #[serde(alias = "elite")]
    elite_count: Option<usize>,
    mutation_rate: Option<f64>,
    validation_ratio: Option<f64>,
    #[serde(alias = "evolution_threads")]
    threads: Option<usize>,
    evolve_features: Option<Value>,
    evolve_image_sizes: Option<Value>,
    evolve_resize_modes: Option<Value>,
    evolve_top_k: Option<Value>,
    memory_penalty_per_mb: Option<f64>,
    inference_penalty_per_ms: Option<f64>,
}

pub fn print_config_report(args: &Args) {
    let Some(report) = &args.config_report else {
        return;
    };

    eprintln!("Loaded config: {}", report.path);
    eprintln!("Parameter sources (CLI overrides config, config overrides defaults):");
    for entry in &report.entries {
        eprintln!(
            "  {:34} {:8} {}",
            entry.name,
            entry.source.as_str(),
            entry.value
        );
    }
}

pub fn load_config(path: &str) -> Result<BenchConfig, Box<dyn Error>> {
    let text = fs::read_to_string(path)
        .map_err(|source| format!("failed to read config {path:?}: {source}"))?;
    serde_json::from_str::<BenchConfig>(&text)
        .map_err(|source| format!("failed to parse config {path:?}: {source}").into())
}

pub fn find_config_path(raw: &[String]) -> Result<Option<String>, Box<dyn Error>> {
    let mut index = 0usize;
    let mut config_path = None;
    while index < raw.len() {
        let value = raw[index].as_str();
        if let Some(path) = value
            .strip_prefix("--config=")
            .or_else(|| value.strip_prefix("--config.location="))
        {
            config_path = Some(path.to_string());
        } else if matches!(
            value,
            "--config" | "--config.location" | "--config-location"
        ) {
            index += 1;
            let path = raw
                .get(index)
                .ok_or_else(|| format!("{value} requires a value"))?;
            config_path = Some(path.clone());
        }
        index += 1;
    }
    Ok(config_path)
}

pub fn skip_config_flag(raw: &[String], index: &mut usize) -> bool {
    let value = raw[*index].as_str();
    if value.starts_with("--config=") || value.starts_with("--config.location=") {
        *index += 1;
        return true;
    }
    if matches!(
        value,
        "--config" | "--config.location" | "--config-location"
    ) {
        *index += 2;
        return true;
    }
    false
}

pub fn default_sources() -> SourceMap {
    REPORT_FIELDS
        .iter()
        .copied()
        .map(|field| (field, ConfigSource::Default))
        .collect()
}

pub fn set_source(sources: &mut SourceMap, field: &'static str, source: ConfigSource) {
    sources.insert(field, source);
}

pub fn build_config_report(path: String, args: &Args, sources: &SourceMap) -> ConfigReport {
    ConfigReport {
        path,
        entries: REPORT_FIELDS
            .iter()
            .copied()
            .map(|field| ConfigReportEntry {
                name: field,
                value: config_report_value(args, field),
                source: *sources.get(field).unwrap_or(&ConfigSource::Default),
            })
            .collect(),
    }
}

pub fn apply_config(
    args: &mut Args,
    config: &BenchConfig,
    sources: &mut SourceMap,
) -> Result<(), Box<dyn Error>> {
    if let Some(format) = &config.format {
        args.format = parse_format(format)?;
        set_source(sources, "format", ConfigSource::Config);
    }
    if let Some(value) = &config.data {
        args.data_path = Some(value.clone());
        set_source(sources, "data_path", ConfigSource::Config);
    }
    if let Some(value) = &config.eval_data {
        args.eval_data_path = Some(value.clone());
        set_source(sources, "eval_data_path", ConfigSource::Config);
    }
    if let Some(value) = &config.out {
        args.out_path = Some(value.clone());
        set_source(sources, "out_path", ConfigSource::Config);
    }
    if let Some(value) = &config.model {
        args.model_path = Some(value.clone());
        set_source(sources, "model_path", ConfigSource::Config);
    }
    if let Some(value) = &config.image {
        args.image_path = Some(value.clone());
        set_source(sources, "image_path", ConfigSource::Config);
    }
    if let Some(value) = config.epochs {
        args.epochs = value;
        set_source(sources, "epochs", ConfigSource::Config);
    }
    if let Some(value) = config.intervals {
        args.intervals = value;
        set_source(sources, "intervals", ConfigSource::Config);
    }
    if let Some(value) = &config.correction_mode {
        args.correction_mode = parse_correction_mode(value)?;
        set_source(sources, "correction_mode", ConfigSource::Config);
    }
    if let Some(value) = config.seed {
        args.seed = value;
        set_source(sources, "seed", ConfigSource::Config);
    }
    if let Some(value) = config.image_size {
        args.image_width = value;
        args.image_height = value;
        set_source(sources, "image_width", ConfigSource::Config);
        set_source(sources, "image_height", ConfigSource::Config);
    }
    if let Some(value) = config.image_width {
        args.image_width = value;
        set_source(sources, "image_width", ConfigSource::Config);
    }
    if let Some(value) = config.image_height {
        args.image_height = value;
        set_source(sources, "image_height", ConfigSource::Config);
    }
    if let Some(value) = &config.image_features {
        args.image_features = value.parse()?;
        set_source(sources, "image_features", ConfigSource::Config);
    }
    if let Some(value) = &config.image_resize {
        args.image_resize = value.parse()?;
        set_source(sources, "image_resize", ConfigSource::Config);
    }
    if let Some(value) = config.samples_per_class {
        args.samples_per_class = value;
        set_source(sources, "samples_per_class", ConfigSource::Config);
    }
    if let Some(value) = config.top_k {
        args.top_k = value;
        set_source(sources, "top_k", ConfigSource::Config);
    }
    if let Some(value) = config.target_mse {
        args.target_mse = Some(value);
        set_source(sources, "target_mse", ConfigSource::Config);
    }
    if let Some(value) = &config.matrix_models {
        args.matrix_models = parse_matrix_models(&config_value_as_csv(value, "matrix_models")?)?;
        set_source(sources, "matrix_models", ConfigSource::Config);
    }
    if let Some(value) = &config.matrix_features {
        args.matrix_features =
            parse_image_features_list(&config_value_as_csv(value, "matrix_features")?)?;
        set_source(sources, "matrix_features", ConfigSource::Config);
    }
    if let Some(value) = &config.matrix_image_sizes {
        args.matrix_image_sizes =
            parse_number_list(&config_value_as_csv(value, "matrix_image_sizes")?)?;
        set_source(sources, "matrix_image_sizes", ConfigSource::Config);
    }
    if let Some(value) = &config.matrix_intervals {
        args.matrix_intervals =
            parse_number_list(&config_value_as_csv(value, "matrix_intervals")?)?;
        set_source(sources, "matrix_intervals", ConfigSource::Config);
    }
    if let Some(value) = &config.matrix_seeds {
        args.matrix_seeds = parse_number_list(&config_value_as_csv(value, "matrix_seeds")?)?;
        set_source(sources, "matrix_seeds", ConfigSource::Config);
    }
    if let Some(value) = &config.matrix_resize_modes {
        args.matrix_resize_modes =
            parse_image_resize_modes_list(&config_value_as_csv(value, "matrix_resize_modes")?)?;
        set_source(sources, "matrix_resize_modes", ConfigSource::Config);
    }
    if let Some(value) = &config.matrix_correction_modes {
        args.matrix_correction_modes =
            parse_correction_modes_list(&config_value_as_csv(value, "matrix_correction_modes")?)?;
        set_source(sources, "matrix_correction_modes", ConfigSource::Config);
    }
    if let Some(value) = config.matrix_top {
        args.matrix_top = value;
        set_source(sources, "matrix_top", ConfigSource::Config);
    }
    if let Some(value) = &config.debug_out {
        args.debug_out_path = Some(value.clone());
        set_source(sources, "debug_out_path", ConfigSource::Config);
    }
    if let Some(value) = &config.debug_train_data {
        args.debug_train_data_path = Some(value.clone());
        set_source(sources, "debug_train_data_path", ConfigSource::Config);
    }
    if let Some(value) = config.debug_limit {
        args.debug_limit = value;
        set_source(sources, "debug_limit", ConfigSource::Config);
    }
    if let Some(value) = &config.debug_samples {
        args.debug_samples = parse_debug_samples(value)?;
        set_source(sources, "debug_samples", ConfigSource::Config);
    }
    if let Some(value) = config.debug_neighbors {
        args.debug_neighbors = value;
        set_source(sources, "debug_neighbors", ConfigSource::Config);
    }
    if let Some(value) = config.population {
        args.evolution_population = value;
        set_source(sources, "evolution_population", ConfigSource::Config);
    }
    if let Some(value) = config.generations {
        args.evolution_generations = value;
        set_source(sources, "evolution_generations", ConfigSource::Config);
    }
    if let Some(value) = config.elite_count {
        args.evolution_elite_count = value;
        set_source(sources, "evolution_elite_count", ConfigSource::Config);
    }
    if let Some(value) = config.mutation_rate {
        args.evolution_mutation_rate = value;
        set_source(sources, "evolution_mutation_rate", ConfigSource::Config);
    }
    if let Some(value) = config.validation_ratio {
        args.evolution_validation_ratio = value;
        set_source(sources, "evolution_validation_ratio", ConfigSource::Config);
    }
    if let Some(value) = config.threads {
        args.evolution_threads = value;
        set_source(sources, "evolution_threads", ConfigSource::Config);
    }
    if let Some(value) = &config.evolve_features {
        args.evolution_feature_modes =
            parse_image_features_list(&config_value_as_csv(value, "evolve_features")?)?;
        set_source(sources, "evolution_feature_modes", ConfigSource::Config);
    }
    if let Some(value) = &config.evolve_image_sizes {
        args.evolution_image_sizes =
            parse_number_list(&config_value_as_csv(value, "evolve_image_sizes")?)?;
        set_source(sources, "evolution_image_sizes", ConfigSource::Config);
    }
    if let Some(value) = &config.evolve_resize_modes {
        args.evolution_resize_modes =
            parse_image_resize_modes_list(&config_value_as_csv(value, "evolve_resize_modes")?)?;
        set_source(sources, "evolution_resize_modes", ConfigSource::Config);
    }
    if let Some(value) = &config.evolve_top_k {
        args.evolution_top_k_values =
            parse_number_list(&config_value_as_csv(value, "evolve_top_k")?)?;
        set_source(sources, "evolution_top_k_values", ConfigSource::Config);
    }
    if let Some(value) = config.memory_penalty_per_mb {
        args.evolution_memory_penalty_per_mb = value;
        set_source(
            sources,
            "evolution_memory_penalty_per_mb",
            ConfigSource::Config,
        );
    }
    if let Some(value) = config.inference_penalty_per_ms {
        args.evolution_inference_penalty_per_ms = value;
        set_source(
            sources,
            "evolution_inference_penalty_per_ms",
            ConfigSource::Config,
        );
    }

    Ok(())
}

fn config_value_as_csv(value: &Value, field: &str) -> Result<String, Box<dyn Error>> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Array(values) => values
            .iter()
            .map(|value| config_atom_as_string(value, field))
            .collect::<Result<Vec<_>, _>>()
            .map(|values| values.join(",")),
        other => {
            Err(format!("config field {field:?} must be a string or array, got {other}").into())
        }
    }
}

fn config_atom_as_string(value: &Value, field: &str) -> Result<String, Box<dyn Error>> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => Ok(value.to_string()),
        other => {
            Err(format!("config field {field:?} array contains unsupported value {other}").into())
        }
    }
}

fn config_report_value(args: &Args, field: &str) -> String {
    match field {
        "command" => args.command.clone(),
        "format" => output_format_name(args.format).to_string(),
        "data_path" => option_string(&args.data_path),
        "eval_data_path" => option_string(&args.eval_data_path),
        "out_path" => option_string(&args.out_path),
        "model_path" => option_string(&args.model_path),
        "image_path" => option_string(&args.image_path),
        "epochs" => args.epochs.to_string(),
        "intervals" => args.intervals.to_string(),
        "correction_mode" => correction_mode_name(args.correction_mode).to_string(),
        "seed" => args.seed.to_string(),
        "image_width" => args.image_width.to_string(),
        "image_height" => args.image_height.to_string(),
        "image_features" => args.image_features.as_str().to_string(),
        "image_resize" => args.image_resize.as_str().to_string(),
        "samples_per_class" => args.samples_per_class.to_string(),
        "top_k" => args.top_k.to_string(),
        "target_mse" => args
            .target_mse
            .map(|value| value.to_string())
            .unwrap_or_else(|| "<none>".to_string()),
        "matrix_models" => list_string(
            args.matrix_models
                .iter()
                .map(|value| matrix_model_name(*value).to_string()),
        ),
        "matrix_features" => list_string(
            args.matrix_features
                .iter()
                .map(|value| value.as_str().to_string()),
        ),
        "matrix_image_sizes" => list_string(
            args.matrix_image_sizes
                .iter()
                .map(|value| value.to_string()),
        ),
        "matrix_intervals" => {
            list_string(args.matrix_intervals.iter().map(|value| value.to_string()))
        }
        "matrix_seeds" => list_string(args.matrix_seeds.iter().map(|value| value.to_string())),
        "matrix_resize_modes" => list_string(
            args.matrix_resize_modes
                .iter()
                .map(|value| value.as_str().to_string()),
        ),
        "matrix_correction_modes" => list_string(
            args.matrix_correction_modes
                .iter()
                .map(|value| correction_mode_name(*value).to_string()),
        ),
        "matrix_top" => args.matrix_top.to_string(),
        "debug_out_path" => option_string(&args.debug_out_path),
        "debug_train_data_path" => option_string(&args.debug_train_data_path),
        "debug_limit" => args.debug_limit.to_string(),
        "debug_samples" => args.debug_samples.as_str().to_string(),
        "debug_neighbors" => args.debug_neighbors.to_string(),
        "evolution_population" => args.evolution_population.to_string(),
        "evolution_generations" => args.evolution_generations.to_string(),
        "evolution_elite_count" => args.evolution_elite_count.to_string(),
        "evolution_mutation_rate" => args.evolution_mutation_rate.to_string(),
        "evolution_validation_ratio" => args.evolution_validation_ratio.to_string(),
        "evolution_threads" => args.evolution_threads.to_string(),
        "evolution_feature_modes" => list_string(
            args.evolution_feature_modes
                .iter()
                .map(|value| value.as_str().to_string()),
        ),
        "evolution_image_sizes" => list_string(
            args.evolution_image_sizes
                .iter()
                .map(|value| value.to_string()),
        ),
        "evolution_resize_modes" => list_string(
            args.evolution_resize_modes
                .iter()
                .map(|value| value.as_str().to_string()),
        ),
        "evolution_top_k_values" => list_string(
            args.evolution_top_k_values
                .iter()
                .map(|value| value.to_string()),
        ),
        "evolution_memory_penalty_per_mb" => args.evolution_memory_penalty_per_mb.to_string(),
        "evolution_inference_penalty_per_ms" => args.evolution_inference_penalty_per_ms.to_string(),
        _ => "<unknown>".to_string(),
    }
}

fn matrix_model_name(model: MatrixModel) -> &'static str {
    match model {
        MatrixModel::Pann => "pann",
        MatrixModel::Panc => "panc",
        MatrixModel::Centroid => "centroid",
    }
}

fn option_string(value: &Option<String>) -> String {
    value.clone().unwrap_or_else(|| "<none>".to_string())
}

fn list_string(values: impl Iterator<Item = String>) -> String {
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        "[]".to_string()
    } else {
        values.join(",")
    }
}
