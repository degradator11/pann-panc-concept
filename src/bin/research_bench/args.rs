use std::env;
use std::error::Error;

use progress_ai::vision::{ImageFeatureMode, ImageVectorConfig};

#[derive(Debug)]
pub struct Args {
    pub command: String,
    pub format: OutputFormat,
    pub data_path: Option<String>,
    pub eval_data_path: Option<String>,
    pub epochs: usize,
    pub intervals: usize,
    pub seed: u64,
    pub image_width: u32,
    pub image_height: u32,
    pub image_features: ImageFeatureMode,
    pub samples_per_class: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
    Csv,
}

pub fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut raw = env::args().skip(1);
    let command = raw.next().ok_or(
        "usage: research-bench <pann-iris|pann-synthetic|pann-image-synthetic|pann-image-folder|panc-iris|panc-synthetic|panc-image-synthetic|panc-image-folder> [--format json|csv] [--data path] [--eval-data path] [--epochs n] [--intervals n] [--seed n] [--image-size n] [--image-features pixels|color|hog|combined] [--samples-per-class n]",
    )?;

    let mut args = Args {
        command,
        format: OutputFormat::Json,
        data_path: None,
        eval_data_path: None,
        epochs: 12,
        intervals: 8,
        seed: 42,
        image_width: 16,
        image_height: 16,
        image_features: ImageFeatureMode::Pixels,
        samples_per_class: 80,
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
            "--samples-per-class" => {
                args.samples_per_class = raw
                    .next()
                    .ok_or("--samples-per-class requires a value")?
                    .parse::<usize>()?;
            }
            other => return Err(format!("unknown option {other}").into()),
        }
    }

    Ok(args)
}

pub fn image_config(args: &Args) -> ImageVectorConfig {
    ImageVectorConfig::new(args.image_width, args.image_height)
        .with_feature_mode(args.image_features)
}

pub fn required_data_path(args: &Args) -> Result<&str, Box<dyn Error>> {
    args.data_path
        .as_deref()
        .ok_or_else(|| "--data path is required for image-folder benchmarks".into())
}
