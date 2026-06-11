use std::error::Error;
use std::fs;
use std::path::Path;

use progress_ai::panc::SimilarityMetric;
use progress_ai::pann::{FeatureRange, PannModelSnapshot};
use progress_ai::vision::{ImageFeatureMode, ImageResizeMode, ImageVectorConfig};
use serde::{Deserialize, Serialize};

pub const ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelArtifact {
    PannImage(PannImageArtifact),
    PancImage(PancImageArtifact),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PannImageArtifact {
    pub version: u32,
    pub class_names: Vec<String>,
    pub image: ImageArtifact,
    pub preprocessing: PreprocessingArtifact,
    pub model: PannModelSnapshot,
    pub epochs_trained: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PancImageArtifact {
    pub version: u32,
    pub class_names: Vec<String>,
    pub image: ImageArtifact,
    pub preprocessing: PreprocessingArtifact,
    pub metric: SimilarityMetric,
    pub references: Vec<PancReferenceArtifact>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageArtifact {
    pub width: u32,
    pub height: u32,
    pub feature_mode: String,
    #[serde(default = "default_resize_mode")]
    pub resize_mode: String,
}

impl ImageArtifact {
    pub fn from_config(config: ImageVectorConfig) -> Self {
        Self {
            width: config.width,
            height: config.height,
            feature_mode: config.feature_mode.as_str().to_string(),
            resize_mode: config.resize_mode.as_str().to_string(),
        }
    }

    pub fn to_config(&self) -> Result<ImageVectorConfig, Box<dyn Error>> {
        let feature_mode = self
            .feature_mode
            .parse::<ImageFeatureMode>()
            .map_err(|error| format!("invalid artifact image feature mode: {error}"))?;
        let resize_mode = self
            .resize_mode
            .parse::<ImageResizeMode>()
            .map_err(|error| format!("invalid artifact image resize mode: {error}"))?;
        Ok(ImageVectorConfig::new(self.width, self.height)
            .with_feature_mode(feature_mode)
            .with_resize_mode(resize_mode))
    }
}

fn default_resize_mode() -> String {
    ImageResizeMode::Stretch.as_str().to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessingArtifact {
    pub min_max_ranges: Vec<FeatureRange>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PancReferenceArtifact {
    pub vector: Vec<f64>,
    pub label: usize,
}

pub fn save_artifact(
    path: impl AsRef<Path>,
    artifact: &ModelArtifact,
) -> Result<(), Box<dyn Error>> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(artifact)?)?;
    Ok(())
}

pub fn load_artifact(path: impl AsRef<Path>) -> Result<ModelArtifact, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&contents)?)
}
