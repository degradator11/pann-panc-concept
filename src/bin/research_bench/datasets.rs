use std::collections::HashMap;
use std::error::Error;
use std::fs;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use progress_ai::preprocess::Dataset;

pub fn load_iris(path: Option<&str>) -> Result<Dataset, Box<dyn Error>> {
    let contents = if let Some(path) = path {
        fs::read_to_string(path)?
    } else {
        include_str!("../../../data/iris.csv").to_string()
    };

    let mut rdr = csv::Reader::from_reader(contents.as_bytes());
    let mut samples = Vec::new();
    let mut labels = Vec::new();
    let mut class_to_label = HashMap::<String, usize>::new();
    let mut class_names = Vec::<String>::new();

    for record in rdr.records() {
        let record = record?;
        let sample = (0..4)
            .map(|index| record[index].parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        let species = record[4].to_string();
        let label = if let Some(label) = class_to_label.get(&species) {
            *label
        } else {
            let label = class_names.len();
            class_to_label.insert(species.clone(), label);
            class_names.push(species);
            label
        };
        samples.push(sample);
        labels.push(label);
    }

    Ok(Dataset {
        samples,
        labels,
        class_names,
    })
}

pub fn load_embedding_csv(path: &str) -> Result<Dataset, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let label_index = headers
        .iter()
        .position(|header| {
            matches!(
                header.trim().to_ascii_lowercase().as_str(),
                "label" | "class" | "class_name" | "target"
            )
        })
        .ok_or("embedding CSV needs a label, class, class_name, or target column")?;
    let path_index = headers.iter().position(|header| {
        matches!(
            header.trim().to_ascii_lowercase().as_str(),
            "path" | "image" | "image_path" | "file"
        )
    });
    let feature_indexes = headers
        .iter()
        .enumerate()
        .filter_map(|(index, _)| {
            if index == label_index || Some(index) == path_index {
                None
            } else {
                Some(index)
            }
        })
        .collect::<Vec<_>>();
    if feature_indexes.is_empty() {
        return Err("embedding CSV needs at least one feature column".into());
    }

    let mut samples = Vec::new();
    let mut labels = Vec::new();
    let mut class_to_label = HashMap::<String, usize>::new();
    let mut class_names = Vec::<String>::new();

    for record in rdr.records() {
        let record = record?;
        let sample = feature_indexes
            .iter()
            .map(|index| record[*index].parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        let class_name = record[label_index].to_string();
        let label = if let Some(label) = class_to_label.get(&class_name) {
            *label
        } else {
            let label = class_names.len();
            class_to_label.insert(class_name.clone(), label);
            class_names.push(class_name);
            label
        };
        samples.push(sample);
        labels.push(label);
    }

    if samples.is_empty() {
        return Err("embedding CSV contains no rows".into());
    }

    Ok(Dataset {
        samples,
        labels,
        class_names,
    })
}

pub fn synthetic_dataset(seed: u64) -> Dataset {
    let centers: [[f64; 2]; 3] = [[0.15, 0.15], [0.85, 0.2], [0.5, 0.85]];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut samples = Vec::new();
    let mut labels = Vec::new();

    for (label, center) in centers.iter().enumerate() {
        for _ in 0..80 {
            let x = (center[0] + rng.gen_range(-0.05_f64..0.05_f64)).clamp(0.0, 1.0);
            let y = (center[1] + rng.gen_range(-0.05_f64..0.05_f64)).clamp(0.0, 1.0);
            samples.push(vec![x, y]);
            labels.push(label);
        }
    }

    Dataset {
        samples,
        labels,
        class_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn embedding_csv_loader_reads_labels_and_features() {
        let path = std::env::temp_dir().join(format!(
            "progress_ai_embeddings_{}_{}.csv",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(
            &path,
            "path,label,e0,e1\ncat-1.jpg,Cat,0.1,0.2\ndog-1.jpg,Dog,0.8,0.7\n",
        )
        .unwrap();

        let dataset = load_embedding_csv(path.to_str().unwrap()).unwrap();

        assert_eq!(dataset.samples, vec![vec![0.1, 0.2], vec![0.8, 0.7]]);
        assert_eq!(dataset.labels, vec![0, 1]);
        assert_eq!(dataset.class_names, vec!["Cat", "Dog"]);
        fs::remove_file(path).unwrap();
    }
}
