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
