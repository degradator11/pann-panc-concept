# PANN/PANC Concept Prototype

Rust research prototype for public-source reconstructions inspired by Progress
PANN/PANC materials.

This project does **not** claim to reproduce Progress, Inc.'s private software,
benchmarks, or proprietary implementation. It implements benchmarkable,
public-material-based models for experimentation.

## What PANN And PANC Mean

- **PANN**: Progressive Artificial Neural Network. In this prototype, inputs are
  mapped into value intervals, each input/interval/output has a corrective
  weight, and training updates the active corrective weights from output error.
- **PANC**: Progressive Associative Neuromorphic Comparator. In this prototype,
  it is represented as a PANC-like analogue comparator that recognizes inputs by
  comparing them against stored reference examples.

## Requirements

- Rust toolchain with Cargo.
- Git, if you want to commit or push changes.

Check your toolchain:

```powershell
rustc --version
cargo --version
```

## Build

From the repository root:

```powershell
cargo build
```

Optimized release build:

```powershell
cargo build --release
```

Compiled binaries are written to:

```text
target\debug\
target\release\
```

## Test And Lint

Run the full test suite:

```powershell
cargo test
```

Run Clippy with warnings treated as errors:

```powershell
cargo clippy --all-targets -- -D warnings
```

## Run The Demo

```powershell
cargo run --bin progress-ai-demo
```

The demo trains a tiny PANN-like classifier and runs a tiny PANC-like comparator
example.

## Run Research Benchmarks

The benchmark binary supports tabular and image-oriented subcommands:

```powershell
cargo run --bin research-bench -- pann-iris --format json
cargo run --bin research-bench -- panc-iris --format csv
cargo run --bin research-bench -- pann-synthetic --epochs 20 --intervals 16
cargo run --bin research-bench -- panc-synthetic
cargo run --bin research-bench -- pann-image-synthetic --image-size 16
cargo run --bin research-bench -- panc-image-synthetic --image-size 16
```

Useful options:

```text
--format json|csv
--data path\to\iris.csv
--epochs 12
--intervals 8
--seed 42
--image-size 16
--image-width 16
--image-height 16
--samples-per-class 80
```

The built-in Iris dataset is stored at:

```text
data\iris.csv
```

Custom Iris-style CSV input must have this header shape:

```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
```

Example with a custom CSV:

```powershell
cargo run --bin research-bench -- pann-iris --data C:\path\to\iris.csv --format json
```

## Image Recognition Benchmarks

Images are converted into numeric vectors before being passed into PANN or
PANC:

```text
image -> resize -> grayscale pixels -> values in [0, 1] -> classifier
```

For example, a 16x16 image becomes a vector with 256 values. With PANN and 8
intervals over 3 classes, the corrective-weight tensor has:

```text
256 pixels * 8 intervals * 3 classes = 6144 weights
```

Synthetic image benchmarks generate simple vertical, horizontal, and diagonal
line-pattern images in memory:

```powershell
cargo run --bin research-bench -- pann-image-synthetic --image-size 16 --epochs 12 --intervals 8 --format json
cargo run --bin research-bench -- panc-image-synthetic --image-size 16 --format csv
```

Folder-based image benchmarks expect one subdirectory per class:

```text
my-images\
  cat\
    cat-001.png
    cat-002.jpg
  dog\
    dog-001.png
    dog-002.jpg
```

Run PANN or PANC-like recognition over that folder:

```powershell
cargo run --bin research-bench -- pann-image-folder --data C:\path\to\my-images --image-size 32 --epochs 20 --intervals 8
cargo run --bin research-bench -- panc-image-folder --data C:\path\to\my-images --image-size 32
```

Supported image formats are PNG and JPEG. Images are resized to the configured
size and converted to grayscale.

## Library Usage

Minimal PANN example:

```rust
use progress_ai::pann::{Distributor, PannModel, one_hot};

let samples = vec![vec![0.1, 0.1], vec![0.9, 0.9]];
let labels = vec![0, 1];
let targets = labels
    .iter()
    .map(|label| one_hot(*label, 2))
    .collect::<Vec<_>>();

let mut model = PannModel::with_unit_ranges(2, 4, 2, Distributor::HardBin)?;
model.train_epoch_difference(&samples, &targets)?;

let prediction = model.predict(&[0.9, 0.9])?;
```

Config-driven PANN example:

```rust
use progress_ai::pann::{
    CorrectionMode, Distributor, IntervalStrategy, PannConfig, PannModel,
};

let mut config = PannConfig::new(2, 8, 2);
config.distributor = Distributor::Gaussian {
    radius: 1,
    sigma: 1.0,
};
config.interval_strategy = IntervalStrategy::Uniform;
config.correction_mode = CorrectionMode::DifferenceLeastSquares;

let model = PannModel::from_training_data_with_config(&samples, config)?;
```

Minimal PANC-like comparator example:

```rust
use progress_ai::panc::{PancComparator, SimilarityMetric};

let mut comparator = PancComparator::new(SimilarityMetric::Euclidean);
comparator.add_reference(vec![0.1, 0.1], 0, ())?;
comparator.add_reference(vec![0.9, 0.9], 1, ())?;

let prediction = comparator.predict_label(&[0.85, 0.85], 1)?;
let explanation = comparator.explain(&[0.85, 0.85], 1, 2)?;
```

Binary PANC-like comparator example:

```rust
use progress_ai::panc::{BinaryEncoder, BinaryPancComparator, BinarySimilarityMetric};

let encoder = BinaryEncoder::new(0.5);
let mut comparator = BinaryPancComparator::new(BinarySimilarityMetric::Jaccard);

comparator.add_reference(encoder.encode(&[1.0, 0.0, 1.0]), "class-a", ())?;
let query = encoder.encode(&[0.9, 0.1, 0.8]);
let prediction = comparator.predict_label(&query, 1)?;
```

Image vectorization example:

```rust
use progress_ai::vision::{ImageVectorConfig, load_image_as_vector};

let config = ImageVectorConfig::new(32, 32);
let vector = load_image_as_vector("digit.png", config)?;
assert_eq!(vector.len(), 32 * 32);
```

## Project Layout

```text
src\pann\                       PANN-like config, model, activation, training
src\panc\                       PANC-like dense and binary comparators
src\preprocess.rs               Dataset preprocessing utilities
src\vision.rs                   Image loading and vectorization utilities
src\bin\research-bench.rs       Benchmark CLI entrypoint
src\bin\research_bench\         Benchmark args, datasets, metrics, runners
tests\research_integration.rs   Synthetic integration tests
data\iris.csv                   Local Iris benchmark data
technology-implementation\      Public-source analysis and implementation notes
```

## Git Push Helper

If you use `push-with-token.local.bat`, keep it local. Files matching
`*.local.bat` are ignored because they may contain access tokens.

You can also push normally after authenticating with GitHub:

```powershell
git push -u origin master
```

## Notes And Limitations

- PANN implementation is a public-source reconstruction suitable for research.
- PANC implementation is a PANC-like comparator baseline, not a proprietary
  clone.
- Multilayer PANN, GPU kernels, analog/memristor simulation, and exact
  proprietary PANC behavior are out of scope.
- Review patent/licensing concerns before commercial use.
