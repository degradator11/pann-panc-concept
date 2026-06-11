# PANN/PANC Concept Prototype

Rust research prototype for public-source reconstructions inspired by Progress
PANN/PANC materials.

This project does **not** claim to reproduce Progress, Inc.'s private software,
benchmarks, or proprietary implementation. It implements benchmarkable,
public-material-based models for experimentation.

See [ROADMAP.md](ROADMAP.md) for current progress, experiment snapshots, and
planned milestones.

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
--correction-mode difference-ls|patent-proportional|ratio
--seed 42
--image-size 16
--image-width 16
--image-height 16
--image-resize stretch|center-crop|letterbox
--samples-per-class 80
--debug-out reports\debug-folder
--debug-train-data path\to\train-folder
--debug-limit 50
--debug-samples misclassified|all|correct
--debug-neighbors 5
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
image -> resize/normalize -> feature vector -> values in [0, 1] -> classifier
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
cargo run --bin research-bench -- pann-image-folder --data C:\path\to\my-images --image-size 64 --epochs 20 --intervals 8 --image-features rich
cargo run --bin research-bench -- panc-image-folder --data C:\path\to\my-images --image-size 64 --image-features rich
```

Supported image formats are PNG and JPEG. Images are resized to the configured
size, then vectorized. `--image-features pixels` uses raw grayscale pixels.
`--image-features combined` uses a smaller handcrafted vector with color
histograms, spatial intensity statistics, and HOG-like edge buckets. The
additional `color` and `hog` modes are available for ablation runs.
`--image-features rich` adds HSV histograms, RGB/HSV color moments, and local
binary pattern texture features. `--image-features rich-spatial` extends that
with per-region HSV histograms so the classifier can see where colors appear,
at the cost of a larger vector.

Resize modes control how non-square images become fixed-size vectors:

```text
--image-resize stretch       resize directly to width x height; default
--image-resize center-crop   crop the central square, then resize
--image-resize letterbox     preserve aspect ratio with neutral gray padding
```

The resize mode is part of the model's preprocessing. Artifact training stores
it in the JSON file, and `eval-*` / `predict-*` reuse the saved mode.

## Real Image Datasets

The image-folder benchmark works with any classification dataset arranged as
one directory per class. By default it uses a deterministic 80/20 split: it
trains on 80% of the images and reports accuracy on both the training portion
and the held-out 20% evaluation portion.

You can also provide an explicit evaluation folder with `--eval-data`. In that
mode, `--data` is used for training and `--eval-data` is used for evaluation.
Class labels are matched by directory name:

```text
C:\datasets\PetImagesShort\
  Cat\
  Dog\
  Eval\
    Cat\
    Dog\
```

Use:

```powershell
cargo run --release --bin research-bench -- pann-image-folder --data C:\datasets\PetImagesShort --eval-data C:\datasets\PetImagesShort\Eval --image-size 64 --epochs 12 --intervals 8 --image-features rich --format json
cargo run --release --bin research-bench -- panc-image-folder --data C:\datasets\PetImagesShort --eval-data C:\datasets\PetImagesShort\Eval --image-size 64 --image-features rich --format json
```

In-memory image-folder benchmarks can also write the static debug report
without saving a model artifact first:

```powershell
cargo run --release --bin research-bench -- pann-image-folder --data C:\datasets\PetImagesShort --eval-data C:\datasets\PetImagesShort\Eval --image-size 64 --image-features rich --image-resize center-crop --intervals 12 --epochs 12 --debug-out reports\cats-dogs-in-memory-debug --debug-limit 25 --debug-samples misclassified --format json
```

The debug report uses the in-memory training split as nearest-neighbor
references unless `--debug-train-data` is provided.

Nested directories that do not contain images directly, such as `Eval`, are not
treated as training classes.

Good starter datasets:

| Dataset | Use case | Link | Notes |
| --- | --- | --- | --- |
| Microsoft/Kaggle Cats and Dogs | cat vs dog classification | https://www.microsoft.com/en-us/download/details.aspx?id=54765 | About 786 MB; usually extracts to `PetImages\Cat` and `PetImages\Dog`. |
| Oxford-IIIT Pet Dataset | pet species or breed classification | https://www.robots.ox.ac.uk/~vgg/data/pets/ | 37 pet categories, roughly 200 images each, with annotations. |
| Fruits-360 | apple/fruit classification | https://www.kaggle.com/datasets/moltean/fruits | Clean centered fruit images; a good fit for raw-pixel experiments. |
| CIFAR-10 | small multi-class benchmark including cat and dog | https://www.cs.toronto.edu/~kriz/cifar.html | Needs a converter because the official files are batch/binary formats, not class folders. |
| Open Images V7 | large-scale real-world image labels/detection | https://storage.googleapis.com/openimages/web/index.html | Too large for a first run; detection labels need additional pipeline work. |

### Cats vs Dogs

After downloading and extracting the Microsoft/Kaggle archive, point `--data`
at the directory that contains the `Cat` and `Dog` subdirectories:

```text
C:\datasets\PetImages\
  Cat\
  Dog\
```

Train and evaluate PANN:

```powershell
cargo run --bin research-bench -- pann-image-folder --data C:\datasets\PetImages --image-size 64 --epochs 12 --intervals 8 --image-features rich --format json
```

Evaluate the PANC-like comparator:

```powershell
cargo run --bin research-bench -- panc-image-folder --data C:\datasets\PetImages --image-size 64 --image-features rich --format json
```

### Apples Or Fruits

After downloading Fruits-360, use a directory with one folder per fruit class,
for example the `Training` folder:

```text
C:\datasets\fruits-360\Training\
  Apple Braeburn\
  Apple Golden 1\
  Banana\
  Orange\
```

Train and evaluate PANN:

```powershell
cargo run --bin research-bench -- pann-image-folder --data "C:\datasets\fruits-360\Training" --image-size 32 --epochs 12 --intervals 8 --image-features combined --format json
```

Evaluate the PANC-like comparator:

```powershell
cargo run --bin research-bench -- panc-image-folder --data "C:\datasets\fruits-360\Training" --image-size 32 --image-features combined --format json
```

To train a smaller apple-vs-other experiment, create a compact folder such as:

```text
C:\datasets\apple-binary\
  apple\
    apple-001.jpg
  other\
    banana-001.jpg
    orange-001.jpg
```

Then run:

```powershell
cargo run --bin research-bench -- pann-image-folder --data C:\datasets\apple-binary --image-size 32 --epochs 12 --intervals 8 --image-features combined
```

### Reading The Metrics

JSON and CSV output contain:

```text
train_accuracy    accuracy on the training split
test_accuracy     accuracy on the held-out evaluation split
train_ms          training/indexing time
inference_ms      evaluation time
memory_bytes      approximate model/reference memory
image_features    image vectorization mode, or none for non-image datasets
image_resize      resize/normalization mode, or none for non-image datasets
```

For PANN, `epochs` and `interval_count` affect training directly. For PANC-like
comparison, there is no iterative training; references are stored and evaluated
with top-k similarity voting.

### Benchmark Matrix

Use `image-matrix` to run a repeatable grid of image-folder experiments and
write a report file. This is useful when comparing PANN vs PANC-like behavior,
feature modes, image sizes, interval counts, and random seeds.

Small Cats/Dogs matrix:

```powershell
cargo run --release --bin research-bench -- image-matrix --data C:\datasets\cats-dogs\train --eval-data C:\datasets\cats-dogs\eval --out reports\cats-dogs-matrix.csv --format csv --matrix-models pann,panc --matrix-features pixels,combined,rich,rich-spatial --matrix-image-sizes 32,64 --matrix-intervals 8 --matrix-seeds 42 --matrix-resize-modes stretch,letterbox --epochs 12
```

Larger matrix:

```powershell
cargo run --release --bin research-bench -- image-matrix --data C:\datasets\cats-dogs\train --eval-data C:\datasets\cats-dogs\eval --out reports\cats-dogs-matrix.json --format json --matrix-models pann,panc --matrix-features pixels,hog,combined,rich,rich-spatial --matrix-image-sizes 16,32,64 --matrix-intervals 4,8,16 --matrix-seeds 1,2,3 --matrix-resize-modes stretch,center-crop,letterbox --epochs 12
```

Notes:

- PANN runs once per interval count.
- PANC-like comparison ignores interval count and runs once per
  model/feature/image-size/seed.
- `--matrix-resize-modes` adds resize/normalization modes to the sweep.
- `--matrix-correction-modes` adds PANN correction/update modes to the sweep.
  PANC-like comparison ignores this option because it does not use PANN
  correction rules.
- `--matrix-top 10` writes the best N matrix rows sorted by eval accuracy,
  worst-class accuracy, and lower overfit gap.
- CSV output contains per-run rows. When `--out reports\name.csv` is used, the
  command also writes `reports\name.summary.csv`; with `--matrix-top`, it also
  writes `reports\name.top.csv`.
- JSON output contains per-run rows plus grouped summaries with mean/min/max
  accuracy and optional `top_rows`.
- Matrix rows include per-class accuracy, confusion matrix, worst class, most
  common confusion, and train-vs-eval overfit gap.
- Matrix summaries include pooled per-class accuracy, worst mean class, best
  seed, best test accuracy, and mean overfit gap.
- Generated `reports/` files are ignored by git.

Example summary CSV columns:

```text
mean_test_accuracy
best_seed
best_test_accuracy
mean_overfit_gap
pooled_test_per_class_accuracy
worst_mean_class_name
worst_mean_class_accuracy
```

Correction-mode matrix:

```powershell
cargo run --release --bin research-bench -- image-matrix --data C:\datasets\cats-dogs\train --eval-data C:\datasets\cats-dogs\eval --out reports\cats-dogs-correction-matrix.csv --format csv --matrix-models pann --matrix-features rich --matrix-image-sizes 64 --matrix-intervals 12 --matrix-seeds 2 --matrix-resize-modes center-crop --matrix-correction-modes difference-ls,patent-proportional,ratio --epochs 12
```

Top-five matrix report:

```powershell
cargo run --release --bin research-bench -- image-matrix --data C:\datasets\cats-dogs\train --eval-data C:\datasets\cats-dogs\eval --out reports\cats-dogs-matrix.csv --format csv --matrix-models pann --matrix-features rich --matrix-image-sizes 64 --matrix-intervals 8,12 --matrix-seeds 1,2,3 --matrix-resize-modes stretch,center-crop --matrix-top 5 --epochs 12
```

On the short Cats/Dogs dataset, the focused correction-mode smoke run found
`difference-ls` and `patent-proportional` tied at **68.6%** eval accuracy, while
`ratio` fell to **53.9%**. Keep `difference-ls` as the default for image work
until broader evidence says otherwise.

### Learning Curve Reports

The public [Progress tests page](https://progress.ai/tests/) reports training
progress as target MSE, epoch, error, and elapsed training time. This prototype
can emit the same kind of PANN-oriented learning curve with
`pann-learning-curve`.

```powershell
cargo run --release --bin research-bench -- pann-learning-curve --data C:\datasets\cats-dogs\train --eval-data C:\datasets\cats-dogs\eval --out reports\cats-dogs-learning-curve.csv --format csv --image-size 64 --epochs 12 --intervals 8 --image-features rich --target-mse 0.02 --seed 2
```

The report contains one row per epoch:

```text
epoch
mean_mse_before
mean_mse_after
train_accuracy
test_accuracy
elapsed_ms
```

On the short Cats/Dogs dataset, PANN with `rich` features reached target
training MSE `0.02` at epoch 8, but held-out evaluation accuracy remained about
65%. That distinction matters: falling training error proves the model is
fitting the training vectors; it does not by itself prove strong image
recognition.

### Persistent Training Artifacts

The in-memory benchmark commands train and evaluate in one run. To train once
and reuse the result later, use the artifact commands.

Train PANN and write a model JSON file:

```powershell
cargo run --release --bin research-bench -- train-pann-image-folder --data C:\datasets\cats-dogs\train --out models\cats-dogs-pann.json --image-size 64 --epochs 12 --intervals 8 --image-features rich --format json
```

Evaluate the saved PANN artifact:

```powershell
cargo run --release --bin research-bench -- eval-pann --model models\cats-dogs-pann.json --data C:\datasets\cats-dogs\eval --format json
```

Artifact evaluation JSON includes diagnostics:

```text
per_class_accuracy       accuracy, correct count, and total per class
confusion_matrix         actual class rows with predicted-class counts
misclassified_examples   first misclassified image paths with expected/predicted labels
```

Generate a static debug report during artifact evaluation:

```powershell
cargo run --release --bin research-bench -- eval-pann --model models\cats-dogs-pann.json --data C:\datasets\cats-dogs\eval --debug-out reports\cats-dogs-debug --debug-limit 50 --debug-samples misclassified --format json
```

If the eval folder is named `Eval`, the command tries to infer the training
folder from its parent for nearest-neighbor examples. You can also pass it
explicitly:

```powershell
cargo run --release --bin research-bench -- eval-pann --model models\cats-dogs-pann.json --data C:\datasets\cats-dogs\eval --debug-train-data C:\datasets\cats-dogs\train --debug-out reports\cats-dogs-debug --debug-neighbors 5 --format json
```

The debug folder contains a static failure-analysis report:

```text
reports\cats-dogs-debug\
  index.html
  config.json
  metrics.json
  failure_analysis.json
  failure_buckets.csv
  predictions.csv
  predictions.json
  per_class_accuracy.csv
  confusion_matrix.csv
  samples\
    0000_expected_Cat_predicted_Dog_3000\
      step_0_original.png
      step_1_center_crop.png
      step_2_resize_exact.png
      step_3_scaled_feature_vector.csv
      neighbor_0_Cat_123.png
      summary.json
```

Open `index.html` to see:

- weakest class and most common confusion
- high-confidence wrong examples
- ambiguous wrong examples
- brightness/contrast/aspect/crop-loss failure buckets
- stretch vs center-crop vs letterbox prediction sensitivity
- nearest training examples in the same scaled feature space
- original vs processed image for selected samples

`--debug-samples misclassified` now selects a mix of high-confidence wrong and
ambiguous wrong examples. `all` exports the first matching samples regardless
of correctness; `correct` exports only correct predictions.

Predict one image with the saved PANN artifact:

```powershell
cargo run --release --bin research-bench -- predict-pann --model models\cats-dogs-pann.json --image C:\datasets\cats-dogs\eval\Cat\cat-001.jpg --format json
```

PANC-like artifacts use the same pattern:

```powershell
cargo run --release --bin research-bench -- train-panc-image-folder --data C:\datasets\cats-dogs\train --out models\cats-dogs-panc.json --image-size 64 --image-features rich --format json
cargo run --release --bin research-bench -- eval-panc --model models\cats-dogs-panc.json --data C:\datasets\cats-dogs\eval --format json
cargo run --release --bin research-bench -- predict-panc --model models\cats-dogs-panc.json --image C:\datasets\cats-dogs\eval\Dog\dog-001.jpg --format json
```

Artifact JSON stores model kind/version, class names, image size, feature mode,
resize mode, preprocessing ranges, and model data. PANN artifacts store weights
and access counts. PANC-like artifacts store reference vectors and labels.
Generated `models/` files are ignored by git.

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
use progress_ai::vision::{ImageResizeMode, ImageVectorConfig, load_image_as_vector};

let config = ImageVectorConfig::new(32, 32).with_resize_mode(ImageResizeMode::Letterbox);
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
