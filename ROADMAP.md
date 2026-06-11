# PANN/PANC Prototype Roadmap

This file tracks implementation progress, experiment results, and near-term
priorities for the Rust PANN/PANC research prototype.

The project is a public-source reconstruction and benchmark harness. It is not
a proprietary Progress clone.

## Current Status

The prototype can:

- train and evaluate PANN-like models on vector and image-folder datasets
- run PANC-like analogue comparator benchmarks
- load CSV/vector data and class-folder image datasets
- use deterministic train/test splits
- use separate image train/eval folders through `--eval-data`
- skip corrupt image files during folder benchmarks
- compare image feature modes: `pixels`, `color`, `hog`, `combined`, `rich`,
  and `rich-spatial`
- compare image resize modes: `stretch`, `center-crop`, and `letterbox`
- compare PANN correction modes in benchmark matrix runs
- report JSON or CSV metrics
- write static image eval debug reports with original/processed images,
  prediction rows, scaled feature vectors, and HTML summary

Current image recognition quality is early-stage. The pipeline works, but
Cats/Dogs accuracy is still modest with classical features.

The main new finding is that training can drive PANN train error down over
multiple epochs, but held-out Cats/Dogs accuracy still plateaus around 64-66%.
That means the near-term bottleneck is image representation and diagnostics,
not artifact persistence or simply running more epochs.

## Completed Milestones

- Initial Rust crate outside `technology-implementation/`
- Public-material PANN/PANC implementation notes
- PANN core model with configurable intervals, distributors, correction modes,
  and plasticity schedule
- PANC-like dense comparator with multiple similarity metrics
- Preprocessing utilities for scaling, clipping, one-hot labels, and splits
- `research-bench` CLI for Iris, synthetic data, and image folders
- Small Iris CSV committed locally
- Image-folder pipeline for PNG/JPEG datasets
- Corrupt image skip behavior for folder benchmarks
- Handcrafted image features:
  - raw grayscale pixels
  - RGB color histograms
  - HOG-like edge buckets
  - combined color/layout/edge vector
  - rich HSV/color-moment/texture vector
  - rich-spatial regional HSV vector
- Separate train/eval image folders with class-name label matching
- Persistent PANN/PANC-like image artifacts for train/eval/predict workflows
- Image benchmark matrix command with CSV/JSON report output, including
  correction-mode sweeps for PANN
- PANN learning-curve reports with epoch/MSE/accuracy/time rows
- Image resize modes for stretch, center-crop, and letterbox preprocessing
- Artifact eval diagnostics with per-class accuracy, confusion matrix, and a
  short misclassified-image list
- Static debug reports for image artifact eval via `--debug-out`
- Failure-analysis report v2 with ranked wrong examples, image buckets, resize
  sensitivity, and nearest training examples
- README dataset links and run instructions

## Current Working Interpretation

What is working:

- PANN/PANC image-folder commands are usable for real class-folder datasets.
- Saved image artifacts work for train once, evaluate later, and predict one
  image workflows.
- Rich handcrafted features improve Cats/Dogs eval accuracy over raw pixels,
  HOG-only, and the earlier combined feature vector.
- Learning-curve reports now show whether PANN is actually reducing training
  error step by step.

What is still weak:

- Eval accuracy is far below train accuracy, so the model is overfitting the
  small training sample and/or the feature vector is not general enough.
- Artifact persistence was necessary infrastructure, but it does not improve
  recognition quality by itself.
- We do not yet know whether most failures come from Cat, Dog, corrupt/odd
  inputs, image framing, or weak visual descriptors.

Decision: keep the next milestone focused on diagnostics and normalization.
Only after we can see the failures clearly should we add heavier feature
extractors or more model complexity.

## Latest Cats/Dogs Snapshot

Dataset layout used:

```text
Train: C:\Users\vilex\Downloads\kagglecatsanddogs_5340\PetImages_short
Eval:  C:\Users\vilex\Downloads\kagglecatsanddogs_5340\PetImages_short\Eval
```

Skipped unreadable files:

- Train: 2 Cat, 2 Dog
- Eval: 9 Cat, 10 Dog

Original results with `--image-size 32`:

| Model | Features | Train Accuracy | Eval Accuracy |
| --- | --- | ---: | ---: |
| PANN | pixels | 100.0% | 54.2% |
| PANN | color | 61.9% | 50.9% |
| PANN | hog | 93.5% | 56.1% |
| PANN | combined | 97.7% | 60.7% |
| PANC-like | pixels | 100.0% | 54.1% |
| PANC-like | color | 74.6% | 54.1% |
| PANC-like | hog | 100.0% | 56.7% |
| PANC-like | combined | 100.0% | 57.8% |

Interpretation: `combined` features show real improvement over raw pixels,
especially for PANN, but the model is not yet a strong cat/dog recognizer.

Latest feature-quality result:

| Model | Features | Image Size | Seeds | Mean Eval Accuracy | Best Eval Accuracy |
| --- | --- | ---: | --- | ---: | ---: |
| PANN | combined | 64 | 1,2,3 | 62.4% | 62.8% |
| PANN | rich, stretch resize | 64 | 1,2,3 | 64.9% | 65.9% |
| PANN | rich, center-crop resize | 64 | 1,2,3 | 68.2% | 68.6% |
| PANN | rich-spatial, center-crop resize | 64 | 1,2,3 | 68.5% | 70.3% |
| PANC-like | rich | 64 | 1 | 59.6% | 59.6% |

Interpretation: `rich` features and 64px vectorization produced the first
repeatable movement beyond the old 60.7% Cats/Dogs ceiling. Center-crop
normalization raised the stable PANN result to about 68%. `rich-spatial`
produced the first 70%+ short-dataset run, but its mean Dog accuracy was weaker
than plain `rich`, so it is promising rather than a clean replacement.

Latest learning-curve result, modeled after the public Progress tests page's
target-MSE/epoch/error/time reporting:

| Setting | Target MSE | Epochs Completed | Final Train MSE | Eval Accuracy |
| --- | ---: | ---: | ---: | ---: |
| PANN rich 64px, seed 2 | 0.02 | 8 | 0.0176 | 65.4% |

Interpretation: training error can fall to a low target quickly while held-out
accuracy stays much lower. The next bottleneck is generalization and feature
quality, not just lowering training MSE.

Learning-curve takeaway:

- Multiple training steps are visible and measurable.
- Train MSE reached the target by epoch 8 in the smoke run.
- Eval accuracy did not rise at the same rate, so more epochs alone are not
  the next best lever.

## Implemented Milestone: Persistent Artifacts

Training can now produce reusable model files.

PANN commands:

```powershell
cargo run --release --bin research-bench -- train-pann-image-folder --data C:\datasets\cats-dogs\train --out models\cats-dogs-pann.json --image-size 64 --epochs 12 --intervals 8 --image-features rich --format json
cargo run --release --bin research-bench -- eval-pann --model models\cats-dogs-pann.json --data C:\datasets\cats-dogs\eval --format json
cargo run --release --bin research-bench -- predict-pann --model models\cats-dogs-pann.json --image C:\datasets\cats-dogs\eval\Cat\cat-001.jpg --format json
```

PANC-like commands:

```powershell
cargo run --release --bin research-bench -- train-panc-image-folder --data C:\datasets\cats-dogs\train --out models\cats-dogs-panc.json --image-size 64 --image-features rich --format json
cargo run --release --bin research-bench -- eval-panc --model models\cats-dogs-panc.json --data C:\datasets\cats-dogs\eval --format json
cargo run --release --bin research-bench -- predict-panc --model models\cats-dogs-panc.json --image C:\datasets\cats-dogs\eval\Dog\dog-001.jpg --format json
```

Artifact contents include:

- model kind and version
- class names
- image size and feature mode
- preprocessing ranges
- PANN config
- PANN weights and access counts
- PANC reference vectors and labels, for PANC artifacts

Implemented success criteria:

- train once, evaluate later without retraining
- predict one image from a saved artifact
- artifact load validates class names, feature dimensions, and config
- JSON artifact format is deterministic enough for tests

Latest smoke result on the short Cats/Dogs train/eval folders:

| Command | Accuracy |
| --- | ---: |
| `eval-pann`, combined 32px artifact | 60.7% |
| `eval-pann`, rich 64px artifact | 64.3% |
| `eval-panc` from saved artifact | 57.8% |

The saved-artifact eval results match the corresponding in-memory benchmark
settings. This confirms persistence correctness, not a quality improvement.

## Implemented Milestone: Benchmark Matrix

The CLI can now run repeatable experiment grids instead of one command at a
time.

Small matrix command:

```powershell
cargo run --release --bin research-bench -- image-matrix --data C:\datasets\cats-dogs\train --eval-data C:\datasets\cats-dogs\eval --out reports\cats-dogs-matrix.csv --format csv --matrix-models pann,panc --matrix-features pixels,combined --matrix-image-sizes 32 --matrix-intervals 8 --matrix-seeds 42 --epochs 12
```

Implemented success criteria:

- run PANN/PANC-like benchmarks across multiple seeds
- compare feature modes, image sizes, and interval counts
- write CSV and JSON report files
- include mean/min/max accuracy
- include train/inference time and memory estimates
- keep generated reports under ignored `reports/`

Latest smoke report on the short Cats/Dogs train/eval folders:

| Model | Features | Image Size | Seed | Intervals | Eval Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| PANN | pixels | 32 | 42 | 8 | 54.2% |
| PANN | combined | 32 | 42 | 8 | 60.7% |
| PANC-like | pixels | 32 | 42 | 0 | 54.1% |
| PANC-like | combined | 32 | 42 | 0 | 57.8% |

## Implemented Milestone: Rich Feature Mode

Goal: improve recognition quality beyond the 60% Cats/Dogs ceiling using
stronger classical image preprocessing before considering pretrained
embeddings.

Implemented additions:

- HSV histograms
- color moments
- local binary pattern texture features
- `--image-features rich`

Implemented success criteria:

- matrix report shows a repeatable gain across at least three seeds
- best Cats/Dogs eval accuracy improves over 60.7%
- feature implementation remains deterministic and dependency-light
- README and roadmap record the new best result

## In-Progress Milestone: Image Normalization And Diagnostics

Goal: understand the remaining error and improve input normalization before
adding pretrained embeddings.

Implemented in the first pass:

- explicit image resize modes:
  - `stretch`, the current behavior
  - `center-crop`, useful when the object is centered and background varies
  - `letterbox`, useful when aspect ratio should be preserved
- resize mode stored inside persisted image artifacts
- `image-matrix` can sweep `--matrix-resize-modes`
- artifact eval reports per-class accuracy
- artifact eval reports confusion matrix output
- artifact eval reports a short misclassified-example list with path, expected label,
  predicted label, and confidence/margin where available
- `image-matrix` rows now include per-class accuracy, confusion matrix, worst
  class, most common confusion, and train-vs-eval overfit gap
- `image-matrix` grouped summaries now include pooled per-class accuracy,
  worst mean class, best seed, best test accuracy, and mean overfit gap
- CSV matrix output writes a sibling `*.summary.csv` file beside the per-run
  row CSV
- `image-matrix` can sweep PANN correction modes with
  `--matrix-correction-modes`
- `image-matrix` can write sorted top-N rows with `--matrix-top`, plus a
  sibling `*.top.csv` file for CSV runs
- artifact eval can write static debug reports:
  - `index.html`
  - `config.json`
  - `metrics.json`
  - `failure_analysis.json`
  - `failure_buckets.csv`
  - `predictions.csv` and `predictions.json`
  - `per_class_accuracy.csv`
  - `confusion_matrix.csv`
  - selected sample folders with original/processed images
  - scaled feature vector CSVs
- debug reports now rank high-confidence wrong and ambiguous wrong samples
- debug reports bucket failures by brightness, contrast, orientation, and
  center-crop loss
- debug reports compare stretch, center-crop, and letterbox predictions for
  selected samples
- debug reports can show nearest training images in the same scaled feature
  space through inferred or explicit train-data folders
- in-memory `pann-image-folder` and `panc-image-folder` commands can write the
  same static debug reports with `--debug-out`, without saving a model artifact
  first

Remaining planned work:

- optionally add an interactive UI later if the static report is still not
  enough

Success criteria:

- report shows whether Cat or Dog is driving most errors
- report includes enough image paths to inspect repeated failure patterns
- report makes preprocessing mistakes visible by comparing original and
  processed images
- report answers which failure modes are most suspicious without manually
  opening every sample folder
- resize/crop mode gives a repeatable gain, or is documented as not helpful
- best Cats/Dogs eval accuracy improves beyond the old 65.9% feature-only best
  run

Normalization matrix command:

```powershell
cargo run --release --bin research-bench -- image-matrix --data C:\Users\vilex\Downloads\kagglecatsanddogs_5340\PetImages_short --eval-data C:\Users\vilex\Downloads\kagglecatsanddogs_5340\PetImages_short\Eval --out reports\cats-dogs-normalization-matrix.csv --format csv --matrix-models pann --matrix-features rich --matrix-image-sizes 64 --matrix-intervals 6,8,12 --matrix-seeds 1,2,3 --matrix-resize-modes stretch,center-crop,letterbox --epochs 12
```

Latest normalization matrix on the short Cats/Dogs train/eval folders:

| Resize | Intervals | Runs | Mean Eval Accuracy | Best Eval Accuracy | Worst Mean Class |
| --- | ---: | ---: | ---: | ---: | --- |
| center-crop | 12 | 3 | 68.2% | 68.6% | Dog, 65.1% |
| center-crop | 8 | 3 | 67.6% | 68.7% | Dog, 57.4% |
| center-crop | 6 | 3 | 65.9% | 66.2% | Dog, 64.1% |
| stretch | 12 | 3 | 65.9% | 66.6% | Cat, 65.8% |
| letterbox | 12 | 3 | 65.4% | 65.9% | Dog, 61.7% |

Interpretation: center-crop is the first repeatable normalization gain on the
short Cats/Dogs set, with the strongest mean at 12 intervals and best single
run at seed 3 / 8 intervals. It still leaves Dogs as the weaker class in most
center-crop settings, so the next quality lever is feature extraction or class
calibration, not more epochs alone.

Focused correction-mode comparison, using PANN rich 64px center-crop,
12 intervals, seed 2, and 12 epochs:

| Correction Mode | Eval Accuracy | Worst Class |
| --- | ---: | --- |
| difference least squares | 68.6% | Dog, 68.3% |
| difference patent proportional | 68.6% | Dog, 68.3% |
| ratio | 53.9% | Cat, 46.7% |

Interpretation: the two difference-style update rules tie on this smoke run,
while ratio update is much weaker for the current image vectors. Keep
least-squares difference as the default image benchmark mode.

## Benchmark Roadmap

Planned benchmark improvements:

- none currently open

## Feature Roadmap

Classical image features to try before pretrained embeddings:

- improved HSV histograms with spatial regions: implemented as
  `--image-features rich-spatial`
- normalized color moments
- improved HOG cell/block normalization
- multi-scale local binary patterns or other texture descriptors
- edge density by image region
- simple symmetry/layout features
- simple foreground/background normalization for object datasets

Optional later feature path:

- pretrained image embeddings as input vectors for PANN/PANC
- keep this optional and clearly separate from the public-source classical
  reconstruction
- compare against a small conventional baseline so we know whether the
  bottleneck is PANN/PANC or the feature vector itself

## Dataset Roadmap

Useful datasets to test:

- Microsoft/Kaggle Cats and Dogs: hard real-world binary task
- Fruits-360: easier object classification with cleaner centered images
- Oxford-IIIT Pets: harder multi-class pet recognition
- CIFAR-10: standard small-image benchmark, needs importer/converter
- synthetic image patterns: regression tests and sanity checks

## Open Questions

- Which feature family gives stable gains across seeds?
- Does PANN benefit more from feature engineering than PANC-like nearest
  analogue comparison?
- Which PANN interval count is best for low-dimensional handcrafted features?
- Should PANN training use matrix/batch updates for image benchmarks?
- What is the most useful default benchmark matrix for fast iteration?
- Does preserving aspect ratio help Cats/Dogs more than stretching?
- Are failures concentrated in one class or in ambiguous/corrupt images?
- At what point do classical features stop paying off compared with using an
  external embedding model as a fixed vectorizer?

## Out Of Scope For Now

- exact proprietary Progress/PANN/PANC compatibility
- GPU kernels
- memristor or analogue hardware simulation
- object detection bounding boxes
- multilayer neural network training
- commercial freedom-to-operate conclusions
