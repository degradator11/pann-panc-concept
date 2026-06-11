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
- compare image feature modes: `pixels`, `color`, `hog`, and `combined`
- report JSON or CSV metrics

Current image recognition quality is early-stage. The pipeline works, but
Cats/Dogs accuracy is still modest with classical features.

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
- Separate train/eval image folders with class-name label matching
- README dataset links and run instructions

## Latest Cats/Dogs Snapshot

Dataset layout used:

```text
Train: C:\Users\vilex\Downloads\kagglecatsanddogs_5340\PetImages_short
Eval:  C:\Users\vilex\Downloads\kagglecatsanddogs_5340\PetImages_short\Eval
```

Skipped unreadable files:

- Train: 2 Cat, 2 Dog
- Eval: 9 Cat, 10 Dog

Results with `--image-size 32`:

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

## Next Milestone: Persistent Artifacts

Goal: make training produce reusable model files.

Planned commands:

```powershell
cargo run --release --bin research-bench -- train-pann-image-folder --data C:\datasets\cats-dogs\train --out models\cats-dogs-pann.json
cargo run --release --bin research-bench -- eval-pann --model models\cats-dogs-pann.json --data C:\datasets\cats-dogs\eval
cargo run --release --bin research-bench -- predict-pann --model models\cats-dogs-pann.json --image C:\datasets\cat.jpg
```

Artifact contents should include:

- model kind and version
- class names
- image size and feature mode
- preprocessing ranges
- PANN config
- PANN weights and access counts
- PANC reference vectors and labels, for PANC artifacts

Success criteria:

- train once, evaluate later without retraining
- predict one image from a saved artifact
- artifact load validates class names, feature dimensions, and config
- JSON artifact format is deterministic enough for tests

## Benchmark Roadmap

Planned benchmark improvements:

- run multiple seeds automatically
- emit summary tables for mean/min/max accuracy
- compare image sizes such as 16, 32, and 64
- compare interval counts such as 4, 8, 16, and 32
- compare PANN correction modes
- add confusion matrix output
- add per-class accuracy output
- store benchmark output files under an ignored `reports/` directory

## Feature Roadmap

Classical image features to try before pretrained embeddings:

- HSV histograms
- color moments
- improved HOG cell/block normalization
- local binary patterns or other texture descriptors
- center crop and square padding modes
- simple foreground/background normalization for object datasets

Optional later feature path:

- pretrained image embeddings as input vectors for PANN/PANC
- keep this optional and clearly separate from the public-source classical
  reconstruction

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
- What is the smallest artifact format that remains useful and inspectable?

## Out Of Scope For Now

- exact proprietary Progress/PANN/PANC compatibility
- GPU kernels
- memristor or analogue hardware simulation
- object detection bounding boxes
- multilayer neural network training
- commercial freedom-to-operate conclusions
