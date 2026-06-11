# Technology Implementation Notes

This directory collects implementation-oriented notes for reproducing a
Progress/PANN-like model from public materials: patents, publications, and the
progress.ai website.

## Short conclusion

It appears possible to build a simplified PANN-like prototype from the published
patents and papers. The public material discloses the core architecture:

- inputs mapped to value intervals
- synapses with multiple corrective weights per input interval
- distributors that select one or more corrective weights from each synapse
- neurons that sum selected corrective weights
- a correction controller that modifies active weights using output error

It is not realistic to reproduce the company's exact model, implementation, or
claimed performance from patents alone. Missing details include preprocessing,
interval selection heuristics, initialization, benchmark configuration, private
software optimizations, GPU implementation choices, and current training code.

## Files

- `patents-and-sources.md`: source map with patent/publication links.
- `pann-core-implementation.md`: architecture, data structures, training loop,
  inference loop, and matrix form.
- `panc-public-implementation-notes.md`: what can be inferred about PANC and
  what remains underspecified.
- `reproduction-plan.md`: practical implementation roadmap and validation plan.
- `missing-parts-analysis.md`: public-information gap analysis and difficulty
  estimates.

## Runnable prototype

The root project now contains a Rust prototype with CSV/vector benchmarks and
image-folder benchmarks. See `../README.md` for build commands, training and
evaluation commands, and real dataset links including Microsoft/Kaggle Cats and
Dogs, Oxford-IIIT Pets, Fruits-360, CIFAR-10, and Open Images.

Image-folder benchmarks expect one class directory per label:

```text
dataset-root/
  cat/
    cat-001.jpg
  dog/
    dog-001.jpg
```

The benchmark performs a deterministic 80/20 train/evaluation split and reports
both `train_accuracy` and `test_accuracy`. For image folders, it can also train
from one folder and evaluate against another folder with matching class names:

```powershell
cargo run --release --bin research-bench -- pann-image-folder --data C:\datasets\PetImagesShort --eval-data C:\datasets\PetImagesShort\Eval --image-size 32 --epochs 12 --intervals 8 --image-features combined --format json
```

Image benchmarks support `--image-features pixels`, `color`, `hog`,
`combined`, and `rich`. The `combined` mode is the compact classical baseline.
The `rich` mode is the current strongest Cats/Dogs classical baseline because
it adds HSV histograms, color moments, and local binary pattern texture
features before PANN/PANC processing.

Image preprocessing now also supports `--image-resize stretch`,
`center-crop`, and `letterbox`. Stretch is the default historical behavior.
Center crop and letterbox are intended for normalization experiments on
real-world photo datasets where aspect ratio and object framing may affect
generalization.

The benchmark CLI also supports persistent image artifacts with
`train-pann-image-folder`, `eval-pann`, `predict-pann`,
`train-panc-image-folder`, `eval-panc`, and `predict-panc`. See the root
README's artifact section for the full command examples. Artifact evaluation
reports per-class accuracy, a confusion matrix, and a short list of
misclassified image paths for diagnostics.

Artifact evaluation can also write a static debug report with `--debug-out`.
The report includes `index.html`, prediction CSV/JSON files, per-class and
confusion CSVs, processed image steps, scaled feature vectors, and per-sample
summaries for selected images.

The debug report is now failure-analysis oriented. It ranks high-confidence
wrong and ambiguous wrong samples, buckets errors by simple image statistics,
compares resize-mode predictions, and can show nearest training examples in
the same scaled feature space.

For repeatable sweeps, use `image-matrix` to compare models, feature modes,
image sizes, resize modes, intervals, and seeds while writing CSV or JSON
reports under the ignored `reports/` directory.

The public Progress tests page reports target MSE, epoch/error, and training
time. The prototype mirrors that reporting style with `pann-learning-curve`,
which emits per-epoch MSE before/after training, train accuracy, test accuracy,
and elapsed milliseconds.

## Legal boundary

These notes are for technical analysis. They are not legal advice. Several
Progress patents appear active. Building or using an implementation for
commercial purposes can require a freedom-to-operate review and/or a license.
