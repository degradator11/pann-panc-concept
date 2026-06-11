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

Image benchmarks support `--image-features pixels`, `color`, `hog`, and
`combined`. The `combined` mode is the current recommended classical baseline
because it adds color histograms, coarse intensity layout, and HOG-like edge
features before PANN/PANC processing.

## Legal boundary

These notes are for technical analysis. They are not legal advice. Several
Progress patents appear active. Building or using an implementation for
commercial purposes can require a freedom-to-operate review and/or a license.
