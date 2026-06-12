# PANN/PANC Breakthrough Assessment

Date: 2026-06-12

## Short Verdict

The current implementation is not a breakthrough by itself. It is a useful
research baseline that confirms one important thing: the low-resolution
analogue-comparison direction is not nonsense.

Progress.ai's claimed PANC technology could be significant if its private
implementation really provides compact encodings, fast large-library comparison,
editable knowledge, low-power hardware paths, and robust recognition. Public
information is not enough to verify that.

## What We Have Proven Locally

The prototype behaves very differently depending on dataset structure:

- Cats vs dogs, natural photos with backgrounds: roughly 60-67% accuracy in our
  previous runs.
- Fruits-small, clean centered objects, 14 classes:
  - PANC-like 32x32 raw pixels, top-k 3: 96.45% eval accuracy.
  - PANN 32x32 raw pixels, 5 epochs, 8 intervals: 98.28% eval accuracy.

That is a strong signal that representation and preprocessing dominate results.
The same general approach struggles when the object is mixed with people,
backgrounds, scale changes, multiple animals, or bad crops, but works well when
the image artifact mostly contains the object to recognize.

## What This Means

The current PANC-like code is close to known families of methods:

- nearest-neighbor classification;
- template/prototype comparison;
- feature-vector similarity search;
- top-k voting over stored analogues.

Those ideas are not new. The interesting part is the Progress.ai-style package:

- tiny image artifacts;
- little or no conventional training;
- editable analogue libraries;
- explainable top matches;
- possible compact binary comparison format;
- possible hardware-friendly comparator.

That combination could be valuable, especially for edge recognition, medical
comparison, industrial inspection, and systems where new examples must become
usable immediately without retraining a large network.

## What Is Still Missing

Public sources do not expose the proprietary PANC core:

- exact binary comparison format;
- exact coefficient-of-similarity formula;
- analogue-library construction algorithm;
- fast comparator data structure or hardware layout;
- proof that large-library comparison scales as claimed.

Because of that, our Rust project should continue to label PANC as a
"PANC-like comparator baseline", not a Progress-equivalent implementation.

## Why Cats/Dogs Failed Relative To Fruits

Cats/dogs images contain background, people, furniture, grass, multiple animals,
scale differences, and crops where the animal's head can disappear. A holistic
comparator will often compare the whole photo rather than the animal identity.

Fruits-360-style data is nearly ideal:

- one object;
- centered;
- clean background;
- consistent scale;
- low-resolution representation still preserves class identity.

So the fruits result is encouraging, but it is not enough to claim general image
recognition breakthrough.

## Current Confidence

- PANN public-source reconstruction: plausible and useful for experiments.
- PANC public-source baseline: directionally aligned with public Progress.ai
  descriptions, but still abstract.
- Breakthrough claim: unproven.
- Research value: high enough to continue.

## Next Proof Points

To make the work more convincing, compare against simple conventional baselines:

- nearest centroid;
- k-nearest neighbors over raw pixels and handcrafted features;
- MobileNet or other pretrained embeddings plus k-NN/centroid;
- small CNN or transfer-learning baseline;
- PANN/PANC on Fruits-360, Oxford-IIIT Pets, CIFAR-10, and cleaned object crops.

The project becomes interesting if PANC-like methods win in low-data,
no-training, editable-library scenarios, especially with compact 32x32 or
binary artifacts.
