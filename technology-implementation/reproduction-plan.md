# Reproduction Plan

This plan is for building and testing a public-source PANN-like prototype.

## Goal

Implement a minimal model that follows the public PANN architecture:

- interval-coded inputs
- corrective weights indexed by input, interval, and output neuron
- distributor-based weight selection
- neuron sums over selected corrective weights
- difference-based one-step correction

The first target is correctness and reproducibility, not matching Progress'
claimed performance.

## Milestone 1: Tiny deterministic prototype

Tasks:

- Implement hard-bin interval encoding.
- Implement dense `W[input, interval, output]`.
- Implement forward pass.
- Implement difference-based sample update.
- Test on synthetic two-class data.

Expected result:

- Model should reduce training error quickly on simple separable data.
- Behavior should be deterministic with fixed data order.

Acceptance checks:

- Unit test: after one update on one sample, that sample's raw output should
  move exactly to the target for hard-bin difference update.
- Unit test: inactive weights remain unchanged.
- Unit test: active weights for each output receive equal correction.

## Milestone 2: IRIS classification

Tasks:

- Normalize four IRIS features.
- Use one-hot targets for three classes.
- Sweep interval counts: 4, 8, 16, 32.
- Compare against logistic regression, k-nearest neighbors, and a small MLP.

Metrics:

- train accuracy
- validation/test accuracy
- training time
- epochs to convergence
- memory use

Expected result:

- Should learn the training set rapidly.
- Generalization may depend strongly on interval count and smoothing.

## Milestone 3: Add smoothing coefficients

Tasks:

- Implement neighbor-bin activation.
- Use triangular or Gaussian-like coefficients.
- Implement coefficient-aware correction:
  `W += error * c / sum(c^2)`.

Reason:

Hard bins can memorize abruptly and generalize poorly. Smoothing gives adjacent
intervals partial responsibility and may improve interpolation.

## Milestone 4: Matrix form

Tasks:

- Flatten `(input, interval)` into activation index `k`.
- Encode samples into sparse activation matrix `A`.
- Implement `Y = A @ W`.
- Start with online sparse updates.
- Later test mini-batch updates with conflict normalization.

Reason:

The later patent describes array/matrix organization. Matrix form is also the
path toward GPU acceleration.

## Milestone 5: Image experiment

Tasks:

- Start with MNIST or a small CIFAR-10 subset.
- Flatten normalized pixels.
- Consider sparse feature selection or patch features to control memory use.
- Compare with nearest neighbor, logistic regression, and a small CNN/MLP.

Risks:

- `features * intervals * outputs` can become large.
- Hard-bin per-pixel representation may memorize without good generalization.
- Claims from old tests may not transfer to modern datasets or baselines.

## Milestone 6: PANC-like comparator experiment

Tasks:

- Build a separate associative comparator baseline.
- Use the same feature vectors.
- Store reference vectors and labels.
- Predict by top-k similarity voting.
- Report matched examples as explanations.

Reason:

This gives a clean experimental comparator baseline even if the true PANC
implementation is not public.

## Technical risks

- Catastrophic interference: correcting one sample can damage previous samples
  that share active bins.
- Interval count tradeoff: too few intervals underfit; too many intervals
  memorize and consume memory.
- Output scale: raw sums may grow without normalization.
- Ratio update instability: division by small outputs can explode.
- Benchmark mismatch: Progress' published comparisons use older baselines and
  incomplete setup details.

## Legal risks

- Public patents can teach implementation details but also define rights.
- Several Progress patents appear active.
- Do not use a clone commercially without a legal freedom-to-operate review.

