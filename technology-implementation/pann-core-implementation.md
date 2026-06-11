# PANN Core Implementation

This file describes a minimal PANN-like software prototype based on public
patent and publication disclosures. It is an engineering reconstruction, not a
claim to reproduce Progress, Inc.'s private implementation.

## Conceptual model

A PANN-like model replaces the single scalar weight normally attached to a
synapse with a set of corrective weights. Each input feature is mapped into a
value interval. The interval selects which corrective weight participates in the
neuron's sum.

For a supervised task:

- input vector: `x` with `m` features
- target vector: `t` with `n` outputs/classes
- number of intervals per input: `D`
- corrective weights: `W[i, d, n]`

Where:

- `i` is input/feature index
- `d` is interval/bin index
- `n` is output neuron index

## Components

### Input preprocessing

Normalize every feature into a stable numeric range. Practical choices:

- min/max scaling into `[0, 1]`
- z-score scaling followed by clipping
- image pixels scaled into `[0, 1]`

For each feature `i`, choose an interval range:

- global `[0, 1]` for normalized features
- per-feature min/max from the training set
- clipped percentile range to avoid outliers dominating interval allocation

### Distributor

The distributor maps an input value to one or more active intervals.

Minimal hard-bin distributor:

```text
d_i = floor((x_i - min_i) / (max_i - min_i) * D)
d_i = clamp(d_i, 0, D - 1)
active(i) = [(d_i, 1.0)]
```

Smoother distributor:

```text
active(i) = [
  (d_i - 1, c_left),
  (d_i, c_center),
  (d_i + 1, c_right)
]
```

The patents describe coefficients of impact and statistical distributions such
as a Gaussian-like distribution. For a first implementation, use hard bins.

### Forward pass

Hard-bin version:

```text
y[n] = sum_i W[i, d_i, n]
```

Coefficient version:

```text
y[n] = sum_i sum_(d,c in active(i)) c * W[i, d, n]
```

Classification can use:

- raw maximum: `argmax(y)`
- softmax over `y`
- winner-take-all with target coding

For a minimal classifier, encode targets one-hot and predict `argmax(y)`.

## Training algorithms

### Difference-based one-step correction

This is the simplest implementable version and matches the public description
in the 2016 "Fast trained neural network" article.

For a sample `(x, t)`:

```text
active = distributor(x)
y = forward(x)
e[n] = t[n] - y[n]
S = number of active feature/interval contributions

for each output neuron n:
  delta = e[n] / S
  for each active (i, d):
    W[i, d, n] += delta
```

With hard bins and one active interval per input, `S = m`.

This exactly compensates the current sample's output error for a linear
hard-bin forward pass, because the output change is:

```text
sum_i delta = m * (e[n] / m) = e[n]
```

Important limitation: correcting one sample can disturb previous samples that
share active weights. Multiple epochs are still needed.

### Coefficient-aware correction

If the distributor activates neighboring bins with coefficients, a practical
least-squares one-step correction is:

```text
denom = sum_over_active(c * c)
delta_base = e[n] / denom
W[i, d, n] += delta_base * c
```

Then the forward output change is:

```text
sum_over_active(c * delta_base * c) = e[n]
```

This is an engineering reconstruction consistent with one-step compensation. It
may not be identical to Progress' private implementation.

### Ratio/deviation-coefficient correction

The patents and publications also describe a ratio-style correction:

```text
factor[n] = t[n] / y[n]
W[i, d, n] *= factor[n]
```

Use this carefully:

- handle `y[n] == 0`
- handle negative values
- handle targets near zero
- consider damping, clipping, or epsilon stabilization

For a first prototype, prefer difference-based updates.

## Epoch loop

```text
initialize W
for epoch in range(max_epochs):
  shuffle training samples
  total_error = 0

  for x, t in samples:
    active = distributor(x)
    y = forward_from_active(active)
    e = t - y
    update active W using e
    total_error += mean(e * e)

  evaluate validation metrics
  stop if target error reached or validation stops improving
```

## Initialization

Possible choices:

- zeros: easiest for difference-based updates
- small random values: useful if ratio updates are used
- class-prior initialization: initialize output weights from target averages

For hard-bin difference updates, zero initialization is a clean starting point.

## Matrix representation

Flatten `(i, d)` into a single feature index `k = i * D + d`.

For a batch of `B` samples:

- `A`: binary or coefficient activation matrix, shape `[B, K]`
- `W`: corrective weight matrix, shape `[K, N]`
- `Y = A @ W`
- `E = T - Y`

Naive batch update:

```text
W += A.T @ normalized(E)
```

This must be normalized carefully because multiple samples may activate the same
weights and request conflicting corrections. Start with online updates before
batch updates.

## Minimal data structures

```text
class PannModel:
  input_ranges: list[(min, max)]
  interval_count: int
  weights: float[m][D][N]

  encode(x) -> list[(i, d, c)]
  forward(x) -> vector[N]
  train_one(x, target) -> metrics
```

Use a dense array for small prototypes. For images or high-dimensional sparse
features, consider sparse dictionaries keyed by `(i, d, n)`.

## What to validate

Start with small known tasks:

- XOR-like synthetic classification
- IRIS classification
- MNIST subset with flattened pixels

Track:

- training MSE
- validation accuracy
- training time
- number of intervals
- number of epochs
- memory use
- performance compared to logistic regression, MLP, and nearest neighbor

## Known implementation gaps

Patents and articles do not fully specify:

- optimal number of intervals
- interval distribution functions
- smoothing/impact coefficient schedules
- initialization used in published tests
- target encoding
- stopping rules
- preprocessing for each benchmark
- how to avoid destructive interference between samples
- exact GPU/matrix implementation
- exact public demo implementation

