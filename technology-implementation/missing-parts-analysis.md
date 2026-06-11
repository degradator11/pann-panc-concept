# Missing Parts Analysis

This note reviews what is still missing after implementing the minimal Rust
prototype and compares those gaps against currently public Progress/PANN/PANC
materials: Progress website pages, Google Patents pages, and public Springer
metadata.

## Summary

The public PANN material is sufficient to implement a defensible prototype of
the main architecture:

- input values are mapped into intervals
- each input/interval/output has a corrective weight
- distributors select one or more active corrective weights
- neurons sum selected corrective weights
- training applies one-step correction from output deviation
- weights can be represented in dense or matrix form

The missing parts are mostly product-quality and benchmark-reproduction
details: exact preprocessing, exact interval heuristics, exact smoothing
coefficients, target coding, stabilization rules, adaptive structure policies,
and GPU implementation choices.

PANC is different. Public information supports only a high-level comparator
baseline. It names binary comparison format, associative libraries, similarity
comparison, and analogue retrieval, but does not disclose enough algorithmic
detail to reproduce the proprietary system.

## Difficulty scale

- Easy: a reasonable engineering version can be built in days.
- Medium: likely weeks of experimentation to get robust behavior.
- Hard: months of research and benchmarking; many design choices are open.
- Very hard: requires substantial R&D or unavailable proprietary detail.
- Not knowable exactly: a clone cannot be verified from public material alone.

## PANN: Publicly specified vs missing

| Area | Publicly available information | Still missing | Reimplementation difficulty |
| --- | --- | --- | --- |
| Core hard-bin model | US9390373B2 and the 2016 article disclose corrective weights selected by input intervals, neuron sums, and difference correction. | Nothing fundamental for a minimal model. | Easy. Already implemented. |
| Difference update | Public formula: divide output error by number of selected synapses/weights, then add correction to active weights. | Details for multi-output target scales and convergence policy. | Easy for raw version; Medium for production behavior. |
| Ratio update | Public formula: desired output divided by actual sum, then multiply active weights. | Zero handling, negative values, damping, clipping, target scaling. | Easy to code raw; Medium to make stable. |
| Impact coefficients / smoothing | Patents disclose coefficients of impact, adjacent intervals, statistical/Gaussian-like distributions, and coefficient-based correction. | Exact distribution, width, normalization, boundary behavior, and whether to use inverse coefficient or least-squares correction in software. | Medium. Exact Progress behavior is not knowable. |
| Interval distribution | Patents disclose uniform, non-uniform, symmetrical/asymmetrical, random, clipped, and statistically limited ranges. | How to choose interval count and distribution per dataset. | Easy for uniform; Medium/Hard for robust automatic selection. |
| Preprocessing | Patents mention min/max from training data, clipping high/low values, and statistical variance reduction; articles mention images/records. | Exact IRIS/image preprocessing, normalization, feature extraction, train/test splits. | Medium for a useful pipeline; Not knowable exactly for published results. |
| Initialization | Patents allow random, function-based, or template-based initialization; one patent passage mentions random values around zero within a correction-weight range. | Exact initial range, seeds, class-prior tricks, whether zero init was used in tests. | Easy to implement variants; Not knowable exactly. |
| Access index `a` | Patents disclose an access counter per corrective weight and describe using it to reduce weight plasticity, freeze old weights, remove noise, and support memory cleanup. | Exact decay schedule, thresholds, freezing policy, interaction with error correction. | Medium for a plausible version; Hard to tune well. |
| Dynamic weights/structure | US9619749B2 describes adding/removing corrective weights, inputs, synapses, and neurons during/supplementary training. | When to add/remove, selection criteria, pruning tolerances, quality checks. | Hard. |
| Winner/group outputs and unsupervised clustering | Patents describe selecting winner outputs or groups above a percentage of max, zeroing others, and retraining. | Exact thresholds, output-map management, class assignment, conflict handling. | Medium/Hard. |
| Matrix/batch training | Article and US10423694B2 disclose matrix recognition and epoch-level averaged deviation using selected-weight usage counts. | Sparse representation choices, conflict normalization details, numerical layout, GPU kernel strategy. | Medium for CPU matrix form; Hard for performant GPU. |
| Multilayer networks | Public article explicitly says multilayer training has its own characteristics and needs separate treatment. | Actual multilayer algorithm. | Very hard / not knowable exactly. |
| Benchmarks and claimed speed | Public pages report old IRIS and custom record-count tests, including hardware and comparison tools. | Datasets, source code, splits, baselines, hyperparameters, downloadable test artifact currently accessible. | Medium to benchmark our prototype; Not knowable exactly for claim reproduction. |
| Analog/memristor implementation | Public material describes resistors/memristors, excitatory/inhibitory circuits, differential amplifiers, and demultiplexers. | Physical device models, pulse calibration, fabrication constraints, circuit-level validation. | Very hard. |

## PANC: Publicly specified vs missing

| Area | Publicly available information | Still missing | Reimplementation difficulty |
| --- | --- | --- | --- |
| Overall paradigm | Public PANC page and Springer abstract describe structured associative comparison against libraries of analogues. | Precise data model and algorithms. | Easy to build a baseline; not enough for a clone. |
| Binary comparison format | Springer metadata mentions BCF for image recognition/classification. | BCF schema, encoding rules, compression/layout, comparison operations. | Hard; exact format not knowable from abstract. |
| Similarity coefficient | Springer keywords include coefficient of similarity; PANC page mentions binary and similarity comparison. | Formula, weighting, thresholds, calibration, class aggregation. | Medium for plausible metrics; not knowable exactly. |
| Associative libraries | Public sources say libraries of examples/analogues and similarity maps are used. | Library structure, indexing, update/merge/delete rules, memory layout. | Medium for brute force/HNSW-style baseline; Hard for proprietary behavior. |
| Explanation | Public sources say decisions are traceable through comparative matches. | Feature-attribution method and UI/reporting semantics. | Easy for matched-neighbor explanations; Medium for trustworthy explanations. |
| PANC_Platform | Springer abstract says a software product illustrates the work and is free for testing/non-commercial use. | A publicly reachable download and inspectable code/spec was not found in this pass. | Cannot evaluate until obtained. |
| Hardware efficiency claims | Public PANC page claims efficient scaling across CPUs, GPUs, edge, and neuromorphic accelerators. | Hardware mapping, kernels, memory bandwidth model, benchmarks. | Hard/Very hard. |

## Practical implementation roadmap

1. Finish PANN first.
   - Add ratio update with epsilon, clipping, and tests.
   - Add explicit public-patent-style coefficient correction mode in addition
     to the current least-squares correction mode.
   - Add access-count plasticity: `learning_scale = f(access_count)`.
   - Add matrix form for dense CPU batches.
   - Add IRIS and small image benchmark harnesses.

2. Treat PANC as an experimental comparator family, not a clone.
   - Keep the current nearest-neighbor module.
   - Add binary vector encoding and Hamming/Jaccard-style BCF experiments.
   - Add top-k explanation reports.
   - Add approximate nearest neighbor indexing only after correctness tests.

3. Do not claim Progress-equivalent behavior.
   - The public disclosures teach enough to build related prototypes.
   - They do not provide enough to reproduce the private software, exact
     benchmarks, or PANC_Platform behavior.

## Highest-value next missing pieces to implement

1. PANN ratio update: low effort, directly public.
2. PANN access index and plasticity decay: medium effort, high relevance to
   reducing destructive interference.
3. PANN matrix epoch update using selected-weight access counts: medium effort,
   directly public and useful for performance.
4. Benchmark harness: medium effort, needed to judge whether the prototype is
   useful.
5. PANC binary comparison experiments: medium effort, but speculative until
   full BCF details or PANC_Platform are available.

## Sources checked

- Progress, "Fast trained neural network":
  https://progress.ai/fast-trained-neural-network/
- Progress, "Analog and Digital Modeling of a Scalable Neural Network":
  https://progress.ai/analog-digital-modeling-scalable-neural-network/
- Progress, "PANC AI Technology":
  https://progress.ai/panc-ai-technology/
- Progress, "Publications":
  https://progress.ai/publications/
- Progress, "Tests":
  https://progress.ai/tests/
- Progress, "Patents":
  https://progress.ai/patents/
- Google Patents, US9390373B2:
  https://patents.google.com/patent/US9390373B2/en
- Google Patents, US9619749B2:
  https://patents.google.com/patent/US9619749B2/en
- Google Patents, US10423694B2:
  https://patents.google.com/patent/US10423694B2/en
- Springer, "Comparative Recognition Technology for Artificial Intelligence":
  https://link.springer.com/chapter/10.1007/978-3-032-13056-3_17
