# PANC Public Implementation Notes

PANC public material is less implementation-complete than the PANN patents.
These notes describe what can be inferred from public pages and the Springer
chapter metadata.

## Publicly described idea

Progress describes PANC as a comparator-centric recognition architecture. The
main public claims are:

- recognition is performed by structured comparison
- the system uses libraries of examples or analogues
- comparisons can be performed in parallel
- decisions can be explained through the matched analogues
- little or no conventional gradient-based training may be needed for many
  recognition tasks

This sounds closer to an associative memory, nearest-neighbor, case-based
reasoning, or similarity-search system than to a backpropagation-trained neural
network.

## Minimal reproducible skeleton

A public-material-based PANC-like prototype could include:

1. Feature encoder
   - Converts raw input into a structured representation.
   - Could be handcrafted features, embeddings, binary features, image patches,
     signal features, or domain-specific descriptors.

2. Analogue library
   - Stores reference examples.
   - Each reference contains features, label, metadata, and optional explanatory
     attributes.

3. Comparator
   - Computes similarity or distance between query and references.
   - Could use Hamming distance, cosine similarity, Euclidean distance,
     correlation, dynamic time warping, or a domain-specific comparator.

4. Retrieval
   - Finds top-k analogues.
   - Can be brute force for small data or indexed by approximate nearest
     neighbor search for larger data.

5. Decision aggregation
   - Classification: weighted vote among top-k references.
   - Regression: weighted average.
   - Detection: threshold on similarity or anomaly score.

6. Explanation
   - Return the matched analogues and the features contributing most to the
     match.

## Pseudocode

```text
library = []

def add_reference(x, label, metadata):
    z = encode(x)
    library.append((z, label, metadata))

def recognize(query):
    zq = encode(query)
    scored = []

    for z, label, metadata in library:
        score = similarity(zq, z)
        scored.append((score, label, metadata))

    top = take_top_k(scored)
    decision = aggregate(top)
    explanation = top
    return decision, explanation
```

## What is not public enough

The public website does not specify:

- exact feature representation
- exact comparator function
- exact analogue-library structure
- exact indexing method
- exact binary comparison format
- exact hardware realization
- learning/update process beyond adding references or maps
- benchmark protocol

Therefore, a PANC-like prototype is possible, but reproducing Progress' PANC
technology is not currently possible from public pages alone without the full
Springer paper, any patent filings, or private documentation.

## Suggested implementation direction

If the project needs a useful experimental comparator model, start with:

- `encode(x)`: normalized vector or binary vector
- `similarity(a, b)`: cosine similarity for dense vectors, Hamming similarity
  for binary vectors
- `top_k`: brute force first, then FAISS/Annoy/HNSW if needed
- `explanation`: matched examples plus per-feature similarity contribution

This would test the practical value of the comparator idea without pretending
to reproduce undisclosed proprietary details.

