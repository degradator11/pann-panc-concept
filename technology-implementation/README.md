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

## Legal boundary

These notes are for technical analysis. They are not legal advice. Several
Progress patents appear active. Building or using an implementation for
commercial purposes can require a freedom-to-operate review and/or a license.

