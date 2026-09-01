---
tags: [audio-similarity, loss-function, differentiable-synthesis, evaluation, listening-test]
date: 2025-06-27
sources: 1
---

# Evaluating Sound Similarity Metrics for Differentiable, Iterative Sound-Matching

**Authors:** Amir Salimi, Abram Hindle, Osmar R. Zaiane
**Year:** 2025 (submitted to IEEE)
**PDF:** SynthMatch-SoundSimilarityMetrics-2506.22628v1.txt

## Key Contributions

- Systematic comparison of **loss functions for iterative sound matching** across multiple synthesizer types
- Framed **differentiable iterative sound-matching** as the natural extension of manual sound design
- Found that loss function performance is **highly dependent on the synthesizer** -- no universal best loss exists
- Included **blind listening tests** alongside objective metrics
- Tested 4 loss functions x 4 synthesizers x 300 trials = comprehensive benchmark

## Method

Four differentiable loss functions (including novel proposals) paired with differentiable subtractive, additive, and AM synthesizers. Performance measured via parameter differences, spectrogram-distance metrics, and human listening scores. Post-hoc analysis examines synthesizer-loss interaction effects.

## Key Results

- Moderate consistency among parameter error, spectral distance, and listening scores
- Loss function performance varies dramatically across synthesizers
- No single "best" loss -- the choice of similarity metric remains a creative/engineering decision conditioned on the synthesis method
- Advocates expanding scope of benchmarks rather than pursuing one-size-fits-all solutions

## Connections

- Directly relevant to loss function design in [Han PNP](han-pnp-taslp-2024.md) and [DDSP matching](masuda-ddsp-ismir21.md)
- Evaluation methodology complements [neural proxies](neural-proxies-2025.md)
- See concept: [audio similarity metrics](../audio-similarity-metrics.md)
