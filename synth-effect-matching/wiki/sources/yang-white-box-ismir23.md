---
tags: [white-box, differentiable-rendering, gradient-descent, fm-synthesis, discontinuities]
date: 2023-11-05
sources: 1
---

# White Box Search over Audio Synthesizer Parameters (Yang et al., ISMIR 2023)

**Authors:** Yuting Yang, Zeyu Jin, Connelly Barnes, Adam Finkelstein
**Venue:** ISMIR 2023
**PDF:** SynthMatching-WhiteBoxParamSearch-ISMIR23.txt

## Key Contributions

- Adapted **differentiable rendering** techniques from computer graphics to directly differentiate a synthesizer as a **white box program**
- Handles discontinuous components (sawtooth, square waveforms, categorical parameters) that thwart standard automatic differentiation
- Built on the **A-delta** method: replaces standard AD rules to enable backpropagation through arbitrary discontinuous programs
- Eliminates need for neural proxies, genetic algorithms, or DDSP reimplementations

## Method

The key insight: discontinuous synthesis functions are band-limited and sampled at a finite rate (e.g., 48kHz). Each sample represents an integral over a small time window. By differentiating this integral (following A-delta from graphics), correct gradients are obtained even at discontinuities.

Applied to a generic FM synth with ADSR envelopes, noise generators, and IIR filters. The white-box approach directly optimizes `theta* = argmin L(f(theta), T)` via gradient descent on the actual synthesizer code.

For categorical parameters (e.g., waveform selection), the method uses continuous relaxation during optimization and rounds to discrete values.

## Key Results

- Outperforms neural proxy baselines and genetic algorithms in both quantitative metrics and qualitative evaluations
- Works on synthesizers with IIR filters, noise, and discontinuous oscillators
- No training data required -- purely optimization-based at inference time

## Connections

- Complements [DiffMoog](diffmoog-2024.md) which re-implements synths as differentiable
- Contrasts with black-box [InverSynth](inversynth-2018.md) and DDSP approaches
- Related to [Yang white-box FM](yang-white-box-fm-2023.md) (same first author)
- See concept: [sound matching approaches](../sound-matching-approaches.md)
