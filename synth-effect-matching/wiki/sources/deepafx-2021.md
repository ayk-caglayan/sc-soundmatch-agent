---
tags: [audio-effects, black-box, stochastic-gradient, differentiable, tube-amp, mastering]
date: 2021-05-11
sources: 1
---

# Differentiable Signal Processing with Black-Box Audio Effects (DeepAFx, 2021)

**Authors:** Marco A. Martinez Ramirez, Paris Smaragdis, Nicholas J. Bryan, Oliver Wang
**Venue:** arXiv 2105.04752 (2021)
**PDF:** DeepAFx21.txt

## Summary

Data-driven approach incorporating **stateful third-party black-box audio effects** as layers in a neural network. Trains a deep encoder to control effect parameters for desired signal manipulation. Uses **fast parallel stochastic gradient approximation** for backpropagation through non-differentiable effects. Demonstrated on tube amplifier emulation, breath/pop removal, and automatic mastering.

## Connections

- Extends sound matching from synthesis to **audio effects processing**
- Stochastic gradient approach contrasts with [DDSP](masuda-ddsp-ismir21.md) (exact gradients) and [white-box](yang-white-box-ismir23.md)
- See concept: [sound matching approaches](../sound-matching-approaches.md)
