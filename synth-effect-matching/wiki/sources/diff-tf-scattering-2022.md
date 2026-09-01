---
tags: [scattering-transform, jtfs, gpu, differentiable, audio-representation]
date: 2022-07-19
sources: 1
---

# Differentiable Time-Frequency Scattering on GPU (Muradeli, Vahidi, Wang, Han et al., DAFx 2022)

**Authors:** John Muradeli, Cyrus Vahidi, Changhong Wang, Han Han, Vincent Lostanlen, Mathieu Lagrange, George Fazekas
**Venue:** DAFx 2022
**PDF:** SynthMatch-DifferentiableTimeFrequencyScattering-2204.08269v4.txt

## Summary

GPU implementation of **Joint Time-Frequency Scattering (JTFS)**, a convolutional operator for audio analysis that captures spectrotemporal modulations. Enables efficient differentiable computation of JTFS, which becomes the perceptual representation Phi used in [PNP loss](han-pnp-taslp-2024.md) for sound matching.

## Connections

- Provides the key audio representation for [Han PNP](han-pnp-taslp-2024.md)
- JTFS used as perceptual metric in [mesostructures](mesostructures-2023.md)
- See concept: [audio similarity metrics](../audio-similarity-metrics.md)
