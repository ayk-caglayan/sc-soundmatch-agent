---
tags: [mesostructure, scattering-transform, arpeggio, ddsp, beyond-spectrogram]
date: 2023-01-24
sources: 1
---

# Mesostructures: Beyond Spectrogram Loss in Differentiable Time-Frequency Analysis (Vahidi, Han et al., 2023)

**Authors:** Cyrus Vahidi, Han Han, Changhong Wang, Mathieu Lagrange, Gyorgy Fazekas, Vincent Lostanlen
**Year:** 2023 (arXiv 2301.10183v1)
**PDF:** SynthMatch-Mesostructures-2301.10183v1.txt

## Summary

Introduces the concept of **mesostructures** -- intermediate levels of musical articulation between waveshape microstructure and musical form macrostructure (e.g., melody, arpeggios, syncopation). Current neural audio synthesizers only train/evaluate at microstructure scale. Proposes a differentiable arpeggiator + time-frequency scattering to model mesostructural audio.

## Connections

- Extends the perceptual loss idea from [PNP](han-pnp-taslp-2024.md) to longer time scales
- Uses JTFS from [diff TF scattering](diff-tf-scattering-2022.md)
- See concept: [audio similarity metrics](../audio-similarity-metrics.md)
