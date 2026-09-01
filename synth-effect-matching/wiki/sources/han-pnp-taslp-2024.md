---
tags: [perceptual-sound-matching, pnp-loss, inverse-problem, jtfs, scattering-transform, physical-modeling]
date: 2024-04-25
sources: 1
---

# Learning to Solve Inverse Problems for Perceptual Sound Matching (Han et al., TASLP 2024)

**Authors:** Han Han, Vincent Lostanlen, Mathieu Lagrange
**Venue:** IEEE/ACM Transactions on Audio, Speech, and Language Processing, Vol. 32, 2024
**PDF:** SynthMatch-HanHan-TASLP-2024.txt

## Key Contributions

- Introduced **Perceptual-Neural-Physical loss (PNP)**: a loss function that balances perceptual relevance with computational efficiency
- PNP linearizes the effect of synthesis parameters on auditory features near each training sample, achieving **100x speedup** over DDSP backpropagation while preserving perceptual fidelity
- Comprehensive evaluation of design choices: parameter rescaling, pretraining, auditory representation, gradient clipping
- State-of-the-art results on AM/FM arpeggiator and rectangular membrane physical model datasets
- Formalized **perceptual sound matching (PSM)** as an inverse problem

## Method

Given a synthesizer g with parameters theta and a psychoacoustic descriptor Phi:
- PSM seeks f such that `theta_hat = f(x)` minimizes `||Phi(g(theta_hat)) - Phi(x)||`
- **PNP loss** precomputes the Jacobian of (Phi o g) at each training sample, creating a quadratic approximation of spectral loss that is as fast as parameter loss during training
- Uses **Joint Time-Frequency Scattering (JTFS)** as the auditory representation Phi
- The linearization is massively parallelizable and precomputable

## Key Results

- PNP-accelerated JTFS has greater influence on PSM performance than any other design choice (parameter rescaling, pretraining, etc.)
- PNP matches the perceptual quality of full DDSP+JTFS at 1/100th the cost
- Gradient clipping is beneficial for PSM optimization

## Connections

- Conference predecessor: [Han PNP ICASSP 2023](han-pnp-icassp-2023.md)
- Thesis wrap-up: [Han thesis 2025](han-thesis-2025.md)
- JTFS implementation: [differentiable TF scattering](diff-tf-scattering-2022.md)
- Gradient clipping study: [Han gradient clipping EUSIPCO 2025](han-gradient-clipping-2025.md)
- PNP is a form of [plug-and-play prior](../plug-and-play-priors.md)
- See concept: [audio similarity metrics](../audio-similarity-metrics.md)
