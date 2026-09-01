---
tags: [concept, loss-function, perceptual, spectral, metrics]
date: 2026-05-29
sources: 14
---

# Audio Similarity Metrics

The choice of loss function `L` in synth parameter estimation is one of the most consequential design decisions. This page surveys the metrics used across the literature.

## Parameter-Domain Metrics

- **Parameter MSE / L2 loss:** `||theta_hat - theta||^2` -- simple, fast, requires ground truth, ignores perceptual significance
- **Classification accuracy (top-k):** quantized parameters treated as classes ([Barkan 2017](sources/barkan-deep-synth-pe-2017.md))
- **Pearson Correlation Coefficient (PCC):** used on STFT reconstructions ([Barkan 2017](sources/barkan-deep-synth-pe-2017.md))

## Spectral-Domain Metrics

- **Multi-scale STFT loss:** L1 or L2 distance between magnitude spectrograms at multiple window sizes; standard in DDSP literature ([Masuda & Saito 2021](sources/masuda-ddsp-ismir21.md))
- **Mel-spectrogram distance:** perceptually weighted frequency scale
- **MFCC distance:** compact timbral descriptor, used in genetic algorithm fitness ([Shier thesis](sources/shier-thesis-2021.md))
- **Log-magnitude spectrogram:** better captures quiet components

## Perceptual / Psychoacoustic Metrics

- **Joint Time-Frequency Scattering (JTFS):** captures spectrotemporal modulations (AM/FM patterns); key representation in PNP loss ([Han PNP](sources/han-pnp-taslp-2024.md), [diff TF scattering](sources/diff-tf-scattering-2022.md))
- **Mesostructural loss:** extends scattering beyond microstructure to arpeggios, rhythm ([Mesostructures 2023](sources/mesostructures-2023.md))

## Learned Metrics

- **Pretrained audio embeddings:** VGGish, PANNs, CLAP features used as proxy targets ([Neural Proxies 2025](sources/neural-proxies-2025.md))
- **RL reward functions:** sound similarity encoded as reward signal, shown to correlate with human perception ([SynthRL 2025](sources/synthrl-2025.md))

## Composite / Hybrid Losses

- **PNP loss (Perceptual-Neural-Physical):** linearized approximation of spectral loss around each training sample; runs as fast as parameter loss but approximates perceptual fidelity ([Han PNP TASLP 2024](sources/han-pnp-taslp-2024.md))
- **Signal-chain loss:** accounts for modular architecture topology ([DiffMoog 2024](sources/diffmoog-2024.md))
- **Combined parameter + spectral loss:** joint training throughout ([Masuda & Saito TASLP 2023](sources/masuda-saito-taslp-2023.md))
- **Entropy-regularised (diversity) loss:** add the parameter distribution's conditional entropy to the audio reconstruction loss (β-VAE form) so the model returns *diverse* valid parameter sets instead of one point ([Peladeau et al. 2025](sources/peladeau-param-distributions-2025.md))
- **Spectral Optimal Transport (SOT):** measures displacement of spectral energy across frequency rather than magnitude difference; convex w.r.t. pitch/frequency, fixing the well-known failure of MR-STFT on frequency estimation ([Peladeau et al. 2025](sources/peladeau-param-distributions-2025.md))

## Key Finding: No Universal Best Metric

[Salimi et al. 2025](sources/sound-similarity-metrics-2025.md) conducted the most comprehensive comparison (4 losses x 4 synthesizers x 300 trials with listening tests) and found:

- Loss function performance is **highly dependent on the synthesizer**
- No single "best" metric exists across all synthesis methods
- Parameter error, spectral distance, and human listening scores show only **moderate consistency**
- The choice of metric remains a creative/engineering decision

## Contradictions and Open Questions

- Parameter loss can produce low error but poor perceptual match (many-to-one parameter-audio mapping). [Peladeau & Peeters 2024](sources/peladeau-blind-afx-2024.md) make this explicit: training on a log-Mel audio loss beats training on parameter MSE for *audio* fidelity, even when the latter wins on *parameter* numbers — accurate `p_hat` does not guarantee accurate `y_hat`.
- Spectral loss can produce good perceptual match but wrong parameters (non-identifiability); the principled response is to estimate a *distribution* of parameters rather than a point ([Peladeau et al. 2025](sources/peladeau-param-distributions-2025.md), see [DRC parameter estimation](drc-parameter-estimation.md))
- JTFS is expensive but most perceptually faithful; PNP provides the only known acceleration path
- How to evaluate temporal/dynamic aspects of sound matching remains under-explored

## Related Concepts

- [Synth Parameter Estimation](synth-parameter-estimation.md)
- [Plug-and-Play Priors](plug-and-play-priors.md) -- PNP loss formulation
- [Sound Matching Approaches](sound-matching-approaches.md)
