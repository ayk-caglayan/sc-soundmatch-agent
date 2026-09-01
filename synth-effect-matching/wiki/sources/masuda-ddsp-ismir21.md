---
tags: [synth-parameter-estimation, ddsp, spectral-loss, semi-supervised, out-of-domain]
date: 2021-11-07
sources: 1
---

# Synthesizer Sound Matching with Differentiable DSP (Masuda & Saito, ISMIR 2021)

**Authors:** Naotake Masuda, Daisuke Saito
**Venue:** ISMIR 2021
**PDF:** SynthMatchingDDSP-ISMIR21-53.txt

## Key Contributions

- First to implement a **differentiable subtractive synthesizer** for sound matching, enabling end-to-end training with spectral loss
- Demonstrated **semi-supervised** training: pre-train with parameter loss on in-domain data, fine-tune with spectral loss on out-of-domain real-world sounds
- Showed spectral loss produces better audio matches than parameter loss alone
- Bridged the gap between conventional synthesizer control and DDSP

## Method

An additive-subtractive synthesizer is implemented in PyTorch: two oscillators (saw/square interpolation via additive harmonics), resonant low-pass filter (approximated as harmonic amplitude mask), frame-by-frame parameter estimation via GRU network. The resonant filter avoids expensive IIR by applying frequency response as harmonic multiplier.

Training procedure:
1. Pre-train estimator with parameter loss on synthesizer-generated (in-domain) pairs
2. Fine-tune with spectral loss (multi-scale STFT) on out-of-domain sounds (no ground truth parameters needed)

## Key Results

- Spectral loss fine-tuning significantly improves matching of real-world sounds
- Out-of-domain training (real instruments) transfers well to practical applications
- Differentiable synthesis enables audio-domain gradient flow that was impossible with black-box synthesizers

## Connections

- Extended in [Masuda & Saito 2023 (TASLP)](masuda-saito-taslp-2023.md) with effects and envelopes
- Builds on [DDSP](../index.md) framework (Engel et al. 2020)
- Contrasts with black-box approaches in [InverSynth](inversynth-2018.md)
- Key methodological advance cited by [Han et al.](han-pnp-icassp-2023.md), [DiffMoog](diffmoog-2024.md), [SynthRL](synthrl-2025.md)
- See concept: [sound matching approaches](../sound-matching-approaches.md)
