---
tags: [drc, dynamic-range-compression, inversion, neural, hybrid, ast, music-effect-encoder, parameter-estimation, audio-effects]
date: 2024-11-01
sources: 1
---

# Sun, Fourer & Maaref 2024 — Neural-Enhanced DRC Inversion (Hybrid)

**Authors:** Haoran Sun, Dominique Fourer, Hichem Maaref (IBISC, Univ. Évry – Paris-Saclay)
**Venue:** arXiv 2411.04337 (v2, Sep 2025). [code](https://github.com/SunHaoRanCN/DRC-Inversion)
**PDF:** SynthMatch-DRC-NeuralInversion-Sun-2024.pdf

## Summary

A **hybrid** DRC inversion: a neural network first estimates the DRC parameters / profile from the *compressed* signal, then the **model-based inversion of [Gorlow & Reiss 2013](gorlow-drc-inversion-2013.md) is applied unchanged** to restore the original dynamics. This removes Gorlow's assumption that the parameters are known and transmitted as metadata. Two estimation routes:

- **Classification (AST):** when the compressor is one of a known set of presets, a modified **Audio Spectrogram Transformer** (ImageNet-pretrained, + MLP head) classifies the *DRC profile*; the exact parameters follow from the predicted label. STFT input of size 64×431 works best.
- **Regression (MEE):** when no preset is assumed, a **Music Effect Encoder** directly regresses the parameter vector.

Classification dominates regression on reconstruction quality, precisely because of the **many-to-one** parameter→sound mapping (see [Peladeau 2025](peladeau-param-distributions-2025.md)): regression errors get amplified by the inversion math, whereas classification sidesteps point estimation. AST also beats the De-Limiter, Demucs, and HDemucs baselines.

## Model & parameters

Same forward model as Gorlow / Zölzer Ch. 7 (their Eqs. 1–5 reproduce the detector, static curve, and gain smoother; this is the SoX implementation). Parameter vector:

$$\theta = \{\,L,\ R,\ p,\ \tau_v^{\mathrm{att}},\ \tau_v^{\mathrm{rel}},\ \tau_g^{\mathrm{att}},\ \tau_g^{\mathrm{rel}}\,\}$$

— two ballistics pairs (envelope detector + gain smoother), the two cascaded one-poles analysed in [`DRC/DRC.tex`](../../../DRC/DRC.tex) §3.4. They replace Gorlow's secant search with a **Levenberg–Marquardt** root-finder for `CHARFZERO` (69.4 s → 26.1 s mean decompression; MSE $3.2\times10^{-5}\to8.2\times10^{-6}$).

## Key findings

- **Sensitivity hierarchy:** threshold $L$ is by far the most error-sensitive parameter (and best estimated, $R^2=0.95$); ratio $R$ moderate; the timing constants are remarkably robust to error — envelope release $\tau_v^{\mathrm{rel}}$ is the hardest to estimate ($R^2=0.46$). Matches Gorlow's finding that $L$ dominates inversion accuracy.
- Data augmentation (Gaussian noise) + **curriculum learning** (SNR annealed 65→20 dB) improves both tasks.
- Evaluated on MedleyDB, MUSDB18-HQ, DAFX, LibriSpeech; 5- and 30-profile datasets. Limitation: only compressor/limiter tested (no expander/compander).

## Connections

- Built directly on [Gorlow & Reiss 2013](gorlow-drc-inversion-2013.md) (model-based inversion) — this is the precursor → neural-estimator successor pair.
- Uses the [Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778), as in [Bruford et al. 2024](bruford-ast-dafx-2024.md) for synth matching.
- The many-to-one motivation is the explicit subject of [Peladeau et al. 2025](peladeau-param-distributions-2025.md).
- Concept hub: [DRC parameter estimation](../drc-parameter-estimation.md). Forward-model equations: [`DRC/DRC.tex`](../../../DRC/DRC.tex).
