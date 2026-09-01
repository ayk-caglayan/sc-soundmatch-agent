---
tags: [blind-estimation, audio-effects, auto-encoder, ddsp, neural-proxy, mastering, drc, equalizer, clipper, loss-function]
date: 2024-04-01
sources: 1
---

# Peladeau & Peeters 2024 — Blind Estimation of Audio Effects (Auto-Encoder + DDSP)

**Authors:** Côme Peladeau, Geoffroy Peeters (LTCI, Télécom-Paris)
**Venue:** IEEE ICASSP 2024, Seoul, pp. 856–860. [hal-04539329] · [audio](https://peladeaucome.github.io/ICASSP-2024-BEAFX-using-DDSP/)
**PDF:** SynthMatch-BlindEffectEstimation-Peladeau-ICASSP-2024.pdf

## Summary

**Blind Estimation of Audio Effects (BE-AFX):** infer the effects + parameters applied to a *dry* signal $x$ from the *wet* signal $y$ **alone** (the dry signal is unavailable). Proposes an **auto-encoder** framing: an analysis network $f^a$ reads $y$, predicts parameters $\hat p$, applies *analysis* effects $\{e^a\}$ to $x$ to produce $\hat y$, and is trained to minimise an **audio loss** $\mathcal L^{\mathrm{Mel}}_{\hat y,y}$ (log-Mel-spectrogram $\ell_1$) — not a parameter loss. This needs neither ground-truth parameters nor knowledge of the true effect implementation $\{e^s\}$.

**Central finding:** a parameter distance does **not** translate to a perceptual distance. Training on audio loss yields better *audio* reconstruction, while training on parameter MSE yields better *parameter* estimates — accurate $\hat p\approx p$ does **not** guarantee $\hat y\approx y$. (This motivates the perceptual-feature objectives elsewhere in this wiki; see [audio similarity metrics](../audio-similarity-metrics.md).)

## Effects and implementations

Mastering chain = **equalizer → DRC → soft clipper**, each with synthesis $\{e^s\}$ (unknown in reality) and analysis $\{e^a\}$ (must be differentiable, else a neural proxy) variants:

- **EQ:** 5-band parametric (synthesis) vs. parametric / 10-band graphic (analysis), frequency-domain differentiable filters.
- **DRC:** the **Giannoulis–Massberg–Reiss (2012)** DSP compressor (= Zölzer Ch. 7 model; threshold, ratio, attack, release, knee). Analysis uses a *simplified DSP* compressor (attack/release linked) or a **Neural Proxy** (a FiLM-conditioned TCN trained to approximate the DSP compressor). Best result: **Hybrid NP** — use the NP to estimate $\hat p$ but the real DSP compressor to render $\hat y$.
- **Clipper:** parametric (tanh/cubic/hard blend), Taylor, or Chebyshev waveshaper.

Encoders compared for $f^a$: MEE (Music Effects Encoder, 88M), TE (Timbre Encoder, 2.8M), TFE (Time+Frequency Encoder, 3.4M). Dataset: MUSDB18 mixes, 10 s clips.

## Connections

- The DRC used here is the **same forward model** as [Gorlow](gorlow-drc-inversion-2013.md) / [Sun](sun-drc-neural-inversion-2024.md) — see [DRC parameter estimation](../drc-parameter-estimation.md) and [`DRC/DRC.tex`](../../../DRC/DRC.tex).
- Directly extended by [Peladeau, Fourer & Peeters 2025](peladeau-param-distributions-2025.md), which replaces this *deterministic* estimator with a *distribution* over parameters.
- Neural-proxy idea overlaps [Neural Proxies 2025](neural-proxies-2025.md); the parameter-loss-vs-audio-loss point connects to [DeepAFx 2021](deepafx-2021.md) and the [sound matching approaches](../sound-matching-approaches.md) taxonomy (black-box, neural-proxy, DDSP).
