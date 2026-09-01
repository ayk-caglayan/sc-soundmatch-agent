---
tags: [drc, dynamic-range-compression, inversion, model-based, reverse-audio-engineering, audio-effects]
date: 2013-07-01
sources: 1
---

# Gorlow & Reiss 2013 — Model-Based Inversion of Dynamic Range Compression

**Authors:** Stanislaw Gorlow, Joshua D. Reiss
**Venue:** IEEE Transactions on Audio, Speech, and Language Processing, 21(7), pp. 1434–1444 (2013). [hal-00728059]
**PDF:** SynthMatch-DRC-ModelBasedInversion-Gorlow-2013.pdf

## Summary

Shows that a **dynamic, nonlinear, time-variant** operator — a DRC compressor — can be **exactly inverted** given an explicit signal model and the compression parameters. By transmitting a few parameters (threshold, ratio, attack, release, detector type) as metadata alongside the *compressed* "broadcast" signal, the original uncompressed signal is recovered with very high numerical accuracy (RMSE down to **−129 dBFS** on synthetic input; PEMO-Q PSMt = 1.00, i.e. perceptually flawless) and low computational cost (≈0.5× real time). Motivated by the **loudness war** and reverse audio engineering — restoring dynamic range the listener never received.

## The forward model (= Zölzer Ch. 7, hard knee, 0 dB makeup)

Feed-forward broadband compressor: RMS/peak detector → gain computer → gain smoother → broadband multiply (their Fig. 1, from Zölzer/DAFX 2nd ed.). With simplifications (hard knee, makeup fixed at 0 dB) the equations are exactly those documented in [`DRC/DRC.tex`](../../../DRC/DRC.tex):

- Detector (one-pole on $|x|^p$): $\tilde x(n)=\beta|x(n)|^p+\bar\beta\tilde x(n-1)$, $v=\sqrt[p]{\tilde x}$, $p\in\{1,2\}$ (peak/RMS).
- Coefficient: $\beta=1-\exp[-2.2/(f_s\tau_v)]$.
- Static curve: $F=-S(V-L)$, $S=1-1/R$; linear form $f=\kappa v^{-S}=(v/l)^{1/R-1}$, $l=10^{L/20}$.
- Gain smoother (one-pole): $g(n)=\gamma f(n)+\bar\gamma g(n-1)$, $\gamma=1-\exp[-2.2/(f_s\tau_g)]$.
- Output: $y(n)=g(n)x(n)$.

## The inversion

Defines a **characteristic function** $\zeta_p(v)$ (Eq. 19) whose zero-crossing is the unknown envelope $v(n)$; solved per-sample by a **secant-style iterative search** (Sec. V, Alg. 3, typically ≈1 iteration/sample). Attack/release phase toggles for both the detector and the gain smoother are *predicted* from the wet signal (Eqs. 24, 29). Threshold $L$ is the dominant accuracy factor; the RMS detector gives slightly larger error than peak (its $\zeta_2$ has stronger curvature). Lookahead breaks invertibility (future samples unavailable); clipping/brick-wall limiting (R = ∞, attack = 0) is a one-to-many map and non-invertible.

## Connections

- This is the **forward + inverse model reused unchanged** by [Sun et al. 2024](sun-drc-neural-inversion-2024.md), whose contribution is to *estimate* $\theta$ with neural networks (Gorlow assumed it known/transmitted) and to swap the secant search for a faster Levenberg–Marquardt root-finder.
- Same feed-forward topology that [Wright & Välimäki 2022](../../../ddsp/wiki/sources/wright2022-greybox-drc.md) *fit to data* as a grey-box VA emulator.
- Concept hub: [DRC parameter estimation](../drc-parameter-estimation.md). Forward-model equations: [`DRC/DRC.tex`](../../../DRC/DRC.tex) §5 gives the Zölzer↔Gorlow notation map.
- Contrast with the **blind** problem ([Peladeau 2024](peladeau-blind-afx-2024.md)): Gorlow needs the dry signal *not at all* but needs the parameters *exactly*.
