---
tags: [concept, drc, dynamic-range-compression, inversion, parameter-estimation, ballistics, audio-effects, blind-estimation]
date: 2026-05-29
sources: 5
---

# DRC Parameter Estimation, Inversion & Modelling

Cross-topic hub for **dynamic range compression (DRC)** as a target for [synth/effect parameter estimation](synth-parameter-estimation.md). Links the `synth-effect-matching/` papers (Gorlow, Sun, the two Peladeau works) to the `ddsp/` grey-box modelling work (Wright). The shared forward model is documented self-contained in [`DRC/DRC.tex`](../../DRC/DRC.tex) (Zölzer Ch. 7: static curve, PEAK/RMS detection, attack/release ballistics, Eqs. 7.1–7.31).

## The forward model (one model, three uses)

A feed-forward broadband compressor: **level detector** (PEAK or RMS one-pole on $|x|^p$) → **gain computer** (static curve $F=-S(V-L)$, $S=1-1/R$) → **gain smoother** (one-pole) → **broadband multiply** $y(n)=g(n)x(n)$. The same model is approached three ways:

| Use | Paper | Knows dry $x$? | Knows params $\theta$? | Goal |
|-----|-------|----------------|------------------------|------|
| **Inversion** | [Gorlow & Reiss 2013](sources/gorlow-drc-inversion-2013.md) | no | **yes** (transmitted as metadata) | recover $x$ from $y$ |
| **Neural inversion** | [Sun et al. 2024](sources/sun-drc-neural-inversion-2024.md) | no | **no** (estimate, then invert) | recover $x$ from $y$ |
| **Blind estimation** | [Peladeau & Peeters 2024](sources/peladeau-blind-afx-2024.md) | no | no | estimate $\hat p$ s.t. $\hat y\approx y$ |
| **Grey-box VA modelling** | [Wright & Välimäki 2022](../../ddsp/wiki/sources/wright2022-greybox-drc.md) | yes (paired) | no | *fit* the model to a device |

## Key findings (distilled this session; not all stated in the PDFs)

1. **Gorlow 2013 = Zölzer's forward model, exactly.** With hard knee and 0 dB makeup their equations match equation-for-equation: $\beta,\gamma=1-\exp[-2.2/(f_s\tau)]$, $F=-S(V-L)$, $S=1-1/R$, $f=(v/l)^{1/R-1}$, $g(n)=\gamma f(n)+\bar\gamma g(n-1)$. The Zölzer↔Gorlow notation map is [`DRC/DRC.tex`](../../DRC/DRC.tex) §5.
2. **Sun 2024 reuses Gorlow's model unchanged**; its contribution is the neural *estimator* (AST classifies the profile, MEE regresses the parameters) feeding the 2013 inversion, plus a Levenberg–Marquardt root-finder in place of the secant search. Parameter vector $\theta=\{L,R,p,\tau_v^{\mathrm{att}},\tau_v^{\mathrm{rel}},\tau_g^{\mathrm{att}},\tau_g^{\mathrm{rel}}\}$.
3. **Two ballistics pairs (the "two time constants").** Both the envelope detector ($\tau_v$) and the gain smoother ($\tau_g$) carry independent attack/release — two cascaded one-poles. A two-time-constant (attack/release) detector is textbook-universal; the *cascaded / decoupled* detector topology is canonically analysed by **Giannoulis, Massberg & Reiss (JAES 2012)**, originally **McNally (JAES 1984)** and **Stikvoort (JAES 1986)**. Wright's switching / RNN-modulated one-poles are learned realisations of this asymmetric, signal-dependent smoothing. The cascade's effective ballistics ≈ rise-times-add-in-quadrature: see [`DRC/DRC.tex`](../../DRC/DRC.tex) §3.4.
4. **Threshold dominates.** Both Gorlow (RMSE strongly correlates with $L$) and Sun (sensitivity analysis; $L$ has $R^2=0.95$, highest impact and best estimated) find threshold the critical parameter; the time constants are comparatively forgiving. Inversion is impossible for lookahead (future samples) and for brick-wall limiting / hard clipping (one-to-many, non-invertible).
5. **Many-to-one mapping ⇒ estimate distributions.** Multiple parameter sets give the same sound, so deterministic regression is ill-posed — which is *why* Sun finds DRC profile **classification** beats parameter **regression** (regression error is amplified by the inversion math). [Peladeau et al. 2025](sources/peladeau-param-distributions-2025.md) confronts this directly, learning an entropy-maximising *distribution* of parameters via normalizing flows.
6. **Parameter distance ≠ perceptual distance.** [Peladeau & Peeters 2024](sources/peladeau-blind-afx-2024.md) show training on an audio (log-Mel) loss beats training on parameter MSE for *audio* fidelity, even when the latter gives better *parameter* numbers — accurate $\hat p$ does not guarantee accurate $\hat y$.

## Prior-art landscape (perceptual / black-box optimisation angle)

- **Black-box effect + deep-feature objective:** Ramírez *DeepAFx* ([summary](sources/deepafx-2021.md); [paper](https://doi.org/10.1109/ICASSP39728.2021.9415103)) — but uses SPSA stochastic-gradient + a neural encoder, **not** derivative-free search.
- **Derivative-free / evolutionary optimisation + perceptual loss** lives in the synthesizer sound-matching tradition (Yee-King et al. 2018; recent CMA-ES + perceptual-loss work ~2026, concurrent). See [sound matching approaches](sound-matching-approaches.md) (genetic-algorithm branch).
- The full triple **{derivative-free optimisation + DRC + perceptual feature space}** appears unprecedented (gap noted for the ISMIR-2026 submission that prompted this ingest).

## See Also

- [`DRC/DRC.tex`](../../DRC/DRC.tex) — self-contained DRC forward-model equations + Zölzer↔inversion-literature notation map
- [Synth Parameter Estimation](synth-parameter-estimation.md) · [Sound Matching Approaches](sound-matching-approaches.md) · [Audio Similarity Metrics](audio-similarity-metrics.md)
- ddsp: [Differentiable Audio Effects](../../ddsp/wiki/differentiable-audio-effects.md) · [Wright grey-box DRC](../../ddsp/wiki/sources/wright2022-greybox-drc.md)
