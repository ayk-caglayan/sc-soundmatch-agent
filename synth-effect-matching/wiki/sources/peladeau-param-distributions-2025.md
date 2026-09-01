---
tags: [parameter-distributions, normalizing-flows, entropy, many-to-one, blind-estimation, ddsp, fm-synthesis, equalizer, loss-function]
date: 2025-09-02
sources: 1
---

# Peladeau, Fourer & Peeters 2025 — Audio Processor Parameters: Estimating Distributions Instead of Deterministic Values

**Authors:** Côme Peladeau, Dominique Fourer, Geoffroy Peeters
**Venue:** DAFx 2025 (DAFx25), Ancona, pp. 275–282. [code](https://github.com/peladeaucome/DAFx_Params_Distrib)
**PDF:** SynthMatch-ParameterDistributions-Peladeau-DAFx-2025.pdf

## Summary

Tackles the **many-to-one** problem head-on: multiple parameter settings produce (nearly) the same sound, so a *deterministic* estimator must arbitrarily pick one. Instead, model the parameters as a **probability distribution** $p_\theta(z\mid y)$ and learn it by optimising two objectives at once:

1. **Reconstruction** — minimise the audio loss between target $y$ and $\hat y=f_{\mathrm{DDSP}}(x;z)$ (as in the deterministic [Peladeau 2024](peladeau-blind-afx-2024.md) auto-encoder).
2. **Diversity** — *maximise* the conditional **entropy** $H_{p_\theta}[z\mid y]$ of the parameters.

Derived as a KL divergence to a Boltzmann–Gibbs posterior $p(z\mid x,y)\propto\exp(-\ell(\hat y,y))$, yielding a β-VAE-style loss $\mathcal L=-\beta H_{p_\theta}[z\mid y]+\mathbb E_{z\sim p_\theta}[\ell(\hat y,y)]$ (Eq. 7). The distribution is a diagonal Gaussian pushed through a **normalizing flow** (deep sigmoidal flow layers) so its entropy is tractable.

## Experiments & findings

- **Exp. 1 — Equalizer:** match a 1-band high-shelf using a 2-band cascade (deliberately redundant). The model learns that the two **gains anti-correlate** ($g_1+g_2=g$, correlation −0.51) while frequencies/Q stay uncorrelated — it discovers the redundancy. The deterministic baseline wins single-shot audio quality, but **best-of-$N$** sampling from the distribution overtakes it by $N=2$ and keeps improving.
- **Exp. 2 — FM synth** (1 carrier + 1 modulator, à la DDX7): uses a **Spectral Optimal Transport + MR-STFT** loss so the modulator *frequency* is learnable. **Entropy tracks identifiability:** at low modulation index (harmonics negligible) the parameters barely matter and entropy is high; at high index the estimate must be precise and entropy collapses.

## Significance

Reframes parameter estimation as learning the *set* of valid solutions, an explicit answer to the many-to-one mapping that [Sun et al. 2024](sun-drc-neural-inversion-2024.md) cite as why DRC *regression* underperforms *classification*. One DSF flow layer suffices for these simple effect chains; future work targets complex chains and real productions.

## Connections

- Direct successor to [Peladeau & Peeters 2024](peladeau-blind-afx-2024.md) (deterministic → distributional).
- The many-to-one motivation is shared with [Sun et al. 2024](sun-drc-neural-inversion-2024.md) (citing this paper) and the ill-posed-inversion view of [Hayes et al. equivariant flow matching 2025](hayes-equivariant-flow-2025.md), which also uses normalizing flows / flow matching to capture multiple solutions.
- Loss-function angle (entropy regularisation, SOT for pitch): [audio similarity metrics](../audio-similarity-metrics.md). Concept hub: [DRC parameter estimation](../drc-parameter-estimation.md).
