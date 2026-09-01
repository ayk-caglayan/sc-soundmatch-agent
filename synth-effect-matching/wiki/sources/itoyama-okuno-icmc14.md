---
tags: [parameter-estimation, linear-regression, handcrafted-features, vst, early]
date: 2014-01-01
sources: 1
---

# Parameter Estimation of Virtual Musical Instrument Synthesizers (Itoyama & Okuno, ICMC 2014)

**Authors:** Katsutoshi Itoyama, Hiroshi G. Okuno
**Venue:** ICMC 2014
**PDF:** SynthMatch-ItoyamaAndOkuno-ICMC14.txt

## Summary

**Multiple linear regression** from handcrafted acoustic features (low-level spectral + delta features) to VST synthesizer parameters. Best case error of 0.004 and 17.35 dB SDR. Alternative approach to sound source separation: obtain isolated instrument sounds by re-synthesizing with estimated parameters rather than separating mixtures.

## Connections

- Baseline method compared against in [Barkan 2017](barkan-deep-synth-pe-2017.md) (handcrafted features + linear regression)
- Represents pre-deep-learning era of synth parameter estimation
