---
tags: [modulation, ddsp, lfo, envelope, sound-matching, interpretability]
date: 2025-10-07
sources: 1
---

# Modulation Discovery with Differentiable Digital Signal Processing

**Authors:** Christopher Mitcheltree, Hao Hao Tan, Joshua D. Reiss
**Venue:** arXiv 2510.06204v1 (2025)
**PDF:** SynthMatch-ModulationDiscovery-DDSP-2510.06204v1.txt

## Key Contributions

- First system to **discover modulation signals** (envelopes, LFOs) present in a sound using neural sound matching
- Leverages modulation extraction, constrained control signal parameterizations, and DDSP
- Addresses the interpretability gap: existing sound-matching systems predict high-dimensional framewise parameters without considering the shape/structure of underlying modulation curves
- Investigates the trade-off between **interpretability and sound-matching accuracy**
- Released trained DDSP synthesizers as a **VST plugin**

## Method

1. Extract modulation signals from target audio (envelopes, LFO patterns)
2. Parameterize control signals with structured representations (e.g., ADSR envelopes, LFO shapes)
3. Use DDSP synthesis with constrained modulation routing
4. Optimize parameters to match target audio while maintaining interpretable modulation structure

## Key Results

- Effective on highly modulated synthetic and real audio samples
- Applicable across different DDSP synth architectures
- Interpretable modulation curves with modest accuracy trade-off vs. unconstrained framewise prediction
- ~98% of presets in Serum use modulation, motivating this work

## Connections

- Extends DDSP framework from [Masuda & Saito](masuda-ddsp-ismir21.md) with modulation structure
- Modular signal chains also in [DiffMoog](diffmoog-2024.md)
- Addresses practical sound design needs highlighted in [Shier thesis](shier-thesis-2021.md)
- See concept: [sound matching approaches](../sound-matching-approaches.md)
