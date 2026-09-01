---
tags: [ddsp, semi-supervised, practical-synthesizer, effects, envelope]
date: 2023-01-16
sources: 1
---

# Improving Semi-Supervised Differentiable Synthesizer Sound Matching for Practical Applications (Masuda & Saito, TASLP 2023)

**Authors:** Naotake Masuda, Daisuke Saito
**Venue:** IEEE/ACM TASLP, Vol. 31, 2023
**PDF:** SynthMatchingDDSP-2023.txt (also MasudaSaito23-...)

## Summary

Extended version of [ISMIR 2021](masuda-ddsp-ismir21.md). Adds **effect modules** (chorus) and **envelope generators** to the differentiable synthesizer for practical music production use. Proposes combined parameter + spectral loss training strategy throughout (rather than switching). Shows that full combined training enables effective chorus utilization.

## Connections

- Extension of [Masuda & Saito ISMIR 2021](masuda-ddsp-ismir21.md)
- Practical synthesizer design informs [DiffMoog](diffmoog-2024.md)
