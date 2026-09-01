---
tags: [vae, preset-generation, dx7, fm-synthesis, categorical-parameters]
date: 2021-09-08
sources: 1
---

# Improving Synthesizer Programming from Variational Autoencoders Latent Space (Le Vaillant et al., DAFx 2021)

**Authors:** Gwendal Le Vaillant, Thierry Dutoit, Sebastien Dekeyser
**Venue:** DAFx 2021
**PDF:** preset-gen-vae-DAFx20in21.txt

## Summary

VAE-based approach for automatic synthesizer programming on the DX7 FM synthesizer. Introduces heterogeneous parameter representations (numerical + categorical), multi-channel spectral input for pitch/intensity variation, and a curated 30K DX7 preset dataset. Achieves significant improvements in parameter inference and audio accuracy.

## Connections

- DX7 also targeted by [Sound2Synth](sound2synth-2022.md)
- VAE approach relates to [Flow Synthesizer](flowsynth-2020.md)
