---
tags: [synth-parameter-estimation, vae, normalizing-flows, latent-space, macro-control]
date: 2020-01-01
sources: 1
---

# Flow Synthesizer: Universal Audio Synthesizer Control with Normalizing Flows

**Authors:** Philippe Esling, Naotake Masuda, Adrien Bardet, Romeo Despres, Axel Chemla-Romeu-Santos
**Venue:** Applied Sciences 2020 (extended from DAFx 2019)
**PDF:** FlowSynth-applsci-10-00302-v2.txt

## Key Contributions

- Novel formulation: learn an **organized latent audio space** of a synthesizer's capabilities, then construct an **invertible mapping** to parameter space via normalizing flows
- Simultaneously addresses three tasks: (1) parameter inference, (2) macro-control learning, (3) audio-based preset exploration
- Introduced **disentangling flows** that steer latent dimensions to match target variation factors (e.g., brightness, noisiness)
- Open-source Max4Live real-time implementation

## Method

1. A **VAE** learns a compressed latent space z from synthesizer audio
2. A **Normalizing Flow** (NF) learns an invertible mapping between latent space z and parameter space v
3. Disentangling flows split the objective for partial density evaluation, enforcing that specific latent dimensions match interpretable audio descriptors

The approach explicitly models that a synthesizer cannot reproduce every possible sound (unlike direct regression approaches).

## Key Results

- Superior parameter inference and audio reconstruction vs. MLP, CNN, and bidirectional LSTM baselines
- Learned macro-parameters correspond to interpretable audio qualities (brightness, richness)
- Smooth interpolation in latent space produces perceptually smooth parameter transitions

## Connections

- Builds on VAE ideas later refined in [preset-gen-vae](preset-gen-vae-2021.md)
- Latent space exploration contrasts with direct parameter prediction in [Barkan](barkan-deep-synth-pe-2017.md)
- Normalizing flows for audio synthesis relate to ../../ddsp/wiki/ concepts
- See concept: [sound matching approaches](../sound-matching-approaches.md)
