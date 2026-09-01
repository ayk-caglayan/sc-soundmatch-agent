---
tags: [synth-inversion, flow-matching, symmetry, permutation-equivariance, generative-model, surge-xt]
date: 2025-06-08
sources: 1
---

# Audio Synthesizer Inversion in Symmetric Parameter Spaces with Approximately Equivariant Flow Matching

**Authors:** Ben Hayes, Charalampos Saitis, Gyorgy Fazekas
**Venue:** ISMIR 2025
**PDF:** SynthMatch-Hayes-2506.07199v1.txt

## Key Contributions

- Identified that synthesizer inversion is **ill-posed** due to intrinsic symmetries (especially permutation invariance of oscillators)
- Showed that regressing point estimates under permutation symmetry degrades performance, even with permutation-invariant losses or symmetry-breaking heuristics
- Proposed viewing equivalent solutions as **modes of a probability distribution** and using a **conditional generative model** (flow matching)
- Introduced **permutation equivariant continuous normalizing flow** for improved performance
- Proposed **relaxed equivariance** strategy that adaptively discovers symmetries from data
- Evaluated on **Surge XT**, a full-featured open-source synthesizer used in real production

## Method

1. Frame synth inversion as conditional density estimation p(theta | audio)
2. Use continuous normalizing flows (flow matching) as the generative model
3. Enforce approximate permutation equivariance in the flow architecture
4. Relaxed equivariance allows the model to discover and exploit symmetries without requiring exact group specification

## Key Results

- Generative approach (flow matching) substantially outperforms regression baselines
- Permutation equivariant flow further improves over non-equivariant generative model
- Relaxed equivariance handles complex real-world symmetries in Surge XT
- State-of-the-art audio reconstruction metrics on a production synthesizer

## Connections

- Addresses the multi-solution problem also noted in [SynthRL](synthrl-2025.md)
- Flow-based approach related to [Flow Synthesizer](flowsynth-2020.md) but with equivariance
- Surge XT evaluation represents a step toward practical synthesizer matching
- See concept: [sound matching approaches](../sound-matching-approaches.md)
