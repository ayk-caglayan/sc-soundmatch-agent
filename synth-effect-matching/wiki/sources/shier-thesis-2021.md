---
tags: [synth-parameter-estimation, thesis, genetic-algorithm, hybrid, spiegelib, torchsynth, survey]
date: 2021-01-01
sources: 1
---

# The Synthesizer Programming Problem: Improving the Usability of Sound Synthesizers

**Author:** Jordie Shier (M.Sc. thesis, University of Victoria)
**Supervisors:** George Tzanetakis, Kirk McNally
**Year:** 2021
**PDF:** SynthMatch_JordieShier2021synthesizer_thesis.txt

## Key Contributions

- Comprehensive thesis covering the synthesizer programming problem from multiple angles
- Compared **deep learning** and **evolutionary programming** (genetic algorithms) for inverse FM synthesis
- Proposed a **hybrid** approach combining neural network warm-start with genetic refinement, achieving high quality in less than half the computation of pure GA
- Developed **SpiegeLib**: open-source library for reproducible inverse synthesis evaluation
- Developed **Synth Explorer**: a novel 2D visual interface for exploring synthesizer sounds
- Co-created **torchsynth** and **synth1B1** dataset (1 billion synthesized sounds)

## Method

The thesis addresses five research questions:
1. Automatic synthesizer programming
2. Inverse synthesis (parameter estimation)
3. Representing synthesized sounds (2D layout)
4. Generating synthesized sounds
5. Developing supportive tools

For inverse FM synthesis, the hybrid approach uses a CNN to produce an initial parameter estimate, then refines it with a genetic algorithm optimizing spectral features.

## Key Results

- Hybrid neural + GA approach outperforms pure GA in speed and matches quality
- SpiegeLib enables standardized benchmarking across methods
- Synth Explorer interface received positive user evaluations from novice synthesizer users

## Connections

- SpiegeLib and torchsynth became standard tools used by [Masuda & Saito](masuda-ddsp-ismir21.md), [SynthRL](synthrl-2025.md), and others
- Hybrid approach anticipates the iterative refinement in [SynthRL](synthrl-2025.md)
- See also [torchsynth/synth1B1](torchsynth-synth1b1-2021.md)
- See concept: [sound matching approaches](../sound-matching-approaches.md)
