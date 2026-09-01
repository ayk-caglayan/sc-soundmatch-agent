---
tags: [reinforcement-learning, cross-domain, sound-matching, transformer, reward-design]
date: 2025-08-01
sources: 1
---

# SynthRL: Cross-domain Synthesizer Sound Matching via Reinforcement Learning

**Authors:** Wonchul Shin, Kyogu Lee
**Venue:** IJCAI 2025
**PDF:** SynthRL-IJCAI-25.txt

## Key Contributions

- First application of **reinforcement learning (RL)** to synthesizer sound matching
- Solves the **cross-domain** generalization problem: train on out-of-domain sounds without ground-truth parameters
- Sound similarity incorporated directly into the **reward function**, bypassing the non-differentiability of conventional synthesizers
- Introduced **transformer-based model architecture** for RL-based parameter estimation
- Proposed **reward-based prioritized experience replay** for training efficiency

## Method

SynthRL frames sound matching as a Markov Decision Process:
- **State:** current audio features
- **Action:** synthesizer parameter adjustments
- **Reward:** sound similarity between generated and target audio (using multi-scale spectral features)

The RL formulation allows fine-tuning on out-of-domain sounds without requiring parameter labels. A transformer architecture processes the audio embedding and iteratively refines parameters.

## Key Results

- Outperforms SOTA on both in-domain and out-of-domain tasks
- Reward function correlates strongly with human perception of sound similarity
- Cross-domain capability demonstrated on real instrument sounds matched to synthesizer

## Connections

- Addresses the same cross-domain problem as [DDSP matching](masuda-ddsp-ismir21.md) but without requiring differentiable synthesizer
- Iterative refinement echoes the hybrid approach in [Shier thesis](shier-thesis-2021.md)
- Contrasts with single-pass prediction in [InverSynth](inversynth-2018.md)
- See concept: [reinforcement learning for sound matching](../reinforcement-learning-sound-matching.md)
