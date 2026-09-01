---
tags: [concept, reinforcement-learning, cross-domain, reward, iterative]
date: 2026-04-06
sources: 2
---

# Reinforcement Learning for Sound Matching

Reinforcement learning (RL) offers an alternative paradigm for synth parameter estimation that avoids the differentiability requirement entirely.

## Motivation

Traditional approaches face a dilemma:
- **Parameter loss** requires ground-truth parameters (only available for in-domain sounds)
- **Spectral loss** requires a differentiable synthesizer (limits to DDSP reimplementations)

RL sidesteps both: the agent receives a **reward** based on audio similarity between its output and the target, without needing parameter labels or differentiable synthesis.

## Formulation ([SynthRL 2025](sources/synthrl-2025.md))

Sound matching as a Markov Decision Process:
- **State:** audio features of current synthesizer output and target
- **Action:** parameter adjustments to the synthesizer
- **Reward:** multi-scale spectral similarity between generated and target audio
- **Episode:** iterative refinement until convergence or max steps

Key innovations:
- **Transformer-based architecture** for processing audio state
- **Reward-based prioritized experience replay** for training efficiency
- **Fine-tuning on out-of-domain sounds** via reward signal only

## Advantages

1. Works with **any** synthesizer (no differentiability needed)
2. Naturally handles **cross-domain** generalization
3. **Iterative refinement** -- can progressively improve matches
4. Reward function can incorporate arbitrary perceptual measures

## Limitations

1. Training is less sample-efficient than supervised learning
2. Exploration in high-dimensional parameter spaces is challenging
3. Reward design requires careful engineering
4. Slower than single-pass feedforward prediction at inference

## Connection to Other Approaches

- RL reward function design relates to [audio similarity metrics](audio-similarity-metrics.md)
- Iterative refinement echoes the hybrid neural+GA approach in [Shier thesis](sources/shier-thesis-2021.md)
- Complements generative approaches like [equivariant flow matching](sources/hayes-equivariant-flow-2025.md) which also handle multi-modal solutions

## Related Concepts

- [Synth Parameter Estimation](synth-parameter-estimation.md)
- [Sound Matching Approaches](sound-matching-approaches.md)
- [Audio Similarity Metrics](audio-similarity-metrics.md)
