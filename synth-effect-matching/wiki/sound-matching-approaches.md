---
tags: [concept, taxonomy, black-box, white-box, ddsp, methods, inversion, blind-estimation]
date: 2026-05-29
sources: 41
---

# Sound Matching Approaches

A taxonomy of methods for [synth parameter estimation](synth-parameter-estimation.md), organized by how they handle the synthesizer.

## Taxonomy

### 1. Black-Box Methods

The synthesizer is treated as an opaque function. No gradients flow through it.

**a) Direct parameter prediction (feedforward)**
- Train a neural network to map audio features to parameters in a single forward pass
- Loss is computed in parameter space only
- Fast at inference, but limited by parameter loss inadequacy
- Examples: [Barkan 2017](sources/barkan-deep-synth-pe-2017.md), [InverSynth 2018](sources/inversynth-2018.md), [Sound2Synth 2022](sources/sound2synth-2022.md), [AST matching](sources/bruford-ast-dafx-2024.md)

**b) Genetic algorithms**
- Evolutionary search over parameter space, evaluating fitness by rendering audio
- Expensive (thousands of synthesizer renders per target) but flexible
- Can work with any synthesizer, no training data needed
- Examples: GA approaches in [Shier thesis](sources/shier-thesis-2021.md), [Quality-Diversity](sources/masuda-qd-2023.md)

**c) Neural proxies**
- Train a differentiable approximation of the synthesizer (or effect)
- Enables gradient-based optimization without modifying the synthesizer
- [Neural Proxies 2025](sources/neural-proxies-2025.md): map presets to pretrained audio embeddings
- [Peladeau & Peeters 2024](sources/peladeau-blind-afx-2024.md): a FiLM-conditioned TCN proxy for a DSP compressor; the "Hybrid NP" estimates parameters with the proxy but renders with the real DSP effect

**d) Stochastic gradient approximation**
- Estimate gradients through black-box effects via random perturbations
- [DeepAFx 2021](sources/deepafx-2021.md): parallel stochastic gradient scheme

**e) Reinforcement learning**
- Frame matching as sequential decision-making; reward = audio similarity
- No gradient through synthesizer needed; can train on out-of-domain sounds
- [SynthRL 2025](sources/synthrl-2025.md)

### 2. White-Box Methods

The synthesizer source code is available and directly differentiated.

**a) Differentiable rendering adaptation**
- Borrow techniques from computer graphics to differentiate through discontinuities
- The A-delta method handles sawtooth, square waves, and categorical parameters
- No training data or neural proxy needed; pure optimization at inference time
- [Yang et al. ISMIR 2023](sources/yang-white-box-ismir23.md)

### 3. DDSP-Based Methods

The synthesizer is **reimplemented** as a differentiable program, enabling end-to-end training.

**a) Differentiable synthesizer + spectral loss**
- Implement synth in PyTorch/JAX; backpropagate audio loss through synthesis
- Enables semi-supervised training on out-of-domain sounds
- [Masuda & Saito ISMIR 2021](sources/masuda-ddsp-ismir21.md), [TASLP 2023](sources/masuda-saito-taslp-2023.md)

**b) Differentiable modular synthesizers**
- Full modular signal chain (FM/AM, LFOs, filters, envelopes) made differentiable
- [DiffMoog 2024](sources/diffmoog-2024.md)

**c) PNP-accelerated DDSP**
- Precompute Jacobian of (synth o perception) to create fast quadratic loss approximation
- 100x speedup over naive DDSP backpropagation
- [Han PNP TASLP 2024](sources/han-pnp-taslp-2024.md)

**d) Modulation-aware DDSP**
- Discover structured modulation signals (envelopes, LFOs) rather than framewise parameters
- [Modulation Discovery 2025](sources/modulation-discovery-2025.md)

### 4. Generative / Latent-Space Methods

Learn an organized representation of the synthesizer's audio capabilities.

- VAE latent space + normalizing flows for invertible audio-to-parameter mapping
- [Flow Synthesizer 2020](sources/flowsynth-2020.md), [Preset VAE 2021](sources/preset-gen-vae-2021.md)
- Conditional flow matching with symmetry awareness: [Hayes 2025](sources/hayes-equivariant-flow-2025.md)

### 5. Hybrid Methods

Combine multiple approaches.

- **Neural warm-start + GA refinement**: [Shier thesis 2021](sources/shier-thesis-2021.md)
- **Parameter loss pretraining + spectral loss fine-tuning**: [Masuda & Saito](sources/masuda-ddsp-ismir21.md)
- **PNP precomputation + neural prediction**: [Han et al.](sources/han-pnp-taslp-2024.md)

### 6. Model-Based Inversion & Blind Effect Estimation

A distinct branch when the target is an **audio effect** with a known DSP structure (e.g. a compressor), rather than a synthesizer.

**a) Model-based inversion** — given the effect's parameters, *analytically invert* the forward model to recover the dry signal. [Gorlow & Reiss 2013](sources/gorlow-drc-inversion-2013.md) invert DRC via a per-sample characteristic-function root-search; [Sun et al. 2024](sources/sun-drc-neural-inversion-2024.md) keep that inversion but estimate the parameters first with a neural network (AST classifier / MEE regressor).

**b) Blind estimation** — recover effect + parameters from the wet signal alone, with no dry reference, by training an auto-encoder on an audio loss ([Peladeau & Peeters 2024](sources/peladeau-blind-afx-2024.md)).

**c) Distributional estimation** — because the parameter→sound map is many-to-one, return a *distribution* of valid parameter sets via normalizing flows + entropy maximisation ([Peladeau et al. 2025](sources/peladeau-param-distributions-2025.md)). Compare the conditional-generative view of [Hayes 2025](sources/hayes-equivariant-flow-2025.md).

See the cross-topic hub [DRC parameter estimation](drc-parameter-estimation.md) for the shared forward model and the grey-box modelling counterpart ([Wright & Välimäki 2022](../../ddsp/wiki/sources/wright2022-greybox-drc.md)).

## Comparison

| Approach | Needs training data? | Needs differentiable synth? | Cross-domain? | Speed |
|----------|---------------------|---------------------------|---------------|-------|
| Black-box NN | Yes (parameter pairs) | No | Limited | Fast inference |
| Genetic algorithm | No | No | Yes | Very slow |
| Neural proxy | Yes (audio pairs) | Proxy only | Possible | Medium |
| RL | Yes (reward signal) | No | Yes | Medium |
| White-box | No | Source code needed | Yes | Medium |
| DDSP | Yes (optional for spectral loss) | Yes (reimplemented) | Yes | Fast training |
| Generative | Yes | No | Possible | Fast inference |

## Open Problems

- Scaling to commercial synthesizers with hundreds of parameters
- Cross-domain generalization to arbitrary target sounds
- Real-time interactive parameter estimation
- Handling temporal evolution (modulation, envelopes) vs. static tones
- Perceptually meaningful evaluation metrics

## Related Concepts

- [Synth Parameter Estimation](synth-parameter-estimation.md) -- the problem definition
- [Audio Similarity Metrics](audio-similarity-metrics.md) -- the loss functions
- [Reinforcement Learning for Sound Matching](reinforcement-learning-sound-matching.md)
- [Plug-and-Play Priors](plug-and-play-priors.md)
- [DRC Parameter Estimation](drc-parameter-estimation.md) -- effect inversion, blind & distributional estimation
