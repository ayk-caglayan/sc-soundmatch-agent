---
tags: [index, meta]
date: 2026-05-29
sources: 41
---

# synth-effect-matching -- Wiki Index

Content catalog for the synth-effect-matching knowledge base (41 papers).

## Special Pages

| Page | Summary |
|------|---------|
| [overview](overview.md) | Top-level synthesis: from linear regression to equivariant flows |

## Concept Pages

| Page | Summary | Tags |
|------|---------|------|
| [Synth Parameter Estimation](synth-parameter-estimation.md) | The core inverse problem: target sound to synth parameters | core, inverse-problem |
| [Sound Matching Approaches](sound-matching-approaches.md) | Taxonomy: black-box, white-box, DDSP, generative, RL, hybrid | taxonomy, methods |
| [Audio Similarity Metrics](audio-similarity-metrics.md) | Loss functions: parameter, spectral, JTFS, learned embeddings, PNP | loss-function, perceptual |
| [Plug-and-Play Priors](plug-and-play-priors.md) | PNP loss: linearized perceptual loss for 100x speedup | pnp-loss, efficiency |
| [RL for Sound Matching](reinforcement-learning-sound-matching.md) | Reinforcement learning for cross-domain matching | reinforcement-learning |
| [DRC Parameter Estimation](drc-parameter-estimation.md) | Cross-topic hub: DRC forward model, ballistics, inversion (Gorlow/Sun), blind estimation (Peladeau), grey-box modelling (Wright) | drc, inversion, ballistics |

## Source Summaries

### Foundational / Early Work

| Page | Paper | Year |
|------|-------|------|
| [su-rnn-strings-2002](sources/su-rnn-strings-2002.md) | Physical Modeling Recurrent Networks for Plucked Strings (Su & Liang) | 2002 |
| [sterling-bowed-string-2010](sources/sterling-bowed-string-2010.md) | Empirical Physical Modeling for Bowed String Instruments (Sterling & Bocko) | 2010 |
| [gorlow-drc-inversion-2013](sources/gorlow-drc-inversion-2013.md) | Model-Based Inversion of Dynamic Range Compression (Gorlow & Reiss) | 2013 |
| [itoyama-okuno-icmc14](sources/itoyama-okuno-icmc14.md) | Parameter Estimation of Virtual Musical Instrument Synthesizers (Itoyama & Okuno) | 2014 |

### Deep Learning for Parameter Prediction (2017--2020)

| Page | Paper | Year |
|------|-------|------|
| [barkan-deep-synth-pe-2017](sources/barkan-deep-synth-pe-2017.md) | Deep Synthesizer Parameter Estimation (Barkan & Tsiris) | 2017 |
| [inversynth-2018](sources/inversynth-2018.md) | InverSynth: Deep Estimation of Synthesizer Parameters (Barkan et al.) | 2018 |
| [neural-source-filter-2019](sources/neural-source-filter-2019.md) | Neural Source-Filter Waveform Models (Wang et al.) | 2019 |
| [flowsynth-2020](sources/flowsynth-2020.md) | Flow Synthesizer: Universal Audio Synthesizer Control (Esling et al.) | 2020 |
| [wav2shape-2020](sources/wav2shape-2020.md) | wav2shape: Hearing the Shape of a Drum Machine (Han & Lostanlen) | 2020 |

### DDSP and Differentiable Synthesis (2021--2023)

| Page | Paper | Year |
|------|-------|------|
| [masuda-ddsp-ismir21](sources/masuda-ddsp-ismir21.md) | Synthesizer Sound Matching with Differentiable DSP (Masuda & Saito) | 2021 |
| [deepafx-2021](sources/deepafx-2021.md) | Differentiable Signal Processing with Black-Box Audio Effects (DeepAFx) | 2021 |
| [preset-gen-vae-2021](sources/preset-gen-vae-2021.md) | Improving Synthesizer Programming from VAE Latent Space (Le Vaillant et al.) | 2021 |
| [shier-thesis-2021](sources/shier-thesis-2021.md) | The Synthesizer Programming Problem (Shier, M.Sc. thesis) | 2021 |
| [torchsynth-synth1b1-2021](sources/torchsynth-synth1b1-2021.md) | One Billion Audio Sounds / torchsynth (Turian, Shier et al.) | 2021 |
| [diff-tf-scattering-2022](sources/diff-tf-scattering-2022.md) | Differentiable Time-Frequency Scattering on GPU (Muradeli et al.) | 2022 |
| [sound2synth-2022](sources/sound2synth-2022.md) | SOUND2SYNTH: FM Synthesizer Parameters Estimation (Chen et al.) | 2022 |
| [mesostructures-2023](sources/mesostructures-2023.md) | Mesostructures: Beyond Spectrogram Loss (Vahidi, Han et al.) | 2023 |
| [han-pnp-icassp-2023](sources/han-pnp-icassp-2023.md) | Perceptual-Neural-Physical Sound Matching (Han et al., ICASSP) | 2023 |
| [masuda-saito-taslp-2023](sources/masuda-saito-taslp-2023.md) | Improving Semi-Supervised Differentiable Synth Matching (Masuda & Saito) | 2023 |
| [masuda-qd-2023](sources/masuda-qd-2023.md) | Quality-Diversity for Synthesizer Sound Matching (Masuda & Saito) | 2023 |
| [diaz-hayes-rigid-body-icassp23](sources/diaz-hayes-rigid-body-icassp23.md) | Rigid Body Sound Synthesis with Differentiable Modal Resonators | 2023 |
| [han-inverse-problem-2023](sources/han-inverse-problem-2023.md) | Learning to Solve Inverse Problems for PSM (Han et al., preprint) | 2023 |

### White-Box Methods (2023)

| Page | Paper | Year |
|------|-------|------|
| [yang-white-box-ismir23](sources/yang-white-box-ismir23.md) | White Box Search over Audio Synthesizer Parameters (Yang et al.) | 2023 |
| [yang-white-box-fm-2023](sources/yang-white-box-fm-2023.md) | White-Box FM Synthesis Parameter Estimation (Yang et al.) | 2023 |

### Current Frontiers (2024--2025)

| Page | Paper | Year |
|------|-------|------|
| [peladeau-blind-afx-2024](sources/peladeau-blind-afx-2024.md) | Blind Estimation of Audio Effects: Auto-Encoder + DDSP (Peladeau & Peeters) | 2024 |
| [sun-drc-neural-inversion-2024](sources/sun-drc-neural-inversion-2024.md) | Neural-Enhanced DRC Inversion, Hybrid (Sun, Fourer & Maaref) | 2024 |
| [peladeau-param-distributions-2025](sources/peladeau-param-distributions-2025.md) | Audio Processor Parameters: Estimating Distributions Instead of Deterministic Values (Peladeau et al.) | 2025 |
| [han-pnp-taslp-2024](sources/han-pnp-taslp-2024.md) | Learning to Solve Inverse Problems for PSM (Han et al., TASLP) | 2024 |
| [diffmoog-2024](sources/diffmoog-2024.md) | DiffMoog: Differentiable Modular Synthesizer (Uzrad, Barkan et al.) | 2024 |
| [bruford-ast-dafx-2024](sources/bruford-ast-dafx-2024.md) | Synth Sound Matching Using Audio Spectrogram Transformers (Bruford et al.) | 2024 |
| [han-thesis-2025](sources/han-thesis-2025.md) | Unearthing the Physicality of Instrumental Timbre (Han, Ph.D. thesis) | 2025 |
| [han-gradient-clipping-2025](sources/han-gradient-clipping-2025.md) | Gradient Clipping for PSM Optimization (Han et al., EUSIPCO) | 2025 |
| [hayes-equivariant-flow-2025](sources/hayes-equivariant-flow-2025.md) | Audio Synth Inversion with Equivariant Flow Matching (Hayes et al.) | 2025 |
| [synthrl-2025](sources/synthrl-2025.md) | SynthRL: Cross-domain Matching via RL (Shin & Lee) | 2025 |
| [sound-similarity-metrics-2025](sources/sound-similarity-metrics-2025.md) | Evaluating Sound Similarity Metrics (Salimi et al.) | 2025 |
| [neural-proxies-2025](sources/neural-proxies-2025.md) | Neural Proxies for Sound Synthesizers (Combes et al.) | 2025 |
| [modulation-discovery-2025](sources/modulation-discovery-2025.md) | Modulation Discovery with DDSP (Mitcheltree et al.) | 2025 |
| [fdn-reverb-matching-2025](sources/fdn-reverb-matching-2025.md) | Reverb Matching via Learned Embeddings + FDN (Gotz et al.) | 2025 |

### Not Yet Converted (PDF text file unavailable)

| PDF | Notes |
|-----|-------|
| ASA-2025-SynthMatch-handout | ASA 2025 conference overheads; text file not found in txt/ |
