---
tags: [synth-parameter-estimation, fm-synthesis, dexed, multi-modal, deep-learning]
date: 2022-07-28
sources: 1
---

# SOUND2SYNTH: Interpreting Sound via FM Synthesizer Parameters Estimation

**Authors:** Zui Chen, Yansen Jing, Shengcheng Yuan, Yifei Xu, Jian Wu, Hang Zhao
**Venue:** arXiv 2205.03043v2 (2022)
**PDF:** Sound2Synth-2205.03043v2.txt

## Key Contributions

- First real-world applicable results on the **Dexed** synthesizer (DX7 FM synth clone)
- Proposed **Prime-Dilated Convolution (PDC)** network structure for parameter estimation
- Multi-modal pipeline combining spectral and temporal features
- Achieved SOTA on Dexed, a complex FM synthesizer with 155 parameters

## Method

SOUND2SYNTH uses a multi-modal approach with both spectral (mel-spectrogram) and waveform inputs. The PDC architecture uses dilated convolutions with prime-number dilation rates to capture multi-scale temporal patterns without redundancy. Parameters are predicted as a regression task.

## Key Results

- First method to produce perceptually acceptable matches on the full Dexed synthesizer
- PDC outperforms standard CNN and RNN architectures on FM synthesis parameter estimation
- Released code, VST plugin, and demo materials

## Connections

- Dexed/DX7 also targeted by [preset-gen-vae](preset-gen-vae-2021.md)
- FM synthesis parameter estimation also addressed by [white-box methods](yang-white-box-fm-2023.md)
- See concept: [synth parameter estimation](../synth-parameter-estimation.md)
