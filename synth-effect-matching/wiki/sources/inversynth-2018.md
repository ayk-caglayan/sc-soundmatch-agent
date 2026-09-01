---
tags: [synth-parameter-estimation, cnn, intra-domain, foundational]
date: 2018-12-15
sources: 1
---

# InverSynth: Deep Estimation of Synthesizer Parameter Configurations from Audio Signals

**Authors:** Oren Barkan, David Tsiris, Noam Koenigstein, Ori Katz
**Year:** 2018 (arXiv 1812.06349)
**PDF:** InverSynth-1812.06349.txt

## Key Contributions

- Journal-length extension of [Barkan 2017](barkan-deep-synth-pe-2017.md) with expanded experiments
- Named the **InverSynth** framework: strided CNN for synth parameter inference
- Comprehensive comparison of spectrogram-based CNN, end-to-end CNN, FC with BoW, and FC with handcrafted features
- Confirmed depth as a key factor; log STFT input improves over linear STFT

## Method

Same 23-parameter JSyn synthesizer architecture. The paper provides more detailed architecture descriptions and ablations than the conference version. InverSynth uses binary cross-entropy loss over one-hot parameter encodings.

## Key Results

- Confirms spectrogram-based deep CNN superiority over all baselines
- End-to-end CNN outperforms handcrafted-feature FC networks but trails spectrogram CNN
- PCC reconstruction quality correlates with human-perceived similarity

## Connections

- Foundation for [DiffMoog](diffmoog-2024.md) (same group, Tel Aviv University)
- Motivates the need for audio-domain losses explored in [DDSP matching](masuda-ddsp-ismir21.md)
- See concept: [synth parameter estimation](../synth-parameter-estimation.md)
