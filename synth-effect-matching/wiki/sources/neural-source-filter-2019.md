---
tags: [speech-synthesis, source-filter, wavenet, neural-waveform]
date: 2019-11-17
sources: 1
---

# Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis (Wang et al., 2019)

**Authors:** Xin Wang, Shinji Takaki, Junichi Yamagishi
**Year:** 2019 (arXiv 1904.12088v2)
**PDF:** SynthMatch-NeuralSourceFilterModel-1904.12088v2.txt

## Summary

Neural waveform models combining source-filter theory with neural networks for speech synthesis. Proposes models that generate waveforms in a one-shot manner (avoiding WaveNet's sequential generation) while maintaining quality. Uses a source excitation signal filtered by a neural network, bridging classical DSP source-filter models and modern neural synthesis.

## Connections

- Source-filter model is a classical DSP concept relevant to synthesizer architecture
- Neural waveform generation relates to [DDSP](../index.md) approaches
- Speech synthesis perspective on the analysis-synthesis problem
