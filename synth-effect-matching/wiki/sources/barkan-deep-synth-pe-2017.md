---
tags: [synth-parameter-estimation, cnn, classification, foundational]
date: 2019-05-12
sources: 1
---

# Deep Synthesizer Parameter Estimation (Barkan & Tsiris, ICASSP 2019)

**Authors:** Oren Barkan, David Tsiris
**Venue:** ICASSP 2019
**PDF:** SynthMatch_Barkan_DeepSynthPE_ICASSP-2017.txt

## Key Contributions

- First deep learning approach to synthesizer parameter estimation using strided CNNs
- Demonstrated both spectrogram-based and end-to-end (raw audio) inference pipelines
- Formulated parameter estimation as a **classification** problem (16 discrete levels per parameter) rather than regression, finding this easier to optimize
- Showed that **network depth** is the critical factor for prediction accuracy
- Built a 23-parameter subtractive + FM synthesizer in JSyn with 200K training samples

## Method

A custom subtractive/FM synthesizer with 4 oscillators, ADSR envelope, resonant low-pass filter, and LFO gater produces 23 parameters. Each parameter is quantized to 16 levels (classification). Two CNN pipelines are compared:

1. **Spectrogram CNN (Conv1--Conv6):** STFT matrix input, 2D strided convolutions (no pooling)
2. **End-to-end CNN (ConvE2E):** Raw audio input, 1D conv layers learn STFT-like transform, followed by 2D conv layers

Baselines include FC networks with BoW spectrograms and handcrafted features from Itoyama & Okuno (2014).

## Key Results

- Conv6 achieves 92% mean STFT PCC and 74% mean FT PCC on reconstruction
- Spectrogram-based CNNs significantly outperform end-to-end and FC baselines
- Depth saturates around 5--6 layers for CNNs; increasing parameters from 1.2M to 2.3M has negligible effect
- CNNs show no overfitting (weight sharing), while FC models overfit after ~20 epochs

## Limitations

- Intra-domain only (same synthesizer for train and test)
- Synthesizer is not differentiable; no audio-domain loss
- Custom toy synthesizer, not a commercial instrument

## Connections

- Extended in [InverSynth](inversynth-2018.md) (same authors, journal version)
- Baseline compared against in most subsequent work
- Formulation as classification vs regression debate continued in [Shier thesis](shier-thesis-2021.md)
- The non-differentiable synthesizer limitation motivates [DDSP-based matching](masuda-ddsp-ismir21.md)
- See concept: [synth parameter estimation](../synth-parameter-estimation.md)
