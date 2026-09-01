---
tags: [differentiable-synthesis, modular-synthesizer, sound-matching, signal-chain-loss, open-source]
date: 2024-01-23
sources: 1
---

# DiffMoog: A Differentiable Modular Synthesizer for Sound Matching

**Authors:** Noy Uzrad, Oren Barkan, Almog Elharar, Shlomi Shvartzman, Moshe Laufer, Lior Wolf, Noam Koenigstein
**Venue:** arXiv 2401.12570 (2024)
**PDF:** DiffMoog-2401.12570.txt

## Key Contributions

- **DiffMoog**: a differentiable modular synthesizer with comprehensive modules (FM/AM, LFOs, filters, envelopes, custom signal chains)
- Novel **signal-chain loss** that accounts for modular architecture during training
- **Encoder network** that self-programs outputs to predict parameters based on user-defined modular architecture
- Open-source platform combining DiffMoog with end-to-end sound matching framework
- Practical insights and lessons learned for differentiable synthesis research

## Method

DiffMoog implements the full signal chain of a commercial-grade modular synthesizer in a differentiable framework:
- Oscillators with FM/AM modulation capabilities
- LFOs, ADSR envelope shapers, resonant filters
- Users can define custom signal chain topologies
- An encoder neural network analyzes input audio and predicts parameters for the user-defined architecture

The signal-chain loss considers the hierarchical structure of the modular architecture.

## Key Results

- Matches the capabilities of real Moog-style modular synthesizers
- End-to-end training with audio-domain objectives
- Released as open-source research platform

## Connections

- Same research group as [InverSynth](inversynth-2018.md) (Barkan, Koenigstein at Tel Aviv)
- Builds on [DDSP concepts](../index.md) but with modular synthesis
- Modular architecture explored further in [modulation discovery](modulation-discovery-2025.md)
- See concept: [sound matching approaches](../sound-matching-approaches.md)
