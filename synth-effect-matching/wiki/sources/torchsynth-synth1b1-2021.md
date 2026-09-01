---
tags: [dataset, gpu-synthesis, modular-synthesizer, benchmark, open-source]
date: 2021-07-20
sources: 1
---

# One Billion Audio Sounds from GPU-Enabled Modular Synthesis (Turian, Shier et al., DAFx 2021)

**Authors:** Joseph Turian, Jordie Shier, George Tzanetakis, Kirk McNally, Max Henry
**Venue:** DAFx 2021
**PDF:** SynthMatch-TorchSynth-OneBillionAudioSounds-2104.12922v2.txt

## Summary

Released **synth1B1**, a 1-billion-sample audio corpus (100x larger than any prior audio dataset), and **torchsynth**, an open-source GPU-enabled modular synthesizer generating samples at 714 MHz (16200x real-time). Also released FM synth timbre and subtractive synth pitch evaluation datasets. Proposed rank-based evaluation criteria for audio representations and synthesizer hyperparameter optimization.

## Connections

- Created by [Shier](shier-thesis-2021.md) and collaborators
- Standard infrastructure used by many subsequent sound matching papers
- GPU synthesis relevant to training efficiency in [DiffMoog](diffmoog-2024.md), [SynthRL](synthrl-2025.md)
