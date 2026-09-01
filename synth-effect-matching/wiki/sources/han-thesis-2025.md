---
tags: [thesis, perceptual-sound-matching, physical-modeling, physicality, inverse-problem, plucked-string, guqin]
date: 2025-03-10
sources: 1
---

# Unearthing the Physicality of Instrumental Timbre (Han Han, Ph.D. Thesis 2025)

**Author:** Han Han
**Institution:** Ecole Centrale de Nantes / LS2N / CNRS
**Supervisors:** Mathieu Lagrange, Vincent Lostanlen
**Defense:** March 10, 2025
**PDF:** SynthMatch-HanHanT-2025-03.txt

## Key Contributions

- Doctoral thesis investigating how **physicality** is implicit in instrumental sound
- Three contributions at different levels of realism:
  1. **Physical model parameter regression from synthetic percussion:** Introduces PNP loss for accelerating perceptually relevant loss functions (→ [TASLP 2024 paper](han-pnp-taslp-2024.md))
  2. **Physical parameter extraction from plucked strings (artificial player):** Addresses sim-to-real transfer under data scarcity
  3. **Audiovisual dataset of guqin playing techniques:** Computational identification of gesture-sound correlations from human performances

## Method

The thesis adopts both physical-model simulation and real-world data acquisition approaches. The key technical contribution (PNP loss) linearizes the composition of synthesizer and perceptual representation to create efficient training objectives for neural parameter estimators. The later chapters extend to real-world scenarios with domain adaptation challenges.

## Key Results

- PNP loss achieves 100x speedup over full DDSP backpropagation
- Sim-to-real transfer demonstrated for plucked string physical model parameter estimation
- Guqin audiovisual dataset enables gesture-sound analysis for traditional instruments

## Connections

- Wraps up the PNP line of work: [ICASSP 2023](han-pnp-icassp-2023.md), [TASLP 2024](han-pnp-taslp-2024.md), [gradient clipping](han-gradient-clipping-2025.md)
- Physical modeling synthesis connects to [bowed string modeling](sterling-bowed-string-2010.md)
- See concept: [synth parameter estimation](../synth-parameter-estimation.md), [audio similarity metrics](../audio-similarity-metrics.md)
