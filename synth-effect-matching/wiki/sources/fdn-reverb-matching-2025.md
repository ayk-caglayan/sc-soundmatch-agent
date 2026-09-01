---
tags: [reverb-matching, fdn, feedback-delay-network, augmented-reality, room-acoustics]
date: 2025-10-27
sources: 1
---

# Matching Reverberant Speech through Learned Acoustic Embeddings and Feedback Delay Networks

**Authors:** Philipp Gotz, Gloria Dal Santo, Sebastian J. Schlecht, Vesa Valimaki, Emanuel A. P. Habets
**Venue:** arXiv 2510.23158v1 (2025)
**PDF:** FDN-ReverbMatching-2510.23158v1.txt

## Key Contributions

- Formulated blind estimation of artificial reverberation parameters as a **reverberant signal matching task**
- Leveraged a **learned room-acoustic prior** (embedding network) for parameter estimation
- Proposed an FDN structure that reproduces both **frequency-dependent decay times** and **direct-to-reverberation ratio**
- Targets auditory augmented reality (AAR) applications

## Method

A learned embedding network extracts room-acoustic features from reverberant speech. These embeddings guide parameter estimation for a feedback delay network (FDN) that generates matching artificial reverberation in real time. The FDN structure is designed to reproduce perceptually important room-acoustic features.

## Key Results

- Improvements over leading automatic FDN tuning methods
- Better estimated room-acoustic parameters and perceptual plausibility
- Suitable for efficient real-time rendering in AAR

## Connections

- Extends sound matching from synthesizers to **room acoustics/reverberation**
- FDN relates to physical modeling in JOS's digital audio signal processing framework
- See concept: [synth parameter estimation](../synth-parameter-estimation.md) (generalized to effects)
