---
tags: [optimization, gradient-clipping, perceptual-sound-matching, adam]
date: 2025-01-01
sources: 1
---

# Gradient Clipping Improves Neural Network Optimization for Perceptual Sound Matching (Han et al., EUSIPCO 2025)

**Authors:** Han Han, Vincent Lostanlen, Mathieu Lagrange
**Venue:** EUSIPCO 2025
**PDF:** SynthMatch-GradientClipping-han2025eusipco.txt

## Summary

Shows that Adam optimizer is unsuited for PSM's non-stationary multi-stage training objectives. Demonstrates that **weight decay + gradient clipping** improves convergence probability and generalization. Analyzes gradient norm and gradient roughness evolution under different optimization setups.

## Connections

- Companion to [Han PNP TASLP 2024](han-pnp-taslp-2024.md)
- Part of [Han thesis](han-thesis-2025.md)
