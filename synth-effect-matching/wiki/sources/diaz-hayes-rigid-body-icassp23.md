---
tags: [physical-modeling, rigid-body, modal-synthesis, iir-filters, ddsp]
date: 2023-06-04
sources: 1
---

# Rigid Body Sound Synthesis with Differentiable Modal Resonators (Diaz, Hayes et al., ICASSP 2023)

**Authors:** Rodrigo Diaz, Ben Hayes et al.
**Venue:** ICASSP 2023
**PDF:** SynthMatch-DDSP-Diaz-Hayes-RigidBodyModal-ICASSP23.txt

## Summary

End-to-end framework training a DNN to generate **modal resonators** for a given 2D shape and material using differentiable IIR filter banks. Paves the way for learning physically-informed synthesizers from real-world recordings. Applied to synthetic rigid body objects.

## Connections

- Audio-domain training objective relates to [DDSP matching](masuda-ddsp-ismir21.md)
- Physical modeling focus shared with [Han thesis](han-thesis-2025.md)
- IIR filter differentiability also addressed in [DiffMoog](diffmoog-2024.md)
